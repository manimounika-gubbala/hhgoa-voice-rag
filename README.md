# Voice-Enabled RAG — #RAGInGoa

Speak a question, get a grounded answer, backed by an actual retrieval
pipeline (not one embedding call and a prayer).

## Architecture

```
mic audio ──▶ ASR (faster-whisper) ──▶ query text
                                            │
                                            ▼
                              embed query (MiniLM, cosine)
                                            │
                                            ▼
                         FAISS top-k retrieval (flat index, exact)
                                            │
                                            ▼
                    cross-encoder rerank (ms-marco-MiniLM)
                                            │
                                            ▼
                    guardrail: abstain if score < threshold
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                            ▼
                   generate (Claude Haiku,          "I don't have enough
                   grounded prompt, retries)          information to answer
                              │                       that confidently."
                              ▼
                      structured RAGResponse
                (answer + citations + latency breakdown)
```

Every stage is timed independently (`app/schemas.py: LatencyBreakdown`)
and every stage that can fail is caught and degrades to a safe abstain
instead of a crash (`app/pipeline.py`).

## Why these choices

- **faster-whisper, int8, small.en by default** — CTranslate2 backend is
  noticeably faster than stock `openai-whisper` at equivalent accuracy.
  Swap to `tiny.en` if you need more speed than accuracy, or `device="cuda"`
  if you get GPU access at the venue.
- **4 real chunking strategies** (`app/chunking.py`): fixed, recursive,
  sentence-window, semantic (embedding similarity-drop topic segmentation).
  Switch via `CHUNK_STRATEGY` env var and re-run `scripts/ingest.py` —
  useful for an A/B slide in your pitch.
- **FAISS flat + cosine** — exact search, no ANN recall loss, plenty fast
  at hackathon corpus scale (thousands of chunks).
- **Cross-encoder rerank** — cosine similarity from a bi-encoder is a
  coarse first pass; the cross-encoder actually reads query+passage
  together and is much better at judging "does this passage answer the
  question," which is also what the guardrail leans on.
- **Guardrail before generation, not after** — abstaining early both
  saves an LLM call (latency + cost) and is the philosophically correct
  place to decide "do we even have grounds to answer."
- **Small/fast generation model** — Haiku-class latency, not Opus-class
  reasoning. RAG over a few short passages doesn't need a large model.

## Honest note on "under 200ms"

Full voice-to-answer latency is dominated by ASR, which scales with
audio length — a 4-second question takes meaningfully longer to
transcribe than 200ms on any model that fits in a hackathon laptop/GPU.
**What's realistically sub-200ms is the retrieval+rerank+generation
stage after transcription** (`post_asr_ms` in the latency breakdown).
`scripts/benchmark.py` reports both numbers separately so you can make
an honest, still-impressive claim: "sub-200ms retrieval-to-answer once
the question is transcribed" beats an inflated number a judge can
immediately disprove by timing your live demo.

If you want to genuinely chase full end-to-end speed, the real levers
are: `tiny.en` Whisper + GPU, streaming ASR with speculative retrieval
on partial transcripts (`app/asr.py: transcribe_stream`), and caching
query embeddings for repeated demo questions.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

# 1. Build the index (drop your own .txt files into data/sample_docs/ first)
python scripts/ingest.py

# 2. Run the server
uvicorn app.server:app --reload --port 8000

# 3. Open frontend/index.html in a browser (serve it or open directly),
#    hold the button, ask a question.
```

## Benchmarking (P50 / P70 / P100)

```bash
# text-only queries — isolates retrieval+generation from ASR
python scripts/benchmark.py --repeats 5

# full voice pipeline — put .wav files named q1.wav, q2.wav... in eval/audio/
python scripts/benchmark.py --audio-dir eval/audio --repeats 3
```

Results print to console and get written to `eval/benchmark_results.json`
per-query, so you can screenshot a percentile table straight into slides.

## Try a different chunking strategy

```bash
CHUNK_STRATEGY=semantic python scripts/ingest.py
CHUNK_STRATEGY=sentence_window python scripts/ingest.py
python scripts/benchmark.py   # compare P50s across strategies
```

## Project layout

```
app/
  asr.py          voice -> text (faster-whisper)
  chunking.py     4 chunking strategies
  vectorstore.py  FAISS + embeddings
  retrieval.py    retrieve, rerank, guardrail
  generation.py   grounded LLM call, manual retry loop
  pipeline.py     the harness — wires every stage together, timed & error-safe
  schemas.py      structured I/O (pydantic) through the whole system
  server.py       FastAPI endpoints
scripts/
  ingest.py       build the index
  benchmark.py    P50/P70/P100 latency report
eval/
  queries.json    sample eval set (includes one intentionally off-topic
                   query to demo the guardrail abstaining)
frontend/
  index.html      hold-to-talk mic recorder, zero build step
```

#RAGInGoa
