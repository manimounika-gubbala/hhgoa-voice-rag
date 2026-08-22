"""
Runs every query in eval/queries.json through the pipeline N times each
(default 3, to smooth out first-call jitter) and reports P50/P70/P100
latency — for total_ms AND post_asr_ms separately, so you can honestly
report "sub-200ms retrieval+generation" without conflating it with ASR
time, which scales with audio length rather than engineering quality.

Usage:
  python scripts/benchmark.py                      # text queries, no ASR
  python scripts/benchmark.py --audio-dir eval/audio  # voice queries, full pipeline
"""
import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table

from app.config import settings
from app.pipeline import run_pipeline
from app.vectorstore import VectorStore

console = Console()


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    if f == c:
        return data[f]
    return data[f] + (data[c] - data[f]) * (k - f)


def report(name: str, values: list[float]):
    table = Table(title=name)
    table.add_column("P50 (ms)")
    table.add_column("P70 (ms)")
    table.add_column("P100 / max (ms)")
    table.add_column("mean (ms)")
    table.add_row(
        f"{percentile(values, 50):.1f}",
        f"{percentile(values, 70):.1f}",
        f"{percentile(values, 100):.1f}",
        f"{statistics.mean(values):.1f}" if values else "0.0",
    )
    console.print(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default=None, help="dir of .wav files named to match query ids")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    with open("eval/queries.json") as f:
        queries = json.load(f)

    store = VectorStore.load(settings.index_dir)

    total_ms_all, post_asr_ms_all, asr_ms_all = [], [], []
    abstain_count = 0
    error_count = 0
    rows = []

    for q in queries:
        for _ in range(args.repeats):
            audio_path = None
            if args.audio_dir:
                candidates = glob.glob(os.path.join(args.audio_dir, f"{q['id']}.*"))
                audio_path = candidates[0] if candidates else None

            resp = run_pipeline(
                store,
                audio_path=audio_path,
                query_text=None if audio_path else q["text"],
            )

            total_ms_all.append(resp.latency.total_ms)
            post_asr_ms_all.append(resp.latency.post_asr_ms)
            if resp.latency.asr_ms:
                asr_ms_all.append(resp.latency.asr_ms)
            if resp.abstained:
                abstain_count += 1
            if resp.error:
                error_count += 1

            rows.append({
                "id": q["id"], "query": resp.query_text, "answer": resp.answer[:120],
                "abstained": resp.abstained, "total_ms": round(resp.latency.total_ms, 1),
                "error": resp.error,
            })

    console.print(f"\n[bold]{len(queries)} queries x {args.repeats} repeats "
                   f"= {len(rows)} runs[/bold]")
    console.print(f"Abstained: {abstain_count}/{len(rows)}   Errors: {error_count}/{len(rows)}\n")

    report("Full pipeline (total_ms)", total_ms_all)
    report("Post-ASR / retrieval+generation only (post_asr_ms)", post_asr_ms_all)
    if asr_ms_all:
        report("ASR stage alone (asr_ms)", asr_ms_all)

    with open("eval/benchmark_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    console.print("\nPer-query results written to eval/benchmark_results.json")


if __name__ == "__main__":
    main()
