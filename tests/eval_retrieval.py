"""
Retrieval evaluation — benchmarks all three chunking strategies.

Metrics:
  Recall@k  — fraction of questions where at least one expected substring
               appears in the top-k retrieved chunks. Measures coverage.
  MRR       — Mean Reciprocal Rank. Rewards finding the right chunk early.
               MRR=1.0 means the relevant chunk is always ranked #1.
               MRR=0.5 means it's ranked #2 on average.

Run:
    python -m tests.eval_retrieval
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingestion_service import run_ingestion
from core.embedder import embed_query
from core.faiss_store import build_index, load_index, search
from core.config_loader import load_config

GROUND_TRUTH_PATH = Path("tests/ground_truth.json")
STRATEGIES        = ["fixed", "sentence", "sliding"]


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def is_hit(chunks: list[dict], expected_substrings: list[str]) -> bool:
    """Return True if ANY expected substring appears in ANY retrieved chunk."""
    combined = " ".join(c["text"] for c in chunks).lower()
    return any(s.lower() in combined for s in expected_substrings)


def reciprocal_rank(chunks: list[dict], expected_substrings: list[str]) -> float:
    """
    Return 1/rank of the first chunk containing any expected substring.
    Returns 0.0 if no chunk matches.
    """
    for rank, chunk in enumerate(chunks, start=1):
        text = chunk["text"].lower()
        if any(s.lower() in text for s in expected_substrings):
            return 1.0 / rank
    return 0.0


def evaluate_strategy(strategy: str, ground_truth: list[dict], cfg: dict) -> dict:
    k = cfg["eval"]["recall_k"]

    print(f"\n  Re-indexing with strategy: '{strategy}'...")
    t0 = time.time()
    chunks = run_ingestion("data/pdfs", strategy)
    index_time = round(time.time() - t0, 2)
    print(f"  Indexed {len(chunks)} chunks in {index_time}s")

    hits  = 0
    rr_scores = []

    for item in ground_truth:
        query_vec = embed_query(item["question"])
        retrieved = search(query_vec, top_k=k)

        if is_hit(retrieved, item["expected_chunks"]):
            hits += 1

        rr_scores.append(reciprocal_rank(retrieved, item["expected_chunks"]))

    total     = len(ground_truth)
    recall_at_k = round(hits / total, 3)
    mrr         = round(sum(rr_scores) / total, 3)

    return {
        "strategy":    strategy,
        "chunks":      len(chunks),
        "recall_at_k": recall_at_k,
        "mrr":         mrr,
        "hits":        hits,
        "total":       total,
        "k":           k,
        "index_time":  index_time,
    }


def print_results(results: list[dict]) -> None:
    k = results[0]["k"]
    print("\n" + "=" * 65)
    print(f"  CHUNKING STRATEGY BENCHMARK  —  Recall@{k} + MRR")
    print("=" * 65)
    header = f"  {'Strategy':<12} {'Chunks':>6}  {'Recall@'+str(k):>10}  {'MRR':>6}  {'Hits':>10}  {'Index time':>10}"
    print(header)
    print("  " + "-" * 61)

    best_recall = max(r["recall_at_k"] for r in results)
    best_mrr    = max(r["mrr"]         for r in results)

    for r in results:
        recall_str = f"{r['recall_at_k']:.3f}" + (" ◀ best" if r["recall_at_k"] == best_recall else "")
        mrr_str    = f"{r['mrr']:.3f}"          + (" ◀ best" if r["mrr"]         == best_mrr    else "")
        print(
            f"  {r['strategy']:<12} {r['chunks']:>6}  "
            f"{recall_str:>16}  {mrr_str:>12}  "
            f"{r['hits']}/{r['total']:>6}  {r['index_time']:>8}s"
        )

    print("=" * 65)

    # Re-index with the best recall strategy so the server is ready after eval
    best = max(results, key=lambda r: r["recall_at_k"])
    print(f"\n  Best strategy by Recall@{k}: '{best['strategy']}'")
    print(f"  Re-indexing server with '{best['strategy']}' for subsequent use...")
    run_ingestion("data/pdfs", best["strategy"])
    print(f"  Done. Server index is now: '{best['strategy']}'")


def main():
    cfg          = load_config()
    ground_truth = load_ground_truth()

    print(f"Loaded {len(ground_truth)} ground-truth questions.")
    print(f"Evaluating {len(STRATEGIES)} strategies at Recall@{cfg['eval']['recall_k']}...\n")

    results = []
    for strategy in STRATEGIES:
        result = evaluate_strategy(strategy, ground_truth, cfg)
        results.append(result)

    print_results(results)

    # Save results for README
    out_path = Path("tests/results/retrieval_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()