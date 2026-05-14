"""
Latency load simulation — runs 50 queries against the live FastAPI server,
measures per-component timing, and identifies the bottleneck.

Requires the server to be running:
    uvicorn main:app --host 0.0.0.0 --port 8000

Run:
    python -m tests.eval_latency

Output:
  - Per-query latency log (retrieval vs LLM vs total)
  - Summary statistics (mean, median, p95, p99)
  - Bottleneck identification
  - Cache hit rate after the run
  - Results saved to tests/results/latency_log.json
"""

import json
import sys
import time
import statistics
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

SERVER_URL = "http://localhost:8000"

# 50 queries — mix of repeated (to exercise cache) and unique
QUERIES = [
    "What is the attention mechanism in the Transformer?",
    "How does multi-head attention work?",
    "What is positional encoding?",
    "What datasets were used to pre-train BERT?",
    "What does BERT stand for?",
    "How does RAG combine retrieval with generation?",
    "What retriever does the RAG model use?",
    "What is the difference between RAG-sequence and RAG-token?",
    "What optimizer was used to train the Transformer?",
    "What is masked language modelling?",
    "How many layers does the Transformer encoder have?",
    "What is the role of the feed-forward layer?",
    "What is dropout regularisation?",
    "How is BERT fine-tuned for classification?",
    "What is next sentence prediction?",
    "What is the BLEU score of the Transformer?",
    "How does the Transformer handle variable-length sequences?",
    "What is dense passage retrieval?",
    "What is the difference between parametric and non-parametric memory?",
    "How are attention weights computed?",
    # Repeated queries below — these hit the embedding cache
    "What is the attention mechanism in the Transformer?",
    "How does multi-head attention work?",
    "What is positional encoding?",
    "What datasets were used to pre-train BERT?",
    "What does BERT stand for?",
    "How does RAG combine retrieval with generation?",
    "What retriever does the RAG model use?",
    "What optimizer was used to train the Transformer?",
    "What is masked language modelling?",
    "How many layers does the Transformer encoder have?",
    "What is the role of the feed-forward layer?",
    "What is dropout regularisation?",
    "How is BERT fine-tuned for classification?",
    "What is next sentence prediction?",
    "What is the BLEU score of the Transformer?",
    "What is the attention mechanism in the Transformer?",
    "How does multi-head attention work?",
    "What is positional encoding?",
    "What datasets were used to pre-train BERT?",
    "What does BERT stand for?",
    "How does RAG combine retrieval with generation?",
    "What retriever does the RAG model use?",
    "What optimizer was used to train the Transformer?",
    "What is masked language modelling?",
    "How many layers does the Transformer encoder have?",
    "What is the attention mechanism in the Transformer?",
    "How does multi-head attention work?",
    "What is positional encoding?",
    "What does BERT stand for?",
    "How does RAG combine retrieval with generation?",
]

assert len(QUERIES) == 50, f"Expected 50 queries, got {len(QUERIES)}"


def run_query(client: httpx.Client, question: str) -> dict:
    t0 = time.time()
    resp = client.post(
        f"{SERVER_URL}/query",
        json={"question": question, "top_k": 5},
        timeout=120.0,
    )
    resp.raise_for_status()
    total_ms = int((time.time() - t0) * 1000)
    data = resp.json()
    return {
        "question":        question[:60],
        "total_ms":        total_ms,
        "retrieval_ms":    data["retrieval_time_ms"],
        "llm_ms":          data["llm_time_ms"],
        "cache_hit":       data["cache_hit"],
    }


def percentile(data: list[float], p: int) -> float:
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def print_summary(logs: list[dict]) -> None:
    totals      = [r["total_ms"]     for r in logs]
    retrievals  = [r["retrieval_ms"] for r in logs]
    llms        = [r["llm_ms"]       for r in logs]
    cache_hits  = sum(1 for r in logs if r["cache_hit"])

    print("\n" + "=" * 65)
    print("  LATENCY SIMULATION RESULTS  —  50 queries")
    print("=" * 65)
    print(f"  {'Metric':<28} {'Retrieval':>10} {'LLM':>10} {'Total':>10}")
    print("  " + "-" * 61)

    def row(label, fn, data_r, data_l, data_t):
        print(f"  {label:<28} {fn(data_r):>9}ms {fn(data_l):>9}ms {fn(data_t):>9}ms")

    row("Mean",   lambda d: int(statistics.mean(d)),   retrievals, llms, totals)
    row("Median", lambda d: int(statistics.median(d)), retrievals, llms, totals)
    row("P95",    lambda d: int(percentile(d, 95)),    retrievals, llms, totals)
    row("P99",    lambda d: int(percentile(d, 99)),    retrievals, llms, totals)
    row("Min",    lambda d: int(min(d)),               retrievals, llms, totals)
    row("Max",    lambda d: int(max(d)),               retrievals, llms, totals)

    print("  " + "-" * 61)
    print(f"  Cache hits: {cache_hits}/50  ({cache_hits*2}% hit rate)")

    # Bottleneck identification
    mean_r = statistics.mean(retrievals)
    mean_l = statistics.mean(llms)
    bottleneck = "LLM generation" if mean_l > mean_r else "Embedding + retrieval"
    pct = int(max(mean_l, mean_r) / statistics.mean(totals) * 100)
    print(f"\n  Bottleneck: {bottleneck} ({pct}% of total latency)")
    print(f"  → See README for full analysis and scaling recommendations.")
    print("=" * 65)


def main():
    # Verify server is up
    try:
        httpx.get(f"{SERVER_URL}/health", timeout=5.0).raise_for_status()
    except Exception:
        print(f"ERROR: Server not reachable at {SERVER_URL}")
        print("Start it with: uvicorn main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    print(f"Running {len(QUERIES)} queries against {SERVER_URL}...")
    print("(This will take a while — each query calls Mistral 7B)\n")

    logs = []
    with httpx.Client() as client:
        for i, question in enumerate(QUERIES, start=1):
            result = run_query(client, question)
            logs.append(result)
            hit_marker = "● cache" if result["cache_hit"] else "○ embed"
            print(
                f"  [{i:02d}/50] {hit_marker} | "
                f"retrieval {result['retrieval_ms']:>4}ms | "
                f"llm {result['llm_ms']:>5}ms | "
                f"total {result['total_ms']:>5}ms | "
                f"{result['question']}"
            )

    print_summary(logs)

    out_path = Path("tests/results/latency_log.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"\n  Full log saved to: {out_path}")


if __name__ == "__main__":
    main()