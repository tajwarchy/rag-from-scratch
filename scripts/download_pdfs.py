"""
Download a small set of ArXiv papers into data/pdfs/.

Papers chosen for this project:
  - "Attention Is All You Need" (Transformer architecture)
  - "BERT: Pre-training of Deep Bidirectional Transformers"
  - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

These are ideal for RAG Q&A testing:
  - Dense technical content → chunking strategies behave differently
  - Well-known facts → easy to write ground-truth eval questions (Phase 6)
  - All open-access on ArXiv — no login or API key required
"""

import urllib.request
import time
from pathlib import Path

PAPERS = [
    {
        "name": "attention_is_all_you_need.pdf",
        "url": "https://arxiv.org/pdf/1706.03762",
        "desc": "Attention Is All You Need (Vaswani et al., 2017)",
    },
    {
        "name": "bert.pdf",
        "url": "https://arxiv.org/pdf/1810.04805",
        "desc": "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
    },
    {
        "name": "rag_paper.pdf",
        "url": "https://arxiv.org/pdf/2005.11401",
        "desc": "Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020)",
    },
]

SAVE_DIR = Path("data/pdfs")


def download(paper: dict) -> None:
    dest = SAVE_DIR / paper["name"]
    if dest.exists():
        print(f"  [skip] already exists: {dest}")
        return

    print(f"  Downloading: {paper['desc']}")
    req = urllib.request.Request(
        paper["url"],
        headers={"User-Agent": "Mozilla/5.0"},  # arxiv requires a UA header
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()

    dest.write_bytes(data)
    size_kb = dest.stat().st_size // 1024
    print(f"  Saved → {dest}  ({size_kb} KB)")


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving PDFs to: {SAVE_DIR.resolve()}\n")

    for i, paper in enumerate(PAPERS):
        download(paper)
        if i < len(PAPERS) - 1:
            time.sleep(1)  # be polite to arxiv

    print(f"\nDone. {len(list(SAVE_DIR.glob('*.pdf')))} PDF(s) in {SAVE_DIR}")


if __name__ == "__main__":
    main()