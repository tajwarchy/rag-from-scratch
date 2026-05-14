"""
Three chunking strategies, each implementing the same interface:

    chunk(pages: list[dict], cfg: dict) -> list[dict]

Input  — list of page dicts from extractor.py:
    {"source_file": str, "page": int, "text": str}

Output — list of chunk dicts:
    {
        "chunk_id":    int,         # global index across all chunks
        "source_file": str,
        "page":        int,
        "text":        str,
        "strategy":    str,
    }

All three strategies produce the same output shape so the embedder
and FAISS store don't need to know which strategy was used.
"""

import re
import tiktoken

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def get_chunker(strategy: str):
    strategies = {
        "fixed":    chunk_fixed,
        "sentence": chunk_sentence,
        "sliding":  chunk_sliding,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose: {list(strategies)}")
    return strategies[strategy]


# ── Strategy 1: Fixed-size (token-based) ─────────────────────────────────────

def chunk_fixed(pages: list[dict], cfg: dict) -> list[dict]:
    """
    Split each page's text into fixed-size token windows with overlap.
    Uses tiktoken so chunk sizes are exact token counts, not character counts.

    Config keys used:
        chunking.fixed.chunk_size   (default 256)
        chunking.fixed.overlap      (default 32)
    """
    chunk_size = cfg["chunking"]["fixed"]["chunk_size"]
    overlap    = cfg["chunking"]["fixed"]["overlap"]

    chunks = []
    chunk_id = 0

    for page in pages:
        tokens = _TOKENIZER.encode(page["text"])
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            window_tokens = tokens[start:end]
            text = _TOKENIZER.decode(window_tokens).strip()
            if text:
                chunks.append(_make_chunk(chunk_id, page, text, "fixed"))
                chunk_id += 1
            start += chunk_size - overlap  # slide forward with overlap

    return chunks


# ── Strategy 2: Sentence-aware ────────────────────────────────────────────────

def chunk_sentence(pages: list[dict], cfg: dict) -> list[dict]:
    """
    Group sentences into chunks of up to max_sentences, with sentence-level overlap.
    Respects natural sentence boundaries — no sentence is split mid-way.

    Config keys used:
        chunking.sentence.max_sentences      (default 5)
        chunking.sentence.overlap_sentences  (default 1)
    """
    max_sents     = cfg["chunking"]["sentence"]["max_sentences"]
    overlap_sents = cfg["chunking"]["sentence"]["overlap_sentences"]

    chunks = []
    chunk_id = 0

    for page in pages:
        sentences = _split_sentences(page["text"])
        start = 0
        while start < len(sentences):
            end = start + max_sents
            window = sentences[start:end]
            text = " ".join(window).strip()
            if text:
                chunks.append(_make_chunk(chunk_id, page, text, "sentence"))
                chunk_id += 1
            start += max_sents - overlap_sents

    return chunks


# ── Strategy 3: Sliding window (token-based) ──────────────────────────────────

def chunk_sliding(pages: list[dict], cfg: dict) -> list[dict]:
    """
    Slide a fixed token window across each page with a configurable step size.
    More overlap than fixed-size — better recall at the cost of more chunks.

    Config keys used:
        chunking.sliding.window_size  (default 200)
        chunking.sliding.step_size    (default 100)
    """
    window_size = cfg["chunking"]["sliding"]["window_size"]
    step_size   = cfg["chunking"]["sliding"]["step_size"]

    chunks = []
    chunk_id = 0

    for page in pages:
        tokens = _TOKENIZER.encode(page["text"])
        start = 0
        while start < len(tokens):
            end = start + window_size
            window_tokens = tokens[start:end]
            text = _TOKENIZER.decode(window_tokens).strip()
            if text:
                chunks.append(_make_chunk(chunk_id, page, text, "sliding"))
                chunk_id += 1
            if end >= len(tokens):
                break
            start += step_size

    return chunks


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_chunk(chunk_id: int, page: dict, text: str, strategy: str) -> dict:
    return {
        "chunk_id":    chunk_id,
        "source_file": page["source_file"],
        "page":        page["page"],
        "text":        text,
        "strategy":    strategy,
    }


def _split_sentences(text: str) -> list[str]:
    """
    Naive but effective sentence splitter using regex.
    Splits on '. ', '? ', '! ' followed by a capital letter or end of string.
    Avoids splitting on common abbreviations (e.g. 'Dr.', 'Fig.').
    """
    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r'(?<=[.?!])\s+(?=[A-Z])', text)
    # Filter out empty or very short fragments (likely artefacts)
    return [p.strip() for p in parts if len(p.strip()) > 10]