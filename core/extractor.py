import re
import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract cleaned text from every page of a PDF.

    Returns a list of page dicts:
        [{"source_file": "paper.pdf", "page": 1, "text": "..."}, ...]

    Cleaning steps applied per page:
        1. Collapse runs of whitespace / newlines into single spaces.
        2. Remove lines that are purely numeric (page numbers, footnote indices).
        3. Strip leading/trailing whitespace.
    Pages that are empty after cleaning are skipped.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")

    pages = []
    doc = fitz.open(str(path))

    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        cleaned = _clean(raw)
        if cleaned:
            pages.append({
                "source_file": path.name,
                "page": page_num,
                "text": cleaned,
            })

    doc.close()
    return pages


def extract_text_from_dir(pdf_dir: str) -> list[dict]:
    """
    Extract text from every PDF in a directory.
    Returns a flat list of page dicts across all files.
    """
    dir_path = Path(pdf_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {dir_path.resolve()}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {dir_path.resolve()}")

    all_pages = []
    for pdf_file in pdf_files:
        print(f"  Extracting: {pdf_file.name}")
        pages = extract_text_from_pdf(str(pdf_file))
        all_pages.extend(pages)
        print(f"    → {len(pages)} pages extracted")

    return all_pages


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    # Collapse all whitespace (tabs, newlines, multiple spaces) into single space
    text = re.sub(r"\s+", " ", text)
    # Remove tokens that are purely numeric (page numbers, indices)
    text = re.sub(r"\b\d+\b", "", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text