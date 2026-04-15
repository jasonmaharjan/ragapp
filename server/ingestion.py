"""Document Ingestion Pipeline: Load raw text from .pdf / .docx / .txt files for chunking and embedding"""

from pathlib import Path
from chunker import Chunk, chunk_text


# Loaders for different file types. Each returns the full text content as a single string.
def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_num}]\n{text}")
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

# Can extend to other formats (e.g. html, csv, pptx) by registering new loaders here
_LOADERS = {
    ".txt": _load_txt,
    ".md": _load_txt,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
}

SUPPORTED_EXTENSIONS = list(_LOADERS.keys())


# Public API
"""Load file_path and return a list of overlapping text chunks"""
def ingest_document(
    file_path: str,
    max_chars: int = 500,
    overlap_chars: int = 100,
) -> list[Chunk]:
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"The Supported file types are: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    raw_text = _LOADERS[extension](path)
    return chunk_text(
        raw_text,
        source=path.name,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
