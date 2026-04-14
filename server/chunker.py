import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    index: int
    source: str = ""
    metadata: dict = field(default_factory=dict)


def _split_paragraphs(text: str) -> list[str]:
    # Split on blank lines (e.g. paragraph / section boundaries)
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    source: str = "",
    max_chars: int = 500,
    overlap_chars: int = 100,
) -> list[Chunk]:
    paragraphs = _split_paragraphs(text)
    raw_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate_paragraph = (current + "\n\n" + para).strip() if current else para

        if len(candidate_paragraph) <= max_chars:
            current = candidate_paragraph   
        else:
            if current:
                raw_chunks.append(current)
                # Seed the next chunk with overlap from the end of current
                tail = current[-overlap_chars:] if overlap_chars else ""
                current = (tail + "\n\n" + para).strip() if tail else para
            else:
                # Single paragraph is already larger than max_chars — keep as-is
                raw_chunks.append(para)
                current = para[-overlap_chars:] if overlap_chars else ""

    if current.strip():
        raw_chunks.append(current.strip())

    return [
        Chunk(text=t, index=i, source=source) for i, t in enumerate(raw_chunks)
    ]
