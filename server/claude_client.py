import os
from collections.abc import Iterator

import anthropic

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = (
    "You are an assistant that answers questions based strictly on the provided context."
    "If the answer cannot be found in the given context, Do Not Hallucinate, simply declare you don't know."
    "Always cite which source document your answer is taken from."
)


def stream_answer(query: str, context_chunks: list[dict]) -> Iterator[str]:
    """
    Stream a Claude answer grounded in context_chunks.
    Each chunk dict has keys: text, source, chunk_index, score.
    Yields text tokens as they arrive.
    """
    context = "\n\n".join(
        f"[{c['source']} chunk {c['chunk_index']}]\n{c['text']}" for c in context_chunks
    )

    with _client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Context:\n\n{context}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"Question: {query}",
                    },
                ],
            }
        ],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    ) as stream:
        for text in stream.text_stream:
            yield text
