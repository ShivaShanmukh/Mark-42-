import os
from pathlib import Path
from typing import AsyncGenerator

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = (
    "You are a codebase expert. Answer questions about the codebase using ONLY the provided context. "
    "If the answer is not in the context, say so clearly. "
    "Always cite which file your answer comes from."
)

MODEL = "claude-sonnet-4-20250514"


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"--- Chunk {i} | File: {chunk['file_path']} | Lines {chunk['line_start']}-{chunk['line_end']} ---\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(parts)


async def answer(query: str, context: list[dict]) -> AsyncGenerator[dict, None]:
    context_block = _build_context_block(context)
    user_message = f"Context from codebase:\n\n{context_block}\n\nQuestion: {query}"

    citations = list({c["file_path"] for c in context if c.get("file_path")})

    async with _client.messages.stream(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text in stream.text_stream:
            yield {"type": "token", "content": text}

    yield {"type": "citations", "files": sorted(citations)}
