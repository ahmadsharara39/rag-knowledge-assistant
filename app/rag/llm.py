"""LLM providers behind a common interface.

- ``OpenAILLM`` and ``AnthropicLLM`` generate grounded answers from retrieved
  context.
- ``ExtractiveLLM`` is a dependency-free fallback that composes an answer from the
  most relevant retrieved sentences and always cites its sources. It lets the
  full pipeline run and be evaluated offline, and it never hallucinates because
  it only returns text that appears in the context.

Every provider receives the same grounded prompt and is instructed to answer
ONLY from the provided context and to cite sources with ``[n]`` markers.
"""

from __future__ import annotations

import re
from typing import Protocol

SYSTEM_PROMPT = (
    "You are a precise enterprise knowledge assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. Cite every claim with its "
    "source marker like [1] or [2]. If the answer is not in the context, say you "
    "don't have enough information. Be concise and factual."
)


class LLMProvider(Protocol):
    def generate(self, question: str, context_blocks: list[str]) -> str:
        ...


def _build_user_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n".join(context_blocks)
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (cite sources with [n]):"
    )


class OpenAILLM:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, question: str, context_blocks: list[str]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(question, context_blocks)},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()


class AnthropicLLM:
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, question: str, context_blocks: list[str]) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": _build_user_prompt(question, context_blocks),
                }
            ],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()


class ExtractiveLLM:
    """Offline, non-hallucinating fallback.

    Scores each sentence in the retrieved context by word overlap with the
    question and returns the best few, each tagged with the source marker it came
    from. Deterministic and citation-complete.
    """

    _WORD = re.compile(r"[a-z0-9]+")
    _SENT = re.compile(r"(?<=[.!?])\s+")

    def generate(self, question: str, context_blocks: list[str]) -> str:
        q_words = set(self._WORD.findall(question.lower()))
        scored: list[tuple[float, int, str]] = []
        for block in context_blocks:
            marker_match = re.match(r"\s*(\[\d+\])", block)
            marker = marker_match.group(1) if marker_match else "[?]"
            body = block[marker_match.end():] if marker_match else block
            for sentence in self._SENT.split(body.strip()):
                sentence = sentence.strip()
                if not sentence:
                    continue
                s_words = set(self._WORD.findall(sentence.lower()))
                if not s_words:
                    continue
                overlap = len(q_words & s_words) / (len(q_words) or 1)
                if overlap > 0:
                    scored.append((overlap, len(sentence), f"{sentence} {marker}"))

        if not scored:
            return "I don't have enough information in the provided context to answer that."

        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [s for _, _, s in scored[:3]]
        return " ".join(top)


def build_llm_provider(settings) -> LLMProvider:
    provider = settings.llm_provider.lower()
    if provider == "openai" and settings.openai_api_key:
        return OpenAILLM(settings.openai_api_key, settings.openai_chat_model)
    if provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    return ExtractiveLLM()
