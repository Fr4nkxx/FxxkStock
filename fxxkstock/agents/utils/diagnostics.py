"""Non-content diagnostics for model calls and targeted stage replay."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from fxxkstock.agents.utils.structured import extract_response_text


def prompt_characters(prompt: Any) -> int:
    """Count user-visible prompt characters across strings and messages."""
    if isinstance(prompt, str):
        return len(prompt)
    if isinstance(prompt, dict):
        return sum(prompt_characters(value) for value in prompt.values())
    if isinstance(prompt, (list, tuple)):
        return sum(prompt_characters(value) for value in prompt)
    content = getattr(prompt, "content", None)
    if content is not None:
        return prompt_characters(content)
    return len(str(prompt or ""))


def invoke_plain_with_diagnostics(
    llm: Any,
    prompt: Any,
    agent: str,
    *,
    input_characters: dict[str, int] | None = None,
    sequence: int = 1,
) -> tuple[Any, dict[str, Any]]:
    """Invoke an unchanged plain-model call and return timing metadata."""
    started_at = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = round(time.perf_counter() - started_at, 3)
    sizes = dict(input_characters or {})
    sizes["prompt"] = prompt_characters(prompt)
    return response, {
        "agent": agent,
        "sequence": sequence,
        "input_characters": sizes,
        "model_attempts": 1,
        "fallback_used": False,
        "output_characters": len(extract_response_text(response)),
        "total_model_duration_seconds": elapsed,
    }


def append_stage_replay_context(
    state: dict[str, Any],
    stage: str,
    snapshot: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Append an immutable pre-call state slice for later local replay."""
    contexts = deepcopy(state.get("stage_replay_contexts") or {})
    entries = list(contexts.get(stage) or [])
    entries.append(deepcopy(snapshot))
    contexts[stage] = entries
    return contexts
