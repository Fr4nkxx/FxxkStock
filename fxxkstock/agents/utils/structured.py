"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_response_text(response: Any) -> str:
    """Extract readable text without exposing provider reasoning blocks."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content or "").strip()

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def bind_structured(
    llm: Any,
    schema: type[T],
    agent_name: str,
    *,
    include_raw: bool = False,
    method: str | None = None,
) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        kwargs: dict[str, Any] = {}
        if include_raw:
            kwargs["include_raw"] = True
        if method:
            kwargs["method"] = method
        return llm.with_structured_output(schema, **kwargs)
    except (NotImplementedError, AttributeError, TypeError, ValueError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name,
            exc,
        )
        return None


def summarize_diagnostic_error(error: object, limit: int = 400) -> str:
    """Return a compact error summary suitable for persisted run diagnostics."""
    text = " ".join(str(error).split())
    text = re.sub(
        r"(?i)\b(api[_-]?key|token|authorization)\b\s*[:=]\s*([^\s,;]+)",
        r"\1=<redacted>",
        text,
    )
    text = re.sub(
        r"(?i)\bbearer\s+[a-z0-9._~+/=-]+",
        "Bearer <redacted>",
        text,
    )
    return text[:limit]


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    diagnostics: dict[str, Any] | None = None,
    *,
    raw_parser: Callable[[str], T] | None = None,
    reuse_raw_response: bool = False,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    started_at = time.perf_counter()
    if diagnostics is not None:
        diagnostics.update(
            {
                "agent": agent_name,
                "structured_available": structured_llm is not None,
                "structured_attempts": 0,
                "fallback_attempts": 0,
                "fallback_used": False,
            }
        )

    if structured_llm is not None:
        structured_started_at = time.perf_counter()
        if diagnostics is not None:
            diagnostics["structured_attempts"] = 1
        try:
            result = structured_llm.invoke(prompt)
            raw_text = ""
            parsing_error: object | None = None
            if isinstance(result, dict) and {
                "parsed",
                "raw",
                "parsing_error",
            }.intersection(result):
                raw_message = result.get("raw")
                raw_text = extract_response_text(raw_message)
                parsing_error = result.get("parsing_error")
                result = result.get("parsed")
                if diagnostics is not None:
                    diagnostics.update(
                        {
                            "raw_response_available": raw_message is not None,
                            "raw_content_characters": len(raw_text),
                            "raw_output_reused": False,
                        }
                    )

            if result is None and raw_text and raw_parser is not None:
                try:
                    result = raw_parser(raw_text)
                    if diagnostics is not None:
                        diagnostics["structured_recovered_from_raw"] = True
                except Exception as exc:  # noqa: BLE001
                    parsing_error = parsing_error or exc

            if result is None and raw_text and reuse_raw_response:
                elapsed = round(time.perf_counter() - structured_started_at, 3)
                if diagnostics is not None:
                    diagnostics.update(
                        {
                            "structured_success": False,
                            "structured_duration_seconds": elapsed,
                            "fallback_reason": "structured_output_unparsed",
                            "fallback_error": summarize_diagnostic_error(
                                parsing_error or "structured response was not parsed"
                            ),
                            "fallback_attempts": 0,
                            "fallback_used": True,
                            "fallback_duration_seconds": 0.0,
                            "raw_output_reused": True,
                            "model_attempts": 1,
                            "output_characters": len(raw_text),
                            "total_model_duration_seconds": round(
                                time.perf_counter() - started_at,
                                3,
                            ),
                        }
                    )
                logger.warning(
                    "%s: structured response was not parsed; reusing its readable text",
                    agent_name,
                )
                return raw_text

            if result is None:
                # A thinking model can answer in plain text instead of calling
                # the tool, leaving the parser with nothing to return. Treat it
                # as a structured miss and fall back, with a clear reason.
                raise ValueError("structured output returned no parsed result") from (
                    parsing_error if isinstance(parsing_error, BaseException) else None
                )
            output = render(result)
            if diagnostics is not None:
                diagnostics.update(
                    {
                        "structured_success": True,
                        "structured_duration_seconds": round(
                            time.perf_counter() - structured_started_at,
                            3,
                        ),
                        "model_attempts": 1,
                        "output_characters": len(output),
                        "total_model_duration_seconds": round(
                            time.perf_counter() - started_at,
                            3,
                        ),
                    }
                )
            return output
        except Exception as exc:
            if diagnostics is not None:
                diagnostics.update(
                    {
                        "structured_success": False,
                        "structured_duration_seconds": round(
                            time.perf_counter() - structured_started_at,
                            3,
                        ),
                        "fallback_reason": type(exc).__name__,
                        "fallback_error": summarize_diagnostic_error(exc),
                    }
                )
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name,
                exc,
            )
    elif diagnostics is not None:
        diagnostics.update(
            {
                "structured_success": False,
                "structured_duration_seconds": 0.0,
                "fallback_reason": "structured_output_unavailable",
            }
        )

    fallback_started_at = time.perf_counter()
    if diagnostics is not None:
        diagnostics["fallback_attempts"] = 1
        diagnostics["fallback_used"] = True
    response = plain_llm.invoke(prompt)
    output = response.content
    if diagnostics is not None:
        diagnostics.update(
            {
                "fallback_duration_seconds": round(
                    time.perf_counter() - fallback_started_at,
                    3,
                ),
                "model_attempts": int(diagnostics["structured_attempts"]) + 1,
                "output_characters": len(str(output)),
                "total_model_duration_seconds": round(
                    time.perf_counter() - started_at,
                    3,
                ),
            }
        )
    return output
