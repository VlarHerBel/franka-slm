"""Cliente HTTP para Ollama (/api/chat) — inferencia SLM schema v1.1."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from .intent_schema import ROBOT_INTENT_JSON_SCHEMA
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_CHAT_URL = "%s/api/chat" % DEFAULT_OLLAMA_BASE_URL


def resolve_chat_url(ollama_url: Optional[str] = None) -> str:
    """Base Ollama (p. ej. http://127.0.0.1:11434) o URL /api/chat completa."""
    if ollama_url is None:
        return OLLAMA_CHAT_URL
    base = ollama_url.rstrip("/")
    if base.endswith("/api/chat"):
        return base
    return "%s/api/chat" % base
OLLAMA_KEEP_ALIVE = "30m"
DEFAULT_TIMEOUT_S = 90.0

EVAL_NUM_CTX = 2048
EVAL_NUM_PREDICT = 256
WARMUP_NUM_CTX = 1024
WARMUP_NUM_PREDICT = 128

WARMUP_PROMPTS: List[str] = [
    "vuelve a home",
    "coge la caja de galletas y déjala en el primer hueco",
]


@dataclass
class OllamaGenerateResult:
    """Resultado de una inferencia a Ollama."""

    raw_response: str = ""
    parsed_json: Optional[Dict[str, Any]] = None
    parse_error: str = ""
    latency_s: float = 0.0
    request_error: str = ""
    timing: Dict[str, Any] = field(default_factory=dict)


def build_ollama_payload(
    model: str,
    user_text: str,
    *,
    num_ctx: int = EVAL_NUM_CTX,
    num_predict: int = EVAL_NUM_PREDICT,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "format": ROBOT_INTENT_JSON_SCHEMA,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(text=user_text),
            },
        ],
    }
    return payload


def _warmup_options(index: int, text: str) -> Tuple[int, int]:
    if index >= 2 or text != WARMUP_PROMPTS[0]:
        return EVAL_NUM_CTX, EVAL_NUM_PREDICT
    return WARMUP_NUM_CTX, WARMUP_NUM_PREDICT


def _parse_ollama_timing(data: Dict[str, Any]) -> Dict[str, Any]:
    def ns_to_s(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return round(int(value) / 1_000_000_000.0, 6)
        except (TypeError, ValueError):
            return None

    eval_duration_s = ns_to_s(data.get("eval_duration"))
    eval_count = data.get("eval_count")
    tps: Any = ""
    if eval_duration_s and eval_duration_s > 0 and eval_count is not None:
        try:
            tps = round(int(eval_count) / eval_duration_s, 4)
        except (TypeError, ValueError):
            tps = ""

    return {
        "total_duration_s": ns_to_s(data.get("total_duration")),
        "load_duration_s": ns_to_s(data.get("load_duration")),
        "prompt_eval_duration_s": ns_to_s(data.get("prompt_eval_duration")),
        "eval_duration_s": eval_duration_s,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": eval_count,
        "tokens_per_second": tps,
    }


def call_ollama_chat(
    user_text: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    ollama_url: Optional[str] = None,
    num_ctx: int = EVAL_NUM_CTX,
    num_predict: int = EVAL_NUM_PREDICT,
) -> OllamaGenerateResult:
    """Llama a Ollama y devuelve texto crudo + JSON parseado (si es posible)."""
    payload = build_ollama_payload(
        model, user_text, num_ctx=num_ctx, num_predict=num_predict
    )
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            resolve_chat_url(ollama_url), json=payload, timeout=timeout_s
        )
        latency = time.perf_counter() - t0
    except requests.Timeout:
        return OllamaGenerateResult(
            request_error="timeout",
            latency_s=time.perf_counter() - t0,
        )
    except requests.RequestException as exc:
        return OllamaGenerateResult(
            request_error="request_error:%s" % exc,
            latency_s=time.perf_counter() - t0,
        )

    if resp.status_code != 200:
        return OllamaGenerateResult(
            request_error="request_error:HTTP_%d" % resp.status_code,
            latency_s=latency,
        )

    try:
        data = resp.json()
    except ValueError:
        return OllamaGenerateResult(
            request_error="request_error:invalid_json_response",
            latency_s=latency,
        )

    timing = _parse_ollama_timing(data)
    message = data.get("message") or {}
    content = str(message.get("content") or "")
    if not content.strip():
        return OllamaGenerateResult(
            request_error="empty_response",
            latency_s=latency,
            timing=timing,
        )

    from .json_extract import extract_json_object

    parsed, parse_err = extract_json_object(content)
    return OllamaGenerateResult(
        raw_response=content,
        parsed_json=parsed,
        parse_error=parse_err,
        latency_s=latency,
        request_error="",
        timing=timing,
    )


def warmup_model(
    model_name: str = DEFAULT_MODEL,
    warmup_count: int = 2,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    ollama_url: Optional[str] = None,
) -> List[OllamaGenerateResult]:
    """Ejecuta inferencias de calentamiento (no miden accuracy del usuario)."""
    results: List[OllamaGenerateResult] = []
    for idx in range(1, max(0, warmup_count) + 1):
        text = WARMUP_PROMPTS[(idx - 1) % len(WARMUP_PROMPTS)]
        num_ctx, num_predict = _warmup_options(idx, text)
        print(
            "[WARMUP] model=%s index=%d/%d text=%r"
            % (model_name, idx, warmup_count, text),
            flush=True,
        )
        result = call_ollama_chat(
            text,
            model=model_name,
            timeout_s=timeout_s,
            ollama_url=ollama_url,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        results.append(result)
        ok = result.request_error == "" and result.parsed_json is not None
        print(
            "[WARMUP] model=%s index=%d latency_s=%.2f success=%s"
            % (model_name, idx, result.latency_s, str(ok).lower()),
            flush=True,
        )
    return results


def generate_intent(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    ollama_url: Optional[str] = None,
) -> OllamaGenerateResult:
    """Genera intent JSON v1.1 para una orden en lenguaje natural."""
    return call_ollama_chat(
        text, model=model, timeout_s=timeout_s, ollama_url=ollama_url
    )
