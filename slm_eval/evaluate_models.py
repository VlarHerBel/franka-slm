#!/usr/bin/env python3
"""Evaluación offline de SLMs locales (Ollama HTTP) → JSON robótico v1.1 validable."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from intent_schema import (
    ROBOT_INTENT_JSON_SCHEMA,
    RobotIntentModel,
    predicted_from_model,
    slot_order_to_csv,
    slot_orders_match,
    validate_intent_payload,
)
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

from semantic_guardrails import apply_semantic_guardrails

DEFAULT_MODELS: List[str] = [
    "qwen3.5:4b-q4_K_M",
    "granite4.1:3b-q4_K_M",
    "qwen3:4b-instruct-2507-q4_K_M",
    "llama3.2:3b",
    "phi4-mini:3.8b-q4_K_M",
]

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
OLLAMA_KEEP_ALIVE = "30m"
REQUEST_TIMEOUT_S = 90.0
SLOW_LATENCY_THRESHOLD_S = 30.0
WARMUP_PROMPTS: List[str] = [
    "vuelve a home",
    "coge la caja de galletas y déjala en el primer hueco",
]
DEFAULT_WARMUP_COUNT = 2
EVAL_NUM_CTX = 2048
EVAL_NUM_PREDICT = 256
WARMUP_NUM_CTX = 1024
WARMUP_NUM_PREDICT = 128

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "commands_dataset.jsonl"
RESULTS_DIR = ROOT / "results"
RAW_OUTPUTS_PATH = RESULTS_DIR / "raw_outputs.jsonl"
METRICS_CMD_PATH = RESULTS_DIR / "metrics_by_command.csv"
METRICS_CMD_RAW_PATH = RESULTS_DIR / "metrics_by_command_raw.csv"
METRICS_MODEL_PATH = RESULTS_DIR / "metrics_by_model.csv"
METRICS_MODEL_RAW_PATH = RESULTS_DIR / "metrics_by_model_raw.csv"
WARMUP_MODEL_PATH = RESULTS_DIR / "warmup_by_model.csv"
WARMUP_DETAILS_PATH = RESULTS_DIR / "warmup_details_by_model.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

OLLAMA_TIMING_FIELDS = [
    "ollama_total_duration_s",
    "ollama_load_duration_s",
    "ollama_prompt_eval_duration_s",
    "ollama_eval_duration_s",
    "ollama_prompt_eval_count",
    "ollama_eval_count",
    "ollama_tokens_per_second",
]

CMD_CSV_FIELDS = [
    "model",
    "command_id",
    "text",
    "expected_intent",
    "predicted_intent",
    "expected_target_label",
    "predicted_target_label",
    "expected_target_selector_type",
    "predicted_target_selector_type",
    "expected_destination_type",
    "predicted_destination_type",
    "expected_slot_index",
    "predicted_slot_index",
    "expected_slot_order",
    "predicted_slot_order",
    "json_valid",
    "schema_valid",
    "intent_correct",
    "target_correct",
    "slot_correct",
    "target_selector_correct",
    "destination_type_correct",
    "slot_order_correct",
    "clear_table_correct",
    "latency_s",
    "response_chars",
    "slow_response",
    "error_type",
    "raw_response",
] + OLLAMA_TIMING_FIELDS

MODEL_CSV_FIELDS = [
    "model",
    "total_commands",
    "json_valid_rate",
    "schema_valid_rate",
    "intent_accuracy",
    "target_accuracy",
    "slot_accuracy",
    "target_selector_accuracy",
    "destination_type_accuracy",
    "slot_order_accuracy",
    "clear_table_accuracy",
    "warmup_latency_s",
    "warmup_latency_mean_s",
    "latency_mean_s",
    "latency_warm_mean_s",
    "latency_mean_excl_first_s",
    "latency_std_s",
    "slow_response_count",
    "cold_start_excluded",
    "guardrails_applied",
    "ollama_load_duration_mean_s",
    "ollama_prompt_eval_duration_mean_s",
    "ollama_eval_duration_mean_s",
    "ollama_tokens_per_second_mean",
]

WARMUP_SUMMARY_FIELDS = [
    "model",
    "warmup_count",
    "warmup_latency_total_s",
    "warmup_latency_mean_s",
    "warmup_success_all",
    "ollama_load_duration_mean_s",
    "ollama_prompt_eval_duration_mean_s",
    "ollama_eval_duration_mean_s",
    "ollama_tokens_per_second_mean",
]

WARMUP_DETAIL_FIELDS = [
    "model",
    "warmup_index",
    "warmup_text",
    "warmup_latency_s",
    "warmup_success",
    "warmup_error_type",
    "warmup_response_chars",
] + OLLAMA_TIMING_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación offline SLM → JSON robótico v1.1 (Ollama API)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Primer modelo + 3 primeros comandos",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Lista de modelos Ollama (sustituye la lista por defecto)",
    )
    parser.add_argument(
        "--max-commands",
        type=int,
        default=None,
        help="Limitar número de comandos del dataset",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=REQUEST_TIMEOUT_S,
        help="Timeout HTTP por request (segundos)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Desactivar warm-up por modelo antes de medir comandos",
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=DEFAULT_WARMUP_COUNT,
        help="Número de inferencias de warm-up por modelo (default: 2)",
    )
    parser.add_argument(
        "--exclude-first-latency",
        action="store_true",
        help="Excluir el primer comando medido de la media de latencia (no afecta accuracy)",
    )
    parser.add_argument(
        "--apply-guardrails",
        action="store_true",
        help="Aplicar semantic_guardrails deterministas (pipeline: SLM -> schema -> guardrails).",
    )
    return parser.parse_args()


def check_ollama_available() -> None:
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=5.0)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            "Ollama no está accesible en %s.\n"
            "Arranca el servicio (`ollama serve`) y verifica modelos.\nError: %s"
            % (OLLAMA_TAGS_URL, exc)
        ) from exc


def load_dataset(path: Path, max_commands: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("JSON inválido en %s línea %d: %s" % (path, line_no, exc))
    if not rows:
        raise ValueError("Dataset vacío: %s" % path)
    if max_commands is not None and max_commands > 0:
        rows = rows[: int(max_commands)]
    return rows


def extract_json_object(raw_text: str) -> Tuple[Optional[dict], str]:
    if raw_text is None or not str(raw_text).strip():
        return None, "empty_response"

    raw = str(raw_text).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, ""
        return None, "json_parse_error:not_object"
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    if start < 0:
        return None, "json_parse_error:no_brace"

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = raw[start : i + 1]
                try:
                    parsed = json.loads(chunk)
                    if isinstance(parsed, dict):
                        return parsed, ""
                    return None, "json_parse_error:not_object"
                except json.JSONDecodeError as exc:
                    return None, "json_parse_error:%s" % exc
    return None, "json_parse_error:unbalanced_braces"


def empty_ollama_timing() -> Dict[str, Any]:
    return {key: "" for key in OLLAMA_TIMING_FIELDS}


def ns_to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(int(value) / 1_000_000_000.0, 6)
    except (TypeError, ValueError):
        return None


def parse_ollama_timing(data: Dict[str, Any]) -> Dict[str, Any]:
    eval_duration_s = ns_to_seconds(data.get("eval_duration"))
    eval_count = data.get("eval_count")
    tokens_per_second: Any = ""
    if eval_duration_s is not None and eval_duration_s > 0 and eval_count is not None:
        try:
            tokens_per_second = round(int(eval_count) / eval_duration_s, 4)
        except (TypeError, ValueError):
            tokens_per_second = ""

    prompt_eval_count = data.get("prompt_eval_count")
    return {
        "ollama_total_duration_s": ns_to_seconds(data.get("total_duration")) or "",
        "ollama_load_duration_s": ns_to_seconds(data.get("load_duration")) or "",
        "ollama_prompt_eval_duration_s": ns_to_seconds(data.get("prompt_eval_duration"))
        or "",
        "ollama_eval_duration_s": eval_duration_s or "",
        "ollama_prompt_eval_count": prompt_eval_count if prompt_eval_count is not None else "",
        "ollama_eval_count": eval_count if eval_count is not None else "",
        "ollama_tokens_per_second": tokens_per_second,
    }


def warmup_texts_for_count(count: int) -> List[str]:
    if count <= 0:
        return []
    texts: List[str] = []
    for i in range(count):
        texts.append(WARMUP_PROMPTS[i % len(WARMUP_PROMPTS)])
    return texts


def warmup_inference_options(warmup_index: int, warmup_text: str) -> Tuple[int, int]:
    """Warm-up 1 ligero; warm-up 2+ con parámetros de evaluación (más representativo)."""
    if warmup_index >= 2 or warmup_text != WARMUP_PROMPTS[0]:
        return EVAL_NUM_CTX, EVAL_NUM_PREDICT
    return WARMUP_NUM_CTX, WARMUP_NUM_PREDICT


def mean_numeric_field(rows: List[Dict[str, Any]], key: str) -> Any:
    values: List[float] = []
    for row in rows:
        raw = row.get(key)
        if raw in ("", None):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return ""
    return round(statistics.mean(values), 4)


def build_ollama_payload(
    model: str,
    user_text: str,
    *,
    use_think: bool,
    num_ctx: int,
    num_predict: int,
) -> dict:
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
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
    if use_think:
        payload["think"] = False
    return payload


def call_ollama_chat(
    model: str,
    user_text: str,
    timeout_s: float,
    *,
    num_ctx: int = EVAL_NUM_CTX,
    num_predict: int = EVAL_NUM_PREDICT,
) -> Tuple[str, float, str, Dict[str, Any]]:
    use_think = True
    for _attempt in range(2):
        payload = build_ollama_payload(
            model,
            user_text,
            use_think=use_think,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                OLLAMA_CHAT_URL,
                json=payload,
                timeout=timeout_s,
            )
            latency = time.perf_counter() - t0
        except requests.Timeout:
            return "", time.perf_counter() - t0, "timeout", empty_ollama_timing()
        except requests.RequestException as exc:
            return (
                "",
                time.perf_counter() - t0,
                "request_error:%s" % exc,
                empty_ollama_timing(),
            )

        if resp.status_code == 200:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return "", latency, "request_error", empty_ollama_timing()
            timing = parse_ollama_timing(data)
            message = data.get("message") or {}
            content = str(message.get("content") or "")
            if not content.strip():
                return "", latency, "empty_response", timing
            return content, latency, "none", timing

        body = resp.text[:500]
        if resp.status_code in (400, 422) and use_think and "think" in body.lower():
            use_think = False
            continue
        return "", latency, "request_error", empty_ollama_timing()

    return "", 0.0, "request_error", empty_ollama_timing()


def normalize_request_error(req_err: str) -> str:
    if req_err == "none":
        return "none"
    if req_err == "timeout":
        return "timeout"
    if req_err == "empty_response":
        return "empty_response"
    if req_err.startswith("request_error"):
        return "request_error"
    return req_err


def run_warmups(
    model: str,
    timeout_s: float,
    warmup_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for warmup_index, warmup_text in enumerate(warmup_texts_for_count(warmup_count), start=1):
        num_ctx, num_predict = warmup_inference_options(warmup_index, warmup_text)
        print(
            "[WARMUP] model=%s index=%d/%d starting text=%r"
            % (model, warmup_index, warmup_count, warmup_text),
            flush=True,
        )
        raw_response, latency_s, req_err, timing = call_ollama_chat(
            model,
            warmup_text,
            timeout_s,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        error_type = normalize_request_error(req_err)
        success = error_type == "none"
        row = {
            "model": model,
            "warmup_index": warmup_index,
            "warmup_text": warmup_text,
            "warmup_latency_s": round(latency_s, 4),
            "warmup_success": success,
            "warmup_error_type": error_type,
            "warmup_response_chars": len(raw_response or ""),
            **timing,
        }
        details.append(row)
        print(
            "[WARMUP] model=%s index=%d latency_s=%.2f success=%s"
            % (model, warmup_index, latency_s, str(success).lower()),
            flush=True,
        )
        if not success:
            print(
                "  WARNING: warm-up %d falló (%s); continuando"
                % (warmup_index, error_type),
                file=sys.stderr,
            )

    latencies = [float(r["warmup_latency_s"]) for r in details]
    summary = {
        "model": model,
        "warmup_count": len(details),
        "warmup_latency_total_s": round(sum(latencies), 4) if latencies else 0.0,
        "warmup_latency_mean_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "warmup_success_all": all(r["warmup_success"] for r in details) if details else False,
        "ollama_load_duration_mean_s": mean_numeric_field(details, "ollama_load_duration_s"),
        "ollama_prompt_eval_duration_mean_s": mean_numeric_field(
            details, "ollama_prompt_eval_duration_s"
        ),
        "ollama_eval_duration_mean_s": mean_numeric_field(
            details, "ollama_eval_duration_s"
        ),
        "ollama_tokens_per_second_mean": mean_numeric_field(
            details, "ollama_tokens_per_second"
        ),
    }
    return details, summary


def labels_match(expected: Any, predicted: Any) -> bool:
    if expected is None and predicted is None:
        return True
    if expected is None or predicted is None:
        return False
    return str(expected) == str(predicted)


def expected_from_command(command: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "intent": command.get("expected_intent"),
        "target_label": command.get("expected_target_label"),
        "target_selector_type": command.get("expected_target_selector_type"),
        "destination_type": command.get("expected_destination_type"),
        "slot_index": command.get("expected_slot_index"),
        "slot_order": command.get("expected_slot_order"),
    }


def compute_clear_table_correct(
    exp: Dict[str, Any],
    pred: Dict[str, Any],
    *,
    intent_correct: bool,
    target_selector_correct: bool,
    destination_type_correct: bool,
    slot_order_correct: bool,
    target_correct: bool,
) -> Optional[bool]:
    if exp.get("intent") != "clear_table":
        return None
    return bool(
        intent_correct
        and target_correct
        and target_selector_correct
        and destination_type_correct
        and slot_order_correct
        and pred.get("intent") == "clear_table"
    )


def base_fail_row(
    model: str,
    command: Dict[str, Any],
    *,
    error_type: str,
    latency_s: float = 0.0,
    raw_response: str = "",
    ollama_timing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    exp = expected_from_command(command)
    row = {
        "model": model,
        "command_id": str(command.get("id", "")),
        "text": str(command.get("text", "")),
        "expected_intent": exp["intent"],
        "predicted_intent": None,
        "expected_target_label": exp["target_label"],
        "predicted_target_label": None,
        "expected_target_selector_type": exp["target_selector_type"],
        "predicted_target_selector_type": None,
        "expected_destination_type": exp["destination_type"],
        "predicted_destination_type": None,
        "expected_slot_index": exp["slot_index"],
        "predicted_slot_index": None,
        "expected_slot_order": slot_order_to_csv(exp["slot_order"]),
        "predicted_slot_order": "",
        "json_valid": False,
        "schema_valid": False,
        "intent_correct": False,
        "target_correct": False,
        "slot_correct": False,
        "target_selector_correct": False,
        "destination_type_correct": False,
        "slot_order_correct": False,
        "clear_table_correct": ""
        if exp["intent"] != "clear_table"
        else False,
        "latency_s": round(latency_s, 4),
        "response_chars": len(raw_response or ""),
        "slow_response": bool(latency_s > SLOW_LATENCY_THRESHOLD_S),
        "error_type": error_type,
        "raw_response": raw_response,
    }
    row.update(ollama_timing or empty_ollama_timing())
    return row


def classify_semantic_errors(row: Dict[str, Any]) -> str:
    if not row["schema_valid"]:
        return str(row.get("error_type") or "schema_validation_error")
    if not row["intent_correct"]:
        return "wrong_intent"
    if not row["target_correct"]:
        return "wrong_target"
    if not row["slot_correct"]:
        return "wrong_slot"
    if not row.get("target_selector_correct", True):
        return "wrong_target_selector"
    if not row.get("destination_type_correct", True):
        return "wrong_destination_type"
    if not row.get("slot_order_correct", True):
        return "wrong_slot_order"
    ct = row.get("clear_table_correct")
    if ct is False:
        return "wrong_clear_table"
    return "none"


def evaluate_one(
    model: str,
    command: Dict[str, Any],
    timeout_s: float,
    *,
    warn_slow: bool = True,
) -> Dict[str, Any]:
    exp = expected_from_command(command)

    raw_response, latency_s, req_err, ollama_timing = call_ollama_chat(
        model, str(command.get("text", "")), timeout_s
    )

    if latency_s > SLOW_LATENCY_THRESHOLD_S and warn_slow:
        print(
            "[SLOW_MODEL] model=%s command_id=%s latency_s=%.1f"
            % (model, command.get("id"), latency_s),
            file=sys.stderr,
        )

    if req_err != "none":
        if req_err == "timeout":
            err = "timeout"
        elif req_err == "empty_response":
            err = "empty_response"
        else:
            err = "request_error"
        return base_fail_row(
            model,
            command,
            error_type=err,
            latency_s=latency_s,
            raw_response=raw_response,
            ollama_timing=ollama_timing,
        )

    row = base_fail_row(
        model,
        command,
        error_type="none",
        latency_s=latency_s,
        raw_response=raw_response,
        ollama_timing=ollama_timing,
    )

    parsed, parse_err = extract_json_object(raw_response)
    if parsed is None:
        row["error_type"] = (
            "empty_response"
            if parse_err == "empty_response"
            else "json_parse_error"
        )
        return row

    row["json_valid"] = True
    ok, intent_model, _schema_err = validate_intent_payload(parsed)
    if not ok or intent_model is None:
        row["error_type"] = "schema_validation_error"
        return row

    row["schema_valid"] = True
    pred = predicted_from_model(intent_model)
    row["predicted_intent"] = pred["intent"]
    row["predicted_target_label"] = pred["target_label"]
    row["predicted_target_selector_type"] = pred["target_selector_type"]
    row["predicted_destination_type"] = pred["destination_type"]
    row["predicted_slot_index"] = pred["slot_index"]
    row["predicted_slot_order"] = slot_order_to_csv(pred["slot_order"])

    row["intent_correct"] = labels_match(exp["intent"], pred["intent"])
    row["target_correct"] = labels_match(exp["target_label"], pred["target_label"])
    row["slot_correct"] = labels_match(exp["slot_index"], pred["slot_index"])
    row["target_selector_correct"] = labels_match(
        exp["target_selector_type"], pred["target_selector_type"]
    )
    row["destination_type_correct"] = labels_match(
        exp["destination_type"], pred["destination_type"]
    )
    row["slot_order_correct"] = slot_orders_match(exp["slot_order"], pred["slot_order"])

    ct = compute_clear_table_correct(
        exp,
        pred,
        intent_correct=row["intent_correct"],
        target_selector_correct=row["target_selector_correct"],
        destination_type_correct=row["destination_type_correct"],
        slot_order_correct=row["slot_order_correct"],
        target_correct=row["target_correct"],
    )
    if ct is None:
        row["clear_table_correct"] = ""
    else:
        row["clear_table_correct"] = ct

    row["error_type"] = classify_semantic_errors(row)
    return row


def resolve_models(args: argparse.Namespace) -> List[str]:
    if args.models:
        return list(args.models)
    if args.smoke_test:
        return [DEFAULT_MODELS[0]]
    return list(DEFAULT_MODELS)


def _bool_metric(value: Any) -> bool:
    if value is True:
        return True
    if value is False:
        return False
    return False


def aggregate_by_model(
    rows: List[Dict[str, Any]],
    warmup_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    cold_start_excluded: bool = False,
    exclude_first_latency: bool = False,
) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(str(r["model"]), []).append(r)

    warmup_summaries = warmup_summaries or {}
    agg: List[Dict[str, Any]] = []
    for model, items in sorted(by_model.items()):
        n = len(items)
        if n == 0:
            continue

        def rate(key: str) -> float:
            return sum(1 for x in items if _bool_metric(x.get(key))) / n

        def rate_subset(key: str, predicate) -> float:
            subset = [x for x in items if predicate(x)]
            if not subset:
                return 0.0
            return sum(1 for x in subset if _bool_metric(x.get(key))) / len(subset)

        latencies = [float(x["latency_s"]) for x in items]
        warm_mean = round(statistics.mean(latencies), 4) if latencies else 0.0
        warm_std = (
            round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else 0.0
        )
        latencies_excl_first = latencies[1:] if len(latencies) > 1 else []
        mean_excl_first = (
            round(statistics.mean(latencies_excl_first), 4)
            if latencies_excl_first
            else ""
        )
        if not exclude_first_latency:
            mean_excl_first = ""

        warmup_info = warmup_summaries.get(model)
        warmup_latency_total = (
            warmup_info.get("warmup_latency_total_s") if warmup_info is not None else ""
        )
        warmup_latency_mean = (
            warmup_info.get("warmup_latency_mean_s") if warmup_info is not None else ""
        )

        agg.append(
            {
                "model": model,
                "total_commands": n,
                "json_valid_rate": round(rate("json_valid"), 4),
                "schema_valid_rate": round(rate("schema_valid"), 4),
                "intent_accuracy": round(rate("intent_correct"), 4),
                "target_accuracy": round(rate("target_correct"), 4),
                "slot_accuracy": round(rate("slot_correct"), 4),
                "target_selector_accuracy": round(rate("target_selector_correct"), 4),
                "destination_type_accuracy": round(rate("destination_type_correct"), 4),
                "slot_order_accuracy": round(rate("slot_order_correct"), 4),
                "clear_table_accuracy": round(
                    rate_subset(
                        "clear_table_correct",
                        lambda x: x.get("expected_intent") == "clear_table",
                    ),
                    4,
                ),
                "warmup_latency_s": warmup_latency_total,
                "warmup_latency_mean_s": warmup_latency_mean,
                "latency_mean_s": warm_mean,
                "latency_warm_mean_s": warm_mean,
                "latency_mean_excl_first_s": mean_excl_first,
                "latency_std_s": warm_std,
                "slow_response_count": sum(1 for x in items if x.get("slow_response")),
                "cold_start_excluded": cold_start_excluded,
                "ollama_load_duration_mean_s": mean_numeric_field(
                    items, "ollama_load_duration_s"
                ),
                "ollama_prompt_eval_duration_mean_s": mean_numeric_field(
                    items, "ollama_prompt_eval_duration_s"
                ),
                "ollama_eval_duration_mean_s": mean_numeric_field(
                    items, "ollama_eval_duration_s"
                ),
                "ollama_tokens_per_second_mean": mean_numeric_field(
                    items, "ollama_tokens_per_second"
                ),
            }
        )
    return agg


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if out.get("clear_table_correct") is None:
                out["clear_table_correct"] = ""
            writer.writerow(out)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_plots(model_metrics: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("matplotlib no instalado; omitiendo gráficas:", exc, file=sys.stderr)
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    short_labels = [m["model"].split(":")[0] for m in model_metrics]

    def bar_plot(values: List[float], title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(values))
        ax.bar(x, values, color="#4C72B0")
        ax.set_xticks(list(x))
        ax.set_xticklabels(short_labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(fig)

    bar_plot(
        [m["json_valid_rate"] for m in model_metrics],
        "JSON válido por modelo",
        "Tasa",
        "json_valid_rate_by_model.png",
    )
    bar_plot(
        [m["schema_valid_rate"] for m in model_metrics],
        "Schema válido (Pydantic v1.1) por modelo",
        "Tasa",
        "schema_valid_rate_by_model.png",
    )
    bar_plot(
        [m["intent_accuracy"] for m in model_metrics],
        "Accuracy de intent por modelo",
        "Accuracy",
        "intent_accuracy_by_model.png",
    )
    bar_plot(
        [m["slot_accuracy"] for m in model_metrics],
        "Accuracy de slot por modelo",
        "Accuracy",
        "slot_accuracy_by_model.png",
    )
    bar_plot(
        [m["clear_table_accuracy"] for m in model_metrics],
        "Accuracy clear_table por modelo",
        "Accuracy",
        "clear_table_accuracy_by_model.png",
    )
    bar_plot(
        [m["target_selector_accuracy"] for m in model_metrics],
        "Accuracy target_selector por modelo",
        "Accuracy",
        "target_selector_accuracy_by_model.png",
    )
    bar_plot(
        [m["destination_type_accuracy"] for m in model_metrics],
        "Accuracy destination.type por modelo",
        "Accuracy",
        "destination_type_accuracy_by_model.png",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(model_metrics))
    ax.bar(
        x,
        [m.get("latency_warm_mean_s", m["latency_mean_s"]) for m in model_metrics],
        color="#DD8452",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels, rotation=25, ha="right")
    ax.set_title("Latencia media (warm, sin warm-up) por modelo")
    ax.set_ylabel("Segundos")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "latency_mean_by_model.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [m.get("latency_warm_mean_s", m["latency_mean_s"]) for m in model_metrics]
    ys = [m["schema_valid_rate"] for m in model_metrics]
    ax.scatter(xs, ys, s=80, c="#55A868")
    for i, lbl in enumerate(short_labels):
        ax.annotate(lbl, (xs[i], ys[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Latencia media (s)")
    ax.set_ylabel("Schema valid rate")
    ax.set_title("Schema válido vs latencia")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "schema_valid_vs_latency.png", dpi=150)
    plt.close(fig)


def print_summary_table(
    model_metrics: List[Dict[str, Any]],
    *,
    exclude_first_latency: bool = False,
) -> None:
    print("\n=== SLM EVALUATION SUMMARY (schema v1.1) ===")
    header = (
        "model | schema_valid | intent | latency_mean_s | latency_excl_first_s | warmup_s"
    )
    print(header)
    print("-" * len(header))
    for m in model_metrics:
        warmup_s = m.get("warmup_latency_s", "")
        warmup_disp = (
            "%.2f" % float(warmup_s) if warmup_s not in ("", None) else "n/a"
        )
        excl = m.get("latency_mean_excl_first_s", "")
        if exclude_first_latency and excl not in ("", None):
            excl_disp = "%.2f" % float(excl)
        else:
            excl_disp = "n/a"
        print(
            "%s | %.3f | %.3f | %.2f | %s | %s"
            % (
                m["model"],
                m["schema_valid_rate"],
                m["intent_accuracy"],
                m.get("latency_mean_s", 0.0),
                excl_disp,
                warmup_disp,
            )
        )


def main() -> None:
    args = parse_args()
    models = resolve_models(args)
    max_cmds = 3 if args.smoke_test else args.max_commands
    apply_guardrails = bool(args.apply_guardrails)

    print("SLM eval v1.1 — Ollama HTTP (think=false, JSON Schema estricto)")
    print("[SYSTEM_PROMPT] chars=%d" % len(SYSTEM_PROMPT))
    print("Dataset:", DATASET_PATH)
    print("Timeout:", args.timeout, "s | Models:", models)
    if args.smoke_test:
        print("Modo: smoke-test (1 modelo, 3 comandos)")
    if max_cmds:
        print("Max commands:", max_cmds)
    warmup_enabled = not args.no_warmup
    warmup_count = max(0, int(args.warmup_count)) if warmup_enabled else 0
    print("Warm-up:", "on" if warmup_enabled else "off ( --no-warmup )")
    if warmup_enabled:
        print("Warm-up count:", warmup_count)
    if args.exclude_first_latency:
        print("Latencia: excluir primer comando de la media agregada")

    check_ollama_available()
    commands = load_dataset(DATASET_PATH, max_cmds)
    print("Comandos:", len(commands))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_OUTPUTS_PATH.exists():
        RAW_OUTPUTS_PATH.unlink()

    per_command_rows_raw: List[Dict[str, Any]] = []
    per_command_rows: List[Dict[str, Any]] = []
    warmup_detail_records: List[Dict[str, Any]] = []
    warmup_summary_records: List[Dict[str, Any]] = []
    warmup_summaries: Dict[str, Dict[str, Any]] = {}
    total = len(models) * len(commands)
    done = 0

    for model in models:
        print("\n=== Modelo:", model, "===")
        if warmup_enabled and warmup_count > 0:
            details, summary = run_warmups(model, float(args.timeout), warmup_count)
            warmup_detail_records.extend(details)
            warmup_summary_records.append(summary)
            warmup_summaries[model] = summary

        for cmd in commands:
            done += 1
            print("[%d/%d] %s | cmd %s" % (done, total, model, cmd.get("id")), flush=True)
            try:
                row_raw = evaluate_one(model, cmd, float(args.timeout))
            except Exception as exc:
                row_raw = base_fail_row(
                    model, cmd, error_type="request_error", raw_response=""
                )
                print("  ERROR:", exc, file=sys.stderr)

            row_final = row_raw
            guardrails_applied = False
            guardrailed_output = None

            if apply_guardrails:
                try:
                    parsed, _parse_err = extract_json_object(row_raw.get("raw_response") or "")
                    if parsed is not None:
                        corrected, guardrails_applied, reasons = apply_semantic_guardrails(
                            str(cmd.get("text", "")), parsed
                        )
                        guardrailed_output = json.dumps(corrected, ensure_ascii=False)

                        if reasons:
                            for r in reasons:
                                print(
                                    "[GUARDRAIL] command_id=%s %s"
                                    % (cmd.get("id"), r),
                                    flush=True,
                                )

                        ok, intent_model, _schema_err = validate_intent_payload(corrected)
                        if ok and intent_model is not None:
                            # Recalcular métricas con la salida corregida.
                            exp = expected_from_command(cmd)
                            row_final = base_fail_row(
                                model,
                                cmd,
                                error_type="none",
                                latency_s=row_raw.get("latency_s", 0.0),
                                raw_response=guardrailed_output,
                            )
                            row_final["json_valid"] = True
                            row_final["schema_valid"] = True

                            pred = predicted_from_model(intent_model)
                            row_final["predicted_intent"] = pred["intent"]
                            row_final["predicted_target_label"] = pred["target_label"]
                            row_final["predicted_target_selector_type"] = pred["target_selector_type"]
                            row_final["predicted_destination_type"] = pred["destination_type"]
                            row_final["predicted_slot_index"] = pred["slot_index"]
                            row_final["predicted_slot_order"] = slot_order_to_csv(pred["slot_order"])

                            row_final["intent_correct"] = labels_match(exp["intent"], pred["intent"])
                            row_final["target_correct"] = labels_match(exp["target_label"], pred["target_label"])
                            row_final["slot_correct"] = labels_match(exp["slot_index"], pred["slot_index"])
                            row_final["target_selector_correct"] = labels_match(
                                exp["target_selector_type"], pred["target_selector_type"]
                            )
                            row_final["destination_type_correct"] = labels_match(
                                exp["destination_type"], pred["destination_type"]
                            )
                            row_final["slot_order_correct"] = slot_orders_match(exp["slot_order"], pred["slot_order"])

                            ct = compute_clear_table_correct(
                                exp,
                                pred,
                                intent_correct=row_final["intent_correct"],
                                target_selector_correct=row_final["target_selector_correct"],
                                destination_type_correct=row_final["destination_type_correct"],
                                slot_order_correct=row_final["slot_order_correct"],
                                target_correct=row_final["target_correct"],
                            )
                            row_final["clear_table_correct"] = "" if ct is None else ct

                            row_final["error_type"] = classify_semantic_errors(row_final)
                        else:
                            # No inventamos: si la salida corregida no valida schema, mantenemos el fallo.
                            row_final = base_fail_row(
                                model,
                                cmd,
                                error_type="schema_validation_error",
                                latency_s=row_raw.get("latency_s", 0.0),
                                raw_response=guardrailed_output,
                            )
                            row_final["json_valid"] = True
                            row_final["schema_valid"] = False
                except Exception as exc:
                    print(
                        "  [GUARDRAIL] command_id=%s fallo: %s"
                        % (cmd.get("id"), exc),
                        file=sys.stderr,
                    )
                    row_final = row_raw
                    guardrails_applied = False

            per_command_rows_raw.append(row_raw)
            per_command_rows.append(row_final)

            append_jsonl(
                RAW_OUTPUTS_PATH,
                {
                    "model": row_raw["model"],
                    "command_id": row_raw["command_id"],
                    "text": row_raw["text"],
                    "raw_model_output": row_raw["raw_response"],
                    "raw_response": row_raw["raw_response"],
                    "guardrailed_response": guardrailed_output,
                    "guardrailed_output": guardrailed_output,
                    "guardrails_applied": guardrails_applied if apply_guardrails else False,
                    "metrics": {
                        k: row_final[k] for k in row_final if k not in ("raw_response", "text")
                    },
                },
            )

    if apply_guardrails:
        write_csv(METRICS_CMD_RAW_PATH, per_command_rows_raw, CMD_CSV_FIELDS)
    write_csv(METRICS_CMD_PATH, per_command_rows, CMD_CSV_FIELDS)
    if warmup_summary_records:
        write_csv(WARMUP_MODEL_PATH, warmup_summary_records, WARMUP_SUMMARY_FIELDS)
    if warmup_detail_records:
        write_csv(WARMUP_DETAILS_PATH, warmup_detail_records, WARMUP_DETAIL_FIELDS)
    model_metrics = aggregate_by_model(
        per_command_rows,
        warmup_summaries,
        cold_start_excluded=warmup_enabled,
        exclude_first_latency=bool(args.exclude_first_latency),
    )
    for m in model_metrics:
        m["guardrails_applied"] = bool(apply_guardrails)

    write_csv(METRICS_MODEL_PATH, model_metrics, MODEL_CSV_FIELDS)

    if apply_guardrails:
        model_metrics_raw = aggregate_by_model(
            per_command_rows_raw,
            warmup_summaries,
            cold_start_excluded=warmup_enabled,
            exclude_first_latency=bool(args.exclude_first_latency),
        )
        for m in model_metrics_raw:
            m["guardrails_applied"] = False
        write_csv(METRICS_MODEL_RAW_PATH, model_metrics_raw, MODEL_CSV_FIELDS)

    if not args.smoke_test:
        generate_plots(model_metrics)

    print_summary_table(
        model_metrics,
        exclude_first_latency=bool(args.exclude_first_latency),
    )

    print("\nResultados:")
    print(" ", METRICS_MODEL_PATH)
    print(" ", METRICS_CMD_PATH)
    if warmup_summary_records:
        print(" ", WARMUP_MODEL_PATH)
    if warmup_detail_records:
        print(" ", WARMUP_DETAILS_PATH)
    print(" ", RAW_OUTPUTS_PATH)
    print(" ", PLOTS_DIR)


if __name__ == "__main__":
    main()
