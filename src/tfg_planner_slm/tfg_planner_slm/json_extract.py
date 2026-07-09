"""Extracción robusta de objeto JSON desde respuestas de SLM."""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple


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
