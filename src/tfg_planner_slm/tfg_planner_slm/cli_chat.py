#!/usr/bin/env python3
"""CLI para probar el pipeline SLM + guardrails.

Por defecto hace solo preview (no ejecuta ROS).
Con flags explícitos permite ejecutar `pick_place` en dry_run o en simulación.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import List, Tuple

from .command_dispatcher import InternalAction, dispatch_command
from .intent_parser import ParsedCommandResult, final_intent_json_string, parse_user_command
from .ollama_client import DEFAULT_MODEL, DEFAULT_OLLAMA_BASE_URL, warmup_model
from .ros_command_executor import ExecutionResult, execute_pick_place_action
from .ros_preflight import format_preflight_result, run_gazebo_preflight_checks
from .slot_state import SlotOccupancy

_EXIT_WORDS = frozenset({"salir", "exit", "quit"})


@dataclass
class CliConfig:
    model: str
    timeout_s: float
    warmup_count: int
    no_warmup: bool
    ollama_url: str
    show_raw: bool
    simulate_slot_fill: bool = False
    execute: bool = False
    execute_sim: bool = False
    yes: bool = False
    ros_timeout_sec: float = 300.0
    skip_preflight: bool = False
    i_understand_this_moves_gazebo_robot: bool = False
    demo_profile: bool = False


def _print_section(title: str) -> None:
    print("\n=== %s ===" % title)


def _partition_errors(parsed: ParsedCommandResult) -> Tuple[List[str], List[str]]:
    """Separa avisos del JSON raw vs errores que persisten tras guardrails."""
    raw_warnings: List[str] = []
    final_errors: List[str] = []

    for err in parsed.errors:
        if err.startswith("schema_validation_error:"):
            if parsed.final_intent is not None:
                detail = err.split(":", 1)[1] if ":" in err else err
                raw_warnings.append("raw_schema_validation_error: %s" % detail)
            else:
                final_errors.append(err)
        elif err.startswith("guardrailed_schema_validation_error:"):
            final_errors.append(err)
        else:
            final_errors.append(err)

    return raw_warnings, final_errors


def _run_warmup(config: CliConfig) -> None:
    if config.no_warmup or config.warmup_count <= 0:
        return
    print("[SLM_STATUS] warming_up", flush=True)
    warmup_model(
        config.model,
        config.warmup_count,
        timeout_s=config.timeout_s,
        ollama_url=config.ollama_url,
    )


def _maybe_update_slot_state_after_preview(
    parsed: ParsedCommandResult,
    action: InternalAction,
    slot_occupancy: SlotOccupancy,
    *,
    simulate_slot_fill: bool,
) -> None:
    """Simulación local (solo preview, sin ROS): opcional."""
    if not simulate_slot_fill:
        if action.intent == "pick_place" and action.execution_supported and parsed.final_intent:
            print("[SLOT_STATE] preview only; slot state not updated", flush=True)
        return

    intent = parsed.final_intent
    if not intent or intent.get("intent") != "pick_place" or not action.execution_supported:
        return

    dest = intent.get("destination") or {}
    slot_index = dest.get("slot_index")
    target = intent.get("target_label")
    if slot_index is None or not target:
        return

    if slot_occupancy.mark_occupied(int(slot_index), str(target)):
        print("[SLOT_STATE] slot %s marked occupied by %s" % (slot_index, target), flush=True)
    else:
        occupant = slot_occupancy.get_occupant(int(slot_index))
        print(
            "[SLOT_STATE] slot %s not updated (occupied by %s)" % (slot_index, occupant),
            flush=True,
        )


def _maybe_update_slot_state_after_execution(
    *,
    action: InternalAction,
    exec_result: ExecutionResult,
    slot_occupancy: SlotOccupancy,
) -> None:
    if not exec_result.started or not exec_result.success:
        return
    if action.intent != "pick_place" or not action.execution_supported:
        return
    if action.slot_index is None or action.target_label is None:
        return
    if slot_occupancy.mark_occupied(int(action.slot_index), str(action.target_label)):
        print(
            "[SLOT_STATE] slot %d marked occupied by %s after successful gazebo execution"
            % (int(action.slot_index), str(action.target_label)),
            flush=True,
        )


def process_user_command(
    text: str,
    config: CliConfig,
    slot_occupancy: SlotOccupancy | None = None,
) -> int:
    """Ejecuta el pipeline para una orden e imprime el resultado."""
    occupancy = slot_occupancy if slot_occupancy is not None else SlotOccupancy()
    print("[SLM_STATUS] parsing_command", flush=True)
    parsed = parse_user_command(
        text,
        model=config.model,
        timeout_s=config.timeout_s,
        ollama_url=config.ollama_url,
        slot_occupancy=occupancy,
    )
    if parsed.final_intent is not None:
        print("[SLM_STATUS] intent_validated", flush=True)
    action = dispatch_command(parsed.final_intent)
    if parsed.guardrails_applied:
        print("[SLM_STATUS] guardrails_applied", flush=True)
    print("[SLM_STATUS] action_ready", flush=True)
    raw_warnings, final_errors = _partition_errors(parsed)

    _print_section("Usuario")
    print(parsed.original_text)

    if config.show_raw:
        _print_section("JSON raw del modelo")
        if parsed.raw_model_json:
            print(json.dumps(parsed.raw_model_json, ensure_ascii=False, indent=2))
        else:
            print(parsed.raw_response or "(vacío)")
            if parsed.errors and not parsed.raw_model_json:
                print("Errores:", ", ".join(parsed.errors))

    _print_section("JSON final (tras guardrails)")
    final_json = final_intent_json_string(parsed)
    if final_json:
        print(final_json)
    else:
        print("(no validado)")
        if parsed.guardrailed_json:
            print("Guardrailed (sin validar schema):")
            print(json.dumps(parsed.guardrailed_json, ensure_ascii=False, indent=2))

    if parsed.guardrails_applied:
        print("\nGuardrails aplicados:", ", ".join(parsed.guardrail_reasons))

    _print_section("Acción interna")
    print("intent:", action.intent)
    print("preview:", action.preview)
    print("execution_supported:", action.execution_supported)
    print("dry_run:", action.dry_run)

    if action.ros_command_preview:
        _print_section("Comando ROS preview")
        print(action.ros_command_preview)

    if raw_warnings:
        _print_section("Avisos del modelo raw")
        for warn in raw_warnings:
            print("-", warn)

    if final_errors:
        _print_section("Errores finales")
        for err in final_errors:
            print("-", err)

    print("\nLatencia Ollama: %.2f s" % parsed.ollama_latency_s)

    # Preview: NO ejecuta ROS.
    if not config.execute and not config.execute_sim:
        _maybe_update_slot_state_after_preview(
            parsed, action, occupancy, simulate_slot_fill=config.simulate_slot_fill
        )
        return 0 if parsed.final_intent else 1

    # Ejecución ROS
    if config.execute_sim and not config.skip_preflight:
        print("[SLM_STATUS] ros_preflight", flush=True)
        pre = run_gazebo_preflight_checks(timeout_s=10.0)
        print("[ROS_PREFLIGHT] %s" % ("ok" if pre.ok else "failed"), flush=True)
        if pre.blocking_errors:
            for e in pre.blocking_errors:
                print("[ROS_PREFLIGHT] blocking_error=%s" % e, flush=True)
        if pre.warnings:
            for w in pre.warnings:
                print("[ROS_PREFLIGHT] warning=%s" % w, flush=True)
        if not pre.ok:
            print(format_preflight_result(pre))
            print("[ROS_EXECUTOR] skipped reason=preflight_failed", flush=True)
            return 1

    print("[SLM_STATUS] executing_ros", flush=True)
    exec_result = execute_pick_place_action(
        action,
        slot_occupancy=occupancy,
        execute=config.execute,
        execute_sim=config.execute_sim,
        assume_yes=config.yes,
        i_understand_this_moves_gazebo_robot=config.i_understand_this_moves_gazebo_robot,
        ros_timeout_sec=config.ros_timeout_sec,
        demo_profile=config.demo_profile,
    )
    if not exec_result.started:
        print("[ROS_EXECUTOR] skipped reason=%s" % exec_result.skipped_reason, flush=True)
    elif exec_result.success:
        print("[SLM_STATUS] execution_success", flush=True)
    else:
        print("[SLM_STATUS] execution_failed", flush=True)

    _print_section("ROS ExecutionResult")
    print(
        json.dumps(
            {
                "started": exec_result.started,
                "success": exec_result.success,
                "returncode": exec_result.returncode,
                "duration_s": exec_result.duration_s,
                "skipped_reason": exec_result.skipped_reason,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if exec_result.stdout.strip():
        _print_section("ROS stdout")
        print(exec_result.stdout)
    if exec_result.stderr.strip():
        _print_section("ROS stderr")
        print(exec_result.stderr)

    # Actualizar SlotOccupancy solo tras éxito real en Gazebo (execute_sim).
    if config.execute_sim and exec_result.success:
        _maybe_update_slot_state_after_execution(
            action=action, exec_result=exec_result, slot_occupancy=occupancy
        )
    else:
        print("[SLOT_STATE] preview only; slot state not updated", flush=True)

    if not exec_result.started:
        return 1
    return 0 if exec_result.success else 1


def run_interactive(config: CliConfig) -> int:
    """Bucle REPL: warm-up una vez, luego reutiliza el modelo en memoria de Ollama."""
    _run_warmup(config)
    slot_occupancy = SlotOccupancy()
    print("[SLM_STATUS] ready", flush=True)
    print("[SLM_CHAT] ready. Escribe una orden o 'salir'.", flush=True)
    print(
        "[SLM_CHAT] Comandos: /slots, /reset_slots, /free_slot N",
        flush=True,
    )
    if config.simulate_slot_fill:
        print("[SLM_CHAT] Simulación de ocupación de slots: activada", flush=True)
    else:
        print("[SLM_CHAT] Simulación de ocupación de slots: desactivada", flush=True)

    exit_code = 0
    while True:
        print("[SLM_STATUS] waiting_next_order", flush=True)
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        text = line.strip()
        if not text:
            continue
        if text.lower() in _EXIT_WORDS:
            break
        if text == "/slots":
            print("[SLOT_STATE] %s" % slot_occupancy.format_status(), flush=True)
            continue
        if text == "/reset_slots":
            slot_occupancy.reset()
            print("[SLOT_STATE] todos los slots liberados", flush=True)
            continue
        if text.startswith("/free_slot"):
            parts = text.split()
            if len(parts) == 2 and parts[1].isdigit():
                n = int(parts[1])
                slot_occupancy.mark_free(n)
                print("[SLOT_STATE] slot %d liberado" % n, flush=True)
            else:
                print("[SLOT_STATE] uso: /free_slot N (N=0..3)", flush=True)
            continue

        code = process_user_command(text, config, slot_occupancy)
        if code != 0:
            exit_code = code

    return exit_code


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parser SLM v1.1 + guardrails (preview seguro, sin ROS)"
    )
    parser.add_argument("text", nargs="?", help="Orden en español (modo one-shot)")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Modo chat: warm-up una vez, varias órdenes sin reiniciar",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Ollama")
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="URL base de Ollama (default: %(default)s)",
    )
    parser.add_argument("--timeout", type=float, default=90.0, help="Timeout HTTP (s)")
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=2,
        help="Inferencias de warm-up al arrancar (0 para desactivar)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="No ejecutar warm-up",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        default=None,
        help="Mostrar JSON raw del modelo (por defecto: sí)",
    )
    parser.add_argument(
        "--hide-raw",
        action="store_true",
        help="Ocultar JSON raw del modelo",
    )
    parser.add_argument(
        "--simulate-slot-fill",
        action="store_true",
        default=None,
        help="Tras pick_place preview, marcar slot ocupado (interactivo: default sí)",
    )
    parser.add_argument(
        "--no-simulate-slot-fill",
        action="store_true",
        help="No actualizar SlotOccupancy tras preview",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Ejecutar pick_place en ROS con dry_run:=true (requiere confirmación)",
    )
    parser.add_argument(
        "--execute-sim",
        action="store_true",
        help="Ejecutar pick_place en Gazebo con dry_run:=false (requiere preflight y confirmación fuerte)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="No preguntar confirmación (NO aplica a --execute-sim sin flag fuerte)",
    )
    parser.add_argument(
        "--i-understand-this-moves-gazebo-robot",
        action="store_true",
        help="Confirmación fuerte obligatoria para --execute-sim",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Saltar preflight (no recomendado) para --execute-sim",
    )
    parser.add_argument(
        "--ros-timeout-sec",
        type=float,
        default=300.0,
        help="Timeout de ejecución ROS (s)",
    )
    parser.add_argument(
        "--demo-profile",
        action="store_true",
        help=(
            "pick_place con perfil demo_scene_02 (golden/authoritative). "
            "Por defecto usa perfil simple validado."
        ),
    )
    return parser


def _resolve_simulate_slot_fill(args: argparse.Namespace) -> bool:
    if args.no_simulate_slot_fill:
        return False
    if args.simulate_slot_fill:
        return True
    return bool(args.interactive)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    show_raw = not args.hide_raw
    if args.show_raw is not None:
        show_raw = args.show_raw

    config = CliConfig(
        model=args.model,
        timeout_s=args.timeout,
        warmup_count=args.warmup_count,
        no_warmup=args.no_warmup,
        ollama_url=args.ollama_url,
        show_raw=show_raw,
        simulate_slot_fill=_resolve_simulate_slot_fill(args),
        execute=bool(args.execute),
        execute_sim=bool(args.execute_sim),
        yes=bool(args.yes),
        ros_timeout_sec=float(args.ros_timeout_sec),
        skip_preflight=bool(args.skip_preflight),
        i_understand_this_moves_gazebo_robot=bool(
            args.i_understand_this_moves_gazebo_robot
        ),
        demo_profile=bool(args.demo_profile),
    )

    if args.interactive:
        return run_interactive(config)

    if not args.text:
        parser.error(
            'Indica una orden o usa --interactive, por ejemplo: '
            '"coge la caja de galletas"'
        )

    _run_warmup(config)
    return process_user_command(args.text, config)


if __name__ == "__main__":
    sys.exit(main())
