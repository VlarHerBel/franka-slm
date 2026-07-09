# tfg_planner_slm — Capa SLM segura (schema v1.1)

Repositorio del TFG: [github.com/VlarHerBel/franka-slm](https://github.com/VlarHerBel/franka-slm)

Planificador de intención en lenguaje natural para el TFG de robótica (Franka Panda, ROS 2).  
**No ejecuta ROS por defecto**: genera intents validados y previews seguros (`dry_run=true`).

## Modelo y endpoint

| Parámetro | Valor |
|-----------|--------|
| Modelo | `qwen3:4b-instruct-2507-q4_K_M` |
| API | `http://127.0.0.1:11434/api/chat` |
| Schema | v1.1 (`intent_schema.py`) |
| Warm-up | 2 peticiones (`vuelve a home` + pick_place representativo) |
| `keep_alive` | `30m` |

Requisito: Ollama en marcha (`ollama serve`) con el modelo descargado.

## Pipeline

```
Usuario (texto)
  → Ollama (JSON schema estricto, think=false)
  → extracción JSON
  → validación Pydantic v1.1
  → semantic_guardrails (reglas cerradas del dominio)
  → resolución «cualquier cajón/slot» (slot_state + estado de sesión)
  → comprobación de ocupación (slot explícito / ya colocado / sin huecos libres)
  → revalidación schema
  → command_dispatcher (acción interna + preview ROS)
```

Los **guardrails semánticos** corrigen solo reglas deterministas y seguras:

- slots ordinales (`tercer hueco` / `tercer cajón` → índice 2, no 3); **cajón/cajon** es sinónimo de slot/hueco/espacio
- «cualquier cajón/slot/hueco» → primer slot libre según `SlotOccupancy` de la sesión (no lo decide el SLM)
- slot explícito ya ocupado por **otro** objeto → `ask_clarification` (no ejecutable, no sobrescribe estado)
- mismo objeto ya en ese cajón → `ask_clarification` («ya contiene …; no hace falta moverlo»)
- objeto concreto + slot → `pick_place` (no `clear_table` erróneo)
- toda la mesa / todos los objetos → `clear_table`
- pronombres sin contexto → `ask_clarification`
- objetos no soportados / fuera de dominio / inseguro → `reject`

No inventan objetos ni ejecutan comandos.

**Resolución lingüística vs estado de slots:** los guardrails interpretan texto (ordinales, objetos, «cualquier cajón»). `SlotOccupancy` mantiene qué cajones están libres u ocupados en la sesión. Tras resolver el destino, `apply_slot_occupancy_checks` impide un `pick_place` ejecutable si el cajón destino está tomado por otro objeto. El JSON final para el dispatcher siempre lleva `destination.type="slot"` y un `slot_index` concreto (`0`–`3`).

## Estructura del paquete

```
tfg_planner_slm/
  intent_schema.py      # Pydantic + JSON Schema Ollama
  prompts.py            # System prompt compacto (~3k chars)
  semantic_guardrails.py
  slot_state.py          # ocupación de slots + «cualquier cajón»
  ros_command_executor.py # ejecución segura ros2 (solo pick_place)
  ros_preflight.py       # checks antes de mover Gazebo
  json_extract.py
  ollama_client.py      # warm-up + generate_intent
  intent_parser.py      # parse_user_command()
  command_dispatcher.py # preview seguro (sin ROS real)
  slm_backend_session.py # estado ready/warming_up para futuras UIs
  cli_chat.py           # prueba por terminal
```

El código antiguo (`llm_node.py`, contrato PICK/UNKNOWN) se mantiene pero **no es la base** de este pipeline.

## Cómo probar la CLI

Desde el directorio del paquete (o con el workspace en `PYTHONPATH`):

```bash
cd ~/tfg_robotics_ws/src/tfg_planner_slm
pip3 install pydantic requests   # si no están instalados
```

### Build (ROS 2)

```bash
cd ~/tfg_robotics_ws
colcon build --packages-select tfg_planner_slm
source install/setup.bash
```

### Modo one-shot

Cada invocación arranca un proceso Python nuevo: se ejecuta el warm-up (salvo `--no-warmup`) y luego una sola orden.

```bash
python3 -m tfg_planner_slm.cli_chat "coge la caja de galletas y déjala en el tercer hueco"
python3 -m tfg_planner_slm.cli_chat "deja el azúcar en slot 0"
python3 -m tfg_planner_slm.cli_chat "recógeme toda la mesa"
python3 -m tfg_planner_slm.cli_chat "ponlo allí"
python3 -m tfg_planner_slm.cli_chat "coge el bote amarillo"
```

También disponible como comando instalado:

```bash
tfg_planner_cli "coge la caja de galletas y déjala en el tercer cajón"
```

### Modo interactivo (simula web/backend)

El warm-up se hace **una sola vez al arrancar**; las órdenes siguientes reutilizan el modelo ya cargado en Ollama (`keep_alive`), sin repetir calentamiento. Es el comportamiento esperado en un backend o chat web persistente.

Para una UI web nueva, `SlmBackendSession` expone `status="warming_up"` y `message="Iniciando asistente..."` mientras se precalienta el modelo. El frontend debe bloquear el chat hasta recibir `ready=true`; después cada mensaje entra directamente al pipeline SLM sin repetir warm-up.

```bash
python3 -m tfg_planner_slm.cli_chat --interactive
```

Tras el mensaje `[SLM_CHAT] ready. Escribe una orden o 'salir'.`, escribe órdenes en bucle. Salida: `salir`, `exit`, `quit` o `Ctrl+C`.

Comandos internos en modo interactivo:

- `/slots` — muestra ocupación simulada (`slot 0: libre`, `slot 1: sugar_box`, …)
- `/reset_slots` — libera todos los slots
- `/free_slot N` — libera el slot `N` (0–3) para pruebas

**Simulación de ocupación en CLI:** con `--simulate-slot-fill` (por defecto en `--interactive`), tras un `pick_place` preview ejecutable se marca el slot (`[SLOT_STATE] slot N marked occupied by …`). Con `--no-simulate-slot-fill` solo se muestra `[SLOT_STATE] preview only; slot state not updated`. En **one-shot** la simulación está desactivada por defecto. Cuando se conecte ROS real, el slot solo debe marcarse ocupado tras **éxito** del movimiento (`returncode=0`), no tras el preview.

Un slot explícito ocupado por otro objeto no se sobrescribe: la orden pasa a `ask_clarification` con un mensaje del tipo *«El tercer cajón ya está ocupado por chips_can…»*.

En web/backend el mismo `SlotOccupancy` debe vivir durante toda la conversación. Antes de cualquier ejecución real, el JSON final siempre lleva `destination.type="slot"` y un `slot_index` concreto (`0`–`3`); nunca `slots_ordered` en `pick_place`.

### Ejecución ROS (Fase 1/2)

- **Por defecto** la CLI solo imprime preview; **no** lanza ROS.
- `--execute`: ejecuta `pick_place` con `dry_run:=true` (no mueve Gazebo).
- `--execute-sim`: ejecuta `pick_place` con `dry_run:=false` (mueve el robot en Gazebo) y hace preflight (salvo `--skip-preflight`).

Ejemplos:

```bash
# Dry-run real (no mueve el robot); pide confirmación salvo --yes
tfg_planner_cli --execute "coge la caja de galletas y déjala en el tercer cajón"
tfg_planner_cli --execute --yes "deja el azúcar en el primer cajón"

# Gazebo real: requiere confirmación fuerte
tfg_planner_cli --execute-sim --i-understand-this-moves-gazebo-robot "deja las galletas en el primer cajón"
```

### Opciones CLI

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--interactive` | — | Bucle REPL; warm-up solo al inicio |
| `--no-warmup` | — | Omitir calentamiento |
| `--warmup-count N` | `2` | Inferencias de warm-up al arrancar |
| `--model NAME` | `qwen3:4b-instruct-2507-q4_K_M` | Modelo Ollama |
| `--ollama-url URL` | `http://127.0.0.1:11434` | URL base de Ollama |
| `--timeout 90` | `90` | Timeout HTTP (s) |
| `--hide-raw` | — | No imprimir JSON raw del modelo |
| `--show-raw` | (sí por defecto) | Forzar mostrar JSON raw |
| `--simulate-slot-fill` | interactivo: sí | Marcar slots tras pick_place preview |
| `--no-simulate-slot-fill` | — | No actualizar `SlotOccupancy` tras preview |
| `--execute` | — | Ejecutar `pick_place` con `dry_run:=true` |
| `--execute-sim` | — | Ejecutar `pick_place` con `dry_run:=false` (mueve Gazebo) |
| `--yes` | — | No pedir confirmación (excepto `--execute-sim` sin flag fuerte) |
| `--i-understand-this-moves-gazebo-robot` | — | Flag obligatorio para `--execute-sim` |
| `--skip-preflight` | — | Saltar checks antes de `--execute-sim` |
| `--ros-timeout-sec` | `300` | Timeout de ejecución ROS (s) |

Si el JSON **raw** no pasa el schema pero los guardrails producen un JSON final válido (p. ej. `ponlo allí`), la CLI muestra `=== Avisos del modelo raw ===` y **no** `=== Errores finales ===`. Los errores finales solo aparecen cuando el intent validado sigue fallando tras guardrails.

Salida esperada (resumen):

| Orden | Intent | Notas |
|-------|--------|--------|
| galletas + tercer hueco | `pick_place` | `cracker_box`, `slot_index=2` |
| azúcar + slot 0 | `pick_place` | `sugar_box`, `slot_index=0` |
| patatas + tercer hueco | `pick_place` | `chips_can`, `slot_index=2` |
| mostaza + slot 1 | `pick_place` | `mustard_bottle`, `slot_index=1` |
| recógeme toda la mesa | `clear_table` | preview, sin ejecución |
| ponlo allí | `ask_clarification` | pregunta de clarificación |
| bote amarillo | `reject` | `object_not_supported` |
| azúcar + tercer cajón | `pick_place` | `sugar_box`, `slot_index=2` |
| galletas en cualquier cajón (sesión vacía) | `pick_place` | `cracker_box`, `slot_index=0` |

Comprobación offline (sin Ollama):

```bash
python3 -m tfg_planner_slm.guardrails_self_check
```

## Seguridad

- La SLM **no** ejecuta terminal ni ROS directamente.
- `dry_run=true` y `require_confirmation=true` por defecto en el contrato (la confirmación se hace en CLI; `perception_to_pregrasp_test` no expone `require_confirmation` como parámetro).
- Objetos permitidos: `cracker_box`, `sugar_box`, `chips_can`, `mustard_bottle`.
- Slots permitidos: `0`, `1`, `2`, `3`.
- Solo `pick_place` genera un preview de comando ROS (`perception_to_pregrasp_test` con `dry_run:=true`).

## Integración con `panda_controller` (perfil pick_place validado)

Los **defaults** de `perception_to_pregrasp_test.py` en `panda_controller` corresponden al flujo validado en Gazebo:

- `execution_cycle_mode=snapshot_execute_home`
- `plan_before_prelude=true`, `enable_safe_pregrasp_stage=false` (direct_pregrasp)
- gripper `joint7_direct`, transporte `distance_based`, place slots ordenados
- `dry_run=true` por seguridad en el nodo

Comando mínimo manual (con Gazebo/MoveIt/percepción lanzados):

```bash
ros2 run panda_controller perception_to_pregrasp_test --ros-args \
  -p dry_run:=false \
  -p target_label:=cracker_box \
  -p place_slot_index:=0
```

`tfg_planner_cli --execute-sim` envía el mismo perfil de forma explícita (trazabilidad) además de `target_label` y `place_slot_index` desde el JSON final guardrailed.

## Relación con `slm_eval/`

La evaluación offline (`~/tfg_robotics_ws/slm_eval/`) midió el modelo y el pipeline con guardrails (146 comandos, métricas 1.0 con guardrails).  
Este paquete **reutiliza** el mismo contrato y guardrails para integración futura con nodos ROS, sin modificar `panda_controller`, percepción ni Gazebo en esta fase.

## Próximos pasos (fuera de este paquete)

- Nodo ROS que llame a `parse_user_command` + `dispatch_command`
- Confirmación explícita del usuario antes de `dry_run=false`
- Ejecución real de `clear_table`, `go_home`, gripper
