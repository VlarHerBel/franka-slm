# Evaluación offline SLM → JSON robótico (v1.1)

Herramienta del TFG ([franka-slm](https://github.com/VlarHerBel/franka-slm)) para **comparar modelos de lenguaje pequeños (SLM) locales ya entrenados** vía [Ollama](https://ollama.com): el usuario escribe órdenes naturales en español y el evaluador mide si el modelo produce un **JSON del contrato robótico seguro y validable**.

**No se entrena ningún modelo aquí.** Solo inferencia + validación + métricas.

## Dos niveles del proyecto

| Nivel | Qué hace | Dónde |
|-------|----------|--------|
| **1. Evaluación offline (este directorio)** | Español → JSON v1.1 → Pydantic estricto → CSV y gráficas | `slm_eval/` |
| **2. Integración futura (fuera de este eval)** | JSON validado → nodos ROS 2 / MoveIt / controlador | `panda_ws/` (no tocado por este eval) |

El evaluador **no ejecuta ROS, Gazebo ni MoveIt**. Solo la API HTTP de Ollama (`/api/chat`).

## Schema v1.1 (resumen)

- **Intents:** `pick_place`, `clear_table`, `go_home`, `open_gripper`, `close_gripper`, `status`, `ask_clarification`, `reject`
- **Objetos ejecutables:** `cracker_box`, `sugar_box`, `chips_can`, `mustard_bottle`
- **`clear_table`:** recoger todos los objetos soportados visibles y colocarlos en huecos `[0,1,2,3]` en orden (`slots_ordered`)
- **Objetos no soportados** (banana, tomate, atún…): `reject` con `reject_reason=object_not_supported` (no mapear a un objeto válido)
- Claves españolas/genéricas (`tarea`, `objeto`, `accion`…) → **fallo de schema** (no se corrigen)

Las órdenes con pronombres o referencias genéricas sin contexto se tratan como `ask_clarification` para evitar inferencias inseguras. Objetos no permitidos, órdenes conversacionales o inseguras → `reject` (no `ask_clarification`). `"slot 0"` = índice 0; un solo objeto con slot → `pick_place`, no `clear_table`. `clear_table` exige `all_supported_visible_objects` + `slots_ordered` + `slot_order` [0,1,2,3] (ver few-shot en `prompts.py`).

## Requisitos

```bash
cd ~/tfg_robotics_ws/slm_eval
pip3 install -r requirements.txt
ollama serve   # otra terminal
```

Regenerar el dataset (opcional, 146 ejemplos):

```bash
python3 generate_commands_dataset.py
```

## Prompt compacto

El `SYSTEM_PROMPT` se mantiene **corto** (reglas esenciales + 3 ejemplos few-shot) para evitar timeouts en inferencia **CPU-only** con modelos como Qwen3. Al arrancar, `evaluate_models.py` imprime `[SYSTEM_PROMPT] chars=...`.

## Warm-up múltiple y latencias

Antes de medir comandos del dataset, el evaluador ejecuta **varias inferencias de calentamiento** por modelo (no entran en accuracy ni en `raw_outputs.jsonl` de evaluación):

| # | Orden de warm-up | Parámetros |
|---|------------------|------------|
| 1 | `"vuelve a home"` | `num_ctx=1024`, `num_predict=128` |
| 2 | `"coge la caja de galletas y déjala en el primer hueco"` | `num_ctx=2048`, `num_predict=256` (como evaluación real) |
| 3+ | Alterna las dos anteriores | Índice ≥2 usa parámetros de evaluación |

- Por defecto: `--warmup-count 2`. Desactivar todo con `--no-warmup` (ignora el count).
- `keep_alive: "30m"` en warm-up y evaluación (evita descargar el modelo entre peticiones).
- Consola: `[WARMUP] model=... index=1/2 ...`
- Si un warm-up falla → warning y se continúa.

### Tipos de latencia

| Métrica | Archivo / columna | Qué mide |
|---------|-------------------|----------|
| **Cold-start (warm-up)** | `warmup_by_model.csv` → `warmup_latency_total_s` | Suma de peticiones de calentamiento |
| **Wall-clock por comando** | `metrics_by_command.csv` → `latency_s` | Tiempo HTTP total (incluye posible primer comando más lento) |
| **Media evaluación** | `metrics_by_model.csv` → `latency_mean_s` | Media de todos los comandos medidos |
| **Sin primer comando** | `latency_mean_excl_first_s` (con `--exclude-first-latency`) | Media cmds 2…N; **accuracy sigue contando todos** |

El primer comando medido puede seguir siendo más lento que el resto aunque haya warm-up; `--exclude-first-latency` ayuda a reportar la latencia “ya caliente” sin ocultar el cmd 001 en `metrics_by_command.csv`.

### Métricas internas de Ollama

Cada respuesta de `/api/chat` puede incluir tiempos en nanosegundos (convertidos a segundos en CSV):

- `ollama_load_duration_s` — carga del modelo en memoria
- `ollama_prompt_eval_duration_s` — evaluación del prompt (system + user + schema)
- `ollama_eval_duration_s` — generación de la respuesta JSON
- `ollama_tokens_per_second` — `eval_count / eval_duration_s`

Detalle por warm-up: `results/warmup_details_by_model.csv`. Resumen por modelo: `results/warmup_by_model.csv` y columnas agregadas en `metrics_by_model.csv`.

## Evaluación raw vs evaluación con guardrails

- **Raw**: el JSON generado por el SLM se parsea y valida contra el schema v1.1; métricas miden solo el modelo.
- **Guardrailed** (opcional): tras validar el JSON, se aplica una fase determinista `semantic_guardrails.py` que solo corrige reglas cerradas del dominio (slots, objetos soportados, clear_table, ambigüedad segura).
  - Esto evita el “whack-a-mole” en prompts: el modelo puede equivocarse, pero el pipeline final fuerza salidas seguras cuando la regla es inequívoca.

Activa la fase guardrailed con `--apply-guardrails`.

## Controles anti-thinking / lentitud

- `think: false` en el payload (Qwen3 / Qwen3.5)
- `format`: JSON Schema estricto (`ROBOT_INTENT_JSON_SCHEMA`)
- Evaluación: `num_predict: 256`, `num_ctx: 2048`; warm-up 1 ligero: `128` / `1024`
- `temperature: 0`, `keep_alive: 30m`
- Timeout HTTP: 90 s por request
- Claves prohibidas → fallan validación (no se remapean)

## Flujo recomendado

```bash
cd ~/tfg_robotics_ws/slm_eval

# 1) Prueba rápida (primer modelo + 3 comandos)
python3 evaluate_models.py --smoke-test

# 2) Un modelo, 10 comandos
python3 evaluate_models.py --models granite4.1:3b-q4_K_M --max-commands 10

# 3) Evaluación completa (5 modelos × dataset)
python3 evaluate_models.py
```

Argumentos: `--smoke-test`, `--models M1 M2`, `--max-commands N`, `--timeout 90`, `--no-warmup`, `--warmup-count N`, `--exclude-first-latency`, `--apply-guardrails`.

## Dataset (`commands_dataset.jsonl`)

146 órdenes en español con etiquetas esperadas: `pick_place` (4 objetos + sinónimos), `clear_table`, gripper, home, status, ambiguas, rechazos (`object_not_supported`, `out_of_domain`, `unsafe_request`).

## Salidas

- `results/raw_outputs.jsonl`
- `results/metrics_by_command.csv` — por comando (salida final; si `--apply-guardrails` está activo, es la salida guardrailed)
- `results/metrics_by_command_raw.csv` — (solo si `--apply-guardrails`) métricas por comando del SLM puro (sin guardrails)
- `results/warmup_details_by_model.csv` — cada inferencia de warm-up con tiempos Ollama
- `results/warmup_by_model.csv` — resumen de warm-up por modelo
- `results/metrics_by_model.csv` — agregados: accuracies, `latency_mean_s`, `latency_mean_excl_first_s`, medias Ollama, … (incluye `guardrails_applied`)
- `results/metrics_by_model_raw.csv` — (solo si `--apply-guardrails`) agregados del SLM puro (sin guardrails)
- `results/plots/*.png` (omitidas en `--smoke-test`):
  - `json_valid_rate_by_model.png`
  - `schema_valid_rate_by_model.png`
  - `intent_accuracy_by_model.png`
  - `slot_accuracy_by_model.png`
  - `clear_table_accuracy_by_model.png`
  - `target_selector_accuracy_by_model.png`
  - `destination_type_accuracy_by_model.png`
  - `latency_mean_by_model.png`
  - `schema_valid_vs_latency.png`

## error_type

`none`, `timeout`, `request_error`, `empty_response`, `json_parse_error`, `schema_validation_error`, `wrong_intent`, `wrong_target`, `wrong_slot`, `wrong_target_selector`, `wrong_destination_type`, `wrong_slot_order`, `wrong_clear_table`

Si `latency_s > 30` → `slow_response=true` y log `[SLOW_MODEL]`.

## Modelos por defecto

`qwen3.5:4b-q4_K_M`, `granite4.1:3b-q4_K_M`, `qwen3:4b-instruct-2507-q4_K_M`, `llama3.2:3b`, `phi4-mini:3.8b-q4_K_M`

## Archivos principales

| Archivo | Rol |
|---------|-----|
| `intent_schema.py` | Pydantic v1.1 + JSON Schema Ollama |
| `prompts.py` | System prompt compacto (mundo cerrado, clear_table, rechazos) |
| `evaluate_models.py` | Bucle de evaluación, métricas, gráficas |
| `commands_dataset.jsonl` | Ground truth offline |
| `generate_commands_dataset.py` | Regenera el JSONL |
