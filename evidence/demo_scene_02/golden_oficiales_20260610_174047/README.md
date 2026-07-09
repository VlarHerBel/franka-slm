# demo_scene_02 — Goldens oficiales (bundle 2026-06-10)

**Directorio canónico** para enlazar SLM ↔ Gazebo ↔ pick/place de los 4 objetos YCB.

## Estructura

| Orden | Objeto | YAML oficial | Log |
|------:|--------|--------------|-----|
| 1 | cracker_box | `cracker_box/01_cracker_box_OFFICIAL_VALIDATED.yaml` | `01_cracker_box_OFFICIAL_VALIDATED_RUN.log` |
| 2 | chips_can | `chips_can/02_chips_can_OFFICIAL_VALIDATED.yaml` | `02_chips_can_OFFICIAL_VALIDATED_RUN.log` |
| 3 | sugar_box | `sugar_box/03_sugar_box_OFFICIAL_VALIDATED.yaml` | `03_sugar_box_OFFICIAL_VALIDATED_RUN.log` |
| 4 | mustard_bottle | `mustard_bottle/04_mustard_bottle_OFFICIAL_VALIDATED.yaml` | `04_mustard_bottle_OFFICIAL_VALIDATED_RUN.log` |

Índice máquina: **`MANIFEST.yaml`** (rutas relativas a este directorio).

Enlace estable (symlink): `~/tfg_robotics_ws/evidence/demo_scene_02/golden_oficiales` → este bundle.

## Uso SLM / scripts

```python
from pathlib import Path
import os
import yaml

ROOT = Path(os.environ.get("TFG_WS", "~/tfg_robotics_ws")).expanduser() / "evidence/demo_scene_02/golden_oficiales"
manifest = yaml.safe_load((ROOT / "MANIFEST.yaml").read_text())
for label, spec in manifest["objects"].items():
    golden_path = ROOT / spec["official_yaml"]
```

Variable de entorno recomendada:

```bash
export TFG_DEMO_GOLDEN_ROOT=$TFG_WS/evidence/demo_scene_02/golden_oficiales
```

## Runtime ROS (panda_controller)

El nodo sigue cargando perfiles desde `config/demo_profiles/` y goldens desde
`config/demo_candidate_cache/` (copias espejo de los YAML oficiales de aquí).

No usar como replay ejecutable:
`demo_scene_02_cracker_box_slot_0_full_execution_golden.yaml` (phase_count_mismatch).

## Otros sitios (histórico, no canónico)

- `evidence/demo_scene_02/cracker_box/` — benchmarks y backups antiguos
- `evidence/demo_scene_02/sugar_box/golden/` — origen del log sugar (copiado aquí)
- `logs/demo_scene_02/` — runs de desarrollo sueltos
