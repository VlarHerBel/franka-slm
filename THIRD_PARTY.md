# Atribución de código de terceros

Este repositorio del TFG integra componentes externos. El desarrollo propio
se concentra en `tfg_planner_slm/`, `slm_eval/`,
extensiones de `panda_vision/` y `panda_controller/`, launches `tfg_*`, goldens
y documentación asociada.

## Componentes upstream

### pymoveit2

- **Origen:** [AndrejOrsula/pymoveit2](https://github.com/AndrejOrsula/pymoveit2)
- **Ruta:** `src/panda_ws/pymoveit2/`
- **Licencia:** BSD 3-Clause (`src/panda_ws/pymoveit2/LICENSE`)
- **Uso en el TFG:** interfaz Python para planificación MoveIt 2

### gazebo_ycb

- **Origen:** [CentralLabFacilities/gazebo_ycb](https://github.com/CentralLabFacilities/gazebo_ycb)
- **Ruta:** `src/gazebo_ycb/`
- **Licencia:** MIT (`src/gazebo_ycb/LICENSE`)
- **Uso en el TFG:** modelos SDF/meshes YCB para Gazebo

### YCB Benchmarks (meshes)

- **Origen:** [YCB Object and Model Set](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/)
- **Ruta:** `src/gazebo_ycb/models/*/meshes/`
- **Licencia:** ver `src/gazebo_ycb/LICENSE_YCB`

### Stack Panda simulación (base)

- **Ruta:** `src/panda_ws/panda_description/`, `panda_moveit/`, `panda_bringup/`, `panda_controller/` (parte base)
- **Maintainer original:** Utkarsh (`kutkarsh706@gmail.com`)
- **Licencias:** Apache 2.0 (`panda_bringup`), MIT (`panda_vision`, `panda_vision_interfaces`), pendiente en otros paquetes
- **Nota:** estos paquetes se tomaron como base de simulación Franka Panda en ROS 2 Gazebo y fueron **ampliados sustancialmente** para el TFG (visión OBB, perfiles demo, integración SLM, spawn runtime YCB).

### Ultralytics YOLO

- **Uso:** backend de detección OBB (`yolo26n-obb.pt` y pesos fine-tuned)
- **Licencia:** [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) — revisar compatibilidad si redistribuyes pesos entrenados

### Ollama / modelos SLM

- **Uso:** inferencia local de SLMs (p. ej. Qwen3) vía API HTTP
- **Modelos:** descargados por el usuario con `ollama pull`; no se incluyen en este repositorio

## Qué no es código del TFG

- `src/gazebo_ycb/` — upstream (vendored)
- `src/panda_ws/pymoveit2/` — upstream
- Pesos `.pt` / `.pth` — generados o descargados; ver `models/vision/README.md`
- `datasets/` — generados localmente; ignorados por git
- `build/`, `install/`, `log/` — artefactos colcon
