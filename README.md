# franka-slm — TFG de Robótica

[github.com/VlarHerBel/franka-slm](https://github.com/VlarHerBel/franka-slm)

Workspace ROS 2: SLM + simulación Franka Panda (Gazebo), visión YCB/OBB y pick-and-place.

Código del TFG: `tfg_planner_slm`, `slm_eval`, `panda_vision`, `panda_controller`. Dependencias de terceros: ver [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Estructura

```
tfg_robotics_ws/
├── iniciar_tfg.sh
├── models/vision/       # pesos YOLO (no versionados)
├── slm_eval/
├── docs/
└── src/
    ├── tfg_planner_slm/
    ├── gazebo_ycb/
    └── panda_ws/
```

## Requisitos

Ubuntu 22.04 · ROS 2 Humble · Gazebo (ros_gz) · MoveIt 2 · Python 3.10+ · Ollama · Ultralytics YOLO

## Instalación

```bash
git clone https://github.com/VlarHerBel/franka-slm.git ~/tfg_robotics_ws
cd ~/tfg_robotics_ws

# Entorno virtual propio (no va en git; cada usuario crea el suyo)
python3 -m venv ~/tfg_env
source ~/tfg_env/bin/activate
pip install -r slm_eval/requirements.txt
# Visión YOLO/Ultralytics: instalar según src/panda_ws/panda_vision/README_YCB_OBB.md

mkdir -p models/vision    # ver models/vision/README.md

source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

`iniciar_tfg.sh` activa `~/tfg_env` si existe; puedes usar otra ruta o nombre de venv.

## Uso

```bash
export TFG_WS=~/tfg_robotics_ws
source $TFG_WS/iniciar_tfg.sh

ros2 launch panda_bringup tfg_ycb_pick_place.launch.py \
  scene_preset:=demo_scene_02 \
  model_path:=$TFG_WS/models/vision/yolo_obb_best.pt

# SLM web (otra terminal)
ros2 launch tfg_planner_slm tfg_web_api.launch.py
```

Evaluación offline: `cd slm_eval && python3 evaluate_models.py` (con `ollama serve` en otra terminal).

## Más información

- [`src/tfg_planner_slm/README.md`](src/tfg_planner_slm/README.md)
- [`slm_eval/README.md`](slm_eval/README.md)
- [`docs/ycb_dataset_pipeline.md`](docs/ycb_dataset_pipeline.md)

## Licencia

[MIT](LICENSE) · terceros en `THIRD_PARTY.md`
