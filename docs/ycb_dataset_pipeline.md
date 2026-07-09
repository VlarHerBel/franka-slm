# Pipeline YCB OBB

## Objetivo

Este pipeline genera un dataset cenital Ultralytics OBB desde Gazebo Sim con:

- semilla reproducible por escena
- etiquetas OBB sin clipping y con orden estable
- pool persistente de entidades YCB movidas por servicios ROS2
- herramientas de validación, auditoría visual y deduplicación

## Compilación

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
cd ~/tfg_robotics_ws
colcon build --packages-select panda_description panda_vision
source ~/tfg_robotics_ws/install/setup.bash
```

## Lanzar Gazebo

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
ros2 launch panda_description gazebo.launch.py \
  world_name:=vision_test_ycb \
  ycb_models_path:=$HOME/tfg_robotics_ws/src/gazebo_ycb/models
```

El launch publica además estos servicios ROS2 del mundo:

- `/world/<world>_world/create`
- `/world/<world>_world/remove`
- `/world/<world>_world/set_pose`
- `/world/<world>_world/pose/info`

## Generar dataset

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
ros2 run panda_vision generate_ycb_dataset --ros-args \
  -p seed:=-1 \
  -p scene_count:=200 \
  -p output_dir:=$HOME/tfg_robotics_ws/datasets/ycb_obb_v3 \
  -p world_name:=vision_test_ycb \
  -p reject_scene_if_incomplete_labels:=true \
  -p labels_use_sim_pose:=true \
  -p labels_sim_pose_source:=ros_tf
```

Parámetros nuevos o relevantes:

- `seed`: `-1` autoseed; si continúas un dataset con `start_scene_index>0`, reutiliza `run_meta.json`.
- `start_scene_index`: continuar escenas sin repetir prefijos.
- `clear_output`: limpia el dataset solo al arrancar desde cero.
- `labels_use_sim_pose`: usa pose real post-settle.
- `labels_sim_pose_source`: `ros_tf` recomendado, `gz_cli` como fallback.
- `reject_scene_if_incomplete_labels`: rechaza escenas con cualquier OBB inválida.
- `table_texture_dir`, `background_texture_dir`, `light_intensity_range`,
  `light_direction_jitter_deg`, `camera_jitter_xyz_mm`, `camera_jitter_rpy_deg`,
  `num_distractors`: hooks de domain randomization.

## Validación

```bash
python3 ~/tfg_robotics_ws/src/panda_ws/panda_vision/panda_vision/validate_obb_labels.py \
  --dataset ~/tfg_robotics_ws/datasets/ycb_obb_v3
```

## Auditoría visual

```bash
python3 ~/tfg_robotics_ws/src/panda_ws/panda_vision/panda_vision/audit_overlay_obb.py \
  --dataset ~/tfg_robotics_ws/datasets/ycb_obb_v3 \
  --sample-count 50
```

Las muestras quedan en `audit_overlay/`.

## Deduplicación perceptual

```bash
python3 ~/tfg_robotics_ws/src/panda_ws/panda_vision/panda_vision/dedup_phash.py \
  --dataset ~/tfg_robotics_ws/datasets/ycb_obb_v3 \
  --threshold 1
```

## Entrenamiento YOLO OBB

Smoke test:

```bash
yolo obb train \
  data=$HOME/tfg_robotics_ws/datasets/ycb_obb_v3/data.yaml \
  model=yolo26n-obb.pt \
  epochs=5 \
  imgsz=640 \
  batch=8 \
  device=0
```

Entrenamiento normal:

```bash
yolo obb train \
  data=$HOME/tfg_robotics_ws/datasets/ycb_obb_v3/data.yaml \
  model=yolo26n-obb.pt \
  epochs=80 \
  imgsz=640 \
  batch=16 \
  device=0
```

Alternativa mayor:

```bash
yolo obb train \
  data=$HOME/tfg_robotics_ws/datasets/ycb_obb_v3/data.yaml \
  model=yolo26s-obb.pt \
  epochs=80 \
  imgsz=640 \
  batch=12 \
  device=0
```

Validación:

```bash
yolo obb val \
  model=/ruta/a/best.pt \
  data=$HOME/tfg_robotics_ws/datasets/ycb_obb_v3/data.yaml \
  imgsz=640 \
  device=0
```

Predicción:

```bash
yolo obb predict \
  model=/ruta/a/best.pt \
  source=$HOME/tfg_robotics_ws/datasets/ycb_obb_v3/images/val \
  imgsz=640 \
  device=0
```

## OOM

- Baja `batch`.
- Baja `imgsz`.
- Usa `device=0` si hay GPU disponible.
- Usa `device=cpu` si quieres un smoke test funcional aunque sea lento.
