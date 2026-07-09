# panda_ws — Stack de simulación Franka Panda (ROS 2)

Agrupa los paquetes ROS 2 para simular un brazo Franka Panda en Gazebo con MoveIt 2,
percepción y control pick-and-place.

## Origen y extensiones del TFG

| Paquete | Origen | Trabajo del TFG |
|---------|--------|-----------------|
| `panda_description` | Base simulación (Utkarsh) | Worlds y assets para escenas YCB |
| `panda_moveit` | Base simulación (Utkarsh) | Config adaptada a sim |
| `panda_bringup` | Base simulación (Utkarsh) | **`tfg_ycb_pick_place.launch.py`**, `tfg_web_api` |
| `panda_controller` | Base simulación (Utkarsh) | **Perfiles demo, goldens, lógica pick-and-place validada** |
| `panda_vision` | Parcialmente nuevo | **Pipeline YCB/OBB, spawn runtime, percepción** |
| `panda_vision_interfaces` | Nuevo (TFG) | Mensajes/servicios de visión |
| `pymoveit2` | [AndrejOrsula/pymoveit2](https://github.com/AndrejOrsula/pymoveit2) | Sin cambios sustanciales |

## Paquetes

```
panda_ws/
├── panda_description/   # URDF, worlds Gazebo
├── panda_moveit/        # Configuración MoveIt 2
├── panda_bringup/       # Launch files (ver tfg_*.launch.py)
├── panda_controller/    # Control, move_to_home, perfiles demo_scene_02
├── panda_vision/        # perception_node, runtime_scene_spawner, dataset OBB
├── panda_vision_interfaces/
└── pymoveit2/           # Upstream — ver LICENSE
```

## Launch principal del TFG

```bash
ros2 launch panda_bringup tfg_ycb_pick_place.launch.py \
  scene_preset:=demo_scene_02 \
  model_path:=$TFG_WS/models/vision/yolo_obb_best.pt \
  ycb_models_path:=$TFG_WS/src/gazebo_ycb/models
```

Argumentos relevantes: `with_perception`, `with_moveit`, `with_controller`, `scene_preset`, `model_path`.

## Documentación relacionada

- Visión YCB/OBB: [`panda_vision/README_YCB_OBB.md`](panda_vision/README_YCB_OBB.md)
- Pipeline dataset: [`../../docs/ycb_dataset_pipeline.md`](../../docs/ycb_dataset_pipeline.md)
- Goldens validados: [`../../evidence/demo_scene_02/golden_oficiales/`](../../evidence/demo_scene_02/golden_oficiales/)
