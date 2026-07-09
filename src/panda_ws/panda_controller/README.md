# panda_controller — Control pick-and-place

Paquete ROS 2 de control del brazo Franka Panda en simulación. Base de simulación (Utkarsh) **extendida para el TFG** con perfiles de demo validados, goldens `demo_scene_02` e integración con visión OBB.

Repositorio del TFG: [github.com/VlarHerBel/franka-slm](https://github.com/VlarHerBel/franka-slm)

## Estructura relevante

```
panda_controller/
├── config/
│   ├── demo_profiles/          # Perfiles YAML por objeto (demo_scene_02)
│   └── demo_candidate_cache/     # Caché espejo de goldens oficiales
├── launch/
│   └── panda_controller.launch.py
└── panda_controller/
    ├── move_to_home.py           # Homing al arranque
    ├── perception_to_pregrasp_test.py  # Nodo principal pick-and-place + validación
    └── ...
```

## Goldens oficiales

Los YAML validados viven en `evidence/demo_scene_02/golden_oficiales/`. El controlador carga copias desde `config/demo_profiles/` y `config/demo_candidate_cache/`.

## Launch

Incluido en el launch integrador del TFG:

```bash
ros2 launch panda_bringup tfg_ycb_pick_place.launch.py scene_preset:=demo_scene_02
```

Ver también [`../panda_bringup/launch/tfg_ycb_pick_place.launch.py`](../panda_bringup/launch/tfg_ycb_pick_place.launch.py).
