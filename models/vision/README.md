# Pesos de visión YOLO (no versionados)

Los modelos `.pt` no se incluyen en el repositorio por tamaño y licencia.

## Opción A — Pesos fine-tuned del TFG

Tras entrenar con el pipeline descrito en `docs/ycb_dataset_pipeline.md`, copia el mejor checkpoint aquí:

```bash
mkdir -p ~/tfg_robotics_ws/models/vision
cp /ruta/a/tu/entrenamiento/weights/best.pt \
   ~/tfg_robotics_ws/models/vision/yolo_obb_best.pt
```

El launch `tfg_ycb_pick_place.launch.py` usa por defecto:

```
$TFG_WS/models/vision/yolo_obb_best.pt
```

## Opción B — Modelo base Ultralytics (pruebas rápidas)

```bash
mkdir -p ~/tfg_robotics_ws/models/vision
# Ultralytics descarga automáticamente al primer uso, o:
wget -O ~/tfg_robotics_ws/models/vision/yolo26n-obb.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n-obb.pt
```

Luego pasa `model_path` explícitamente al launch.

## Segmentation (opcional)

Para experimentos con segmentación (`yolo11n-seg.pt`), coloca el archivo en este directorio o indica la ruta con el parámetro `model_path` del nodo de percepción.
