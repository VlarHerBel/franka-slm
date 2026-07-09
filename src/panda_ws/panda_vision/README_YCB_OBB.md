# YCB OBB en Gazebo Sim

## Qué se ha modificado

- Se amplió `panda_description/launch/gazebo.launch.py` para añadir un parámetro
  `ycb_models_path` y concatenarlo a `GZ_SIM_RESOURCE_PATH`.
- Se añadió `config/ycb_obb_dataset.yaml` con **14** modelos YCB de `gazebo_ycb`
  (más variedad de color y forma) y sus dimensiones para spawn y OBB.
- Se añadió el ejecutable `generate_ycb_dataset` en `panda_vision` para generar
  imágenes PNG y labels OBB Ultralytics desde Gazebo Sim.
- Se añadió `launch/perception_obb.launch.py` para cargar después un modelo OBB
  entrenado con `perception_node`.
- Se ajustó `perception_node` para que su topic legacy pueda exponer `id` y
  `position`, facilitando el uso futuro con `vision_bridge_node`.

## Cargar el entorno

El código vive en `~/tfg_robotics_ws` y las dependencias Python viven en
`~/tfg_env`. Carga siempre el entorno con:

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
```

## Compilar

Desde el workspace real del proyecto:

```bash
cd ~/tfg_robotics_ws
colcon build --packages-select panda_description panda_vision
source ~/tfg_robotics_ws/install/setup.bash
```

## Lanzar Gazebo Sim

Para el dataset usa el mundo **`vision_test_ycb`**: misma mesa que `vision_test` pero **sin**
las piezas geométricas de color encima del tablero.

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
ros2 launch panda_description gazebo.launch.py \
  world_name:=vision_test_ycb \
  ycb_models_path:=$TFG_WS/src/gazebo_ycb/models
```

Por defecto **`gazebo_gui:=false`**: Gazebo arranca en modo **servidor** (`gz sim -s`), sin
ventana gráfica (menos carga CPU/GPU, suele ir más rápido al generar el dataset). La cámara
sigue publicando en local con GPU. Para depurar a ojo: `gazebo_gui:=true`. En un servidor
sin GPU/display, si la cámara falla, prueba GUI o consulta la documentación de Gazebo sobre
**headless rendering** (`--headless-rendering`).

El generador asume por defecto `world_name:=vision_test_ycb` y coloca los YCB en la
**superficie del tablero** (`table_surface_z_m`, por defecto `0.26` m para este mundo).

En `vision_test_ycb.world` la iluminación está pensada para **dataset**: ambiente alto,
**sin sombras proyectadas** y una segunda luz de relleno, para reducir caras casi negras
por sombra dura (sigue pudiendo haber artefactos de mallas DAE).

**Texturas repetidas entre objetos:** en Ignition/Ogre, muchos YCB comparten el nombre de
archivo `texture_map.png`; el motor puede **reutilizar la primera textura cargada** para
todos los modelos. El generador activa por defecto `texture_unique_cache:=true`: copia
cada modelo bajo `~/.cache/panda_vision/ycb_texture_unique/<modelo>/` con un PNG renombrado
y un `model.sdf` con rutas `file://`. Para desactivarlo: `-p texture_unique_cache:=false`.

Si sigues usando `vision_test` (con primitivas), las fotos mezclarán cajas de colores con YCB.

Si quieres usar el demo completo sin SLM:

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
ros2 launch tfg_planner_slm tfg_demo.launch.py \
  with_slm:=false \
  world_name:=vision_test
```

## Generar el dataset

Salida por defecto (**v2**):

- `~/tfg_robotics_ws/datasets/ycb_obb_v2/images/train`
- `~/tfg_robotics_ws/datasets/ycb_obb_v2/images/val`
- `~/tfg_robotics_ws/datasets/ycb_obb_v2/labels/train`
- `~/tfg_robotics_ws/datasets/ycb_obb_v2/labels/val`
- `~/tfg_robotics_ws/datasets/ycb_obb_v2/data.yaml`

### Reanudar sin pisar escenas (`start_scene_index`)

Si ya tienes hasta `scene_00244` y quieres seguir con `scene_00245`, usa el mismo
`output_dir`, **`clear_output:=false`** (o déjalo en `true`: si `start_scene_index>0`,
el nodo **fuerza** `clear_output=false` y avisa), y por ejemplo:

```bash
-p start_scene_index:=245 \
-p scene_count:=2255 \
```

`scene_count` es **cuántas escenas genera esta ejecución**, no el total del dataset.
El reparto train/val de las nuevas escenas sigue siendo aleatorio con tu `seed` desde
el inicio del generador (no reproduce exactamente el stream del run anterior).

**Gazebo:** no es obligatorio reiniciarlo; conviene `initial_full_purge:=true` una vez
para limpiar slots `ycb_dataset_obj_*`. Si la sim va rara, reinicia Gazebo.

### Imagen vs etiquetas (que coincidan)

Las etiquetas antiguas usaban la **pose planificada** y la imagen la **física tras
`settle_time`**: si un objeto cae, se sale del plano o quedan modelos viejos sin borrar,
el `.txt` y la PNG **no cuadran**.

Por defecto el generador usa **`labels_use_gz_model_pose:=true`** y **`labels_sim_pose_source:=ros_tf`**:
tras la captura lee las poses del topic puenteado `/world/<mundo>/pose/info` (más rápido que
`gz model -p`). Con **`labels_sim_pose_source:=gz_cli`** se usa el subprocess clásico.
Si falla la lectura, **reintenta la escena** (`gz_pose_scene_max_retries`).
Si tras reintentos sigue fallando: con **`allow_planned_pose_labels_fallback:=false`**
**no guarda** esa escena; con **`true`** guarda con pose planificada (menos fiable).

Sigue siendo importante **no dejar basura** en Gazebo (borrados lentos / entidades
fantasma): reinicia la sim o `initial_full_purge` si la mesa no queda limpia entre escenas.

### Train vs val (automático)

No hace falta elegir tú las imágenes. **Cada escena** se asigna a `train` o `val` con una
moneda aleatoria según `val_split` (por ejemplo `0.2` → ~20 % val). El `seed` fija la
secuencia para poder reproducir el reparto. Con `val_split:=0.0` todo va a `train` (útil
para pruebas rápidas como `ycb_obb_check10`). Las carpetas `ycb_obb_check10` y
`ycb_obb_smoketest` son solo **rutas de salida distintas** que tú eliges con
`output_dir`; el script no las mezcla entre sí.

### ¿Cuántas imágenes generar?

Orientativo: con **varias clases** (p. ej. 14 en el YAML actual) y 2–5 objetos por
escena, suele ir bien partir de **varios miles** de imágenes (2k–10k) para un modelo
sólido; con **500–1000** ya sirve para prototipos.

Si `ign service remove` va lento, sube `delete_timeout_sec` (por defecto ya es generoso)
o deja que el generador registre un warning y continúe.

Ejemplo de generación:

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
ros2 run panda_vision generate_ycb_dataset --ros-args \
  -p world_name:=vision_test_ycb \
  -p scene_count:=200 \
  -p output_dir:=$TFG_WS/datasets/ycb_obb_v2 \
  -p min_objects:=2 \
  -p max_objects:=5 \
  -p val_split:=0.2 \
  -p seed:=7 \
  -p settle_time:=1.5 \
  -p spawn_x_min:=0.42 \
  -p spawn_x_max:=0.72 \
  -p spawn_y_min:=-0.22 \
  -p spawn_y_max:=0.22 \
  -p ycb_models_path:=$TFG_WS/src/gazebo_ycb/models
```

Parámetros útiles si las escenas salen muy amontonadas o siempre con el mismo color:

- `diverse_classes_per_scene` (por defecto `true`): con 2–5 objetos y 5 clases en el YAML,
  evita repetir clase en la misma foto cuando hay suficientes clases distintas.
- `min_surface_gap_m`: hueco extra entre “discos” XY de cada objeto (`r_i+r_j+gap`).
- `global_spawn_z_lift_m`: sube un poco todos los spawns para reducir penetración en la mesa.
- `post_settle_camera_frames`: descarta varios frames tras `settle_time` antes de guardar la PNG.
- `table_center_*`, `table_half_extent_*`, `table_edge_margin_m`: recorte según el tablero
  real para que **ningún objeto** quede planeado fuera de la madera (huella + margen).
- `lying_down_probability`: fracción de objetos con **pitch ~90°** (tumbados); la huella
  OBB usa las **dos dimensiones menores** del AABB. `lying_random_roll`: más variedad
  visual a costa de peor ajuste caja–malla; por defecto `false`.

## Entrenar YOLO OBB

Comando de entrenamiento:

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
yolo obb train \
  data=$TFG_WS/datasets/ycb_obb_v2/data.yaml \
  model=yolo26n-obb.pt \
  epochs=80 \
  imgsz=640 \
  batch=16 \
  device=0
```

Validación:

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
yolo obb val \
  model=/ruta/a/best.pt \
  data=$TFG_WS/datasets/ycb_obb_v2/data.yaml \
  imgsz=640 \
  device=0
```

## Conectar luego al pipeline de visión

El punto natural de integración es `panda_vision/perception_node`, no
`object_detector`. Puedes lanzar el backend OBB así:

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
ros2 launch panda_vision perception_obb.launch.py \
  model_path:=/ruta/a/best.pt \
  vision_backend:=yolo26_obb
```

Si quieres probar compatibilidad con el bridge actual:

```bash
source ~/tfg_robotics_ws/iniciar_tfg.sh
ros2 launch panda_vision perception_obb.launch.py \
  model_path:=/ruta/a/best.pt \
  vision_backend:=yolo26_obb \
  publish_legacy_topic:=true
```

Eso hará que `perception_node` publique también en `/detections_3d`, que es el
topic que ya escucha `tfg_planner_slm/vision_bridge_node.py`.

## Limpieza entre escenas y mensajes "Entity ... not found"

El generador solo pide borrar los slots que se usaron en la **escena anterior**, para no
inundar la consola al intentar borrar `ycb_dataset_obj_2`…`4` cuando solo había dos objetos.

Al **inicio** puede hacer un purge de todos los slots (`initial_full_purge`, por defecto
`true`) por si quedaron entidades de un run abortado; en simulación limpia Gazebo puede
avisar "not found" (es esperado e inofensivo). Para silenciarlo:

```text
-p initial_full_purge:=false
```

## Limitaciones de esta primera versión

- Las OBB se aproximan con `pose XY + yaw + width_m + length_m`.
- Se asume que los objetos están erguidos sobre la mesa.
- La proyección usa el frame de referencia `panda_link0`, que en esta
  simulación coincide con el robot spawneado en el origen.
- El generador usa servicios/CLI de Gazebo Sim (`gz service`), no Gazebo Classic.
