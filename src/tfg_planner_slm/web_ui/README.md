# web_ui — Frontend del asistente SLM

Interfaz web integrada en el paquete `tfg_planner_slm`. Conecta con el backend
HTTP (`tfg_planner_slm/web_api.py`) y muestra **solo mensajes humanos** (nunca el
JSON interno del contrato v1.1).

La UI activa del TFG vive en este directorio (`web_ui/`). Prototipos visuales externos
no forman parte del repositorio.

## Estructura

```
web_ui/
  index.html              # landing + chat
  styles.css              # estilos de la interfaz
  src/app.js              # lógica de UI (polling salud, chat, sin JSON)
  src/utils/robotApi.js   # cliente API (getBackendHealth, sendRobotCommand)
```

Es una app estática sin paso de build (no requiere npm). El backend la sirve en
el mismo origen, por lo que no hace falta configurar CORS ni la URL.

## Arranque (un solo proceso sirve API + UI)

```bash
cd ~/tfg_robotics_ws
source install/setup.bash   # o: export PYTHONPATH=~/tfg_robotics_ws/src/tfg_planner_slm
python3 -m tfg_planner_slm.web_api --host 0.0.0.0 --port 8010
```

Abre `http://localhost:8010`.

- Mientras el warm-up no termina, la UI muestra «Iniciando asistente...» y el
  input queda bloqueado.
- Cuando el backend está `ready`, se habilita el chat.

## Backend en otro host/puerto (opcional)

Si sirves la UI por separado, define la URL del backend antes de cargar `app.js`:

```html
<script>window.ROBOT_BACKEND_URL = "http://localhost:8010";</script>
```

(En una futura migración a Vite se podría usar `VITE_ROBOT_BACKEND_URL`.)

## Importante

- La UI no implementa lógica de objetos ni de slots: la conversión
  «cajón humano → slot_index» y la detección de objetos las hace el backend.
- La UI nunca muestra `intent`, `target_label`, `slot_index` ni JSON interno.
