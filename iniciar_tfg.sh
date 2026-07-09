#!/bin/bash
# Carga el entorno de trabajo del TFG (ROS 2 Humble + venv + workspace).
# Uso: source ~/tfg_robotics_ws/iniciar_tfg.sh

export TFG_WS="${TFG_WS:-$HOME/tfg_robotics_ws}"

# 1. ROS 2 Humble
source /opt/ros/humble/setup.bash

# 2. Entorno virtual Python (ajusta la ruta si usas otro venv)
if [ -f "$HOME/tfg_env/bin/activate" ]; then
  source "$HOME/tfg_env/bin/activate"
fi

# 3. Workspace compilado
if [ -f "$TFG_WS/install/setup.bash" ]; then
  source "$TFG_WS/install/setup.bash"
else
  echo "⚠️  No se encontró $TFG_WS/install/setup.bash — ejecuta 'colcon build' primero."
fi

echo "✅ Entorno TFG cargado (TFG_WS=$TFG_WS)"
echo "🤖 Demo integrada: ros2 launch panda_bringup tfg_ycb_pick_place.launch.py scene_preset:=demo_scene_02"
echo "🌐 UI SLM:          ros2 launch tfg_planner_slm tfg_web_api.launch.py"
