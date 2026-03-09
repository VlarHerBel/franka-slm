#!/bin/bash
# 1. Cargar ROS 2 Humble
source /opt/ros/humble/setup.bash
# 2. Cargar Entorno Virtual
source ~/tfg_env/bin/activate
# 3. Cargar Workspace
source ~/tfg_robotics_ws/install/setup.bash

echo "✅ Entorno TFG cargado correctamente."
echo "🤖 Recuerda: Si ros2 run falla, usa: python3 src/base_simulation/pymoveit2/examples/pick_and_place.py"
