#!/usr/bin/env bash
# Reinicia demo_scene_02 y ejecuta clear_table (4 objetos, 1 por ciclo perception).
set -euo pipefail

WS="${TFG_WS:-$HOME/tfg_robotics_ws}"
source "$WS/install/setup.bash"

echo "[1/4] Reset estado de depósitos completados"
echo '[]' > /tmp/tfg_demo_completed_objects.json

echo "[2/4] Borrar YCB runtime en Gazebo"
ros2 run panda_vision clear_ycb_objects --ros-args \
  -p delete_all_runtime_ycb:=true \
  -p world_name:=vision_test_ycb_world

echo "[3/4] Respawn demo_scene_02 (requiere runtime_scene_spawner activo)"
if ros2 service list 2>/dev/null | grep -q /runtime_scene/spawn_random_scene; then
  ros2 service call /runtime_scene/clear_scene std_srvs/srv/Trigger "{}"
  sleep 1
  ros2 param set /runtime_scene_spawner scene_preset demo_scene_02 2>/dev/null \
    || ros2 param set /runtime_scene_spawner scene_preset demo_scene_02
  ros2 service call /runtime_scene/spawn_random_scene std_srvs/srv/Trigger "{}"
else
  echo "WARN: runtime_scene_spawner no activo."
  echo "  Arranca el stack Gazebo+spawner y relanza este script,"
  echo "  o respawnea manualmente con scene_preset:=demo_scene_02"
fi

echo "[4/4] Lanzar clear_table (un objeto por snapshot; repetir tras cada HOME)"
echo "Comando:"
echo "  ros2 run panda_controller perception_to_pregrasp_test --ros-args \\"
echo "    -p dry_run:=false \\"
echo "    -p scene_id:=demo_scene_02 \\"
echo "    -p execution_mode:=clear_table \\"
echo "    -p clear_table_manual_step:=true \\"
echo "    -p demo_authoritative_scene:=true \\"
echo "    -p demo_persist_completed_objects:=true \\"
echo "    -p execute_once:=true \\"
echo "    -p return_home_after_execution:=true"
