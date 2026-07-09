# mustard_bottle - Golden oficial validado

Estado: OFICIAL OPERATIVO (pipeline pick-lift-transport-place-home completo).

YAML oficial:
`04_mustard_bottle_OFFICIAL_VALIDATED.yaml`

Origen (runtime ROS, espejo):
`src/panda_ws/panda_controller/config/demo_candidate_cache/demo_scene_02_mustard_bottle_golden.yaml`

Log de validación:
`04_mustard_bottle_OFFICIAL_VALIDATED_RUN.log`

Resultado del run (2026-06-10):
- prevalidation_source=geometric_fallback
- POST_PICK_TRANSPORT_ENTRY_VERIFY result=OK
- ATTACHED_DIRECT_ACTION_ROUTE_VALIDATE result=OK
- PLACE_RELEASE: fallo en z=0.3084 (STATUS_ABORTED); fallback z=0.3284 OK
- PLACE_OPEN + DETACH result=OK
- PLACE_RETREAT result=OK
- MOVE_HOME result=OK
- GAZEBO_POST_PLACE_VERIFY: FAIL_OBJECT_Z_LOW (objeto bajo pared caja; revisar antes demo SLM)

Slot:
slot_4 / index 3
deposit_xy=(-0.5400, -0.1000)
release_tcp_z=0.3284

Pick order en escena:
4º objeto (último en clear_table demo_scene_02).

Attach:
mode=planning_scene_only (fricción Gazebo, sin set_pose).
