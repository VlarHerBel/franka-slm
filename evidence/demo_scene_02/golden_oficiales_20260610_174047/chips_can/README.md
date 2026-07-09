# chips_can - Golden oficial validado

Estado: OFICIAL 100% OPERATIVO.

YAML oficial:
02_chips_can_OFFICIAL_VALIDATED.yaml

Origen:
src/panda_ws/panda_controller/config/demo_candidate_cache/demo_scene_02_chips_can_golden.yaml

Log de validación:
02_chips_can_OFFICIAL_VALIDATED_RUN.log

Resultado del run:
- CHIPS_CAN legacy high->low robustecido con tolerancia TF
- POST_PICK_TRANSPORT_ENTRY_VERIFY result=OK
- ATTACHED_DIRECT_ACTION_ROUTE_VALIDATE result=OK
- PLACE_RELEASE_SELECTED result=OK
- DETACH result=OK
- RETURN_HOME_AFTER_PLACE result=OK
- [MODE] execution_mode='pick_place' completado

Slot:
slot_2 / index 1
deposit_xy=(-0.5400, 0.0800)
release_tcp_z=0.4005

Nota:
El primer release candidate falló a 0.3905, pero el fallback dinámico seleccionó 0.4005 y el place terminó correctamente.
