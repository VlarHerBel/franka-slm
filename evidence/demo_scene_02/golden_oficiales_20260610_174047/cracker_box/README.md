# cracker_box - Golden oficial validado

Estado: OFICIAL 100% OPERATIVO.

YAML oficial:
01_cracker_box_OFFICIAL_VALIDATED.yaml

Origen:
src/panda_ws/panda_controller/config/demo_candidate_cache/demo_scene_02_cracker_box_golden.yaml

Log de validación:
01_cracker_box_OFFICIAL_VALIDATED_RUN.log

Resultado del run:
- POST_PICK_TRANSPORT_ENTRY_VERIFY result=OK
- PLACE_RELEASE_SELECTED result=OK
- DETACH result=OK
- RETURN_HOME_AFTER_PLACE result=OK
- [MODE] execution_mode='pick_place' completado

Slot:
slot_1 / index 0
deposit_xy=(-0.3700, 0.0800)
release_tcp_z=0.3292

Importante:
NO usar como oficial:
demo_scene_02_cracker_box_slot_0_full_execution_golden.yaml

Motivo:
Ese full_execution_golden falla en replay por phase_count_mismatch.
