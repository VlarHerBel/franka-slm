import math

from panda_controller.math_tf_utils import (
    quaternion_from_euler,
    quaternion_matrix,
    yaw_from_quaternion,
)


def test_quaternion_from_euler_identity() -> None:
    q = quaternion_from_euler(0.0, 0.0, 0.0)
    assert abs(q[0]) < 1e-12
    assert abs(q[1]) < 1e-12
    assert abs(q[2]) < 1e-12
    assert abs(q[3] - 1.0) < 1e-12


def test_yaw_roundtrip() -> None:
    yaw = 1.234
    q = quaternion_from_euler(0.0, 0.0, yaw)
    recovered = yaw_from_quaternion(q)
    diff = (recovered - yaw + math.pi) % (2.0 * math.pi) - math.pi
    assert abs(diff) < 1e-9


def test_top_down_quaternion_is_normalized() -> None:
    base_down = (0.0, 1.0, 0.0, 0.0)
    rot = quaternion_matrix(base_down)
    # Debe ser matriz de rotación válida ortonormal.
    for i in range(3):
        n = float(sum(rot[j, i] * rot[j, i] for j in range(3)))
        assert abs(n - 1.0) < 1e-9
