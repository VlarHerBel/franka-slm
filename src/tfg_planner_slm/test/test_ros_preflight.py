"""Tests para parse_active_controllers y strip_ansi."""

from tfg_planner_slm.ros_preflight import parse_active_controllers, strip_ansi


def test_parse_active_controllers_standard_output() -> None:
    stdout = """
arm_controller          joint_trajectory_controller/JointTrajectoryController  active
gripper_controller      joint_trajectory_controller/JointTrajectoryController  active
joint_state_broadcaster joint_state_broadcaster/JointStateBroadcaster          active
"""
    active = parse_active_controllers(stdout)
    assert active == {
        "arm_controller",
        "gripper_controller",
        "joint_state_broadcaster",
    }
    assert "arm_controller" in active
    assert "gripper_controller" in active


def test_parse_active_controllers_ignores_inactive() -> None:
    stdout = """
arm_controller joint_trajectory_controller/JointTrajectoryController inactive
gripper_controller joint_trajectory_controller/JointTrajectoryController active
"""
    active = parse_active_controllers(stdout)
    assert active == {"gripper_controller"}


def test_parse_active_controllers_ansi_colored_output() -> None:
    stdout = (
        "\x1b[92marm_controller\x1b[0m "
        "joint_trajectory_controller/JointTrajectoryController "
        "\x1b[92mactive\x1b[0m\n"
        "\x1b[92mgripper_controller\x1b[0m "
        "joint_trajectory_controller/JointTrajectoryController "
        "\x1b[92mactive\x1b[0m\n"
        "\x1b[92mjoint_state_broadcaster\x1b[0m "
        "joint_state_broadcaster/JointStateBroadcaster "
        "\x1b[92mactive\x1b[0m"
    )
    active = parse_active_controllers(stdout)
    assert active == {
        "arm_controller",
        "gripper_controller",
        "joint_state_broadcaster",
    }


def test_strip_ansi_removes_color_codes() -> None:
    raw = "\x1b[92mactive\x1b[0m"
    assert strip_ansi(raw) == "active"
