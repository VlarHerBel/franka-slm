"""Lanza solo web_api (SLM + UI) sin Gazebo. Usar en terminal aparte."""

import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _create_web_api_process(context, *args, **kwargs):
    port = LaunchConfiguration("web_port").perform(context)
    host = LaunchConfiguration("host").perform(context)
    scene_id = LaunchConfiguration("scene_id").perform(context)
    log_json = _truthy(LaunchConfiguration("log_json").perform(context))
    open_browser = _truthy(LaunchConfiguration("open_browser").perform(context))
    no_warmup = _truthy(LaunchConfiguration("no_warmup_on_start").perform(context))
    no_auto_execute = _truthy(LaunchConfiguration("no_auto_execute").perform(context))

    exe = os.path.join(
        get_package_prefix("tfg_planner_slm"),
        "lib",
        "tfg_planner_slm",
        "web_api",
    )
    cmd = [exe, "--host", host, "--port", port, "--scene-id", scene_id]
    if log_json:
        cmd.append("--log-json")
    if open_browser:
        cmd.append("--open-browser")
    if no_warmup:
        cmd.append("--no-warmup-on-start")
    if no_auto_execute:
        cmd.append("--no-auto-execute")

    browser_url = "http://localhost:%s" % port
    actions = [
        LogInfo(
            msg=(
                "[TFG_WEB_API] host=%s port=%s warmup=%s open_browser=%s url=%s"
                % (
                    host,
                    port,
                    "off" if no_warmup else "on",
                    str(open_browser).lower(),
                    browser_url,
                )
            )
        ),
        ExecuteProcess(cmd=cmd, output="screen"),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("host", default_value="0.0.0.0"),
            DeclareLaunchArgument("web_port", default_value="8000"),
            DeclareLaunchArgument(
                "open_browser",
                default_value="true",
                description="Abrir http://localhost:<web_port> al arrancar.",
            ),
            DeclareLaunchArgument(
                "log_json",
                default_value="true",
                description="Imprimir JSON del SLM en terminal.",
            ),
            DeclareLaunchArgument(
                "no_warmup_on_start",
                default_value="false",
                description=(
                    "Sin inferencias Ollama al arrancar (~2 min menos); "
                    "el SLM queda listo al instante (primera orden puede ir más lenta)."
                ),
            ),
            DeclareLaunchArgument(
                "no_auto_execute",
                default_value="false",
                description="Solo interpretar; no ejecutar en Gazebo.",
            ),
            DeclareLaunchArgument(
                "scene_id",
                default_value="two_boxes_01",
                description="scene_id para pick_place (debe coincidir con scene_preset de Gazebo).",
            ),
            OpaqueFunction(function=_create_web_api_process),
        ]
    )
