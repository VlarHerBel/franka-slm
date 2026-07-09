from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = 'tfg_planner_slm'


def _web_ui_data_files():
    """Instala web_ui/ en share/tfg_planner_slm/web_ui (preserva subcarpetas)."""
    pkg_root = Path(__file__).resolve().parent
    base = pkg_root / "web_ui"
    if not base.is_dir():
        return []
    entries = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel_parent = path.parent.relative_to(base)
        dest = "share/%s/web_ui" % package_name
        if str(rel_parent) != ".":
            dest = "%s/%s" % (dest, rel_parent)
        # colcon exige rutas relativas al manifiesto del paquete
        entries.append((dest, [str(path.relative_to(pkg_root))]))
    return entries


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ] + _web_ui_data_files(),
    install_requires=['setuptools', 'requests', 'PyYAML', 'pydantic'],
    zip_safe=True,
    maintainer='vlar',
    maintainer_email='alvarohbell@gmail.com',
    description='Planificador SLM seguro para TFG de robótica (Franka Panda, ROS 2)',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'llm_node = tfg_planner_slm.llm_node:main',
            'executor_node = tfg_planner_slm.executor_node:main',
            'vision_simulator = tfg_planner_slm.vision_simulator:main',
            'vision_bridge_node = tfg_planner_slm.vision_bridge_node:main',
            'web_bridge_node = tfg_planner_slm.web_bridge_node:main',
            'tfg_planner_cli = tfg_planner_slm.cli_chat:main',
            'web_api = tfg_planner_slm.web_api:main',
        ],
    },
)
