from setuptools import find_packages, setup

package_name = 'tfg_planner_slm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vlar',
    maintainer_email='alvarohbell@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
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

        ],
    },
)
