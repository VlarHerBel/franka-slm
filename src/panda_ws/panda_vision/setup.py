from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'panda_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'ultralytics',
        'opencv-python',
        'open3d',
        'PyYAML',
        'Pillow',
        'ImageHash',
    ],
    zip_safe=True,
    maintainer='utk',
    maintainer_email='kutkarsh706@gmail.com',
    description='Modular Panda perception (Open3D, YOLO OBB, grasp clients).',
    license='MIT',
    extras_require={
        'test': ['pytest'],
        # Optional stack for real Grounded SAM 2 (user installs torch + project deps).
        'grounded_sam2': [],
    },
    entry_points={
        'console_scripts': [
            'object_detector = panda_vision.object_detector:main',
            'perception_node = panda_vision.nodes.perception_node:main',
            'runtime_scene_spawner = panda_vision.runtime_scene_spawner:main',
            'generate_ycb_dataset = panda_vision.generate_ycb_dataset:main',
            'validate_obb_labels = panda_vision.validate_obb_labels:main',
            'audit_overlay_obb = panda_vision.audit_overlay_obb:main',
            'dedup_phash = panda_vision.dedup_phash:main',
            'spawn_ycb_object = panda_vision.spawn.spawn_ycb_object:main',
            'spawn_ycb_catalog_photo = panda_vision.spawn.spawn_ycb_catalog_photo:main',
            'clear_ycb_objects = panda_vision.spawn.clear_ycb_objects:main',
            'runtime_scene_gt_node = panda_vision.spawn.runtime_scene_gt_node:main',
            'measure_ycb_yaw_table = panda_vision.tools.measure_ycb_yaw_table:main',
            'print_camera_pose = panda_vision.tools.print_camera_pose:main',
            'validate_ycb_model_geometry = panda_vision.tools.validate_ycb_model_geometry:main',
        ],
    },
)
