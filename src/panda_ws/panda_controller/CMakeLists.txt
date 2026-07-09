cmake_minimum_required(VERSION 3.8)
project(panda_controller)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_manager REQUIRED)
find_package(joint_trajectory_controller REQUIRED)
find_package(joint_state_broadcaster REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/run_perception_to_pregrasp_test.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME perception_to_pregrasp_test
)

install(PROGRAMS
  panda_controller/move_to_home.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME move_to_home
)

install(PROGRAMS
  panda_controller/record_joint_waypoint.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME record_joint_waypoint
)

install(PROGRAMS
  panda_controller/demo_scene_reachability_scan.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME demo_scene_reachability_scan
)

install(PROGRAMS
  panda_controller/demo_scene_layout_validator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME demo_scene_layout_validator
)

install(PROGRAMS
  panda_controller/demo_scene_plan_preflight_certifier.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME demo_scene_plan_preflight_certifier
)

install(PROGRAMS
  panda_controller/static_waypoint_joint_state_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME static_waypoint_joint_state_publisher
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()