cmake_minimum_required(VERSION 3.8)
project(tb3p_calibration)

# colcon build without --cmake-args -DCMAKE_BUILD_TYPE=... compiles with no
# optimization; Ceres autodiff (Jet templates) is then ~50x slower
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)

# LiDAR-camera point-to-plane solver in C++ (Ceres), same problem as
# scripts/lidar_cam_calib.py (session 14)
add_executable(lidar_cam_calib_ceres src/lidar_cam_calib_ceres.cpp)
target_compile_features(lidar_cam_calib_ceres PRIVATE cxx_std_17)
target_link_libraries(lidar_cam_calib_ceres
  Ceres::ceres Eigen3::Eigen yaml-cpp::yaml-cpp)
install(TARGETS lidar_cam_calib_ceres DESTINATION lib/${PROJECT_NAME})

# Online time offset estimation between IMU, wheels and LiDAR (session 19),
# same method as scripts/time_offset_estimate.py
add_executable(time_offset_node src/time_offset_node.cpp)
target_compile_features(time_offset_node PRIVATE cxx_std_17)
target_link_libraries(time_offset_node
  Eigen3::Eigen
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS})
install(TARGETS time_offset_node DESTINATION lib/${PROJECT_NAME})

# Online calibration drift monitor (session 21): camera-LiDAR yaw from depth
# images (scripts/drift_check.py), yaw-rate consistency and time offsets,
# reported on /diagnostics
add_executable(drift_monitor_node src/drift_monitor_node.cpp)
target_compile_features(drift_monitor_node PRIVATE cxx_std_17)
target_link_libraries(drift_monitor_node
  Eigen3::Eigen
  yaml-cpp::yaml-cpp
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  ${diagnostic_msgs_TARGETS}
  ${tf2_msgs_TARGETS})
install(TARGETS drift_monitor_node DESTINATION lib/${PROJECT_NAME})

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

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_time_offset test/test_time_offset.cpp)
  target_include_directories(test_time_offset PRIVATE src)
  target_link_libraries(test_time_offset Eigen3::Eigen)

  ament_add_gtest(test_drift_core test/test_drift_core.cpp)
  target_include_directories(test_drift_core PRIVATE src)
  target_link_libraries(test_drift_core Eigen3::Eigen)

  # Python scripts (session 23): calibration merge math and URDF check
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_calib_merge test/test_calib_merge.py)
  ament_add_pytest_test(test_lidar_base_calib test/test_lidar_base_calib.py)
  ament_add_pytest_test(test_estimators test/test_estimators.py)
endif()

ament_package()
