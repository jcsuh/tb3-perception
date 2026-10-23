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

# LiDAR-camera point-to-plane solver in C++ (Ceres), same problem as
# scripts/lidar_cam_calib.py (session 14)
add_executable(lidar_cam_calib_ceres src/lidar_cam_calib_ceres.cpp)
target_compile_features(lidar_cam_calib_ceres PRIVATE cxx_std_17)
target_link_libraries(lidar_cam_calib_ceres
  Ceres::ceres Eigen3::Eigen yaml-cpp::yaml-cpp)
install(TARGETS lidar_cam_calib_ceres DESTINATION lib/${PROJECT_NAME})

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
