cmake_minimum_required(VERSION 3.8)
project(tb3p_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)

# 손에 든 캘리브레이션 리그를 흔드는 Gazebo 시스템 플러그인
add_library(tb3p_excitation SHARED src/excitation.cpp)
target_compile_features(tb3p_excitation PRIVATE cxx_std_17)
target_link_libraries(tb3p_excitation gz-sim::core gz-plugin::core)
install(TARGETS tb3p_excitation LIBRARY DESTINATION lib)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY launch config models worlds urdf DESTINATION share/${PROJECT_NAME})

ament_package()
