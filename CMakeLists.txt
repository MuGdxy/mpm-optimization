cmake_minimum_required(VERSION 3.27)
project(mpm LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# ###########################################################
#
# Libraries
#
# ###########################################################

# Find and link dependencies using Vcpkg
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
# libigl[core,opengl,glfw]
find_package(libigl CONFIG REQUIRED)

# ###########################################################
#
# Binaries
#
# ###########################################################

add_executable(sphere)

file(GLOB_RECURSE MPM_SRCS "*.cpp" "*/*.cpp" "*/*.h" "*/*.hpp")
target_sources(sphere PRIVATE ${MPM_SRCS})

# Link libraries
target_link_libraries(sphere PRIVATE glad::glad glfw OpenMP::OpenMP_CXX)
target_link_libraries(sphere PRIVATE igl::igl_core igl::igl_glfw igl::igl_opengl igl_copyleft::igl_copyleft_core)
target_link_libraries(sphere PRIVATE Eigen3::Eigen)

# Include directories
target_include_directories(sphere
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/mcloptlib/include
)

# openmp:llvm flag
target_compile_options(sphere PRIVATE "/openmp:llvm")
target_compile_definitions(sphere PRIVATE MPM_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}")