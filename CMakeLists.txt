cmake_minimum_required(VERSION 3.4...3.18)
project(kalman_cpp)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(kalman_cpp src/KalmanTracker.cpp)
target_link_libraries(kalman_cpp PRIVATE Eigen3::Eigen)
