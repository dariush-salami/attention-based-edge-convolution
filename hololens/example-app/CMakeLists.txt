cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(custom_ops)

find_package(Torch REQUIRED)
find_package(TorchSparse REQUIRED)
find_package(TorchScatter REQUIRED)
add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(example-app "${TorchSparse_LIBRARIES}")
target_link_libraries(example-app "${TorchScatter_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)