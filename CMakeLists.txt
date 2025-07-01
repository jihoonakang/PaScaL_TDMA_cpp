cmake_minimum_required(VERSION 3.19)

option(CUDA "Enable GPU (CUDA) support" ON)
option(CUDA_AWARE_MPI "Enable CUDA-Aware MPI support" ON)
option(TESTING "Enable building and running tests" ON)
option(PYTHON "Enable Python bindings using pybind11" on)

if(CUDA)
    project(PaScaL_TDMA_CPP LANGUAGES CXX CUDA)
else()
    project(PaScaL_TDMA_CPP LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Add compile options
add_compile_options(-O2 -DDEBUG)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(examples/convection3D)

# Enable device linking
if(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust to your GPU (e.g., 86 for Ampere)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)  # 또는 실제 설치 경로
    include_directories(${CUDA_INCLUDE_DIRS})
    target_compile_definitions(PaScaL_TDMA PRIVATE CUDA)
    message(STATUS "GPU support ENABLED (CUDA)")
else()
    message(STATUS "GPU support DISABLED")
endif()

# Enable testing
if(TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Tests ENABLED")
else()
    message(STATUS "Tests DISABLED")
endif()

# Python binding
if(PYTHON)
    add_subdirectory(src/pybind)
endif()

