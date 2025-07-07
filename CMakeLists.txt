# 0. Set minimum required CMake version
cmake_minimum_required(VERSION 3.19)

# 1. Option definitions
option(CUDA "Enable GPU (CUDA) support" ON)
option(CUDA_AWARE_MPI "Enable CUDA-Aware MPI support" ON)
option(TESTING "Enable building and running tests" ON)
option(PYTHON "Enable Python bindings using pybind11" on)

# 2. Project declaration based on CUDA support
if(CUDA)
    project(PaScaL_TDMA LANGUAGES CXX CUDA)
else()
    project(PaScaL_TDMA LANGUAGES CXX)
endif()

# 3. Language standards and required packages
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(MPI REQUIRED)

# 4. Global compile options (optimization, debug)
#add_compile_options(-O2 -DDEBUG)
add_compile_options(-O2)

# 5. Add source and example subdirectories
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(examples/convection3D)

# 6. CUDA-specific configuration
if(CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES 75)          # adjust to target GPU
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda) # adjust path as needed
    include_directories(${CUDA_INCLUDE_DIRS})
    target_compile_definitions(PaScaL_TDMA PRIVATE CUDA)

    if(CUDA_AWARE_MPI)
        message(STATUS "CUDA-Aware MPI support: ENABLED")
    else()
        message(STATUS "CUDA-Aware MPI support: DISABLED")
    endif()
else()
    message(STATUS "GPU support: DISABLED")
endif()

# 7. Testing framework
if(TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Tests ENABLED")
else()
    message(STATUS "Tests DISABLED")
endif()

# 8. Python binding 
if(PYTHON)
    add_subdirectory(src/pybind)
    message(STATUS "Python binding ENABLED")
else()
    message(STATUS "Python binding DISABLED")
endif()
