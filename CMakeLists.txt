# 0. Set minimum required CMake version
cmake_minimum_required(VERSION 3.19)

# 1. Option definitions
option(CUDA "Enable GPU (CUDA) support" OFF)
option(CUDA_AWARE_MPI "Enable CUDA-Aware MPI support" OFF)
option(TESTING "Enable building and running tests" ON)
option(PYTHON "Enable Python bindings using pybind11" on)
option(FORTRAN "Enable Fortran interface" on)

# 2. Project declaration based on CUDA and FORTRAN support
if(CUDA AND FORTRAN)
    project(PaScaL_TDMA LANGUAGES CXX CUDA Fortran)
elseif(CUDA)
    project(PaScaL_TDMA LANGUAGES CXX CUDA)
elseif(FORTRAN)
    project(PaScaL_TDMA LANGUAGES CXX Fortran)
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
    add_compile_options(-DCUDA)

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

# 9. Fortran interface 
if(FORTRAN)
    add_subdirectory(src/fortran)
    message(STATUS "Fortran interface ENABLED")
else()
    message(STATUS "Fortran interface DISABLED")
endif()
