cmake_minimum_required(VERSION 3.19)
option(CUDA "Enable GPU (CUDA) support" OFF)
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

# Enable testing
include(CTest)
enable_testing()

# Enable device linking
if(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust to your GPU (e.g., 86 for Ampere)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)  # 또는 실제 설치 경로
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "GPU support ENABLED (CUDA)")
else()
    message(STATUS "GPU support DISABLED")
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
add_subdirectory(examples/convection3D)

# Python binding
option(PYTHON "Enable compilation of the Python module" ON)
if(PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    # Try to get CONDA_PREFIX from environment
    if(DEFINED ENV{CONDA_PREFIX})
        set(PYTHON_ENV_PATH "$ENV{CONDA_PREFIX}" CACHE PATH "Python environment path")
        message(STATUS "Detected conda environment: ${PYTHON_ENV_PATH}")
    else()
        # Fallback: manually set or use default
        set(PYTHON_ENV_PATH "/usr" CACHE PATH "Default python env path")
        message(WARNING "CONDA_PREFIX not set, using default path: ${PYTHON_ENV_PATH}")
    endif()

    set(CMAKE_PREFIX_PATH "${PYTHON_ENV_PATH}/lib/python3.11/site-packages/pybind11")
    set(Python_EXECUTABLE "${PYTHON_ENV_PATH}/bin/python" CACHE FILEPATH "Python 3.11 interpreter path")

    find_package(pybind11 REQUIRED) # or add_subdirectory(pybind11)

    pybind11_add_module(PaScaL_TDMA_pybind ${CMAKE_SOURCE_DIR}/src/pybind/pyBinding.cpp ${CMAKE_SOURCE_DIR}/src/PaScaL_TDMA.cpp)
    target_include_directories(PaScaL_TDMA_pybind PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/utils)
    target_link_libraries(PaScaL_TDMA_pybind PUBLIC MPI::MPI_CXX)

    add_custom_command(TARGET PaScaL_TDMA_pybind POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/examples/python
        COMMAND ${CMAKE_COMMAND} -E copy 
                $<TARGET_FILE:PaScaL_TDMA_pybind>
                ${CMAKE_SOURCE_DIR}/examples/python/
        COMMENT " Copying PaScaL_TDMA_pybind module to examples/python"
    )

endif()



# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/refs/heads/main.zip
#   DOWNLOAD_EXTRACT_TIMESTAMP true
# )

