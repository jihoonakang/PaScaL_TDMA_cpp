# src/pybind/CMakeLists.txt

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
set(Python_EXECUTABLE "${PYTHON_ENV_PATH}/bin/python" CACHE FILEPATH "Python interpreter")

find_package(pybind11 REQUIRED)

pybind11_add_module(PaScaL_TDMA_pybind
    pyBinding.cpp
    ${CMAKE_SOURCE_DIR}/src/PaScaL_TDMA.cpp
)

target_include_directories(PaScaL_TDMA_pybind PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
)

target_link_libraries(PaScaL_TDMA_pybind PUBLIC MPI::MPI_CXX)

add_custom_command(TARGET PaScaL_TDMA_pybind POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/examples/python
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:PaScaL_TDMA_pybind>
        ${CMAKE_SOURCE_DIR}/examples/python/
    COMMENT "Copying pybind module to examples/python"
)