# src/pybind/CMakeLists.txt

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")

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

# Optional CUDA version
if(CUDA)
    enable_language(CUDA)

    # pybind11_add_module can't handle .cu files in cases.
    # Instead, We use add_library for CUDA support.
    add_library(PaScaL_TDMA_cuda_pybind MODULE
        cupyBinding.cpp
        ${CMAKE_SOURCE_DIR}/src/PaScaL_TDMA.cu
        ${CMAKE_SOURCE_DIR}/src/TDMASolver.cu
    )

    set_target_properties(PaScaL_TDMA_cuda_pybind PROPERTIES
        OUTPUT_NAME "PaScaL_TDMA_cuda_pybind"
        PREFIX ""  # for Python import (no lib prefix)
        SUFFIX ".cpython-311-x86_64-linux-gnu.so"  # Python ABI suffix
    )

    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/src/PaScaL_TDMA.cu
        ${CMAKE_SOURCE_DIR}/src/TDMASolver.cu
        PROPERTIES LANGUAGE CUDA
    )

    target_compile_options(PaScaL_TDMA_cuda_pybind PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
    )
    
    target_include_directories(PaScaL_TDMA_cuda_pybind PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        /usr/local/cuda/include
    )

    target_compile_definitions(PaScaL_TDMA_cuda_pybind PRIVATE CUDA=1)
    target_link_libraries(PaScaL_TDMA_cuda_pybind PRIVATE pybind11::module MPI::MPI_CXX cuda)

    add_custom_command(TARGET PaScaL_TDMA_cuda_pybind POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:PaScaL_TDMA_cuda_pybind>
            ${CMAKE_SOURCE_DIR}/examples/python/
        COMMENT "Copying pybind CUDA module to examples/python"
    )
endif()