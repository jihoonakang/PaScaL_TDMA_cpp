# Build the PaScaL_TDMA static library

# Source files common to both CPU and CUDA builds
set(PASCAL_TDMA_SRC PaScaL_TDMA.cpp)

if(CUDA)
    list(APPEND PASCAL_TDMA_SRC TDMASolver.cu PaScaL_TDMA.cu)

    # Ensure .cuh is treated as a CUDA file (header only)
    set_source_files_properties(TDMASolver.cuh PaScaL_TDMA.cuh PROPERTIES LANGUAGE CUDA)
endif()

add_library(PaScaL_TDMA STATIC ${PASCAL_TDMA_SRC})

# Include directories
target_include_directories(PaScaL_TDMA PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/utils
)

# Link libraries
target_link_libraries(PaScaL_TDMA PUBLIC MPI::MPI_CXX)

if(CUDA)
    target_compile_options(PaScaL_TDMA PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
    target_link_libraries(PaScaL_TDMA PUBLIC cuda)
    target_compile_definitions(PaScaL_TDMA PRIVATE USE_CUDA=1)
    if(CUDA_AWARE_MPI)
        target_compile_definitions(PaScaL_TDMA PRIVATE USE_CUDA_AWARE_MPI=1)
    endif()

    message(STATUS "CUDA flags: ${CUDA}")
    message(STATUS "CUDA_AWARE_MPI flags: ${CUDA_AWARE_MPI}")
endif()