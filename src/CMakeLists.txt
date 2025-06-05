# Build the PaScaL_TDMA static library

# Source files common to both CPU and CUDA builds
set(PASCAL_TDMA_SRC PaScaL_TDMA.cpp)

if(CUDA)
    list(APPEND PASCAL_TDMA_SRC TDMASolver.cu PaScaL_TDMA.cu)

    # Ensure .cuh is treated as a CUDA file (header only)
    set_source_files_properties(TDMASolver.cuh PROPERTIES LANGUAGE CUDA)
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
    target_link_libraries(PaScaL_TDMA PUBLIC cuda)
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
endif()