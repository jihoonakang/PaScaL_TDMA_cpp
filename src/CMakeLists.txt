### Build the PaScaL_TDMA static library

# Collect common source files for C++ and CUDA
set(PASCAL_TDMA_SRC PaScaL_TDMA.cpp)

# Append CUDA sources and treat headers as CUDA if CUDA support is enabled
if(CUDA)
    list(APPEND PASCAL_TDMA_SRC TDMASolver.cu PaScaL_TDMA.cu)
    # Ensure CUDA headers (.cuh) are compiled as CUDA language
    set_source_files_properties(TDMASolver.cuh PaScaL_TDMA.cuh 
                                PROPERTIES LANGUAGE CUDA)
endif()

# Create static library target
add_library(PaScaL_TDMA STATIC ${PASCAL_TDMA_SRC})

# Specify public include directories for the library
target_include_directories(PaScaL_TDMA PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/utils
)

# Link against MPI C++ library
target_link_libraries(PaScaL_TDMA PUBLIC MPI::MPI_CXX)

# CUDA-specific compile options and libraries
if(CUDA)
    target_compile_options( PaScaL_TDMA PRIVATE
                            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
    target_link_libraries(PaScaL_TDMA PUBLIC cuda)

    # Define macro for CUDA code
    target_compile_definitions(PaScaL_TDMA PRIVATE USE_CUDA=1)

    # Define macro if CUDA-Aware MPI is enabled
    if(CUDA_AWARE_MPI)
        target_compile_definitions(PaScaL_TDMA PRIVATE USE_CUDA_AWARE_MPI=1)
    endif()
endif()