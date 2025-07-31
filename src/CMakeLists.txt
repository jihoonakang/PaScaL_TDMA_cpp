### Build the PaScaL_TDMA static library

# Create static library target
add_library(PaScaL_TDMA STATIC)

# Add source files (CUDA sources only when enabled)
target_sources(PaScaL_TDMA PRIVATE
    pascal_tdma.cpp
    $<$<BOOL:${CUDA}>:tdma_solver.cu>
    $<$<BOOL:${CUDA}>:pascal_tdma.cu>
)

# Mark CUDA headers for compilation when CUDA is enabled
if(CUDA)
    set_source_files_properties(tdma_solver.cuh pascal_tdma.cuh 
                                PROPERTIES LANGUAGE CUDA)
endif()

# Include directories: src and shared utilities
target_include_directories(PaScaL_TDMA PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_SOURCE_DIR}/utils
)


# Link libraries: MPI always, CUDA runtime when enabled
target_link_libraries(PaScaL_TDMA PUBLIC
    MPI::MPI_CXX
    $<$<BOOL:${CUDA}>:cuda>
)

# Compile definitions via generator expressions
target_compile_definitions(PaScaL_TDMA PRIVATE
    $<$<BOOL:${CUDA}>:USE_CUDA=1>
    $<$<AND:$<BOOL:${CUDA}>,$<BOOL:${CUDA_AWARE_MPI}>>:USE_CUDA_AWARE_MPI=1>
)