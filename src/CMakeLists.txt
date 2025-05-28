# Build the PaScaL_TDMA static library
add_library(PaScaL_TDMA STATIC
    PaScaL_TDMA.cpp
    TDMASolver.cu     # CUDA header with implementation
)

# Mark .cuh as CUDA (needed if it's not separately compiled)
set_source_files_properties(TDMASolver.cuh PROPERTIES LANGUAGE CUDA)

target_include_directories(PaScaL_TDMA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
                                              ${CMAKE_SOURCE_DIR}/utils)
target_link_libraries(PaScaL_TDMA PUBLIC MPI::MPI_CXX cuda)

# Optional: show CUDA flags
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
