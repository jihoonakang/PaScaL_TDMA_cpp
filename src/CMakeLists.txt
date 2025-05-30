# Build the PaScaL_TDMA static library
if(CUDA)
    # CUDA header with implementation
    add_library(PaScaL_TDMA STATIC PaScaL_TDMA.cpp TDMASolver.cu )
    # Mark .cuh as CUDA (needed if it's not separately compiled)
    set_source_files_properties(TDMASolver.cuh PROPERTIES LANGUAGE CUDA)
else()
    add_library(PaScaL_TDMA STATIC PaScaL_TDMA.cpp )
endif()

target_include_directories(PaScaL_TDMA PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
                                              ${CMAKE_SOURCE_DIR}/utils)
if(CUDA)
    target_link_libraries(PaScaL_TDMA PUBLIC MPI::MPI_CXX cuda)
    # Optional: show CUDA flags
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
else()
    target_link_libraries(PaScaL_TDMA PUBLIC MPI::MPI_CXX)
endif()


