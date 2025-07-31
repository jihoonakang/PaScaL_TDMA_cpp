add_library(fortran_interface
    pascal_tdma_c_interface.cpp
    fortran_interface.f90
)

target_include_directories(fortran_interface PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
)

target_link_libraries(fortran_interface PUBLIC PaScaL_TDMA MPI::MPI_CXX)

if(CUDA)
    add_library(cuda_fortran_interface
        cuda_pascal_tdma_c_interface.cpp
        cuda_fortran_interface.f90
    )

    target_include_directories(cuda_fortran_interface PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        ${MPI_Fortran_INCLUDE_PATH}
    )

    target_link_libraries(cuda_fortran_interface PUBLIC
        PaScaL_TDMA     # CUDA-based TDMA library
        MPI::MPI_CXX    # CUDA-aware MPI C++ interface
    )
endif()