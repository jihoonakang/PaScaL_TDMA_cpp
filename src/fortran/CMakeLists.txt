add_library(fortranInterface
    PaScaL_TDMA_C_interface.cpp
    fortran_interface.f90
)

target_include_directories(fortranInterface PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
)

target_link_libraries(fortranInterface PUBLIC PaScaL_TDMA MPI::MPI_CXX)

if(CUDA)
    add_library(cuFortranInterface
        cuPaScaL_TDMA_C_interface.cpp
        cuFortran_interface.f90
    )

    target_include_directories(cuFortranInterface PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/utils
        ${MPI_Fortran_INCLUDE_PATH}
    )

    target_link_libraries(cuFortranInterface PUBLIC
        PaScaL_TDMA     # CUDA-based TDMA library
        MPI::MPI_CXX    # CUDA-aware MPI C++ interface
    )
endif()