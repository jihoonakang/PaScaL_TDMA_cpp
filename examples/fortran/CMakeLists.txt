# Function to add a CPU-based Fortran example with optional multiple sources
function(add_fortran_example target_name)
    # All additional arguments after the target name are source files
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

    add_executable(${target_name} ${ARG_SOURCES})

    # Set the directory to place compiled Fortran modules (.mod)
    set_target_properties(${target_name} PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran_modules
        LINKER_LANGUAGE Fortran
    )

    # Include both source and build-time module locations
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/fortran
        ${CMAKE_BINARY_DIR}/src/fortran
    )

    target_link_libraries(${target_name} PRIVATE fortranInterface MPI::MPI_Fortran)
endfunction()

# Function to add a CPU-based Fortran example with optional multiple sources
function(add_cufortran_example target_name)
    # All additional arguments after the target name are source files
    cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})

    add_executable(${target_name} ${ARG_SOURCES})

    # Set the directory to place compiled Fortran modules (.mod)
    set_target_properties(${target_name} PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran_modules
        LINKER_LANGUAGE Fortran
    )

    # Include both source and build-time module locations
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/fortran
        ${CMAKE_BINARY_DIR}/src/fortran
    )

    target_link_libraries(${target_name} PRIVATE fortranInterface cuFortranInterface MPI::MPI_Fortran)
endfunction()


# CPU example
add_fortran_example(single_fortran SOURCES single.f90)
add_fortran_example(many_fortran SOURCES mpi_topology_2D.f90 many.f90)
add_fortran_example(many_rhs_fortran SOURCES mpi_topology_2D.f90 many_rhs_3D.f90)

# GPU example
if(CUDA)
    add_cufortran_example(cu_many_fortran SOURCES cuMany.f90)
    add_cufortran_example(cu_many_rhs_fortran SOURCES cuManyRHS.f90)
endif()