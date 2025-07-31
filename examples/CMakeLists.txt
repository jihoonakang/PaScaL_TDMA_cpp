# Function to add a CPU-based example
function(add_pascal_example target_name source_file)
    add_executable(${target_name} ${source_file})
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${target_name} PRIVATE PaScaL_TDMA MPI::MPI_CXX)
endfunction()

# Function to add a CUDA-based example
function(add_cuda_example target_name source_file)
    add_executable(${target_name} ${source_file})
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${target_name} PRIVATE cuda PaScaL_TDMA MPI::MPI_CXX)
endfunction()

# CPU examples
add_pascal_example(single_example   single.cpp)
add_pascal_example(many_example     many.cpp)
add_pascal_example(many_rhs_example many_rhs.cpp)

# CUDA examples
if(CUDA)
    add_cuda_example(cuda_many_example      cuda_many.cpp)
    add_cuda_example(cuda_many_rhs_example  cuda_many_rhs.cpp)
endif()

# Fortran examples
if(FORTRAN)
    add_subdirectory(fortran)
endif()