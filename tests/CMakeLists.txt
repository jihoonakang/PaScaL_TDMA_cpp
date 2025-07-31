# tests/CMakeLists.txt

message(STATUS "Configuring tests/CMakeLists.txt")

# ===================[ Helper Functions ]===================

# Add a CPU-based test target
function(add_pascal_test target_name source_file)
    add_executable(${target_name} ${source_file})
    add_dependencies(${target_name} PaScaL_TDMA)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_directories(${target_name} PRIVATE /usr/local/lib)
    target_link_libraries(${target_name} PRIVATE PaScaL_TDMA MPI::MPI_CXX gtest_main gtest )
endfunction()

# Add a GPU-based test target (CUDA)
function(add_cuPascal_test target_name source_file)
    add_executable(${target_name} ${source_file})
    add_dependencies(${target_name} PaScaL_TDMA)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_directories(${target_name} PRIVATE /usr/local/lib)
    target_link_libraries(${target_name} PRIVATE cuda PaScaL_TDMA MPI::MPI_CXX gtest_main gtest )
endfunction()

# ===================[ Test Targets ]=======================

# CPU tests
add_pascal_test(test_single     test_single.cpp)
add_pascal_test(test_many       test_many.cpp)
add_pascal_test(test_many_rhs   test_many_rhs.cpp)

# GPU tests
if(CUDA)
    add_cuPascal_test(test_cuda_many        test_cuda_many.cpp)
    add_cuPascal_test(test_cuda_many_rhs    test_cuda_many_rhs.cpp)
endif()

# ===================[ CTest Integration ]==================

include(GoogleTest)

# Register tests with mpirun and specific arguments
add_test(NAME PaScaL_TDMA_single_p4_n100_kernel_standard 
         COMMAND mpirun -np 4 $<TARGET_FILE:test_single> 100 standard)
add_test(NAME PaScaL_TDMA_single_p4_n100_kernel_cyclic 
         COMMAND mpirun -np 4 $<TARGET_FILE:test_single> 100 cyclic)
add_test(NAME PaScaL_TDMA_single_p4_n1000_kernel_standard 
         COMMAND mpirun -np 4 $<TARGET_FILE:test_single> 1000 standard)
add_test(NAME PaScaL_TDMA_single_p4_n1000_kernel_cyclic 
         COMMAND mpirun -np 4 $<TARGET_FILE:test_single> 1000 cyclic)

add_test(NAME PaScaL_TDMA_many_p1_n100x100_kernel_standard 
         COMMAND mpirun -np 1 $<TARGET_FILE:test_many> 100 100)
add_test(NAME PaScaL_TDMA_many_p9_n500x500_kernel_standard
         COMMAND mpirun -np 9 $<TARGET_FILE:test_many> 500 500)

add_test(NAME PaScaL_TDMA_many_RHS_p1_n10x10x10_kernel_standard
         COMMAND mpirun -np 1 $<TARGET_FILE:test_many_rhs> 10 10 10)
add_test(NAME PaScaL_TDMA_many_RHS_p8_n100x100x100_kernel_standard
         COMMAND mpirun -np 8 $<TARGET_FILE:test_many_rhs> 20 20 20)

if(CUDA)
    add_test(NAME PaScaL_TDMA_cuda_many_p1_n10x20x20_kernel_standard
             COMMAND mpirun -np 1 $<TARGET_FILE:test_cuda_many> 10 40 40)
    add_test(NAME PaScaL_TDMA_cuda_many_p4_n10x32x64_kernel_standard
             COMMAND mpirun -np 4 $<TARGET_FILE:test_cuda_many> 10 32 64)
    add_test(NAME PaScaL_TDMA_cuda_many_RHS_p1_n10x20x20_kernel_standard
             COMMAND mpirun -np 1 $<TARGET_FILE:test_cuda_many_rhs> 10 40 40)
    add_test(NAME PaScaL_TDMA_cuda_many_RHS_p4_n10x32x64_kernel_standard
             COMMAND mpirun -np 4 $<TARGET_FILE:test_cuda_many_rhs> 10 32 64)
endif()

# For automatic GoogleTest discovery, uncomment if needed:
# gtest_discover_tests(test_single)
