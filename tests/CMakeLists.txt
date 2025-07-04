# tests/CMakeLists.txt

# 메시지로 진입 확인
message(STATUS "Configuring tests/CMakeLists.txt")

# Function to add a CPU-based test
function(add_pascal_test target_name source_file)
    add_executable(${target_name} ${source_file})
    add_dependencies(${target_name} PaScaL_TDMA)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_directories(${target_name} PRIVATE /usr/local/lib)
    target_link_libraries(${target_name} PRIVATE PaScaL_TDMA MPI::MPI_CXX gtest_main gtest )
endfunction()

# Function to add a GPU-based test
function(add_cuPascal_test target_name source_file)
    add_executable(${target_name} ${source_file})
    add_dependencies(${target_name} PaScaL_TDMA)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_directories(${target_name} PRIVATE /usr/local/lib)
    target_link_libraries(${target_name} PRIVATE cuda PaScaL_TDMA MPI::MPI_CXX gtest_main gtest )
endfunction()

# CPU test
add_pascal_test(testSingle    testSingle.cpp)
add_pascal_test(testMany      testMany.cpp)
add_pascal_test(testManyRHS   testManyRHS.cpp)

# GPU test
if(CUDA)
    add_cuPascal_test(testCuMany    testCuMany.cpp)
    add_cuPascal_test(testCuManyRHS testCuManyRHS.cpp)
endif()

# 테스트 등록
# add_test(NAME SolverSingleTest COMMAND testSingle)
include(GoogleTest)
# gtest_discover_tests(testSingle)
# get_property(tests DIRECTORY PROPERTY TESTS)
# message(STATUS "CTest registered tests: ${tests}")


# testSingle을 mpirun으로 실행하도록 등록
add_test(NAME PaScaL_TDMA_single_p4_n100_kernel_standard COMMAND mpirun -np 4 $<TARGET_FILE:testSingle> 100 standard)
add_test(NAME PaScaL_TDMA_single_p4_n100_kernel_cyclic COMMAND mpirun -np 4 $<TARGET_FILE:testSingle> 100 cyclic)
add_test(NAME PaScaL_TDMA_single_p4_n1000_kernel_standard COMMAND mpirun -np 4 $<TARGET_FILE:testSingle> 1000 standard)
add_test(NAME PaScaL_TDMA_single_p4_n1000_kernel_cyclic COMMAND mpirun -np 4 $<TARGET_FILE:testSingle> 1000 cyclic)

add_test(NAME PaScaL_TDMA_many_p1_n100x100_kernel_standard COMMAND mpirun -np 1 $<TARGET_FILE:testMany> 100 100)
add_test(NAME PaScaL_TDMA_many_p9_n500x500_kernel_standard COMMAND mpirun -np 9 $<TARGET_FILE:testMany> 500 500)

add_test(NAME PaScaL_TDMA_many_RHS_p1_n10x10x10_kernel_standard COMMAND mpirun -np 1 $<TARGET_FILE:testManyRHS> 10 10 10)
add_test(NAME PaScaL_TDMA_many_RHS_p8_n100x100x100_kernel_standard COMMAND mpirun -np 8 $<TARGET_FILE:testManyRHS> 20 20 20)

if(CUDA)
    add_test(NAME PaScaL_TDMA_cuda_many_p1_n10x20x20_kernel_standard COMMAND mpirun -np 1 $<TARGET_FILE:testCuMany> 10 40 40)
    add_test(NAME PaScaL_TDMA_cuda_many_p4_n10x32x64_kernel_standard COMMAND mpirun -np 4 $<TARGET_FILE:testCuMany> 10 32 64)
    add_test(NAME PaScaL_TDMA_cuda_many_RHS_p1_n10x20x20_kernel_standard COMMAND mpirun -np 1 $<TARGET_FILE:testCuManyRHS> 10 40 40)
    add_test(NAME PaScaL_TDMA_cuda_many_RHS_p4_n10x32x64_kernel_standard COMMAND mpirun -np 4 $<TARGET_FILE:testCuManyRHS> 10 32 64)
endif()

# get_property(tests DIRECTORY PROPERTY TESTS)
# message(STATUS "CTest registered tests: ${tests}")
