# tests/CMakeLists.txt

# 메시지로 진입 확인
message(STATUS "Configuring tests/CMakeLists.txt")

# testSingle
add_executable(testSingle testSingle.cpp)
target_include_directories(testSingle PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_directories(testSingle PRIVATE /usr/local/lib)
add_dependencies(testSingle PaScaL_TDMA)
target_link_libraries(testSingle PRIVATE PaScaL_TDMA gtest_main gtest MPI::MPI_CXX)

# testMany
add_executable(testMany testMany.cpp)
target_include_directories(testMany PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_directories(testMany PRIVATE /usr/local/lib)
add_dependencies(testMany PaScaL_TDMA)
target_link_libraries(testMany PRIVATE PaScaL_TDMA gtest_main gtest MPI::MPI_CXX)

# testManyRHS
add_executable(testManyRHS testManyRHS.cpp)
target_include_directories(testManyRHS PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_directories(testManyRHS PRIVATE /usr/local/lib)
add_dependencies(testManyRHS PaScaL_TDMA)
target_link_libraries(testManyRHS PRIVATE PaScaL_TDMA gtest_main gtest MPI::MPI_CXX)


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

# get_property(tests DIRECTORY PROPERTY TESTS)
# message(STATUS "CTest registered tests: ${tests}")
