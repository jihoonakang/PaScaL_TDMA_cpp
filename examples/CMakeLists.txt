
# Build the tdm_single example
add_executable(single_example single.cpp)

target_include_directories(single_example PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(single_example PRIVATE PaScaL_TDMA MPI::MPI_CXX)

# Build the tdm_many_2D example
add_executable(many_example many.cpp)

target_include_directories(many_example PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(many_example PRIVATE PaScaL_TDMA MPI::MPI_CXX)

# Build the tdm_many_RHS_3D example
add_executable(manyRHS_example manyRHS.cpp)

target_include_directories(manyRHS_example PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(manyRHS_example PRIVATE PaScaL_TDMA MPI::MPI_CXX)

if(CUDA)

    # 소스 파일
    add_executable(cuMany cuManyTest.cpp)
    set_source_files_properties(cuManyTest.cpp PROPERTIES LANGUAGE CUDA)

    target_include_directories(cuMany PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cuMany PRIVATE cuda PaScaL_TDMA MPI::MPI_CXX)

    # 소스 파일
    add_executable(cuManyRHS cuManyRHSTest.cpp)
    set_source_files_properties(cuManyRHSTest.cpp PROPERTIES LANGUAGE CUDA)

    target_include_directories(cuManyRHS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(cuManyRHS PRIVATE cuda PaScaL_TDMA MPI::MPI_CXX)

endif()
