
# Build the convection_3d
add_executable(convection_3d convection_3d.cpp domain_layout_3d.cpp convection_solver.cpp)

target_include_directories(convection_3d PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(convection_3d PRIVATE PaScaL_TDMA MPI::MPI_CXX)
