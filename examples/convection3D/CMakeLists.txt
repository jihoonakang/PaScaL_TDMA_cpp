
# Build the convection3D
add_executable(convection3D convection3D.cpp domainLayout3D.cpp convectionSolver.cpp)

target_include_directories(convection3D PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(convection3D PRIVATE PaScaL_TDMA MPI::MPI_CXX)
