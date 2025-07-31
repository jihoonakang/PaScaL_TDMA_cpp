/**
 * @file cuda_pascal_tdma_c_interface.hpp
 * @brief C interface for CUDA-based PaScaL_TDMA library (multi-system solvers).
 *
 * This header provides C bindings to create, solve, and destroy
 * CUDA-enabled parallel TDMA solver plans for multiple systems,
 * including variants with multiple right-hand sides (RHS).
 * The functions are designed for interoperability with Fortran (MPI_Fint).
 */

#pragma once
#include <cstddef>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a CUDA plan for solving multiple TDMA systems.
 *
 * @param[out] handle      Pointer to the created plan handle.
 * @param[in]  n_row       Number of rows in each TDMA system.
 * @param[in]  ny_sys      Number of TDMA systems in the y-dimension.
 * @param[in]  nz_sys      Number of TDMA systems in the z-dimension.
 * @param[in]  myrank      Rank of the calling process.
 * @param[in]  nprocs      Total number of MPI processes.
 * @param[in]  mpi_comm_f  Fortran MPI communicator (MPI_Fint).
 * @param[in]  tdma_type   TDMA solver type (0: Standard, 1: Cyclic).
 */
void cu_pascal_tdma_plan_many_create(void** handle, int n_row, int ny_sys, int nz_sys,
                                     int myrank, int nprocs, MPI_Fint mpi_comm_f, int tdma_type);

/**
 * @brief Solve multiple TDMA systems on the GPU using the provided plan.
 *
 * @param[in]     handle   Plan handle created by cu_pascal_tdma_plan_many_create().
 * @param[in,out] a        Sub-diagonal coefficients (device pointer).
 * @param[in,out] b        Main diagonal coefficients (device pointer).
 * @param[in,out] c        Super-diagonal coefficients (device pointer).
 * @param[in,out] d        Right-hand side vectors (solutions are written back to device memory).
 */
void cu_pascal_tdma_many_solve(void* handle, double* a, double* b, double* c, double* d);

/**
 * @brief Destroy a CUDA TDMA plan and free GPU resources.
 *
 * @param[in,out] handle Plan handle to destroy.
 */
void cu_pascal_tdma_plan_many_destroy(void* handle);

/**
 * @brief Create a CUDA plan for solving multiple TDMA systems with multiple RHS vectors.
 *
 * @param[out] handle      Pointer to the created plan handle.
 * @param[in]  n_row       Number of rows in each TDMA system.
 * @param[in]  ny_sys      Number of TDMA systems in the y-dimension.
 * @param[in]  nz_sys      Number of TDMA systems in the z-dimension.
 * @param[in]  myrank      Rank of the calling process.
 * @param[in]  nprocs      Total number of MPI processes.
 * @param[in]  mpi_comm_f  Fortran MPI communicator (MPI_Fint).
 * @param[in]  tdma_type   TDMA solver type (0: Standard, 1: Cyclic).
 */
void cu_pascal_tdma_plan_many_rhs_create(void** handle, int n_row, int ny_sys, int nz_sys,
                                         int myrank, int nprocs, MPI_Fint mpi_comm_f, 
                                         int tdma_type);

/**
 * @brief Solve multiple TDMA systems with multiple RHS vectors on the GPU.
 *
 * @param[in]     handle   Plan handle created by cu_pascal_tdma_plan_many_rhs_create().
 * @param[in,out] a        Sub-diagonal coefficients (device pointer).
 * @param[in,out] b        Main diagonal coefficients (device pointer).
 * @param[in,out] c        Super-diagonal coefficients (device pointer).
 * @param[in,out] d        Right-hand side vectors (solutions are written back to device memory).
 */
void cu_pascal_tdma_many_rhs_solve(void* handle, double* a, double* b, double* c, double* d);

/**
 * @brief Destroy a CUDA TDMA (RHS) plan and free GPU resources.
 *
 * @param[in,out] handle Plan handle to destroy.
 */
void cu_pascal_tdma_plan_many_rhs_destroy(void* handle);

#ifdef __cplusplus
}
#endif