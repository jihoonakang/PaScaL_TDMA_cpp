/**
 * @file pascal_tdma_c_interface.hpp
 * @brief C interface for PaScaL_TDMA library (single/multiple TDMA solvers).
 *
 * This header provides C bindings to create, solve, and destroy
 * parallel TDMA solver plans for both single and multiple systems.
 * The functions are designed for interoperability with Fortran (MPI_Fint).
 */

#pragma once
#include <cstddef>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a plan for solving a single TDMA system in parallel.
 *
 * @param[out] handle      Pointer to the created plan handle.
 * @param[in]  n_row       Number of rows in the TDMA system.
 * @param[in]  myrank      Rank of the calling process.
 * @param[in]  nprocs      Total number of MPI processes.
 * @param[in]  mpi_comm_f  Fortran MPI communicator (MPI_Fint).
 * @param[in]  tdma_type   TDMA solver type (e.g., Thomas, CR, PCR).
 */
void pascal_tdma_plan_single_create(void** handle, int n_row, int myrank, int nprocs,
                                    MPI_Fint mpi_comm_f, int tdma_type);

/**
 * @brief Solve a single TDMA system using the provided plan.
 *
 * @param[in] handle   Plan handle created by pascal_tdma_plan_single_create().
 * @param[in,out] a    Sub-diagonal coefficients (size n_row).
 * @param[in,out] b    Main diagonal coefficients (size n_row).
 * @param[in,out] c    Super-diagonal coefficients (size n_row).
 * @param[in,out] d    Right-hand side vector (solution is written back).
 */
void pascal_tdma_single_solve(void* handle, double* a, double* b, double* c, double* d);

/**
 * @brief Destroy a single TDMA plan and free resources.
 *
 * @param[in,out] handle Plan handle to destroy.
 */
void pascal_tdma_plan_single_destroy(void* handle);

/**
 * @brief Create a plan for solving multiple independent TDMA systems.
 *
 * @param[out] handle     Pointer to the created plan handle.
 * @param[in]  n_row      Number of rows in each TDMA system.
 * @param[in]  n_sys      Number of independent TDMA systems.
 * @param[in]  myrank     Rank of the calling process.
 * @param[in]  nprocs     Total number of MPI processes.
 * @param[in]  mpi_comm   MPI communicator (MPI_Fint).
 * @param[in]  tdma_type  TDMA solver type (e.g., Thomas, CR, PCR).
 */
void pascal_tdma_plan_many_create(void** handle, int n_row, int n_sys, int myrank, int nprocs,
                                  MPI_Fint mpi_comm, int tdma_type);

/**
 * @brief Solve multiple TDMA systems using the provided plan.
 *
 * @param[in] handle   Plan handle created by pascal_tdma_plan_many_create().
 * @param[in,out] a    Sub-diagonal coefficients (size n_row × n_sys).
 * @param[in,out] b    Main diagonal coefficients (size n_row × n_sys).
 * @param[in,out] c    Super-diagonal coefficients (size n_row × n_sys).
 * @param[in,out] d    Right-hand side vectors (solutions are written back).
 */
void pascal_tdma_many_solve(void* handle, double* a, double* b, double* c, double* d);

/**
 * @brief Destroy a multiple TDMA plan and free resources.
 *
 * @param[in,out] handle Plan handle to destroy.
 */
void pascal_tdma_plan_many_destroy(void* handle);

/**
 * @brief Create a plan for solving multiple TDMA systems with RHS partitioning.
 *
 * @param[out] handle     Pointer to the created plan handle.
 * @param[in]  n_row      Number of rows in each TDMA system.
 * @param[in]  n_sys      Number of independent TDMA systems.
 * @param[in]  myrank     Rank of the calling process.
 * @param[in]  nprocs     Total number of MPI processes.
 * @param[in]  mpi_comm   MPI communicator (MPI_Fint).
 * @param[in]  tdma_type  TDMA solver type (e.g., Thomas, CR, PCR).
 */
void pascal_tdma_plan_many_rhs_create(void** handle, int n_row, int n_sys, int myrank, int nprocs,
                                      MPI_Fint mpi_comm, int tdma_type);

/**
 * @brief Solve multiple TDMA systems with RHS partitioning.
 *
 * @param[in] handle   Plan handle created by pascal_tdma_plan_many_rhs_create().
 * @param[in,out] a    Sub-diagonal coefficients (size n_row × n_sys).
 * @param[in,out] b    Main diagonal coefficients (size n_row × n_sys).
 * @param[in,out] c    Super-diagonal coefficients (size n_row × n_sys).
 * @param[in,out] d    Right-hand side vectors (solutions are written back).
 */
void pascal_tdma_many_rhs_solve(void* handle, double* a, double* b, double* c, double* d);


/**
 * @brief Destroy a multiple TDMA (RHS) plan and free resources.
 *
 * @param[in,out] handle Plan handle to destroy.
 */
void pascal_tdma_plan_many_rhs_destroy(void* handle);

#ifdef __cplusplus
}
#endif
