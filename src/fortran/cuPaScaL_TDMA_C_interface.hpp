#pragma once
#include <cstddef>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create a CUDA TDMA plan for many systems
 */
void cu_pascal_tdma_plan_many_create(void** handle,
                                     int n_row, int ny_sys, int nz_sys,
                                     int myrank, int nprocs,
                                     MPI_Fint mpi_comm_f, int tdma_type);

/**
 * @brief Solve batched TDMA systems on GPU
 */
void cu_pascal_tdma_many_solve(void* handle,
                               double* a, double* b, double* c, double* d);

/**
 * @brief Destroy CUDA TDMA plan and free resources
 */
void cu_pascal_tdma_plan_many_destroy(void* handle);

#ifdef __cplusplus
}
#endif
