/**
 * @file cuda_pascal_tdma_c_interface.cpp
 * @brief Implementation of C bindings for CUDA-based PaScaL_TDMA solvers.
 *
 * This file defines the C interface functions for creating, solving,
 * and destroying CUDA-enabled TDMA solver plans. It also provides
 * utility functions for GPU memory allocation and data transfer.
 */

#include "cuda_pascal_tdma_c_interface.hpp"
#include "pascal_tdma.cuh"
#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

using namespace CuPaScaL_TDMA;

extern "C" {

/**
 * @brief Allocate GPU memory.
 *
 * @param[out] ptr    Pointer to allocated device memory.
 * @param[in]  bytes  Number of bytes to allocate.
 */
void cuda_malloc(void **ptr, size_t bytes) {
    cudaMalloc(ptr, bytes);
}

/**
 * @brief Free GPU memory.
 *
 * @param[in] ptr  Pointer to device memory to free.
 */
void cuda_free(void *ptr) {
    cudaFree(ptr);
}

/**
 * @brief Copy data from host to device.
 *
 * @param[out] dst  Device pointer (destination).
 * @param[in]  src  Host pointer (source).
 * @param[in]  n    Number of double elements to copy.
 */
void cuda_memcpy_h2d(void *dst, const double *src, int n) {
    cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice);
}

/**
 * @brief Copy data from device to host.
 *
 * @param[out] dst  Host pointer (destination).
 * @param[in]  src  Device pointer (source).
 * @param[in]  n    Number of double elements to copy.
 */
void cuda_memcpy_d2h(double *dst, const void *src, int n) {
    cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyDeviceToHost);
}

/**
 * @brief Create a CUDA plan for solving multiple TDMA systems.
 */
void cu_pascal_tdma_plan_many_create(void** handle, int n_row, int ny_sys, int nz_sys,
                                     int myrank, int nprocs, MPI_Fint mpi_comm_f, int tdma_type) {
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_f);
    auto* plan = new CuPTDMAPlanMany();
    TDMAType type = (tdma_type == 0) ? TDMAType::Standard : TDMAType::Cyclic;
    plan->create(n_row, ny_sys, nz_sys, comm, type);
    *handle = static_cast<void*>(plan);
}

/**
 * @brief Solve multiple TDMA systems using the created CUDA plan.
 */
void cu_pascal_tdma_many_solve(void* handle, double* a, double* b, double* c, double* d) {
    auto* plan = static_cast<CuPTDMAPlanMany*>(handle);
    CuPTDMASolverMany::cuSolve(*plan, a, b, c, d);
}

/**
 * @brief Destroy a CUDA TDMA plan and free GPU resources.
 */
void cu_pascal_tdma_plan_many_destroy(void* handle) {
    auto* plan = static_cast<CuPTDMAPlanMany*>(handle);
    plan->destroy();
    delete plan;
    handle = nullptr;
}

/**
 * @brief Create a CUDA plan for solving multiple TDMA systems with multiple RHS vectors.
 */
void cu_pascal_tdma_plan_many_rhs_create(void** handle, int n_row, int ny_sys, int nz_sys,
                                         int myrank, int nprocs, MPI_Fint mpi_comm_f, 
                                         int tdma_type) {
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_f);
    auto* plan = new CuPTDMAPlanManyRHS();
    TDMAType type = (tdma_type == 0) ? TDMAType::Standard : TDMAType::Cyclic;
    plan->create(n_row, ny_sys, nz_sys, comm, type);
    *handle = static_cast<void*>(plan);
}

/**
 * @brief Solve multiple TDMA systems with multiple RHS vectors using the created CUDA plan.
 */
void cu_pascal_tdma_many_rhs_solve(void* handle, double* a, double* b, double* c, double* d) {
    auto* plan = static_cast<CuPTDMAPlanManyRHS*>(handle);
    CuPTDMASolverManyRHS::cuSolve(*plan, a, b, c, d);
}

/**
 * @brief Destroy a CUDA TDMA (RHS) plan and free GPU resources.
 */
void cu_pascal_tdma_plan_many_rhs_destroy(void* handle) {
    auto* plan = static_cast<CuPTDMAPlanManyRHS*>(handle);
    plan->destroy();
    delete plan;
    handle = nullptr;
}

} // extern "C"