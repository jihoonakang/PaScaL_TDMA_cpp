#include "cuda_pascal_tdma_c_interface.hpp"
#include "pascal_tdma.cuh"
#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

using namespace CuPaScaL_TDMA;

extern "C" {

void cuda_malloc(void **ptr, size_t bytes) {
    cudaMalloc(ptr, bytes);
}

void cuda_free(void *ptr) {
    cudaFree(ptr);
}

void cuda_memcpy_h2d(void *dst, const double *src, int n) {
    cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice);
}

void cuda_memcpy_d2h(double *dst, const void *src, int n) {
    cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyDeviceToHost);
}

void cu_pascal_tdma_plan_many_create(void** handle,
                                     int n_row, int ny_sys, int nz_sys,
                                     int myrank, int nprocs,
                                     MPI_Fint mpi_comm_f, int tdma_type) {
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_f);
    auto* plan = new CuPTDMAPlanMany();
    TDMAType type = (tdma_type == 0) ? TDMAType::Standard : TDMAType::Cyclic;
    plan->create(n_row, ny_sys, nz_sys, comm, type);
    *handle = static_cast<void*>(plan);
}

void cu_pascal_tdma_many_solve(void* handle,
                                double* a, double* b, double* c, double* d) {
    auto* plan = static_cast<CuPTDMAPlanMany*>(handle);
    CuPTDMASolverMany::cuSolve(*plan, a, b, c, d);
}

void cu_pascal_tdma_plan_many_destroy(void* handle) {
    auto* plan = static_cast<CuPTDMAPlanMany*>(handle);
    plan->destroy();
    delete plan;
    handle = nullptr;
}

void cu_pascal_tdma_plan_many_rhs_create(void** handle,
                                     int n_row, int ny_sys, int nz_sys,
                                     int myrank, int nprocs,
                                     MPI_Fint mpi_comm_f, int tdma_type) {
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_f);
    auto* plan = new CuPTDMAPlanManyRHS();
    TDMAType type = (tdma_type == 0) ? TDMAType::Standard : TDMAType::Cyclic;
    plan->create(n_row, ny_sys, nz_sys, comm, type);
    *handle = static_cast<void*>(plan);
}

void cu_pascal_tdma_many_rhs_solve(void* handle,
                                double* a, double* b, double* c, double* d) {
    auto* plan = static_cast<CuPTDMAPlanManyRHS*>(handle);
    CuPTDMASolverManyRHS::cuSolve(*plan, a, b, c, d);
}

void cu_pascal_tdma_plan_many_rhs_destroy(void* handle) {
    auto* plan = static_cast<CuPTDMAPlanManyRHS*>(handle);
    plan->destroy();
    delete plan;
    handle = nullptr;
}

}