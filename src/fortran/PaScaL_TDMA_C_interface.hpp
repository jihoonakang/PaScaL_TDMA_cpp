#pragma once
#include <cstddef>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

void pascal_tdma_plan_single_create(void** handle, int n_row,
                                    int myrank, int nprocs,
                                    MPI_Fint mpi_comm_f, int tdma_type);

void pascal_tdma_single_solve(void* handle,
                              double* a, double* b, double* c, double* d);

void pascal_tdma_plan_single_destroy(void* handle);

void pascal_tdma_plan_many_create(void** handle, int n_row, int n_sys,
                                    int myrank, int nprocs,
                                    MPI_Fint mpi_comm, int tdma_type);

void pascal_tdma_many_solve(void* handle,
                              double* a, double* b, double* c, double* d);

void pascal_tdma_plan_many_destroy(void* handle);

void pascal_tdma_plan_many_rhs_create(void** handle, int n_row, int n_sys,
                                    int myrank, int nprocs,
                                    MPI_Fint mpi_comm, int tdma_type);

void pascal_tdma_many_rhs_solve(void* handle,
                                double* a, double* b, double* c, double* d);

void pascal_tdma_plan_many_rhs_destroy(void* handle);

#ifdef __cplusplus
}
#endif
