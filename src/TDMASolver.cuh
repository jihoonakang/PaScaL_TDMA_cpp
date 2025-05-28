#pragma once

#include <cuda_runtime.h>
#include <cassert>

class cuTDMASolver {
public:
    static void cuMany(const double* d_a, const double* d_b, double* d_c, double* d_d,
                       int nx, int ny, int nz, cudaStream_t stream = 0);
    static void cuManyCyclic(const double* d_a, const double* d_b, double* d_c, double* d_d,
                       int nx, int ny, int nz, cudaStream_t stream = 0);
    static void cuManyRHS(const double* a_d, const double* b_d, double* c_d, double* d_d,
                          int nx, int ny, int nz, cudaStream_t stream = 0);
    static void cuManyRHSCyclic(const double* a_d, const double* b_d, double* c_d, double* d_d,
                          int nx, int ny, int nz, cudaStream_t stream = 0);
};
