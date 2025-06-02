#include <cuda_runtime.h>
#include "PaScaL_TDMA.hpp"
#include <cassert>
#include <mpi.h>


// CUDA 커널
__global__ void cuForwardMany(double* A, double* B, double* C, double* D, int n_row, int n_sys) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_sys) return;

    A[j] /= B[j]; D[j] /= B[j]; C[j] /= B[j];
    A[j + n_sys] /= B[j + n_sys]; D[j + n_sys] /= B[j + n_sys]; C[j + n_sys] /= B[j + n_sys];

    for (int i = 2; i < n_row; ++i) {
        int idx = i * n_sys + j;
        int idx_prev = idx - n_sys;
        double r = 1.0 / (B[idx] - A[idx] * C[idx_prev]);
        D[idx] = r * (D[idx] - A[idx] * D[idx_prev]);
        C[idx] = r * C[idx];
        A[idx] = -r * A[idx] * A[idx_prev];
    }
}

__global__ void cuBackwardMany(double* A, double* C, double* D, int n_row, int n_sys) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_sys) return;

    for (int i = n_row - 3; i >= 1; --i) {
        int idx = i * n_sys + j;
        int idx_next = idx + n_sys;
        D[idx] -= C[idx] * D[idx_next];
        A[idx] -= C[idx] * A[idx_next];
        C[idx] = -C[idx] * C[idx_next];
    }
}

__global__ void cuReduceMany(const double* A, const double* C, const double* D,
                             double* A0, double* A1, double* C0, double* C1,
                             double* D0, double* D1,
                             int n_row, int n_sys) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_sys) return;

    double r = 1.0 / (1.0 - A[j + n_sys] * C[j]);
    D0[j] = r * (D[j] - C[j] * D[j + n_sys]);
    A0[j] = r * A[j];
    C0[j] = -r * C[j] * C[j + n_sys];
    A1[j] = A[(n_row - 1) * n_sys + j];
    C1[j] = C[(n_row - 1) * n_sys + j];
    D1[j] = D[(n_row - 1) * n_sys + j];
}

__global__ void cuReconstructMany(const double* A, const double* C, double* D,
                                  const double* D0, const double* D1,
                                  int n_row, int n_sys) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_sys) return;

    D[0 * n_sys + j] = D0[j];
    D[(n_row - 1) * n_sys + j] = D1[j];

    for (int i = 1; i < n_row - 1; ++i) {
        int idx = i * n_sys + j;
        D[idx] = D[idx] - A[idx] * D0[j] - C[idx] * D1[j];
    }
}


namespace PaScaL_TDMA {
class cuPTDMASolverMany {

// cuSolve 함수
__host__ void cuSolve(PTDMAPlanMany& plan,
                      double* A, double* B, double* C, double* D) {

    const int n_row = plan.n_row;
    const int n_sys = plan.n_sys;
    assert(n_row > 2);

    if (plan.size == 1) {
        dispatchTDMASolver<BatchType::Many>(plan.type, A, B, C, D, n_row, n_sys);
        return;
    }

    // Device memory
    double *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, sizeof(double) * n_row * n_sys);
    cudaMalloc(&d_B, sizeof(double) * n_row * n_sys);
    cudaMalloc(&d_C, sizeof(double) * n_row * n_sys);
    cudaMalloc(&d_D, sizeof(double) * n_row * n_sys);

    cudaMemcpy(d_A, A, sizeof(double) * n_row * n_sys, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(double) * n_row * n_sys, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(double) * n_row * n_sys, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, sizeof(double) * n_row * n_sys, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n_sys + threads - 1) / threads;

    cuForwardMany<<<blocks, threads>>>(d_A, d_B, d_C, d_D, n_row, n_sys);
    cuBackwardMany<<<blocks, threads>>>(d_A, d_C, d_D, n_row, n_sys);
    cuReduceMany<<<blocks, threads>>>(d_A, d_C, d_D,
                                      plan.d_A_rd, plan.d_A_rd + n_sys,
                                      plan.d_C_rd, plan.d_C_rd + n_sys,
                                      plan.d_D_rd, plan.d_D_rd + n_sys,
                                      n_row, n_sys);

    // MPI reduction transpose
    MPI_Request request[3];
    MPI_Status statuses[3];

    MPI_Ialltoallw(plan.d_A_rd, plan.count_send.data(), plan.displ_send.data(),
                   plan.ddtype_FS.data(),
                   plan.d_A_rt, plan.count_recv.data(), plan.displ_recv.data(),
                   plan.ddtype_BS.data(),
                   plan.comm_ptdma, &request[0]);

    MPI_Ialltoallw(plan.d_C_rd, plan.count_send.data(), plan.displ_send.data(),
                   plan.ddtype_FS.data(),
                   plan.d_C_rt, plan.count_recv.data(), plan.displ_recv.data(),
                   plan.ddtype_BS.data(),
                   plan.comm_ptdma, &request[1]);

    MPI_Ialltoallw(plan.d_D_rd, plan.count_send.data(), plan.displ_send.data(),
                   plan.ddtype_FS.data(),
                   plan.d_D_rt, plan.count_recv.data(), plan.displ_recv.data(),
                   plan.ddtype_BS.data(),
                   plan.comm_ptdma, &request[2]);

    MPI_Waitall(3, request, statuses);

    dispatchTDMASolver<BatchType::Many>(plan.type,
                                        plan.d_A_rt, plan.d_B_rt,
                                        plan.d_C_rt, plan.d_D_rt,
                                        plan.n_row_rt, plan.n_sys_rt);

    MPI_Ialltoallw(plan.d_D_rt, plan.count_recv.data(), plan.displ_recv.data(),
                   plan.ddtype_BS.data(),
                   plan.d_D_rd, plan.count_send.data(), plan.displ_send.data(),
                   plan.ddtype_FS.data(),
                   plan.comm_ptdma, &request[0]);
    MPI_Waitall(1, request, statuses);

    cuReconstructMany<<<blocks, threads>>>(d_A, d_C, d_D,
                                           plan.d_D_rd, plan.d_D_rd + n_sys,
                                           n_row, n_sys);

    cudaMemcpy(D, d_D, sizeof(double) * n_row * n_sys, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
}

};
};