#include <cuda_runtime.h>
#include <cassert>
#include "TDMASolver.cuh"

__global__ static void cuManyKernel(const double* __restrict__ a,
                                    const double* __restrict__ b,
                                    double* __restrict__ c,
                                    double* __restrict__ d,
                                    int nx, int ny, int nz) {
    // Global (j,k) index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int gid = k + j * nz;
    if (j >= ny || k >= nz) return;

    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int tid = tk + tj * blockDim.x;

    extern __shared__ double shared[];
    double* a1 = shared;
    double* b1 = a1 + (blockDim.x + 1) * blockDim.y;
    double* c0 = b1 + (blockDim.x + 1) * blockDim.y;
    double* c1 = c0 + (blockDim.x + 1) * blockDim.y;
    double* d0 = c1 + (blockDim.x + 1) * blockDim.y;
    double* d1 = d0 + (blockDim.x + 1) * blockDim.y;

    // 초기화
    b1[tid] = b[gid];
    c1[tid] = c[gid];
    d1[tid] = d[gid];

    d1[tid] /= b1[tid];
    c1[tid] /= b1[tid];

    d[gid] = d1[tid];
    c[gid] = c1[tid];

    // Forward sweep
    for (int i = 1; i < nx; ++i) {
        c0[tid] = c1[tid];
        d0[tid] = d1[tid];

        gid += ny * nz;
        a1[tid] = a[gid];
        b1[tid] = b[gid];
        c1[tid] = c[gid];
        d1[tid] = d[gid];

        double r = 1.0 / (b1[tid] - a1[tid] * c0[tid]);
        d1[tid] = r * (d1[tid] - a1[tid] * d0[tid]);
        c1[tid] = r * c1[tid];

        d[gid] = d1[tid];
        c[gid] = c1[tid];
    }

    // Backward sweep
    for (int i = nx - 2; i >= 0; --i) {
        gid -= ny * nz;
        c0[tid] = c[gid];
        d0[tid] = d[gid];

        d0[tid] = d0[tid] - c0[tid] * d1[tid];
        d1[tid] = d0[tid];
        d[gid] = d0[tid];
    }
}

void cuTDMASolver::cuMany(const double* a_d, const double* b_d, double* c_d, double* d_d,
            int nx, int ny, int nz, cudaStream_t stream) {

        assert(nz > 1 && ny > 0 && nx > 0);

        dim3 threads(8, 8);
        dim3 blocks((nz + threads.x - 1) / threads.x,
                    (ny + threads.y - 1) / threads.y);

        int sys_size = (threads.x + 1)* threads.y;
        size_t shmem = 6 * sys_size * sizeof(double); 

        cuManyKernel<<<blocks, threads, shmem, stream>>>(
            a_d, b_d, c_d, d_d, nx, ny, nz );
        cudaDeviceSynchronize();
}

__global__ static void cuManyCyclicKernel(const double* __restrict__ a,
                                          const double* __restrict__ b,
                                          double* __restrict__ c,
                                          double* __restrict__ d,
                                          double* __restrict__ e,
                                          int nx, int ny, int nz) {
    // Global (j,k) index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;

    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int tid = tk + tj * blockDim.x;

    int stride = ny * nz;
    int gid = j * nz + k;

    extern __shared__ double shared[];
    double* a1 = shared;
    double* b1 = a1 + (blockDim.x + 1) * blockDim.y;
    double* c0 = b1 + (blockDim.x + 1) * blockDim.y;
    double* c1 = c0 + (blockDim.x + 1) * blockDim.y;
    double* d0 = c1 + (blockDim.x + 1) * blockDim.y;
    double* d1 = d0 + (blockDim.x + 1) * blockDim.y;
    double* e0 = d1 + (blockDim.x + 1) * blockDim.y;
    double* e1 = e0 + (blockDim.x + 1) * blockDim.y;

    // zero e
    for (int i = 0; i < nx; ++i) {
        e[gid + i * stride] = 0.0;
    }

    // initialize e[2], e[nz]
    e[gid + stride] = -a[gid + stride];
    e[gid + (nx - 1) * stride] = -c[gid + (nx - 1) * stride];

    // step 1
    gid += stride;
    d1[tid] = d[gid];
    b1[tid] = b[gid];
    c1[tid] = c[gid];
    e1[tid] = e[gid];

    d1[tid] /= b1[tid];
    c1[tid] /= b1[tid];
    e1[tid] /= b1[tid];

    d[gid] = d1[tid];
    c[gid] = c1[tid];
    e[gid] = e1[tid];

    // forward sweep
    for (int i = 2; i < nx; ++i) {
        gid += stride;
        c0[tid] = c1[tid];
        d0[tid] = d1[tid];
        e0[tid] = e1[tid];

        a1[tid] = a[gid];
        b1[tid] = b[gid];
        c1[tid] = c[gid];
        d1[tid] = d[gid];
        e1[tid] = e[gid];

        double r = 1.0 / (b1[tid] - a1[tid] * c0[tid]);
        d1[tid] = r * (d1[tid] - a1[tid] * d0[tid]);
        e1[tid] = r * (e1[tid] - a1[tid] * e0[tid]);
        c1[tid] = r * c1[tid];

        d[gid] = d1[tid];
        c[gid] = c1[tid];
        e[gid] = e1[tid];
    }

    // backward sweep
    for (int i = nx - 2; i >= 1; --i) {
        gid -= stride;
        c0[tid] = c[gid];
        d0[tid] = d[gid];
        e0[tid] = e[gid];

        d0[tid] -= c0[tid] * d1[tid];
        e0[tid] -= c0[tid] * e1[tid];

        d1[tid] = d0[tid];
        e1[tid] = e0[tid];

        d[gid] = d0[tid];
        e[gid] = e0[tid];
    }

    // final correction step (i=1)
    gid -= stride;
    double a_1 = a[gid];
    double b_1 = b[gid];
    double c_1 = c[gid];
    double d_1 = d[gid];
    double e_last = e[gid + (nx - 1) * stride];

    double numerator = d_1 - a_1 * d[gid + (nx - 1) * stride] - c_1 * d0[tid];
    double denominator = b_1 + a_1 * e_last + c_1 * e0[tid];

    d1[tid] = numerator / denominator;
    d[gid] = d1[tid];

    // apply final correction
    for (int i = 1; i < nx; ++i) {
        gid += stride;
        d[gid] += d1[tid] * e[gid];
    }
}

void cuTDMASolver::cuManyCyclic(const double* a_d, const double* b_d, double* c_d, double* d_d,
                  int nx, int ny, int nz, cudaStream_t stream) {

    assert(nz > 1 && ny > 0 && nx > 0);
    double *e_d;
    cudaMalloc(&e_d, nx * ny * nz * sizeof(double));

    dim3 threads(8, 8);
    dim3 blocks((nz + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);

    int sys_size = (threads.x + 1)* threads.y;
    size_t shmem = 8 * sys_size * sizeof(double); 

    cuManyCyclicKernel<<<blocks, threads, shmem, stream>>>(
        a_d, b_d, c_d, d_d, e_d, nx, ny, nz);

    cudaDeviceSynchronize();
    cudaFree(e_d);
}

__global__ static void cuManyRHSKernel(const double* __restrict__ a,
                                       const double* __restrict__ b,
                                       double* __restrict__ c,
                                       double* __restrict__ d,
                                       int nx, int ny, int nz) {
    // Global index for (j, k)
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;
    int gid = k + j * nz; 

    const int stride = ny * nz;
    double c_local[256]; //Solving race condition. TODO: memory problem
    for (int i = 0; i < nx; i++) c_local[i] = c[i];
    
    extern __shared__ double shared[];
    double* d0 = shared;
    double* d1 = d0 + (blockDim.x + 1) * blockDim.y;

    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int tid = tk + tj * blockDim.x;

    // Forward sweep
    double b1 = b[0];
    d1[tid] = d[gid];

    d1[tid] /= b1;
    c_local[0] /= b1;

    d[gid] = d1[tid];

    for (int i = 1; i < nx; ++i) {
        d0[tid] = d1[tid];
        gid += stride;

        double a1 = a[i];
        d1[tid] = d[gid];

        double r = 1.0 / (b[i] - a1 * c_local[i-1]);
        d1[tid] = r * (d1[tid] - a1 * d0[tid]);

        c_local[i] *= r;
        d[gid] = d1[tid];
    }

    // Backward sweep
    for (int i = nx - 2; i >= 0; --i) {
        gid -= stride;
        d0[tid] = d[gid];
        d0[tid] -= c_local[i] * d1[tid];
        d1[tid] = d0[tid];
        d[gid] = d0[tid];
    }
}

void cuTDMASolver::cuManyRHS(const double* a_d, const double* b_d,
                             double* c_d, double* d_d,
                             int nx, int ny, int nz,
                             cudaStream_t stream) {
    assert(nx > 1 && ny > 0 && nz > 0);

    dim3 threads(8, 8);
    dim3 blocks((nz + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);

    int sys_size = (threads.x + 1) * threads.y;  // +1 padding for bank conflict
    size_t shmem = 2 * sys_size * sizeof(double);

    cuManyRHSKernel<<<blocks, threads, shmem, stream>>>(
        a_d, b_d, c_d, d_d, nx, ny, nz
    );

    cudaDeviceSynchronize();
}

__global__ static void cuManyRHSCyclicKernel(const double* __restrict__ a,
                                             const double* __restrict__ b,
                                             double* __restrict__ c,
                                             double* __restrict__ d,
                                             int nx, int ny, int nz) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = ny * nz;
    int gid = j * nz + k;

    double c_local[256], e[256]; //Solving race condition. TODO: memory problem
    for (int i = 0; i < nx; i++) c_local[i] = c[i];

    extern __shared__ double shared[];
    double* d0 = shared;
    double* d1 = d0 + (blockDim.x + 1) * blockDim.y;

    for (int i = 0; i < nx; ++i)
        e[i] = 0.0;

    e[1]      = -a[1];
    e[nx - 1] = -c_local[nx - 1];

    gid += stride;
    double b1 = b[1];
    d1[tid] = d[gid];
    d1[tid] /= b1;
    c_local[1] /= b1;
    e[1] /= b1;
    d[gid] = d1[tid];

    // Forward sweep
    for (int i = 2; i < nx; ++i) {
        gid += stride;
        d0[tid] = d1[tid];

        double a1 = a[i];
        b1 = b[i];
        d1[tid] = d[gid];

        double r = 1.0 / (b1 - a1 * c_local[i - 1]);
        d1[tid] = r * (d1[tid] - a1 * d0[tid]);
        e[i] = r * (e[i] - a1 * e[i - 1]);
        c_local[i] = r * c_local[i];
        d[gid] = d1[tid];
    }

    // Backward sweep
    for (int i = nx - 2; i >= 1; --i) {
        gid -= stride;
        double c_i = c_local[i];
        d0[tid] = d[gid];
        d0[tid] -= c_i * d1[tid];
        e[i] -= c_i * e[i + 1];
        d1[tid] = d0[tid];
        d[gid] = d0[tid];
    }

    // Correction step (i = 0)
    gid -= stride;
    double a0 = a[0];
    double b0 = b[0];
    double c0 = c_local[0];
    double e_last = e[nx - 1];

    double numerator = d[gid] - a0 * d[gid + (nx - 1) * stride] - c0 * d1[tid];
    double denominator = b0 + a0 * e_last + c0 * e[1];
    d1[tid] = numerator / denominator;
    d[gid] = d1[tid];

    // Final correction
    for (int i = 1; i < nx; ++i) {
        gid += stride;
        d[gid] += d1[tid] * e[i];
    }
}

void cuTDMASolver::cuManyRHSCyclic(const double* a_d, const double* b_d, double* c_d,
                                   double* d_d, int nx, int ny, int nz,
                                   cudaStream_t stream) {
    assert(nx > 2 && ny > 0 && nz > 0);

    // Allocate e (correction vector, same shape as d)
    dim3 threads(16, 16);
    dim3 blocks((nz + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);

    int sys_size = (threads.x + 1) * threads.y;
    size_t shmem = 6 * sys_size * sizeof(double);  // c0, c1, d0, d1, e0, e1

    cuManyRHSCyclicKernel<<<blocks, threads, shmem, stream>>>(
        a_d, b_d, c_d, d_d, nx, ny, nz);

    cudaDeviceSynchronize();  // for development
}