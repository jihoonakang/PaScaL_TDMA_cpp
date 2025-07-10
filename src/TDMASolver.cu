/**
 * @file TDMASolver.cu
 * @brief CUDA implementations for cuTDMASolver methods using custom kernels.
 */

#include <cuda_runtime.h>
#include <cassert>
#include "TDMASolver.cuh"

/**
 * @brief Kernel to solve many independent tridiagonal systems in parallel.
 *
 * Each thread handles one system identified by (j,k) in the 2D grid of systems.
 * Shared memory buffers store intermediate coefficients for forward/backward sweeps.
 *
 * @param a Lower-diagonal array (device pointer).
 * @param b Diagonal array.
 * @param c Upper-diagonal array (in/out).
 * @param d Right-hand side array (in/out).
 * @param nx Number of rows per system.
 * @param ny Number of systems in Y dimension.
 * @param nz Number of systems in Z dimension.
 */
__global__ static void cuManyKernel(const double* __restrict__ a,
                                    const double* __restrict__ b,
                                    double* __restrict__ c,
                                    double* __restrict__ d,
                                    int nx, int ny, int nz) {

    // Compute global system indices
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;
    int gid = k + j * nz;
    const int stride = ny * nz;

    // Local thread ID with padding for shared memory
    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int tid = tk + tj * (blockDim.x + 1);

    // Shared memory layout: a1,b1,c0,c1,d0,d1 buffers per thread
    extern __shared__ double shared[];
    double* a1 = shared;
    double* b1 = a1 + (blockDim.x + 1) * blockDim.y;
    double* c0 = b1 + (blockDim.x + 1) * blockDim.y;
    double* c1 = c0 + (blockDim.x + 1) * blockDim.y;
    double* d0 = c1 + (blockDim.x + 1) * blockDim.y;
    double* d1 = d0 + (blockDim.x + 1) * blockDim.y;

    // --- Initialization using shared memory: first row
    b1[tid] = b[gid];
    c1[tid] = c[gid];
    d1[tid] = d[gid];

    d1[tid] /= b1[tid];
    c1[tid] /= b1[tid];

    // Write back initial values to global memory
    d[gid] = d1[tid];
    c[gid] = c1[tid];

    // --- Forward elimination for rows 1..nx-1
    for (int i = 1; i < nx; i++) {
        // Shift buffers
        c0[tid] = c1[tid];
        d0[tid] = d1[tid];

        // Load next row elements
        gid += stride;             ///< move to next row in device arrays
        a1[tid] = a[gid];
        b1[tid] = b[gid];
        c1[tid] = c[gid];
        d1[tid] = d[gid];

        // Compute on shared memory
        double r = 1.0 / (b1[tid] - a1[tid] * c0[tid]);
        d1[tid] = r * (d1[tid] - a1[tid] * d0[tid]);
        c1[tid] = r * c1[tid];

        // Write back elimination results to global memory
        d[gid] = d1[tid];
        c[gid] = c1[tid];
    }

    // --- Backward substitution
    for (int i = nx - 2; i >= 0; --i) {
        // Load previous row elements
        gid -= stride;             ///< move back to previous row
        c0[tid] = c[gid];
        d0[tid] = d[gid];

        // Compute on shared memory
        d0[tid] = d0[tid] - c0[tid] * d1[tid];
        d1[tid] = d0[tid];

        // Write back substitution results to global memory
        d[gid] = d0[tid];
    }
}

/**
 * @brief Host wrapper launching cuManyKernel.
 *
 * Configures thread blocks, shared memory size, and synchronizes.
 */
void cuTDMASolver::cuMany(const double* a_d, const double* b_d, 
                          double* c_d, double* d_d,
                          int nx, int ny, int nz, 
                          cudaStream_t stream) noexcept {

    assert(nz > 1 && ny > 0 && nx > 0);

    // Launch configuration
    dim3 threads(8, 8);
    dim3 blocks((nz + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);

    int sys_size = (threads.x + 1)* threads.y; // +1 padding for bank conflict
    size_t shmem = 6 * sys_size * sizeof(double); 

    // Kernel launch
    cuManyKernel<<<blocks, threads, shmem, stream>>>(
        a_d, b_d, c_d, d_d, nx, ny, nz );

}

/**
 * @brief Kernel for cyclic tridiagonal systems, including correction vector e.
 *
 * Each thread handles one system identified by (j,k) in the 2D grid of systems.
 * Shared memory buffers store intermediate coefficients for forward/backward sweeps.
 *
 * @param a Lower-diagonal array (device pointer).
 * @param b Diagonal array.
 * @param c Upper-diagonal array (in/out).
 * @param d Right-hand side array (in/out).
 * @param nx Number of rows per system.
 * @param ny Number of systems in Y dimension.
 * @param nz Number of systems in Z dimension.
 */
__global__ static void cuManyCyclicKernel(const double* __restrict__ a,
                                          const double* __restrict__ b,
                                          double* __restrict__ c,
                                          double* __restrict__ d,
                                          double* __restrict__ e,
                                          int nx, int ny, int nz) {
    // Compute global system indices
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;
    int gid = j * nz + k;
    const int stride = ny * nz;

    // Local thread ID with padding for shared memory
    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int tid = tk + tj * (blockDim.x + 1);

    // Shared memory layout: a1,b1,c0,c1,d0,d1,e0,e1 buffers per thread
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
    for (int i = 0; i < nx; i++) {
        e[gid + i * stride] = 0.0;
    }

    // initialize e[2], e[nx]
    e[gid + stride] = -a[gid + stride];
    e[gid + (nx - 1) * stride] = -c[gid + (nx - 1) * stride];

    // --- Initialization using shared memory: first row
    gid += stride;
    d1[tid] = d[gid];
    b1[tid] = b[gid];
    c1[tid] = c[gid];
    e1[tid] = e[gid];

    d1[tid] /= b1[tid];
    c1[tid] /= b1[tid];
    e1[tid] /= b1[tid];

    // Write back initial values to global memory
    d[gid] = d1[tid];
    c[gid] = c1[tid];
    e[gid] = e1[tid];

    // --- Forward elimination for rows 1..nx-1
    for (int i = 2; i < nx; i++) {
        // Shift buffers
        c0[tid] = c1[tid];
        d0[tid] = d1[tid];
        e0[tid] = e1[tid];

        // Load next row elements
        gid += stride;          ///< move to next row in device arrays
        a1[tid] = a[gid];
        b1[tid] = b[gid];
        c1[tid] = c[gid];
        d1[tid] = d[gid];
        e1[tid] = e[gid];

        // Compute on shared memory
        double r = 1.0 / (b1[tid] - a1[tid] * c0[tid]);
        d1[tid] = r * (d1[tid] - a1[tid] * d0[tid]);
        e1[tid] = r * (e1[tid] - a1[tid] * e0[tid]);
        c1[tid] = r * c1[tid];

        // Write back elimination results to global memory
        d[gid] = d1[tid];
        c[gid] = c1[tid];
        e[gid] = e1[tid];
    }

    // --- Backward substitution
    for (int i = nx - 2; i >= 1; --i) {
        // Load previous row elements
        gid -= stride;              ///< move back to previous row
        c0[tid] = c[gid];
        d0[tid] = d[gid];
        e0[tid] = e[gid];

        // Compute on shared memory
        d0[tid] -= c0[tid] * d1[tid];
        e0[tid] -= c0[tid] * e1[tid];

        d1[tid] = d0[tid];
        e1[tid] = e0[tid];

        // Write back substitution results to global memory
        d[gid] = d0[tid];
        e[gid] = e0[tid];
    }

    // Final correction step (i=0)
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

    // Apply final correction
    for (int i = 1; i < nx; i++) {
        gid += stride;
        d[gid] += d1[tid] * e[gid];
    }
}

/**
 * @brief Host wrapper launching cuManyCyclicKernel.
 *
 * Configures thread blocks, shared memory size, and synchronizes.
 */
void cuTDMASolver::cuManyCyclic(const double* a_d, const double* b_d, 
                                double* c_d, double* d_d,
                                int nx, int ny, int nz, 
                                cudaStream_t stream) noexcept {

    assert(nz > 1 && ny > 0 && nx > 0);
    double *e_d;
    cudaMalloc(&e_d, nx * ny * nz * sizeof(double));

    // Launch configuration
    dim3 threads(8, 8);
    dim3 blocks((nz + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);

    int sys_size = (threads.x + 1)* threads.y; // +1 padding for bank conflict
    size_t shmem = 8 * sys_size * sizeof(double); 

    // Kernel launch
    cuManyCyclicKernel<<<blocks, threads, shmem, stream>>>(
        a_d, b_d, c_d, d_d, e_d, nx, ny, nz);

    cudaFree(e_d);
}

/**
 * @brief Kernel to solve batched systems with common diagonals.
 *
 * Each thread handles one system identified by (j,k) in the 2D grid of systems.
 * Shared memory buffers store intermediate coefficients for forward/backward sweeps.
 *
 * @param a Lower-diagonal array (device pointer).
 * @param b Diagonal array.
 * @param c Upper-diagonal array (in/out).
 * @param d Right-hand side array (in/out).
 * @param nx Number of rows per system.
 * @param ny Number of systems in Y dimension.
 * @param nz Number of systems in Z dimension.
 */
__global__ static void cuManyRHSKernel(const double* __restrict__ a,
                                       const double* __restrict__ b,
                                       double* __restrict__ c,
                                       double* __restrict__ d,
                                       int nx, int ny, int nz) {
    // Compute global system indices
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;
    int gid = k + j * nz; 
    const int stride = ny * nz;

    // Local thread ID with padding for shared memory
    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int tid = tk + tj * (blockDim.x + 1);

    // Local register for solving race condition. TODO: memory problem
    double c_local[256];    /// Local register 
    for (int i = 0; i < nx; i++) c_local[i] = c[i];
    
    // Shared memory layout: d0,d1 buffers per thread
    extern __shared__ double shared[];
    double* d0 = shared;
    double* d1 = d0 + (blockDim.x + 1) * blockDim.y;

    // --- Initialization using shared memory: first row
    double b1 = b[0];
    d1[tid] = d[gid];

    d1[tid] /= b1;
    c_local[0] /= b1;

    // Write back initial values to global memory
    d[gid] = d1[tid];

    // --- Forward elimination for rows 1..nx-1
    for (int i = 1; i < nx; i++) {
        // Shift buffers
        d0[tid] = d1[tid];

        // Load next row elements
        gid += stride;
        d1[tid] = d[gid];

        // Compute on shared memory
        double a1 = a[i];
        double r = 1.0 / (b[i] - a1 * c_local[i-1]);
        d1[tid] = r * (d1[tid] - a1 * d0[tid]);
        c_local[i] *= r;

        // Write back elimination results to global memory
        d[gid] = d1[tid];
    }

    // --- Backward substitution
    for (int i = nx - 2; i >= 0; --i) {
        // Load previous row elements
        gid -= stride;              ///< move back to previous row
        d0[tid] = d[gid];

        // Compute on shared memory
        d0[tid] -= c_local[i] * d1[tid];
        d1[tid] = d0[tid];

        // Write back substitution results to global memory
        d[gid] = d0[tid];
    }
}

/**
 * @brief Host wrapper launching cuManyRHSKernel.
 *
 * Configures thread blocks, shared memory size, and synchronizes.
 */
void cuTDMASolver::cuManyRHS(const double* a_d, const double* b_d,
                             double* c_d, double* d_d,
                             int nx, int ny, int nz,
                             cudaStream_t stream) noexcept {

    assert(nx > 1 && ny > 0 && nz > 0);

    // Launch configuration
    dim3 threads(8, 8);
    dim3 blocks((nz + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);

    int sys_size = (threads.x + 1) * threads.y;  // +1 padding for bank conflict
    size_t shmem = 2 * sys_size * sizeof(double);

    // Kernel launch
    cuManyRHSKernel<<<blocks, threads, shmem, stream>>>(
        a_d, b_d, c_d, d_d, nx, ny, nz);

}

/**
 * @brief Kernel to solve cyclic batched systems with common diagonals.
 *
 * Each thread handles one system identified by (j,k) in the 2D grid of systems.
 * Shared memory buffers store intermediate coefficients for forward/backward sweeps.
 *
 * @param a Lower-diagonal array (device pointer).
 * @param b Diagonal array.
 * @param c Upper-diagonal array (in/out).
 * @param d Right-hand side array (in/out).
 * @param nx Number of rows per system.
 * @param ny Number of systems in Y dimension.
 * @param nz Number of systems in Z dimension.
 */
__global__ static void cuManyRHSCyclicKernel(const double* __restrict__ a,
                                             const double* __restrict__ b,
                                             double* __restrict__ c,
                                             double* __restrict__ d,
                                             int nx, int ny, int nz) {
    // Compute global system indices
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;
    int gid = j * nz + k;
    const int stride = ny * nz;

    // Local thread ID with padding for shared memory
    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int tid = tk + tj * (blockDim.x + 1);

    // Local register for solving race condition. TODO: memory problem
    double c_local[256], e[256];
    for (int i = 0; i < nx; i++) c_local[i] = c[i];

    // Shared memory layout: d0,d1 buffers per thread
    extern __shared__ double shared[];
    double* d0 = shared;
    double* d1 = d0 + (blockDim.x + 1) * blockDim.y;

    // zero e
    for (int i = 0; i < nx; i++)
        e[i] = 0.0;

    // initialize e[2], e[nx]
    e[1]      = -a[1];
    e[nx - 1] = -c_local[nx - 1];

    // --- Initialization using shared memory: first row
    gid += stride;
    double b1 = b[1];
    d1[tid] = d[gid];
    d1[tid] /= b1;
    c_local[1] /= b1;
    e[1] /= b1;

    // Write back initial values to global memory
    d[gid] = d1[tid];

    // --- Forward elimination for rows 1..nx-1
    for (int i = 2; i < nx; i++) {
        // Shift buffers
        d0[tid] = d1[tid];

        // Load next row elements
        double a1 = a[i];
        b1 = b[i];
        gid += stride;          ///< move to next row in device arrays
        d1[tid] = d[gid];

        // Compute on shared memory
        double r = 1.0 / (b1 - a1 * c_local[i - 1]);
        d1[tid] = r * (d1[tid] - a1 * d0[tid]);
        e[i] = r * (e[i] - a1 * e[i - 1]);
        c_local[i] = r * c_local[i];

        // Write back elimination results to global memory
        d[gid] = d1[tid];
    }

    // --- Backward substitution
    for (int i = nx - 2; i >= 1; --i) {
        // Load previous row elements
        gid -= stride;              ///< move back to previous row
        double c_i = c_local[i];
        d0[tid] = d[gid];

        // Compute on shared memory
        d0[tid] -= c_i * d1[tid];
        e[i] -= c_i * e[i + 1];
        d1[tid] = d0[tid];

        // Write back substitution results to global memory
        d[gid] = d0[tid];
    }

    // Final correction step (i=0)
    gid -= stride;
    double a0 = a[0];
    double b0 = b[0];
    double c0 = c_local[0];
    double e_last = e[nx - 1];

    double numerator = d[gid] - a0 * d[gid + (nx - 1) * stride] - c0 * d1[tid];
    double denominator = b0 + a0 * e_last + c0 * e[1];
    d1[tid] = numerator / denominator;
    d[gid] = d1[tid];

    // Apply final correction
    for (int i = 1; i < nx; i++) {
        gid += stride;
        d[gid] += d1[tid] * e[i];
    }
}

/**
 * @brief Host wrapper launching cuManyRHSCyclicKernel.
 *
 * Configures thread blocks, shared memory size, and synchronizes.
 */
void cuTDMASolver::cuManyRHSCyclic(const double* a_d, const double* b_d, double* c_d,
                                   double* d_d, int nx, int ny, int nz,
                                   cudaStream_t stream) noexcept {
    assert(nx > 2 && ny > 0 && nz > 0);

    // Launch configuration
    dim3 threads(8, 8);
    dim3 blocks((nz + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y);

    int sys_size = (threads.x + 1) * threads.y; // +1 padding for bank conflict
    size_t shmem = 2 * sys_size * sizeof(double);  // d0, d1

    // Kernel launch
    cuManyRHSCyclicKernel<<<blocks, threads, shmem, stream>>>(
        a_d, b_d, c_d, d_d, nx, ny, nz);
}