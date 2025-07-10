/**
 * @file PaScaL_TDMA.cu
 * @brief Parallel TDMA solver implementation using CUDA-aware MPI.
 *
 * Contains:
 *  - Utility kernel for initializing arrays (cuInitValue)
 *  - Batched tridiagonal solvers (cuManyModifiedThomasMany, cuManyModifiedThomasManyRHS)
 *  - Reconstruction kernels (cuReconstructMany, cuReconstructManyRHS)
 *  - Slab transpose helpers for YZ<->XY communication
 *  - TDMA plan classes (cuPTDMAPlanMany, cuPTDMAPlanManyRHS)
 *  - Solver orchestration (cuPTDMASolverMany, cuPTDMASolverManyRHS)
 *  - Serial dispatch utilities (cuBatchSolver, cuDispatchTDMASolver)
 */

#include <cuda_runtime.h>
#include <cassert>
#include <numeric>
#include <mpi.h>
#include "PaScaL_TDMA.cuh"
#include "TDMASolver.cuh"

/**
 * @brief Initialize an array on the GPU to a given value.
 * @param[in,out] arr  Pointer to the array in device memory.
 * @param[in]     n    Number of elements in the array.
 * @param[in]     val  Value to set each element to.
 */
__global__ void cuInitValue(double* arr, int n, double val) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] = val;
    }
}

/**
 * @brief Batched forward/backward Thomas for multiple independent systems (Many).
 *
 * Splits each 1D tridiagonal system along X into forward elimination and
 * backward substitution. Uses shared memory pipelining for efficient data reuse.
 * Reduces boundary coefficients into a_rd/c_rd/d_rd for inter-rank exchange.
 *
 * @param[in,out] a    Sub-diagonal coefficients (modified).
 * @param[in,out] b    Main diagonal coefficients.
 * @param[in,out] c    Super-diagonal coefficients.
 * @param[in,out] d    Right-hand side values.
 * @param[out]    a_rd Reduced sub-diagonal boundary (two endpoints).
 * @param[out]    c_rd Reduced super-diagonal boundary.
 * @param[out]    d_rd Reduced RHS boundary.
 * @param[in]     n_row Number of equations per system (X dimension).
 * @param[in]     ny    Number of systems in Y dimension.
 * @param[in]     nz    Number of systems in Z dimension.
 */
__global__ void cuManyModifiedThomasMany(
    double* a, double* b, double* c, double* d, 
    double* a_rd, double* c_rd, double* d_rd, int n_row, int ny, int nz) {

    // 2D grid/thread index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;

    int stride = ny * nz;       ///< Number of elements per X-slice
    int gid0   = k + j * nz;    ///< Global index for i=0
    int gid    = gid0;

    // Shared memory tile pitch (avoid bank conflicts)
    int tile_pitch = blockDim.x + 1;
    int tid = threadIdx.x + threadIdx.y * tile_pitch;

    extern __shared__ double shared[];  ///< Shared memory buffer

    // Pointers into shared for pipeline storage
    double* a0 = shared;
    double* b0 = a0 + tile_pitch * blockDim.y;
    double* c0 = b0 + tile_pitch * blockDim.y;
    double* d0 = c0 + tile_pitch * blockDim.y;
    double* a1 = d0 + tile_pitch * blockDim.y;
    double* b1 = a1 + tile_pitch * blockDim.y;
    double* c1 = b1 + tile_pitch * blockDim.y;
    double* d1 = c1 + tile_pitch * blockDim.y;
    double* r0 = d1 + tile_pitch * blockDim.y;

    // i = 0
    a0[tid] = a[gid];
    b0[tid] = b[gid];
    c0[tid] = c[gid];
    d0[tid] = d[gid];

    double inv = 1.0 / b0[tid];
    a0[tid] *= inv; 
    c0[tid] *= inv; 
    d0[tid] *= inv;

    a[gid] = a0[tid];
    c[gid] = c0[tid];
    d[gid] = d0[tid];

    // i = 1
    gid += stride;
    a1[tid] = a[gid];
    b1[tid] = b[gid];
    c1[tid] = c[gid];
    d1[tid] = d[gid];

    inv = 1.0 / b1[tid];
    a1[tid] *= inv; 
    c1[tid] *= inv; 
    d1[tid] *= inv;

    a[gid] = a1[tid];
    c[gid] = c1[tid];
    d[gid] = d1[tid];

    // forward sweep
    for (int i = 2; i < n_row; i++) {
        gid += stride;

        // pipeline copy
        a0[tid] = a1[tid];
        c0[tid] = c1[tid];
        d0[tid] = d1[tid];

        // load next
        a1[tid] = a[gid];
        b1[tid] = b[gid];
        c1[tid] = c[gid];
        d1[tid] = d[gid];

        double r = 1.0 / (b1[tid] - a1[tid] * c0[tid]);
        d1[tid] = r * (d1[tid] - a1[tid] * d0[tid]);
        c1[tid] = r * c1[tid];
        a1[tid] = -r * a1[tid] * a0[tid];

        // store
        a[gid] = a1[tid];
        c[gid] = c1[tid];
        d[gid] = d1[tid];
    }

    a_rd[gid0 + stride] = a1[tid];
    c_rd[gid0 + stride] = c1[tid];
    d_rd[gid0 + stride] = d1[tid];

        // upper elimination (backward)
    a1[tid] = a0[tid];
    c1[tid] = c0[tid];
    d1[tid] = d0[tid];

    gid -= stride; // n_row - 2

    for (int i = n_row - 3; i >= 1; --i) {
        gid -= stride;         // offset = (i, j, k)

        // Load i-th data into shared
        a0[tid] = a[gid];
        c0[tid] = c[gid];
        d0[tid] = d[gid];

        // Elimination using (i+1)th data
        d0[tid] -= c0[tid] * d1[tid];
        a0[tid] -= c0[tid] * a1[tid];
        c0[tid] = -c0[tid] * c1[tid];

        // Pipeline copy
        a1[tid] = a0[tid];
        c1[tid] = c0[tid];
        d1[tid] = d0[tid];

        // Store result to global memory
        a[gid] = a0[tid];
        c[gid] = c0[tid];
        d[gid] = d0[tid];
    }

    // top row correction
    a0[tid] = a[gid0];
    c0[tid] = c[gid0];
    d0[tid] = d[gid0];

    r0[tid] = 1.0 / (1.0 - a1[tid] * c0[tid]);
    d0[tid] = r0[tid] * (d0[tid] - c0[tid] * d1[tid]);
    a0[tid] = r0[tid] * a0[tid];
    c0[tid] = -r0[tid] * c0[tid] * c1[tid];

    d[gid0] = d0[tid];
    a[gid0] = a0[tid];
    c[gid0] = c0[tid];

    a_rd[gid0] = a0[tid];
    c_rd[gid0] = c0[tid];
    d_rd[gid0] = d0[tid];

}

/**
 * @brief Reconstruct full solutions from reduced boundary results.
 *
 * Uses the exchanged boundary data d_rd to update interior solution values.
 *
 * @param[in]     a, c     Modified coefficients from factorization.
 * @param[in,out] d        Solution array to update.
 * @param[in]     d_rd     Boundary solutions.
 * @param[in]     n_row    Number of equations per system.
 * @param[in]     ny, nz   Grid dimensions.
 */
__global__ void cuReconstructMany(const double* a, const double* c, double* d,
                                  const double* d_rd,
                                  int n_row, int ny, int nz) {

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= ny || k >= nz) return;

    // Thread-local indices
    int tj = threadIdx.y;
    int tk = threadIdx.x;

    int tile_pitch = blockDim.x + 1;
    int tid = tk + tj * tile_pitch;

    // Allocate shared memory
    extern __shared__ double shared[];

    double* ds = shared;
    double* de = ds + tile_pitch * blockDim.y;

    int gid = j * nz + k;            // system index
    int stride  = ny * nz;

    // Load reduced system solutions into shared memory
    ds[tid] = d_rd[gid];             // d_rd(:,:,1)
    de[tid] = d_rd[gid + stride];    // d_rd(:,:,2)
    __syncthreads();

    // Update solution vector
    d[gid] = ds[tid];                // d(:,:,1)
    d[gid + (n_row - 1) * stride] = de[tid];  // d(:,:,nz_row)

    for (int i = 1; i < n_row - 1; i++) {
        gid += stride;
        d[gid] -= a[gid] * ds[tid] + c[gid] * de[tid];
    }
}

/**
 * @brief Batched forward/backward Thomas for multiple RHS (ManyRHS).
 *
 * Assumes constant a/c coefficients across systems. Only d and b vary per system.
 * Saves reduced boundaries a_rd/c_rd/d_rd for exchange, then reconstructs.
 *
 * @param[in,out] a    Sub-diagonal (constant across systems).
 * @param[in,out] b    Main diagonal (varies per X but constant across YZ).
 * @param[in,out] c    Super-diagonal (constant across systems).
 * @param[in,out] d    RHS values per system.
 * @param[out]    a_rd Reduced a at system endpoints.
 * @param[out]    c_rd Reduced c at system endpoints.
 * @param[out]    d_rd Reduced d at system endpoints.
 * @param[in]     n_row Number of equations per system.
 * @param[in]     ny    Systems in Y.
 * @param[in]     nz    Systems in Z.
 */
__global__ void cuManyModifiedThomasManyRHS(
    double* a, double* b, double* c, double* d, 
    double* a_rd, double* c_rd, double* d_rd, int n_row, int ny, int nz) {

    // 2D grid/thread index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;
    int gid = k + j * nz;      // system index
    int gid0 =gid;

    const int stride = ny * nz;

    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int tile_pitch = blockDim.x + 1;
    int tid = tk + tj * tile_pitch;

    // Shared memory tile: [nx+1][ny]
    extern __shared__ double shared[];

    double* d0 = shared;
    double* d1 = d0 + tile_pitch * blockDim.y;

    double b0, b1, r;
    double a_local[256], c_local[256]; //Solving race condition. TODO: memory problem

    for (int i = 0; i < n_row; i++) {
        a_local[i] = a[i];
        c_local[i] = c[i];
    }

    // i = 0
    b0 = b[0];
    d0[tid] = d[gid];

    a_local[0] /= b0;
    c_local[0] /= b0;
    d0[tid] /= b0;

    d[gid] = d0[tid];

    // i = 1
    gid += stride;
    b1 = b[1];
    d1[tid] = d[gid];

    a_local[1] /= b1;
    c_local[1] /= b1;
    d1[tid] /= b1;

    d[gid] = d1[tid];

    // forward sweep
    for (int i = 2; i < n_row; i++) {
        gid += stride;

        // pipeline copy
        d0[tid] = d1[tid];

        // load next
        d1[tid] = d[gid];

        double r = 1.0 / (b[i] - a_local[i] * c_local[i - 1]);
        d1[tid] = r * (d1[tid] - a_local[i] * d0[tid]);
        c_local[i] = r * c_local[i];
        a_local[i] = -r * a_local[i] * a_local[i - 1];

        // store
        d[gid] = d1[tid];
    }

    a_rd[1] = a_local[n_row - 1];
    c_rd[1] = c_local[n_row - 1];
    d_rd[gid0 + stride] = d1[tid];

        // upper elimination (backward)
    d1[tid] = d0[tid];

    gid -= stride; // n_row - 2

    for (int i = n_row - 3; i >= 1; --i) {
        gid -= stride;         // offset = (i, j, k)

        // Load i-th data into shared
        d0[tid] = d[gid];

        // Elimination using (i+1)th data
        d0[tid] -= c_local[i] * d1[tid];
        a_local[i] -= c_local[i] * a_local[i + 1];
        c_local[i] = -c_local[i] * c_local[i + 1];

        // Pipeline copy
        d1[tid] = d0[tid];

        // Store result to global memory
        d[gid] = d0[tid];
    }

    // top row correction
    d0[tid] = d[gid0];

    r = 1.0 / (1.0 - a_local[1] * c_local[0]);
    d0[tid] = r * (d0[tid] - c_local[0] * d1[tid]);
    a_rd[0] = r * a_local[0];
    c_rd[0] = -r * c_local[0] * c_local[1];

    d[gid0] = d0[tid];
    d_rd[gid0] = d0[tid];

    for (int i = 0; i < n_row; i++) {
        a[i] = a_local[i];
        c[i] = c_local[i];
    }
}

/**
 * @brief Reconstruct full solutions from reduced boundary results.
 *
 * Uses the exchanged boundary data d_rd to update interior solution values.
 *
 * @param[in]     a, c     Modified coefficients from factorization.
 * @param[in,out] d        Solution array to update.
 * @param[in]     d_rd     Boundary solutions.
 * @param[in]     n_row    Number of equations per system.
 * @param[in]     ny, nz   Grid dimensions.
 */
__global__ void cuReconstructManyRHS(
    const double* a, const double* c, double* d,
    const double* d_rd, int n_row, int ny, int nz) {

    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= ny || k >= nz) return;

    // Thread-local indices
    int tj = threadIdx.y;
    int tk = threadIdx.x;

    int tile_pitch = blockDim.x + 1;
    int tid = tk + tj * tile_pitch;

    // Allocate shared memory
    extern __shared__ double shared[];

    double* ds = shared;
    double* de = ds + tile_pitch * blockDim.y;

    int gid = j * nz + k;            // system index
    int stride  = ny * nz;

    double a_local[256], c_local[256]; //Solving race condition. TODO: memory problem

    for (int i = 0; i < n_row; i++) {
        a_local[i] = a[i];
        c_local[i] = c[i];
    }

    // Load reduced system solutions into shared memory
    ds[tid] = d_rd[gid];             // d_rd(:,:,1)
    de[tid] = d_rd[gid + stride];    // d_rd(:,:,2)
    __syncthreads();

    // Update solution vector
    d[gid] = ds[tid];                // d(:,:,1)
    d[gid + (n_row - 1) * stride] = de[tid];  // d(:,:,nz_row)

    for (int i = 1; i < n_row - 1; i++) {
        gid += stride;
        d[gid] -= a_local[i] * ds[tid] + c_local[i] * de[tid];
    }
}

/**
 * @brief Rebuild Y–Z slab from 1D buffer after MPI exchange.
 *
 * @param[in]  array1D 1D buffer.
 * @param[out] slab_xy Input XY slab in device memory.
 * @param[in]  n1      Number of rows in X per rank.
 * @param[in]  n2      Number of Y systems.
 * @param[in]  n3      Total Z systems.
 * @param[in]  size    Number of ranks.
 */
__global__ void mem_unite_slab_yz(const double* __restrict__ array1D,
                                double* __restrict__ slab_yz,
                                int n1, int n2, int n3, int size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int n3_div = n3 / size;

    if (i >= n1 || j >= n2 || k >= n3_div) return;

    int blksize = n1 * n2 * n3 / size;

    for (int rank = 0; rank < size; rank++) {
        int pos = i * n2 * n3_div + j * n3_div + k + rank * blksize;
        slab_yz[i * n2 * n3 + j * n3 + k + rank * n3_div] = array1D[pos];
    }
}

/**
 * @brief Detach YZ slab into linear buffer for MPI_Alltoall.
 *
 * @param[in]  slab_yz Input YZ slab in device memory.
 * @param[out] array1D 1D buffer.
 * @param[in]  n1      Number of rows in X per rank.
 * @param[in]  n2      Number of Y systems.
 * @param[in]  n3      Total Z systems.
 * @param[in]  size    Number of ranks.
 */
__global__ void mem_detach_slab_yz(const double* __restrict__ slab_yz,
                                double* __restrict__ array1D,
                                int n1, int n2, int n3, int size) {

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int n3_div = n3 / size;

    if (i >= n1 || j >= n2 || k >= n3) return;

    int blksize = n1 * n2 * n3_div;

    for (int rank = 0; rank < size; rank++) {
        int pos = i * n2 * n3_div + j * n3_div + k + rank * blksize;
        array1D[pos] = slab_yz[i * n2 * n3 + j * n3 + k + rank * n3_div];
    }
}

namespace cuPaScaL_TDMA {

    /**
    * @brief Base-level create: not supported.
    * @throws std::runtime_error Always.
    */
    void cuPTDMAPlanBase::create(int n_row, int ny_sys, int nz_sys, 
                                 MPI_Comm comm_ptdma, TDMAType type_) {
        throw std::runtime_error("create(int, int, MPI_Comm, TDMAType) not implemented");
    }

    /**
    * @brief Exchange an XY‐slab and reassemble it into a YZ‐slab.
    *
    * Each rank sends its XY‐slab contiguous buffer to all others,
    * receives a 1D buffer of boundary data, and then unites it back
    * into the 2D YZ layout.
    *
    * @tparam PlanType     Type of the TDMA plan 
    * @param[in]  p        Plan carrying dimensions and block/grid config
    * @param[in]  slab_xy Input slab in X–Y ordering (device pointer).
    * @param[out] slab_yz Output slab in Y–Z ordering (device pointer).
    */
    template<typename PlanType>
    void transposeSlabXYtoYZ(const PlanType& p,
                            const double* slab_xy,
                            double* slab_yz) {
        int blockCount = p.n_row_rd_ * p.n_sys_ / p.size_;
        size_t buf_bytes = sizeof(double) * p.n_row_rd_ * p.n_sys_;

        // Allocate device buffer
        double* buf_dev;
        cudaMalloc(&buf_dev, buf_bytes);

        // Exchange boundaries
        MPI_Alltoall(slab_xy, blockCount, MPI_DOUBLE,
                    buf_dev, blockCount, MPI_DOUBLE,
                    p.comm_ptdma_);
        cudaDeviceSynchronize();

        // Unite 1D buffer into YZ slab
        mem_unite_slab_yz<<<p.blocks_alltoall_, p.threads_>>>(
            buf_dev, slab_yz,
            p.n_row_rd_, p.ny_sys_, p.nz_sys_, p.size_);
        cudaDeviceSynchronize();

        cudaFree(buf_dev);
    }

    //-------------------------------------------------------------------------------
    // Plan Many

    /**
    * @class cuPTDMAPlanMany
    * @brief Configuration and buffers for Many solver.
    */

    /**
    * @brief Allocate device buffers and configure grid for Many.
    *
    * @param[in] n_row     Equations per system.
    * @param[in] ny_sys    Systems in Y.
    * @param[in] nz_sys    Systems in Z.
    * @param[in] comm_ptdma MPI communicator.
    * @param[in] type      Solver type.
    */    
    void cuPTDMAPlanMany::create(int n_row, int ny_sys, int nz_sys, 
                                 MPI_Comm comm_ptdma, TDMAType type) {
        
        n_row_ = n_row;
        ny_sys_ = ny_sys;
        nz_sys_ = nz_sys;
        n_sys_ = ny_sys_ * nz_sys_;

        MPI_Comm_dup(comm_ptdma, &comm_ptdma_);
        MPI_Comm_size(comm_ptdma_, &size_);
        MPI_Comm_rank(comm_ptdma_, &rank_);
        type_ = type;

        nz_sys_rd_ = nz_sys;
        n_row_rd_ = 2;
        std::vector<int> nz_sys_rt_array(size_);
    
        // Compute local and global problem dimensions
        nz_sys_rt_ = Util::para_range_n(1, nz_sys_rd_, size_, rank_);
        n_row_rt_ = n_row_rd_ * size_;

        MPI_Allgather(&nz_sys_rt_, 1, MPI_INT, nz_sys_rt_array.data(), 1, MPI_INT, comm_ptdma);

        n_sys_rd_ = nz_sys_rd_ * ny_sys_;
        n_sys_rt_ = nz_sys_rt_ * ny_sys_;
    
        cudaMalloc((void**)&a_rd_d_, sizeof(double) * n_row_rd_ * n_sys_rd_);
        cudaMalloc((void**)&b_rd_d_, sizeof(double) * n_row_rd_ * n_sys_rd_);
        cudaMalloc((void**)&c_rd_d_, sizeof(double) * n_row_rd_ * n_sys_rd_);
        cudaMalloc((void**)&d_rd_d_, sizeof(double) * n_row_rd_ * n_sys_rd_);

        cudaMalloc((void**)&a_rt_d_, sizeof(double) * n_row_rt_ * n_sys_rt_);
        cudaMalloc((void**)&b_rt_d_, sizeof(double) * n_row_rt_ * n_sys_rt_);
        cudaMalloc((void**)&c_rt_d_, sizeof(double) * n_row_rt_ * n_sys_rt_);
        cudaMalloc((void**)&d_rt_d_, sizeof(double) * n_row_rt_ * n_sys_rt_);

        threads_         = dim3(8, 8, 1);

        int ny_block = ny_sys_ / threads_.x;
        if (ny_block == 0 || (ny_sys % threads_.x != 0)) {
            std::cerr << "[Error] ny_sys should be a multiple of threads.x. "
                    << "threads.x = " << threads_.x << ", ny_sys = " << ny_sys_ << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }
 
        int nz_block = nz_sys_ / threads_.y;
        if (nz_block == 0 || (nz_sys % threads_.y != 0)) {
            std::cerr << "[Error] nz_sys should be a multiple of threads.y. "
                    << "threads.y = " << threads_.y << ", nz_sys = " << nz_sys_ << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }

        int nz_rt_block = nz_sys_rt_ / threads_.y;
        if (nz_rt_block == 0 || (nz_sys_rt_ % threads_.y != 0)) {
            std::cerr << "[Error] nz_sys_rt should be a multiple of threads.y. "
                    << "threads.y = " << threads_.y << ", nz_sys_rt = " << nz_sys_rt_ << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }

        // Set dim3 thread/block configuration
        blocks_          = dim3(nz_block,    ny_block);
        blocks_rt_       = dim3(nz_rt_block, ny_block, 1);
        blocks_alltoall_ = dim3(nz_rt_block, ny_block, n_row_rd_);

        const int thread_1d = threads_.x * threads_.y;
        const int block_1d = n_row_rd_ * n_sys_rd_ / thread_1d;
    
        cuInitValue<<<block_1d, thread_1d>>>(b_rd_d_, n_row_rd_ * n_sys_rd_, 1.0);
        cuInitValue<<<block_1d, thread_1d>>>(b_rt_d_, n_row_rd_ * n_sys_rd_, 1.0);

        return;
    }
    
    /**
    * @brief Free all device buffers for Many plan.
    */
    void cuPTDMAPlanMany::destroy() {

        if (a_rd_d_ != nullptr) {
            cudaFree(a_rd_d_);
            a_rd_d_ = nullptr;
        }
        if (b_rd_d_ != nullptr) {
            cudaFree(b_rd_d_);
            b_rd_d_ = nullptr;
        }
        if (c_rd_d_ != nullptr) {
            cudaFree(c_rd_d_);
            c_rd_d_ = nullptr;
        }
        if (d_rd_d_ != nullptr) {
            cudaFree(d_rd_d_);
            d_rd_d_ = nullptr;
        }
        if (a_rt_d_ != nullptr) {
            cudaFree(a_rt_d_);
            a_rt_d_ = nullptr;
        }
        if (b_rt_d_ != nullptr) {
            cudaFree(b_rt_d_);
            b_rt_d_ = nullptr;
        }
        if (c_rt_d_ != nullptr) {
            cudaFree(c_rt_d_);
            c_rt_d_ = nullptr;
        }
        if (d_rt_d_ != nullptr) {
            cudaFree(d_rt_d_);
            d_rt_d_ = nullptr;
        }
        return;
    }    

    /**
    * @class cuPTDMASolverMany
    * @brief Orchestrates batched solve for Many.
    */
    void cuPTDMASolverMany::cuSolve(cuPTDMAPlanMany& plan,
                        double* a, double* b, double* c, double* d) {

        const int n_row = plan.n_row_;
        const int ny_sys = plan.ny_sys_;
        const int nz_sys = plan.nz_sys_;
        assert(n_row > 2);

        if (plan.size_ == 1) {
            cuDispatchTDMASolver<BatchType::Many>(plan.type_, a, b, c, d, n_row, ny_sys, nz_sys);
            return;
        }

        int shmem_size = 9 * (plan.threads_.x + 1) * plan.threads_.y * sizeof(double);

        cuManyModifiedThomasMany<<<plan.blocks_, plan.threads_, shmem_size>>>
            (a, b, c, d, plan.a_rd_d_, plan.c_rd_d_, plan.d_rd_d_, n_row, ny_sys, nz_sys);

        transposeSlabYZtoXY(plan, plan.a_rd_d_, plan.a_rt_d_);
        transposeSlabYZtoXY(plan, plan.c_rd_d_, plan.c_rt_d_);
        transposeSlabYZtoXY(plan, plan.d_rd_d_, plan.d_rt_d_);

        cuDispatchTDMASolver<BatchType::Many>(plan.type_,
                                            plan.a_rt_d_, plan.b_rt_d_,
                                            plan.c_rt_d_, plan.d_rt_d_,
                                            plan.n_row_rt_, plan.ny_sys_, plan.nz_sys_rt_);

        transposeSlabXYtoYZ(plan, plan.d_rt_d_, plan.d_rd_d_);

        shmem_size = 2 * (plan.threads_.x + 1) * plan.threads_.y * sizeof(double);
        cuReconstructMany<<<plan.blocks_, plan.threads_, shmem_size>>>
            (a, c, d, plan.d_rd_d_, n_row, ny_sys, nz_sys);

        cudaDeviceSynchronize();
    }

    //-------------------------------------------------------------------------------
    // Plan ManyRHS
    /**
    * @class cuPTDMAPlanManyRHS
    * @brief Configuration and buffers for ManyRHS solver.
    */
    void cuPTDMAPlanManyRHS::create(int n_row, int ny_sys, int nz_sys, 
                                    MPI_Comm comm_ptdma, TDMAType type) {
        
        n_row_ = n_row;
        ny_sys_ = ny_sys;
        nz_sys_ = nz_sys;
        n_sys_ = ny_sys_ * nz_sys_;

        MPI_Comm_dup(comm_ptdma, &comm_ptdma_);
        MPI_Comm_size(comm_ptdma_, &size_);
        MPI_Comm_rank(comm_ptdma_, &rank_);
        type_ = type;

        nz_sys_rd_ = nz_sys_;
        n_row_rd_ = 2;
        std::vector<int> nz_sys_rt_array(size_);
    
        // Compute local and global problem dimensions
        nz_sys_rt_ = Util::para_range_n(1, nz_sys_rd_, size_, rank_);
        n_row_rt_ = n_row_rd_ * size_;

        MPI_Allgather(&nz_sys_rt_, 1, MPI_INT, nz_sys_rt_array.data(), 1, MPI_INT, comm_ptdma);

        n_sys_rd_ = nz_sys_rd_ * ny_sys_;
        n_sys_rt_ = nz_sys_rt_ * ny_sys_;

        cudaMalloc((void**)&a_rd_d_, sizeof(double) * n_row_rd_);
        cudaMalloc((void**)&b_rd_d_, sizeof(double) * n_row_rd_);
        cudaMalloc((void**)&c_rd_d_, sizeof(double) * n_row_rd_);
        cudaMalloc((void**)&d_rd_d_, sizeof(double) * n_row_rd_ * n_sys_rd_);

        cudaMalloc((void**)&a_rt_d_, sizeof(double) * n_row_rt_);
        cudaMalloc((void**)&b_rt_d_, sizeof(double) * n_row_rt_);
        cudaMalloc((void**)&c_rt_d_, sizeof(double) * n_row_rt_);
        cudaMalloc((void**)&d_rt_d_, sizeof(double) * n_row_rt_ * n_sys_rt_);

        threads_         = dim3(8, 8, 1);

        int ny_block = ny_sys_ / threads_.x;
        if (ny_block == 0 || (ny_sys % threads_.x != 0)) {
            std::cerr << "[Error] ny_sys should be a multiple of threads.x. "
                    << "threads.x = " << threads_.x << ", ny_sys = " << ny_sys_ << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }
 
        int nz_block = nz_sys_ / threads_.y;
        if (nz_block == 0 || (nz_sys % threads_.y != 0)) {
            std::cerr << "[Error] nz_sys should be a multiple of threads.y. "
                    << "threads.y = " << threads_.y << ", nz_sys = " << nz_sys_ << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }

        int nz_rt_block = nz_sys_rt_ / threads_.y;
        if (nz_rt_block == 0 || (nz_sys_rt_ % threads_.y != 0)) {
            std::cerr << "[Error] nz_sys_rt should be a multiple of threads.y. "
                    << "threads.y = " << threads_.y << ", nz_sys_rt = " << nz_sys_rt_ << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }

        // Set dim3 thread/block configuration
        blocks_          = dim3(nz_block,    ny_block);
        blocks_rt_       = dim3(nz_rt_block, ny_block, 1);
        blocks_alltoall_ = dim3(nz_rt_block, ny_block, n_row_rd_);

        const int thread_1d = threads_.x * threads_.y;
        const int block_1d = n_row_rd_ * n_sys_rd_ / thread_1d;
    
        cuInitValue<<<block_1d, thread_1d>>>(b_rd_d_, n_row_rd_ * n_sys_rd_, 1.0);
        cuInitValue<<<block_1d, thread_1d>>>(b_rt_d_, n_row_rd_ * n_sys_rd_, 1.0);

        return;
    }
    
    void cuPTDMAPlanManyRHS::destroy() {

        if (a_rd_d_ != nullptr) {
            cudaFree(a_rd_d_);
            a_rd_d_ = nullptr;
        }
        if (b_rd_d_ != nullptr) {
            cudaFree(b_rd_d_);
            b_rd_d_ = nullptr;
        }
        if (c_rd_d_ != nullptr) {
            cudaFree(c_rd_d_);
            c_rd_d_ = nullptr;
        }
        if (d_rd_d_ != nullptr) {
            cudaFree(d_rd_d_);
            d_rd_d_ = nullptr;
        }
        if (a_rt_d_ != nullptr) {
            cudaFree(a_rt_d_);
            a_rt_d_ = nullptr;
        }
        if (b_rt_d_ != nullptr) {
            cudaFree(b_rt_d_);
            b_rt_d_ = nullptr;
        }
        if (c_rt_d_ != nullptr) {
            cudaFree(c_rt_d_);
            c_rt_d_ = nullptr;
        }
        if (d_rt_d_ != nullptr) {
            cudaFree(d_rt_d_);
            d_rt_d_ = nullptr;
        }
        return;
    }    

    /**
    * @class cuPTDMASolverManyRHS
    * @brief Orchestrates batched solve for ManyRHS.
    */
    void cuPTDMASolverManyRHS::cuSolve(cuPTDMAPlanManyRHS& plan,
                        double* a, double* b, double* c, double* d) {

        const int n_row = plan.n_row_;
        const int ny_sys = plan.ny_sys_;
        const int nz_sys = plan.nz_sys_;
        const int n_row_rd = plan.n_row_rd_;

        assert(n_row > 2);

        if (plan.size_ == 1) {
            cuDispatchTDMASolver<BatchType::ManyRHS>(plan.type_, a, b, c, d, 
                                                     n_row, ny_sys, nz_sys);
            return;
        }

        int shmem_size = 2 * (plan.threads_.x + 1) * plan.threads_.y * sizeof(double);

        cuManyModifiedThomasManyRHS<<<plan.blocks_, plan.threads_, shmem_size>>>
            (a, b, c, d, plan.a_rd_d_, plan.c_rd_d_, plan.d_rd_d_, n_row, ny_sys, nz_sys);

        MPI_Allgather(plan.a_rd_d_, n_row_rd, MPI_DOUBLE,
                      plan.a_rt_d_, n_row_rd, MPI_DOUBLE,
                      plan.comm_ptdma_);
        MPI_Allgather(plan.c_rd_d_, n_row_rd, MPI_DOUBLE,
                      plan.c_rt_d_, n_row_rd, MPI_DOUBLE,
                      plan.comm_ptdma_);

        transposeSlabYZtoXY(plan, plan.d_rd_d_, plan.d_rt_d_);

        cuDispatchTDMASolver<BatchType::ManyRHS>(plan.type_,
                                            plan.a_rt_d_, plan.b_rt_d_,
                                            plan.c_rt_d_, plan.d_rt_d_,
                                            plan.n_row_rt_, plan.ny_sys_, plan.nz_sys_rt_);

        transposeSlabXYtoYZ(plan, plan.d_rt_d_, plan.d_rd_d_);

        shmem_size = 2 * (plan.threads_.x + 1) * plan.threads_.y * sizeof(double);
        cuReconstructManyRHS<<<plan.blocks_, plan.threads_, shmem_size>>>
            (a, c, d, plan.d_rd_d_, n_row, ny_sys, nz_sys);

        cudaDeviceSynchronize();
    }

    /**
    * @brief Detach an YZ‐slab and reassemble it into a XY‐slab.
    *
    * After local detachment of the 2D YZ‐slab into a contiguous 1D array,
    * each rank exchanges its boundary data with all others via MPI_Alltoall,
    * and the received buffer is directly laid out as an XY‐slab.
    *
    * @tparam PlanType     Type of the TDMA plan 
    * @param[in]  p        Plan carrying dimensions and block/grid config
    * @param[in]  slab_yz  Input slab in Y–Z ordering (device pointer).
    * @param[out] slab_xy  Output slab in X–Y ordering (device pointer).
    */
    template<typename PlanType>
    void transposeSlabYZtoXY(const PlanType& p,
                            const double* slab_yz,
                            double* slab_xy) {
        int blockCount = p.n_row_rd_ * p.n_sys_ / p.size_;
        size_t buf_bytes = sizeof(double) * p.n_row_rd_ * p.n_sys_;

        // Allocate device buffer
        double* buf_dev;
        cudaMalloc(&buf_dev, buf_bytes);

        // Detach YZ slab into 1D buffer
        mem_detach_slab_yz<<<p.blocks_alltoall_, p.threads_>>>(
            slab_yz, buf_dev,
            p.n_row_rd_, p.ny_sys_, p.nz_sys_, p.size_);
        cudaDeviceSynchronize();

        // Exchange boundaries
        MPI_Alltoall(buf_dev, blockCount, MPI_DOUBLE,
                    slab_xy, blockCount, MPI_DOUBLE,
                    p.comm_ptdma_);

        cudaFree(buf_dev);
    }

    //================================
    //  Serial TDMA solver interface
    //================================

    template<typename T>
    constexpr bool dependent_false = false;

    template <TDMAType tdma_type, BatchType batch_type>
    void cuBatchSolver(double* a, double* b, double* c, double* d, int n_row, int ny, int nz) {
        if constexpr (batch_type == BatchType::Many) {
            if constexpr (tdma_type == TDMAType::Standard)
                cuTDMASolver::cuMany(a, b, c, d, n_row, ny, nz);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                cuTDMASolver::cuManyCyclic(a, b, c, d, n_row, ny, nz);
            else 
                static_assert(dependent_false<std::integral_constant<TDMAType, tdma_type>>,
                              "Unsupported TDMAType for BatchType::Many");
        }
        else if constexpr (batch_type == BatchType::ManyRHS) {
            if constexpr (tdma_type == TDMAType::Standard)
                cuTDMASolver::cuManyRHS(a, b, c, d, n_row, ny, nz);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                cuTDMASolver::cuManyRHSCyclic(a, b, c, d, n_row, ny, nz);
            else 
                static_assert(dependent_false<std::integral_constant<TDMAType, tdma_type>>,
                              "Unsupported TDMAType for BatchType::Many");
        }
        else 
            static_assert(dependent_false<std::integral_constant<BatchType, batch_type>>,
                          "Unsupported BatchType in cuBatchSolver");
    }

    template <BatchType batch_type>
    void cuDispatchTDMASolver(TDMAType type, double* a, double* b, double* c, double* d, int n_row, int ny, int nz) {
        switch (type) {
            case TDMAType::Standard:
                cuBatchSolver<TDMAType::Standard, batch_type>(a, b, c, d, n_row, ny, nz);
                break;
            case TDMAType::Cyclic:
                cuBatchSolver<TDMAType::Cyclic, batch_type>(a, b, c, d, n_row, ny, nz);
                break;
            default:
                throw std::invalid_argument("Unknown TDMAType");
        }
    }

    template void cuBatchSolver<TDMAType::Standard, BatchType::Many>(
        double*, double*, double*, double*, int, int, int);

    template void cuBatchSolver<TDMAType::Cyclic, BatchType::Many>(
        double*, double*, double*, double*, int, int, int);

    template void cuBatchSolver<TDMAType::Standard, BatchType::ManyRHS>(
        double*, double*, double*, double*, int, int, int);

    template void cuBatchSolver<TDMAType::Cyclic, BatchType::ManyRHS>(
        double*, double*, double*, double*, int, int, int);

    template void transposeSlabXYtoYZ<cuPTDMAPlanMany>(
        const cuPTDMAPlanMany&, const double*, double* );

    template void transposeSlabXYtoYZ<cuPTDMAPlanManyRHS>(
        const cuPTDMAPlanManyRHS&, const double*, double* );

};