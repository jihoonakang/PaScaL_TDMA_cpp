#include <cuda_runtime.h>
#include <cassert>
#include <numeric>
#include <mpi.h>
#include "PaScaL_TDMA.cuh"
#include "TDMASolver.cuh"

// CUDA 커널
__global__ void cuInitValue(double* arr, int n, double val) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        arr[idx] = val;
    }
}

__global__ void cuManyModifiedThomasMany(double* A, double* B, double* C, double* D, double* A_rd, double* C_rd, double* D_rd, int n_row, int ny, int nz) {

    // 2D grid/thread index
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny || k >= nz) return;
    int gid = k + j * nz;      // system index
    int gid0 =gid;

    int tj = threadIdx.y;
    int tk = threadIdx.x;
    int tile_pitch = blockDim.x + 1;
    int tid = tk + tj * tile_pitch;

    // Shared memory tile: [nx+1][ny]
    extern __shared__ double shared[];

    double* a0 = shared;
    double* b0 = a0 + tile_pitch * blockDim.y;
    double* c0 = b0 + tile_pitch * blockDim.y;
    double* d0 = c0 + tile_pitch * blockDim.y;
    double* a1 = d0 + tile_pitch * blockDim.y;
    double* b1 = a1 + tile_pitch * blockDim.y;
    double* c1 = b1 + tile_pitch * blockDim.y;
    double* d1 = c1 + tile_pitch * blockDim.y;
    double* r0 = d1 + tile_pitch * blockDim.y;

    const int stride = ny * nz;

    // i = 0
    a0[tid] = A[gid];
    b0[tid] = B[gid];
    c0[tid] = C[gid];
    d0[tid] = D[gid];

    a0[tid] /= b0[tid];
    c0[tid] /= b0[tid];
    d0[tid] /= b0[tid];

    A[gid] = a0[tid];
    C[gid] = c0[tid];
    D[gid] = d0[tid];

    // i = 1
    gid += stride;
    a1[tid] = A[gid];
    b1[tid] = B[gid];
    c1[tid] = C[gid];
    d1[tid] = D[gid];

    a1[tid] /= b1[tid];
    c1[tid] /= b1[tid];
    d1[tid] /= b1[tid];

    A[gid] = a1[tid];
    C[gid] = c1[tid];
    D[gid] = d1[tid];

    // forward sweep
    for (int i = 2; i < n_row; i++) {
        gid += stride;

        // pipeline copy
        a0[tid] = a1[tid];
        c0[tid] = c1[tid];
        d0[tid] = d1[tid];

        // load next
        a1[tid] = A[gid];
        b1[tid] = B[gid];
        c1[tid] = C[gid];
        d1[tid] = D[gid];

        double r = 1.0 / (b1[tid] - a1[tid] * c0[tid]);
        d1[tid] = r * (d1[tid] - a1[tid] * d0[tid]);
        c1[tid] = r * c1[tid];
        a1[tid] = -r * a1[tid] * a0[tid];

        // store
        A[gid] = a1[tid];
        C[gid] = c1[tid];
        D[gid] = d1[tid];
    }

    A_rd[gid0 + stride] = a1[tid];
    C_rd[gid0 + stride] = c1[tid];
    D_rd[gid0 + stride] = d1[tid];

        // upper elimination (backward)
    a1[tid] = a0[tid];
    c1[tid] = c0[tid];
    d1[tid] = d0[tid];

    gid -= stride; // n_row - 2

    for (int i = n_row - 3; i >= 1; --i) {
        gid -= stride;         // offset = (i, j, k)

        // Load i-th data into shared
        a0[tid] = A[gid];
        c0[tid] = C[gid];
        d0[tid] = D[gid];

        // Elimination using (i+1)th data
        d0[tid] -= c0[tid] * d1[tid];
        a0[tid] -= c0[tid] * a1[tid];
        c0[tid] = -c0[tid] * c1[tid];

        // Pipeline copy
        a1[tid] = a0[tid];
        c1[tid] = c0[tid];
        d1[tid] = d0[tid];

        // Store result to global memory
        A[gid] = a0[tid];
        C[gid] = c0[tid];
        D[gid] = d0[tid];
    }

    // top row correction
    a0[tid] = A[gid0];
    c0[tid] = C[gid0];
    d0[tid] = D[gid0];

    r0[tid] = 1.0 / (1.0 - a1[tid] * c0[tid]);
    d0[tid] = r0[tid] * (d0[tid] - c0[tid] * d1[tid]);
    a0[tid] = r0[tid] * a0[tid];
    c0[tid] = -r0[tid] * c0[tid] * c1[tid];

    D[gid0] = d0[tid];
    A[gid0] = a0[tid];
    C[gid0] = c0[tid];

    A_rd[gid0] = a0[tid];
    C_rd[gid0] = c0[tid];
    D_rd[gid0] = d0[tid];

}

__global__ void cuReconstructMany(const double* A, const double* C, double* D,
                                  const double* D_rd,
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
    ds[tid] = D_rd[gid];             // d_rd(:,:,1)
    de[tid] = D_rd[gid + stride];    // d_rd(:,:,2)
    __syncthreads();

    // Update solution vector
    D[gid] = ds[tid];                // d(:,:,1)
    D[gid + (n_row - 1) * stride] = de[tid];  // d(:,:,nz_row)

    for (int i = 1; i < n_row - 1; i++) {
        gid += stride;
        D[gid] -= A[gid] * ds[tid] + C[gid] * de[tid];
    }
}

__global__ void cuManyModifiedThomasManyRHS(double* A, double* B, double* C, double* D, double* A_rd, double* C_rd, double* D_rd, int n_row, int ny, int nz) {

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
        a_local[i] = A[i];
        c_local[i] = C[i];
    }

    // i = 0
    b0 = B[0];
    d0[tid] = D[gid];

    a_local[0] /= b0;
    c_local[0] /= b0;
    d0[tid] /= b0;

    D[gid] = d0[tid];

    // i = 1
    gid += stride;
    b1 = B[1];
    d1[tid] = D[gid];

    a_local[1] /= b1;
    c_local[1] /= b1;
    d1[tid] /= b1;

    D[gid] = d1[tid];

    // forward sweep
    for (int i = 2; i < n_row; i++) {
        gid += stride;

        // pipeline copy
        d0[tid] = d1[tid];

        // load next
        d1[tid] = D[gid];

        double r = 1.0 / (B[i] - a_local[i] * c_local[i - 1]);
        d1[tid] = r * (d1[tid] - a_local[i] * d0[tid]);
        c_local[i] = r * c_local[i];
        a_local[i] = -r * a_local[i] * a_local[i - 1];

        // store
        D[gid] = d1[tid];
    }

    A_rd[1] = a_local[n_row - 1];
    C_rd[1] = c_local[n_row - 1];
    D_rd[gid0 + stride] = d1[tid];

        // upper elimination (backward)
    d1[tid] = d0[tid];

    gid -= stride; // n_row - 2

    for (int i = n_row - 3; i >= 1; --i) {
        gid -= stride;         // offset = (i, j, k)

        // Load i-th data into shared
        d0[tid] = D[gid];

        // Elimination using (i+1)th data
        d0[tid] -= c_local[i] * d1[tid];
        a_local[i] -= c_local[i] * a_local[i + 1];
        c_local[i] = -c_local[i] * c_local[i + 1];

        // Pipeline copy
        d1[tid] = d0[tid];

        // Store result to global memory
        D[gid] = d0[tid];
    }

    // top row correction
    d0[tid] = D[gid0];

    r = 1.0 / (1.0 - a_local[1] * c_local[0]);
    d0[tid] = r * (d0[tid] - c_local[0] * d1[tid]);
    A_rd[0] = r * a_local[0];
    C_rd[0] = -r * c_local[0] * c_local[1];

    D[gid0] = d0[tid];
    D_rd[gid0] = d0[tid];

    for (int i = 0; i < n_row; i++) {
        A[i] = a_local[i];
        C[i] = c_local[i];
    }
}

__global__ void cuReconstructManyRHS(const double* A, const double* C, double* D,
                                  const double* D_rd,
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

    double a_local[256], c_local[256]; //Solving race condition. TODO: memory problem

    for (int i = 0; i < n_row; i++) {
        a_local[i] = A[i];
        c_local[i] = C[i];
    }

    // Load reduced system solutions into shared memory
    ds[tid] = D_rd[gid];             // d_rd(:,:,1)
    de[tid] = D_rd[gid + stride];    // d_rd(:,:,2)
    __syncthreads();

    // Update solution vector
    D[gid] = ds[tid];                // d(:,:,1)
    D[gid + (n_row - 1) * stride] = de[tid];  // d(:,:,nz_row)

    for (int i = 1; i < n_row - 1; i++) {
        gid += stride;
        D[gid] -= a_local[i] * ds[tid] + c_local[i] * de[tid];
    }
}

// detach x-y slab into 1D array
__global__ void mem_detach_slab_xy(const double* __restrict__ slab_xy,
                                double* __restrict__ array1D,
                                int n1, int n2, int n3, int size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int n3_div = n3 / size;

    if (i >= n1 * size || j >= n2 || k >= n3) return;

    int blksize = n1 * n2 * n3_div;

    for (int rank = 0; rank < size; rank++) {
        int pos = i * n2 * n3_div + j * n3_div + k + rank * blksize;
        array1D[pos] = slab_xy[(i + rank * n1) * n2 * n3_div + j * n3_div + k];
    }
}

// rebuild y-z slab from 1D array
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

// detach y-z slab into 1D array
// n1 : nx_row_rd = 2
// n2 : ny_sys
// n3 : nz_sys
// block ( nx_row_rd, ny_sys, nz_sys_rt = nz_sys/size)
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

// rebuild x-y slab from 1D array
__global__ void mem_unite_slab_xy(const double* __restrict__ array1D,
                                double* __restrict__ slab_xy,
                                int n1, int n2, int n3, int size) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    int n3_div = n3 / size;

    if (i >= n1 || j >= n2 || k >= n3_div) return;

    int blksize = n1 * n2 * n3_div;

    for (int rank = 0; rank < size; rank++) {
        int pos = i * n2 * n3_div + j * n3_div + k + rank * blksize;
        slab_xy[(i + rank * n1) * n2 * n3_div + j * n3_div + k] = array1D[pos];
    }
}

namespace cuPaScaL_TDMA {

    void cuPTDMAPlanBase::create(int n_row_, int ny_sys_, int nz_sys_, MPI_Comm comm_ptdma_, TDMAType type_) {
        throw std::runtime_error("create(int, int, MPI_Comm, TDMAType) not implemented");
    }

    void cuPTDMAPlanBase::create(int n_row_, int ny_sys_, int nz_sys_, MPI_Comm comm_ptdma_) {
        throw std::runtime_error("create(int, int, MPI_Comm) not implemented");
    }

    // Many
    void cuPTDMAPlanMany::create(int n_row_, int ny_sys_, int nz_sys_, MPI_Comm comm_ptdma_, TDMAType type_) {
        
        n_row = n_row_;
        ny_sys = ny_sys_;
        nz_sys = nz_sys_;
        n_sys = ny_sys * nz_sys;

        MPI_Comm_dup(comm_ptdma_, &comm_ptdma);
        MPI_Comm_size(comm_ptdma, &size);
        MPI_Comm_rank(comm_ptdma, &rank);
        type = type_;

        nz_sys_rd = nz_sys;
        n_row_rd = 2;
        std::vector<int> nz_sys_rt_array(size);
    
        // Compute local and global problem dimensions
        nz_sys_rt = Util::para_range_n(1, nz_sys_rd, size, rank);
        n_row_rt = n_row_rd * size;

        MPI_Allgather(&nz_sys_rt, 1, MPI_INT, nz_sys_rt_array.data(), 1, MPI_INT, comm_ptdma);

        n_sys_rd = nz_sys_rd * ny_sys;
        n_sys_rt = nz_sys_rt * ny_sys;
    
        cudaMalloc((void**)&d_a_rd, sizeof(double) * n_row_rd * n_sys_rd);
        cudaMalloc((void**)&d_b_rd, sizeof(double) * n_row_rd * n_sys_rd);
        cudaMalloc((void**)&d_c_rd, sizeof(double) * n_row_rd * n_sys_rd);
        cudaMalloc((void**)&d_d_rd, sizeof(double) * n_row_rd * n_sys_rd);

        cudaMalloc((void**)&d_a_rt, sizeof(double) * n_row_rt * n_sys_rt);
        cudaMalloc((void**)&d_b_rt, sizeof(double) * n_row_rt * n_sys_rt);
        cudaMalloc((void**)&d_c_rt, sizeof(double) * n_row_rt * n_sys_rt);
        cudaMalloc((void**)&d_d_rt, sizeof(double) * n_row_rt * n_sys_rt);

        threads         = dim3(8, 8, 1);

        int ny_block = ny_sys / threads.x;
        if (ny_block == 0 || (ny_sys % threads.x != 0)) {
            std::cerr << "[Error] ny_sys should be a multiple of threads.x. "
                    << "threads.x = " << threads.x << ", ny_sys = " << ny_sys << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }
 
        int nz_block = nz_sys / threads.y;
        if (nz_block == 0 || (nz_sys % threads.y != 0)) {
            std::cerr << "[Error] nz_sys should be a multiple of threads.y. "
                    << "threads.y = " << threads.y << ", nz_sys = " << nz_sys << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }

        int nz_rt_block = nz_sys_rt / threads.y;
        if (nz_rt_block == 0 || (nz_sys_rt % threads.y != 0)) {
            std::cerr << "[Error] nz_sys_rt should be a multiple of threads.y. "
                    << "threads.y = " << threads.y << ", nz_sys_rt = " << nz_sys_rt << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }

        // Set dim3 thread/block configuration
        blocks          = dim3(nz_block,    ny_block);
        blocks_rt       = dim3(nz_rt_block, ny_block, 1);
        blocks_alltoall = dim3(nz_rt_block, ny_block, n_row_rd);

        const int thread_1d = threads.x * threads.y;
        const int block_1d = n_row_rd * n_sys_rd / thread_1d;
    
        cuInitValue<<<block_1d, thread_1d>>>(d_b_rd, n_row_rd * n_sys_rd, 1.0);
        cuInitValue<<<block_1d, thread_1d>>>(d_b_rt, n_row_rd * n_sys_rd, 1.0);

        return;
    }
    
    void cuPTDMAPlanMany::destroy() {

        if (d_a_rd != nullptr) {
            cudaFree(d_a_rd);
            d_a_rd = nullptr;
        }
        if (d_b_rd != nullptr) {
            cudaFree(d_b_rd);
            d_b_rd = nullptr;
        }
        if (d_c_rd != nullptr) {
            cudaFree(d_c_rd);
            d_c_rd = nullptr;
        }
        if (d_d_rd != nullptr) {
            cudaFree(d_d_rd);
            d_d_rd = nullptr;
        }
        if (d_a_rt != nullptr) {
            cudaFree(d_a_rt);
            d_a_rt = nullptr;
        }
        if (d_b_rt != nullptr) {
            cudaFree(d_b_rt);
            d_b_rt = nullptr;
        }
        if (d_c_rt != nullptr) {
            cudaFree(d_c_rt);
            d_c_rt = nullptr;
        }
        if (d_d_rt != nullptr) {
            cudaFree(d_d_rt);
            d_d_rt = nullptr;
        }
        return;
    }    

    // cuSolve 함수
    void cuPTDMASolverMany::cuSolve(cuPTDMAPlanMany& plan,
                        double* A, double* B, double* C, double* D) {

        const int n_row = plan.n_row;
        const int ny_sys = plan.ny_sys;
        const int nz_sys = plan.nz_sys;
        assert(n_row > 2);

        if (plan.size == 1) {
            cuDispatchTDMASolver<BatchType::Many>(plan.type, A, B, C, D, n_row, ny_sys, nz_sys);
            return;
        }

        int shmem_size = 9 * (plan.threads.x + 1) * plan.threads.y * sizeof(double);

        cuManyModifiedThomasMany<<<plan.blocks, plan.threads, shmem_size>>>
            (A, B, C, D, plan.d_a_rd, plan.d_c_rd, plan.d_d_rd, n_row, ny_sys, nz_sys);

        transposeSlabYZtoXY(plan, plan.d_a_rd, plan.d_a_rt);
        transposeSlabYZtoXY(plan, plan.d_c_rd, plan.d_c_rt);
        transposeSlabYZtoXY(plan, plan.d_d_rd, plan.d_d_rt);

        cuDispatchTDMASolver<BatchType::Many>(plan.type,
                                            plan.d_a_rt, plan.d_b_rt,
                                            plan.d_c_rt, plan.d_d_rt,
                                            plan.n_row_rt, plan.ny_sys, plan.nz_sys_rt);

        transposeSlabXYtoYZ(plan, plan.d_d_rt, plan.d_d_rd);

        shmem_size = 2 * (plan.threads.x + 1) * plan.threads.y * sizeof(double);
        cuReconstructMany<<<plan.blocks, plan.threads, shmem_size>>>
            (A, C, D, plan.d_d_rd, n_row, ny_sys, nz_sys);

        cudaDeviceSynchronize();
    }

    void cuPTDMASolverMany::transposeSlabXYtoYZ(const cuPTDMAPlanMany& p,
                                                    const double* slab_xy,
                                                    double* slab_yz) {

        int blksize = p.n_row_rd * p.n_sys / p.size;
        size_t buffer_size = sizeof(double) * p.n_row_rd * p.n_sys;
        double* recvbuf_dev;

        // GPU memory allocation
        cudaMalloc((void**)&recvbuf_dev, buffer_size);

#ifdef USE_CUDA_AWARE_MPI
        int ierr = MPI_Alltoall(slab_xy, blksize, MPI_DOUBLE,
                                recvbuf_dev, blksize, MPI_DOUBLE,
                                p.comm_ptdma);
        assert(ierr == MPI_SUCCESS);
#else
        double* sendbuf_host;
        double* recvbuf_host;

        // Host (pinned) memory allocation
        cudaMallocHost((void**)&sendbuf_host, buffer_size);
        cudaMallocHost((void**)&recvbuf_host, buffer_size);

        // Device -> Host copy. Detach x-y slab to device buffer is not required.
        cudaMemcpy(sendbuf_host, slab_xy, buffer_size, cudaMemcpyDeviceToHost);

        // MPI Alltoall on host
        int ierr = MPI_Alltoall(sendbuf_host, blksize, MPI_DOUBLE,
                                recvbuf_host, blksize, MPI_DOUBLE,
                                p.comm_ptdma);
        assert(ierr == MPI_SUCCESS);

        // Host -> Device copy
        cudaMemcpy(recvbuf_dev, recvbuf_host, buffer_size, cudaMemcpyHostToDevice);

        cudaFreeHost(sendbuf_host);
        cudaFreeHost(recvbuf_host);
#endif
        // Reconstruct y-z slab from device buffer
        mem_unite_slab_yz<<<p.blocks_alltoall, p.threads>>>(
            recvbuf_dev, slab_yz, p.n_row_rd, p.ny_sys, p.nz_sys, p.size);
        cudaDeviceSynchronize();

        cudaFree(recvbuf_dev);
    }

    // Transpose slab from y-z to x-z direction
    void cuPTDMASolverMany::transposeSlabYZtoXY(const cuPTDMAPlanMany& p,
                                const double* slab_yz,
                                double* slab_xy) {

        int blksize = p.n_row_rd * p.n_sys / p.size;
        size_t buffer_size = sizeof(double) * p.n_row_rd * p.n_sys;

        double* sendbuf_dev;

        // Allocate GPU memory
        cudaMalloc((void**)&sendbuf_dev, buffer_size);

        // Detach y-z slab to 1D buffer (device memory)
        mem_detach_slab_yz<<<p.blocks_alltoall, p.threads>>>(
            slab_yz, sendbuf_dev, p.n_row_rd, p.ny_sys, p.nz_sys, p.size);
        cudaDeviceSynchronize();

#ifdef USE_CUDA_AWARE_MPI
        int ierr = MPI_Alltoall(sendbuf_dev, blksize, MPI_DOUBLE,
                                slab_xy, blksize, MPI_DOUBLE,
                                p.comm_ptdma);
        assert(ierr == MPI_SUCCESS);
#else
        double* sendbuf_host;
        double* recvbuf_host;

        // Allocate host memory (pinned memory for performance)
        cudaMallocHost((void**)&sendbuf_host, buffer_size);
        cudaMallocHost((void**)&recvbuf_host, buffer_size);

        // Copy device -> host
        cudaMemcpy(sendbuf_host, sendbuf_dev, buffer_size, cudaMemcpyDeviceToHost);

        // MPI Alltoall on host
        int ierr = MPI_Alltoall(sendbuf_host, blksize, MPI_DOUBLE,
                                recvbuf_host, blksize, MPI_DOUBLE,
                                p.comm_ptdma);
        assert(ierr == MPI_SUCCESS);
        // Copy host -> device. Reconstruct x-y slab from device memory is not required
        cudaMemcpy(slab_xy, recvbuf_host, buffer_size, cudaMemcpyHostToDevice);

        cudaFreeHost(sendbuf_host);
        cudaFreeHost(recvbuf_host);
#endif
        cudaFree(sendbuf_dev);
    }

    // ManyRHS
    void cuPTDMAPlanManyRHS::create(int n_row_, int ny_sys_, int nz_sys_, MPI_Comm comm_ptdma_, TDMAType type_) {
        
        n_row = n_row_;
        ny_sys = ny_sys_;
        nz_sys = nz_sys_;
        n_sys = ny_sys * nz_sys;

        MPI_Comm_dup(comm_ptdma_, &comm_ptdma);
        MPI_Comm_size(comm_ptdma, &size);
        MPI_Comm_rank(comm_ptdma, &rank);
        type = type_;

        nz_sys_rd = nz_sys;
        n_row_rd = 2;
        std::vector<int> nz_sys_rt_array(size);
    
        // Compute local and global problem dimensions
        nz_sys_rt = Util::para_range_n(1, nz_sys_rd, size, rank);
        n_row_rt = n_row_rd * size;

        MPI_Allgather(&nz_sys_rt, 1, MPI_INT, nz_sys_rt_array.data(), 1, MPI_INT, comm_ptdma);

        n_sys_rd = nz_sys_rd * ny_sys;
        n_sys_rt = nz_sys_rt * ny_sys;

        cudaMalloc((void**)&d_a_rd, sizeof(double) * n_row_rd);
        cudaMalloc((void**)&d_b_rd, sizeof(double) * n_row_rd);
        cudaMalloc((void**)&d_c_rd, sizeof(double) * n_row_rd);
        cudaMalloc((void**)&d_d_rd, sizeof(double) * n_row_rd * n_sys_rd);

        cudaMalloc((void**)&d_a_rt, sizeof(double) * n_row_rt);
        cudaMalloc((void**)&d_b_rt, sizeof(double) * n_row_rt);
        cudaMalloc((void**)&d_c_rt, sizeof(double) * n_row_rt);
        cudaMalloc((void**)&d_d_rt, sizeof(double) * n_row_rt * n_sys_rt);

        threads         = dim3(8, 8, 1);

        int ny_block = ny_sys / threads.x;
        if (ny_block == 0 || (ny_sys % threads.x != 0)) {
            std::cerr << "[Error] ny_sys should be a multiple of threads.x. "
                    << "threads.x = " << threads.x << ", ny_sys = " << ny_sys << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }
 
        int nz_block = nz_sys / threads.y;
        if (nz_block == 0 || (nz_sys % threads.y != 0)) {
            std::cerr << "[Error] nz_sys should be a multiple of threads.y. "
                    << "threads.y = " << threads.y << ", nz_sys = " << nz_sys << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }

        int nz_rt_block = nz_sys_rt / threads.y;
        if (nz_rt_block == 0 || (nz_sys_rt % threads.y != 0)) {
            std::cerr << "[Error] nz_sys_rt should be a multiple of threads.y. "
                    << "threads.y = " << threads.y << ", nz_sys_rt = " << nz_sys_rt << std::endl;
            MPI_Finalize();
            std::exit(EXIT_FAILURE);
        }

        // Set dim3 thread/block configuration
        blocks          = dim3(nz_block,    ny_block);
        blocks_rt       = dim3(nz_rt_block, ny_block, 1);
        blocks_alltoall = dim3(nz_rt_block, ny_block, n_row_rd);

        const int thread_1d = threads.x * threads.y;
        const int block_1d = n_row_rd * n_sys_rd / thread_1d;
    
        cuInitValue<<<block_1d, thread_1d>>>(d_b_rd, n_row_rd * n_sys_rd, 1.0);
        cuInitValue<<<block_1d, thread_1d>>>(d_b_rt, n_row_rd * n_sys_rd, 1.0);

        return;
    }
    
    void cuPTDMAPlanManyRHS::destroy() {
    
        if (d_a_rd != nullptr) {
            cudaFree(d_a_rd);
            d_a_rd = nullptr;
        }
        if (d_b_rd != nullptr) {
            cudaFree(d_b_rd);
            d_b_rd = nullptr;
        }
        if (d_c_rd != nullptr) {
            cudaFree(d_c_rd);
            d_c_rd = nullptr;
        }
        if (d_d_rd != nullptr) {
            cudaFree(d_d_rd);
            d_d_rd = nullptr;
        }
        if (d_a_rt != nullptr) {
            cudaFree(d_a_rt);
            d_a_rt = nullptr;
        }
        if (d_b_rt != nullptr) {
            cudaFree(d_b_rt);
            d_b_rt = nullptr;
        }
        if (d_c_rt != nullptr) {
            cudaFree(d_c_rt);
            d_c_rt = nullptr;
        }
        if (d_d_rt != nullptr) {
            cudaFree(d_d_rt);
            d_d_rt = nullptr;
        }
        return;
    }    

    // cuSolve 함수
    void cuPTDMASolverManyRHS::cuSolve(cuPTDMAPlanManyRHS& plan,
                        double* A, double* B, double* C, double* D) {

        const int n_row = plan.n_row;
        const int ny_sys = plan.ny_sys;
        const int nz_sys = plan.nz_sys;
        const int n_row_rd = plan.n_row_rd;

        assert(n_row > 2);

        if (plan.size == 1) {
            cuDispatchTDMASolver<BatchType::ManyRHS>(plan.type, A, B, C, D, n_row, ny_sys, nz_sys);
            return;
        }

        int shmem_size = 2 * (plan.threads.x + 1) * plan.threads.y * sizeof(double);

        cuManyModifiedThomasManyRHS<<<plan.blocks, plan.threads, shmem_size>>>
            (A, B, C, D, plan.d_a_rd, plan.d_c_rd, plan.d_d_rd, n_row, ny_sys, nz_sys);

        MPI_Allgather(plan.d_a_rd, n_row_rd, MPI_DOUBLE,
                      plan.d_a_rt, n_row_rd, MPI_DOUBLE,
                      plan.comm_ptdma);
        MPI_Allgather(plan.d_c_rd, n_row_rd, MPI_DOUBLE,
                      plan.d_c_rt, n_row_rd, MPI_DOUBLE,
                      plan.comm_ptdma);

        transposeSlabYZtoXY(plan, plan.d_d_rd, plan.d_d_rt);

        cuDispatchTDMASolver<BatchType::ManyRHS>(plan.type,
                                            plan.d_a_rt, plan.d_b_rt,
                                            plan.d_c_rt, plan.d_d_rt,
                                            plan.n_row_rt, plan.ny_sys, plan.nz_sys_rt);

        transposeSlabXYtoYZ(plan, plan.d_d_rt, plan.d_d_rd);

        shmem_size = 2 * (plan.threads.x + 1) * plan.threads.y * sizeof(double);
        cuReconstructManyRHS<<<plan.blocks, plan.threads, shmem_size>>>
            (A, C, D, plan.d_d_rd, n_row, ny_sys, nz_sys);

        cudaDeviceSynchronize();
    }

    void cuPTDMASolverManyRHS::transposeSlabXYtoYZ(const cuPTDMAPlanManyRHS& p,
                                                    const double* slab_xy,
                                                    double* slab_yz) {

        int blksize = p.n_row_rd * p.n_sys / p.size;
        size_t buffer_size = sizeof(double) * p.n_row_rd * p.n_sys;
        double* recvbuf_dev;

        // GPU memory allocation
        cudaMalloc((void**)&recvbuf_dev, buffer_size);

#ifdef USE_CUDA_AWARE_MPI
        int ierr = MPI_Alltoall(slab_xy, blksize, MPI_DOUBLE,
                                recvbuf_dev, blksize, MPI_DOUBLE,
                                p.comm_ptdma);
        assert(ierr == MPI_SUCCESS);
#else
        double* sendbuf_host;
        double* recvbuf_host;

        // Host (pinned) memory allocation
        cudaMallocHost((void**)&sendbuf_host, buffer_size);
        cudaMallocHost((void**)&recvbuf_host, buffer_size);

        // Device -> Host copy. Detach x-y slab to device buffer is not required.
        cudaMemcpy(sendbuf_host, slab_xy, buffer_size, cudaMemcpyDeviceToHost);

        // MPI Alltoall on host
        int ierr = MPI_Alltoall(sendbuf_host, blksize, MPI_DOUBLE,
                                recvbuf_host, blksize, MPI_DOUBLE,
                                p.comm_ptdma);
        assert(ierr == MPI_SUCCESS);

        // Host -> Device copy
        cudaMemcpy(recvbuf_dev, recvbuf_host, buffer_size, cudaMemcpyHostToDevice);

        cudaFreeHost(sendbuf_host);
        cudaFreeHost(recvbuf_host);
#endif
        // Reconstruct y-z slab from device buffer
        mem_unite_slab_yz<<<p.blocks_alltoall, p.threads>>>(
            recvbuf_dev, slab_yz, p.n_row_rd, p.ny_sys, p.nz_sys, p.size);
        cudaDeviceSynchronize();

        cudaFree(recvbuf_dev);
    }

    // Transpose slab from y-z to x-z direction
    void cuPTDMASolverManyRHS::transposeSlabYZtoXY(const cuPTDMAPlanManyRHS& p,
                                const double* slab_yz,
                                double* slab_xy) {

        int blksize = p.n_row_rd * p.n_sys / p.size;
        size_t buffer_size = sizeof(double) * p.n_row_rd * p.n_sys;

        double* sendbuf_dev;

        // Allocate GPU memory
        cudaMalloc((void**)&sendbuf_dev, buffer_size);

        // Detach y-z slab to 1D buffer (device memory)
        mem_detach_slab_yz<<<p.blocks_alltoall, p.threads>>>(
            slab_yz, sendbuf_dev, p.n_row_rd, p.ny_sys, p.nz_sys, p.size);
        cudaDeviceSynchronize();

#ifdef USE_CUDA_AWARE_MPI
        int ierr = MPI_Alltoall(sendbuf_dev, blksize, MPI_DOUBLE,
                                slab_xy, blksize, MPI_DOUBLE,
                                p.comm_ptdma);
        assert(ierr == MPI_SUCCESS);
#else
        double* sendbuf_host;
        double* recvbuf_host;

        // Allocate host memory (pinned memory for performance)
        cudaMallocHost((void**)&sendbuf_host, buffer_size);
        cudaMallocHost((void**)&recvbuf_host, buffer_size);

        // Copy device → host
        cudaMemcpy(sendbuf_host, sendbuf_dev, buffer_size, cudaMemcpyDeviceToHost);

        // MPI Alltoall on host
        int ierr = MPI_Alltoall(sendbuf_host, blksize, MPI_DOUBLE,
                                recvbuf_host, blksize, MPI_DOUBLE,
                                p.comm_ptdma);
        assert(ierr == MPI_SUCCESS);

        // Copy host -> device. Reconstruct x-y slab from device memory is not required
        cudaMemcpy(slab_xy, recvbuf_host, buffer_size, cudaMemcpyHostToDevice);

        cudaFreeHost(sendbuf_host);
        cudaFreeHost(recvbuf_host);
#endif
        cudaFree(sendbuf_dev);
    }

    //================================
    //  Serial TDMA solver interface
    //================================

    template<typename T>
    constexpr bool dependent_false = false;

    template <TDMAType tdma_type, BatchType batch_type>
    void cuBatchSolver(double* A, double* B, double* C, double* D, int n_row, int ny, int nz) {
        if constexpr (batch_type == BatchType::Many) {
            if constexpr (tdma_type == TDMAType::Standard)
                cuTDMASolver::cuMany(A, B, C, D, n_row, ny, nz);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                cuTDMASolver::cuManyCyclic(A, B, C, D, n_row, ny, nz);
            else 
                static_assert(dependent_false<std::integral_constant<TDMAType, tdma_type>>,
                              "Unsupported TDMAType for BatchType::Many");
        }
        else if constexpr (batch_type == BatchType::ManyRHS) {
            if constexpr (tdma_type == TDMAType::Standard)
                cuTDMASolver::cuManyRHS(A, B, C, D, n_row, ny, nz);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                cuTDMASolver::cuManyRHSCyclic(A, B, C, D, n_row, ny, nz);
            else 
                static_assert(dependent_false<std::integral_constant<TDMAType, tdma_type>>,
                              "Unsupported TDMAType for BatchType::Many");
        }
        else 
            static_assert(dependent_false<std::integral_constant<BatchType, batch_type>>,
                          "Unsupported BatchType in cuBatchSolver");
    }

    template <BatchType batch_type>
    void cuDispatchTDMASolver(TDMAType type, double* A, double* B, double* C, double* D, int n_row, int ny, int nz) {
        switch (type) {
            case TDMAType::Standard:
                cuBatchSolver<TDMAType::Standard, batch_type>(A, B, C, D, n_row, ny, nz);
                break;
            case TDMAType::Cyclic:
                cuBatchSolver<TDMAType::Cyclic, batch_type>(A, B, C, D, n_row, ny, nz);
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

};