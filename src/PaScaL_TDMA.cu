#include <cuda_runtime.h>
#include <cassert>
#include <numeric>
#include <mpi.h>
#include "PaScaL_TDMA.cuh"
#include "TDMASolver.cuh"

// CUDA 커널
__global__ void cuManyModifiedThomasKernel(double* A, double* B, double* C, double* D, double* A_rd, double* B_rd, double* C_rd, double* D_rd, int n_row, int ny, int nz) {

    // int j = blockIdx.x * blockDim.x + threadIdx.x;
    // if (j >= n_sys) return;

    // A[j] /= B[j]; D[j] /= B[j]; C[j] /= B[j];
    // A[j + n_sys] /= B[j + n_sys]; D[j + n_sys] /= B[j + n_sys]; C[j + n_sys] /= B[j + n_sys];

    // for (int i = 2; i < n_row; ++i) {
    //     int idx = i * n_sys + j;
    //     int idx_prev = idx - n_sys;
    //     double r = 1.0 / (B[idx] - A[idx] * C[idx_prev]);
    //     D[idx] = r * (D[idx] - A[idx] * D[idx_prev]);
    //     C[idx] = r * C[idx];
    //     A[idx] = -r * A[idx] * A[idx_prev];
    // }

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
    for (int i = 2; i < n_row; ++i) {
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
    B_rd[gid0 + stride] = 1.0;
    C_rd[gid0 + stride] = c1[tid];
    D_rd[gid0 + stride] = d1[tid];

        // upper elimination (backward)
    a1[tid] = a0[tid];
    c1[tid] = c0[tid];
    d1[tid] = d0[tid];

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
    B_rd[gid0] = 1.0;
    C_rd[gid0] = c0[tid];
    D_rd[gid0] = d0[tid];

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

        const int n_sys_rd = n_sys;
        const int n_row_rd = 2;
        std::vector<int> n_sys_rt_array(size);
    
        // Compute local and global problem dimensions
        n_sys_rt = Util::para_range_n(1, n_sys_rd, size, rank);
        n_row_rt = n_row_rd * size;

        MPI_Allgather(&n_sys_rt, 1, MPI_INT, n_sys_rt_array.data(), 1, MPI_INT, comm_ptdma);
    
        count_send.assign(size, 1);
        displ_send.assign(size, 0);
        count_recv.assign(size, 1);
        displ_recv.assign(size, 0);

        cudaMalloc((void**)&d_a_rd, sizeof(double) * n_row_rd * n_sys_rd);
        cudaMalloc((void**)&d_b_rd, sizeof(double) * n_row_rd * n_sys_rd);
        cudaMalloc((void**)&d_c_rd, sizeof(double) * n_row_rd * n_sys_rd);
        cudaMalloc((void**)&d_d_rd, sizeof(double) * n_row_rd * n_sys_rd);

        cudaMalloc((void**)&d_a_rt, sizeof(double) * n_row_rt * n_sys_rt);
        cudaMalloc((void**)&d_b_rt, sizeof(double) * n_row_rt * n_sys_rt);
        cudaMalloc((void**)&d_c_rt, sizeof(double) * n_row_rt * n_sys_rt);
        cudaMalloc((void**)&d_d_rt, sizeof(double) * n_row_rt * n_sys_rt);

        return;
    }
    
    void cuPTDMAPlanMany::destroy() {

        count_send.clear();
        displ_send.clear();
        count_recv.clear();
        displ_recv.clear();
    
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
        const int n_sys = plan.n_sys;
        const int ny_sys = plan.ny_sys;
        const int nz_sys = plan.nz_sys;
        assert(n_row > 2);

        if (plan.size == 1) {
            cuDispatchTDMASolver<BatchType::Many>(plan.type, A, B, C, D, n_row, ny_sys, nz_sys);
            std::cout << n_row << ' ' << nz_sys << ' ' << ny_sys << ' ' << n_sys << std::endl;
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

        cuManyModifiedThomasKernel<<<blocks, threads>>>(d_A, d_B, d_C, d_D, plan.d_a_rd, plan.d_b_rd, plan.d_c_rd, plan.d_d_rd, n_row, ny_sys, nz_sys);
        // cuBackwardMany<<<blocks, threads>>>(d_A, d_C, d_D, n_row, n_sys);
        // cuReduceMany<<<blocks, threads>>>(d_A, d_C, d_D,
        //                                 plan.d_a_rd, plan.d_a_rd + n_sys,
        //                                 plan.d_c_rd, plan.d_c_rd + n_sys,
        //                                 plan.d_d_rd, plan.d_d_rd + n_sys,
        //                                 n_row, n_sys);

        // MPI reduction transpose
        MPI_Request request[3];
        MPI_Status statuses[3];

        // MPI_Ialltoallw(plan.d_a_rd, plan.count_send.data(), plan.displ_send.data(),
        //             plan.ddtype_FS.data(),
        //             plan.d_a_rt, plan.count_recv.data(), plan.displ_recv.data(),
        //             plan.ddtype_BS.data(),
        //             plan.comm_ptdma, &request[0]);

        // MPI_Ialltoallw(plan.d_c_rd, plan.count_send.data(), plan.displ_send.data(),
        //             plan.ddtype_FS.data(),
        //             plan.d_c_rt, plan.count_recv.data(), plan.displ_recv.data(),
        //             plan.ddtype_BS.data(),
        //             plan.comm_ptdma, &request[1]);

        // MPI_Ialltoallw(plan.d_d_rd, plan.count_send.data(), plan.displ_send.data(),
        //             plan.ddtype_FS.data(),
        //             plan.d_d_rt, plan.count_recv.data(), plan.displ_recv.data(),
        //             plan.ddtype_BS.data(),
        //             plan.comm_ptdma, &request[2]);

        MPI_Waitall(3, request, statuses);

        cuDispatchTDMASolver<BatchType::Many>(plan.type,
                                            plan.d_a_rt, plan.d_b_rt,
                                            plan.d_c_rt, plan.d_d_rt,
                                            plan.n_row_rt, plan.n_sys_rt, plan.n_sys_rt);

        // MPI_Ialltoallw(plan.d_d_rt, plan.count_recv.data(), plan.displ_recv.data(),
        //             plan.ddtype_BS.data(),
        //             plan.d_d_rd, plan.count_send.data(), plan.displ_send.data(),
        //             plan.ddtype_FS.data(),
        //             plan.comm_ptdma, &request[0]);
        MPI_Waitall(1, request, statuses);

        cuReconstructMany<<<blocks, threads>>>(d_A, d_C, d_D,
                                            plan.d_d_rd, plan.d_d_rd + n_sys,
                                            n_row, n_sys);

        cudaMemcpy(D, d_D, sizeof(double) * n_row * n_sys, cudaMemcpyDeviceToHost);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    }

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


};