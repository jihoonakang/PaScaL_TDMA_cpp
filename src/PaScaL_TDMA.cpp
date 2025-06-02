#include "PaScaL_TDMA.hpp"
#include "TDMASolver.hpp"
#include <vector>
#include <mpi.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <cassert>
#ifdef __CUDACC__  // CUDA 컴파일러일 때만 포함
#include <cuda_runtime.h>
#endif

namespace PaScaL_TDMA {

    void PTDMAPlanBase::create(int n_row_, MPI_Comm comm_ptdma_, int root_rank_, TDMAType type_) {
        throw std::runtime_error("create(int, MPI_Comm, int, TDMAType) not implemented");
    }

    void PTDMAPlanBase::create(int n_row_, int n_sys_, MPI_Comm comm_ptdma_, TDMAType type_) {
        throw std::runtime_error("create(int, int, MPI_Comm, TDMAType) not implemented");
    }

    void PTDMAPlanBase::create(int n_row_, int n_sys_, MPI_Comm comm_ptdma_) {
        throw std::runtime_error("create(int, int, MPI_Comm) not implemented");
    }

    void PTDMAPlanSingle::create(int n_row_, MPI_Comm comm_ptdma_, int root_rank_, TDMAType type_) {

        n_row = n_row_;
        MPI_Comm_dup(comm_ptdma_, &comm_ptdma);
        MPI_Comm_size(comm_ptdma, &size);
        MPI_Comm_rank(comm_ptdma, &rank);
        root_rank = root_rank_;
        type = type_;

        int n_row_rd = 2;
        n_row_rt = n_row_rd * size;

        A_rd.resize(n_row_rd);
        B_rd.assign(n_row_rd, 1.0);
        C_rd.resize(n_row_rd);
        D_rd.resize(n_row_rd);

        A_rt.resize(n_row_rt);
        B_rt.assign(n_row_rt, 1.0);
        C_rt.resize(n_row_rt);
        D_rt.resize(n_row_rt);
    }

    void PTDMAPlanSingle::destroy() {

        A_rd.clear();
        B_rd.clear();
        C_rd.clear();
        D_rd.clear();

        A_rt.clear();
        B_rt.clear();
        C_rt.clear();
        D_rt.clear();
    }

    static void forwardSingle(double* A, const double* B, double* C, double* D, 
                             const int n_row) {

        double r;

        A[0] /= B[0]; D[0] /= B[0]; C[0] /= B[0];
        A[1] /= B[1]; D[1] /= B[1]; C[1] /= B[1];

        for (int i = 2; i < n_row; i++) {
            r = 1.0 / (B[i] - A[i] * C[i - 1]);
            D[i] = r * (D[i] - A[i] * D[i - 1]);
            C[i] = r * C[i];
            A[i] = -r * A[i] * A[i - 1];
        }
    }

    static void backwardSingle(double* A, double* C, double* D, 
                               const int n_row) {

        for (int i = n_row - 3; i >= 1; i--) {
            D[i] -= C[i] * D[i + 1];
            A[i] -= C[i] * A[i + 1];
            C[i] = -C[i] * C[i + 1];
        }

        double r = 1.0 / (1.0 - A[1] * C[0]);
        D[0] = r * (D[0] - C[0] * D[1]);
        A[0] = r * A[0];
        C[0] = -r * C[0] * C[1];
    }

    void PTDMASolverSingle::solve(PTDMAPlanSingle& plan, 
                                  double* A, 
                                  double* B, 
                                  double* C,
                                  double* D) {

        const int n_row = plan.n_row;
        
        if (n_row <= 2) {
            std::cerr << "Error: n_row must be greater than 2 in rank : " << plan.rank << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (plan.size == 1) {
            dispatchTDMASolver<BatchType::Single>(plan.type, A, B, C, D, n_row, 1);
            return;
        }

        forwardSingle(A, B, C, D, n_row);
        backwardSingle(A, C, D, n_row);

        plan.A_rd[0] = A[0]; plan.A_rd[1] = A[n_row - 1];
        plan.C_rd[0] = C[0]; plan.C_rd[1] = C[n_row - 1];
        plan.D_rd[0] = D[0]; plan.D_rd[1] = D[n_row - 1];

        MPI_Gather(plan.A_rd.data(), 2, MPI_DOUBLE, plan.A_rt.data(), 2, MPI_DOUBLE, plan.root_rank, plan.comm_ptdma);
        MPI_Gather(plan.C_rd.data(), 2, MPI_DOUBLE, plan.C_rt.data(), 2, MPI_DOUBLE, plan.root_rank, plan.comm_ptdma);
        MPI_Gather(plan.D_rd.data(), 2, MPI_DOUBLE, plan.D_rt.data(), 2, MPI_DOUBLE, plan.root_rank, plan.comm_ptdma);

        if (plan.rank == plan.root_rank) {
            dispatchTDMASolver<BatchType::Single>(plan.type, plan.A_rt.data(), plan.B_rt.data(), plan.C_rt.data(), plan.D_rt.data(), plan.n_row_rt, 1);
        }

        MPI_Scatter(plan.D_rt.data(), 2, MPI_DOUBLE, plan.D_rd.data(), 2, MPI_DOUBLE, plan.root_rank, plan.comm_ptdma);

        D[0] = plan.D_rd[0];
        D[n_row - 1] = plan.D_rd[1];

        for (int i = 1; i < n_row - 1; i++) {
            D[i] = D[i] - A[i] * D[0] - C[i] * D[n_row - 1];
        }
        return;
    }

    // Many
    void PTDMAPlanMany::create(int n_row_, int n_sys_, MPI_Comm comm_ptdma_, TDMAType type_) {
        
        n_row = n_row_;
        n_sys = n_sys_;
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
        
        A_rd.resize(n_row_rd, n_sys_rd);
        B_rd.assign(n_row_rd, n_sys_rd, 1.0);
        C_rd.resize(n_row_rd, n_sys_rd);
        D_rd.resize(n_row_rd, n_sys_rd);
    
        A_rt.resize(n_row_rt, n_sys_rt);
        B_rt.assign(n_row_rt, n_sys_rt, 1.0);
        C_rt.resize(n_row_rt, n_sys_rt);
        D_rt.resize(n_row_rt, n_sys_rt);
    
        ddtype_FS.resize(size);
        ddtype_BS.resize(size);
    
        for (int i = 0; i < size; ++i) {
            int bigsize[2] = {n_row_rd, n_sys_rd};
            int subsize[2] = {n_row_rd, n_sys_rt_array[i]};
            int start[2] = {0, std::accumulate(n_sys_rt_array.begin(), n_sys_rt_array.begin() + i, 0)};
    
            MPI_Type_create_subarray(2, bigsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &ddtype_FS[i]);
            MPI_Type_commit(&ddtype_FS[i]);
    
            int rstart[2] = {n_row_rd * i, 0};
            int rsub[2] = {n_row_rd, n_sys_rt};
            int rbig[2] = {n_row_rt, n_sys_rt};
    
            MPI_Type_create_subarray(2, rbig, rsub, rstart, MPI_ORDER_C, MPI_DOUBLE, &ddtype_BS[i]);
            MPI_Type_commit(&ddtype_BS[i]);
        }
    
        count_send.assign(size, 1);
        displ_send.assign(size, 0);
        count_recv.assign(size, 1);
        displ_recv.assign(size, 0);

#ifdef CUDA
        cudaMalloc((void**)&d_A_rd, sizeof(double) * n_row_rd * n_sys_rd);
        cudaMalloc((void**)&d_B_rd, sizeof(double) * n_row_rd * n_sys_rd);
        cudaMalloc((void**)&d_C_rd, sizeof(double) * n_row_rd * n_sys_rd);
        cudaMalloc((void**)&d_D_rd, sizeof(double) * n_row_rd * n_sys_rd);

        cudaMalloc((void**)&d_A_rt, sizeof(double) * n_row_rt * n_sys_rt);
        cudaMalloc((void**)&d_B_rt, sizeof(double) * n_row_rt * n_sys_rt);
        cudaMalloc((void**)&d_C_rt, sizeof(double) * n_row_rt * n_sys_rt);
        cudaMalloc((void**)&d_D_rt, sizeof(double) * n_row_rt * n_sys_rt);
#endif
        return;
    }
    
    void PTDMAPlanMany::destroy() {
        for (int i = 0; i < size; ++i) {
            MPI_Type_free(&ddtype_FS[i]);
            MPI_Type_free(&ddtype_BS[i]);
        }
    
        ddtype_FS.clear();
        ddtype_BS.clear();
        count_send.clear();
        displ_send.clear();
        count_recv.clear();
        displ_recv.clear();
    
        A_rd.clear(); B_rd.clear(); C_rd.clear(); D_rd.clear();
        A_rt.clear(); B_rt.clear(); C_rt.clear(); D_rt.clear();

#ifdef CUDA
    if (d_A_rd != nullptr) {
        cudaFree(d_A_rd);
        d_A_rd = nullptr;
    }
    if (d_B_rd != nullptr) {
        cudaFree(d_B_rd);
        d_B_rd = nullptr;
    }
    if (d_C_rd != nullptr) {
        cudaFree(d_C_rd);
        d_C_rd = nullptr;
    }
    if (d_D_rd != nullptr) {
        cudaFree(d_D_rd);
        d_D_rd = nullptr;
    }
    if (d_A_rt != nullptr) {
        cudaFree(d_A_rt);
        d_A_rt = nullptr;
    }
    if (d_B_rt != nullptr) {
        cudaFree(d_B_rt);
        d_B_rt = nullptr;
    }
    if (d_C_rt != nullptr) {
        cudaFree(d_C_rt);
        d_C_rt = nullptr;
    }
    if (d_D_rt != nullptr) {
        cudaFree(d_D_rt);
        d_D_rt = nullptr;
    }
#endif
        return;
    }    


    static void forwardMany(double* A, const double* B, 
                            double* C, double* D,
                            const int n_row, const int n_sys) {

        for (int j = 0; j < n_sys; j++) {

            A[j] /= B[j];D[j] /= B[j]; C[j] /= B[j];
            A[j + n_sys] /= B[j + n_sys]; 
            D[j + n_sys] /= B[j + n_sys];
            C[j + n_sys] /= B[j + n_sys];
        }

        for (int i = 2; i < n_row; ++i) {
            int idx = i * n_sys;
            int idx_prev = idx - n_sys;
            for (int j = 0; j < n_sys; ++j) {
                double r = 1.0 / (B[idx] - A[idx] * C[idx_prev]);
                D[idx] = r * (D[idx] - A[idx] * D[idx_prev]);
                C[idx] = r * C[idx];
                A[idx] = -r * A[idx] * A[idx_prev];
                ++idx; ++idx_prev;
            }
        }
    }

    static void backwardMany(double* A, double* C, double* D,
                             const int n_row, const int n_sys) {

        for (int i = n_row - 3; i >= 1; --i) {
            int idx = i * n_sys;
            int idx_next = idx + n_sys;
            for (int j = 0; j < n_sys; ++j) {
                D[idx] -= C[idx] * D[idx_next];
                A[idx] -= C[idx] * A[idx_next];
                C[idx] = -C[idx] * C[idx_next];
                ++idx; ++idx_next;
            }
        }
    }

    static void reduceMany(const double* A, const double* C, const double* D,
                           double* A0, double* A1, double* C0, double* C1,
                           double* D0, double* D1,
                           const int n_row, const int n_sys) {

        for (int j = 0; j < n_sys; ++j) {
            double r = 1.0 / (1.0 - A[j + n_sys] * C[j]);
            D0[j] = r * (D[j] - C[j] * D[j + n_sys]);
            A0[j] = r * A[j];
            C0[j] = -r * C[j] * C[j + n_sys];
            A1[j] = A[(n_row - 1) * n_sys + j];
            C1[j] = C[(n_row - 1) * n_sys + j];
            D1[j] = D[(n_row - 1) * n_sys + j];
        }
    }

    static void reconstructMany(const double* A, const double* C, double* D,
                                const double* D0, const double* D1,
                                const int n_row, const int n_sys) {

        for (int j = 0; j < n_sys; ++j) {
            D[0 * n_sys + j]           = D0[j];
            D[(n_row - 1) * n_sys + j] = D1[j];
        }

        for (int i = 1; i < n_row - 1; ++i) {
            int idx = i * n_sys;
            for (int j = 0; j < n_sys; ++j) {
                D[idx] = D[idx] - A[idx] * D0[j] - C[idx] * D1[j];
                ++idx;
            }
        }
    }

    void PTDMASolverMany::solve(PTDMAPlanMany& plan,
                                double* A,double* B, double* C, double* D) {

        const int n_row = plan.n_row;
        const int n_sys = plan.n_sys;

        assert(n_row > 2);

        if (plan.size == 1) {
            dispatchTDMASolver<BatchType::Many>(plan.type, A, B, C, D, n_row, n_sys);
            // switch (plan.type) {
            //     case TDMAType::Standard:
            //         batchSolver<TDMAType::Standard, BatchType::Many>(A, B, C, D, n_row, n_sys);
            //         break;
            //     case TDMAType::Cyclic:
            //         batchSolver<TDMAType::Cyclic, BatchType::Many>(A, B, C, D, n_row, n_sys);
            //         break;
            //     default:
            //         throw std::invalid_argument("Unknown TDMAType");
            // }
            return;
        }

        forwardMany(A, B, C, D, n_row, n_sys);
        backwardMany(A, C, D, n_row, n_sys);
        reduceMany(A, C, D,
                     plan.A_rd.getData(), plan.A_rd.getData() + n_sys,
                     plan.C_rd.getData(), plan.C_rd.getData() + n_sys,
                     plan.D_rd.getData(), plan.D_rd.getData() + n_sys,
                     n_row, n_sys);

        // MPI transpose of reduced system
        MPI_Request request[3];
        MPI_Status statuses[3];

        MPI_Ialltoallw(plan.A_rd.getData(), plan.count_send.data(), plan.displ_send.data(), 
                    plan.ddtype_FS.data(),
                    plan.A_rt.getData(), plan.count_recv.data(), plan.displ_recv.data(), 
                    plan.ddtype_BS.data(),
                    plan.comm_ptdma, &request[0]);

        MPI_Ialltoallw(plan.C_rd.getData(), plan.count_send.data(), plan.displ_send.data(), 
                    plan.ddtype_FS.data(),
                    plan.C_rt.getData(), plan.count_recv.data(), plan.displ_recv.data(), 
                    plan.ddtype_BS.data(),
                    plan.comm_ptdma, &request[1]);

        MPI_Ialltoallw(plan.D_rd.getData(), plan.count_send.data(), plan.displ_send.data(), 
                    plan.ddtype_FS.data(),
                    plan.D_rt.getData(), plan.count_recv.data(), plan.displ_recv.data(), 
                    plan.ddtype_BS.data(),
                    plan.comm_ptdma, &request[2]);

        MPI_Waitall(3, request, statuses);
        // Solve the reduced tridiagonal systems

        dispatchTDMASolver<BatchType::Many>(plan.type, plan.A_rt.getData(), plan.B_rt.getData(), plan.C_rt.getData(), plan.D_rt.getData(), plan.n_row_rt, plan.n_sys_rt);

        MPI_Ialltoallw(plan.D_rt.getData(), plan.count_recv.data(), plan.displ_recv.data(), 
                    plan.ddtype_BS.data(),
                    plan.D_rd.getData(), plan.count_send.data(), plan.displ_send.data(), 
                    plan.ddtype_FS.data(),
                    plan.comm_ptdma, &request[0]);
        MPI_Waitall(1, request, statuses);

        reconstructMany(A, C, D, 
                            plan.D_rd.getData(), plan.D_rd.getData() + n_sys,
                            n_row, n_sys);
    }

    void PTDMAPlanManyRHS::create(int n_row_, int n_sys_, MPI_Comm comm_ptdma_, TDMAType type_) {

        n_row = n_row_;
        n_sys = n_sys_;
        type = type_;

        MPI_Comm_dup(comm_ptdma_, &comm_ptdma);
        MPI_Comm_size(comm_ptdma, &size);
        MPI_Comm_rank(comm_ptdma, &rank);

        int n_sys_rd = n_sys;
        int n_row_rd = 2;
        std::vector<int> n_sys_rt_array(size);

        // Compute local and global problem dimensions
        n_sys_rt = Util::para_range_n(1, n_sys_rd, size, rank);
        n_row_rt = n_row_rd * size;

        MPI_Allgather(&n_sys_rt, 1, MPI_INT, n_sys_rt_array.data(), 1, MPI_INT, comm_ptdma);
        
        A_rd.resize(n_row_rd);
        B_rd.assign(n_row_rd, 1);
        C_rd.resize(n_row_rd);
        D_rd.resize(n_row_rd, n_sys_rd);

        A_rt.resize(n_row_rt);
        B_rt.assign(n_row_rt, 1);
        C_rt.resize(n_row_rt);
        D_rt.resize(n_row_rt, n_sys_rt);

        ddtype_FS.resize(size);
        ddtype_BS.resize(size);

        for (int i = 0; i < size; ++i) {
            int bigsize[2] = {n_row_rd, n_sys_rd};
            int subsize[2] = {n_row_rd, n_sys_rt_array[i]};
            int start[2] = {0, std::accumulate(n_sys_rt_array.begin(), n_sys_rt_array.begin() + i, 0)};

            MPI_Type_create_subarray(2, bigsize, subsize, start, MPI_ORDER_C, MPI_DOUBLE, &ddtype_FS[i]);
            MPI_Type_commit(&ddtype_FS[i]);

            int rstart[2] = {n_row_rd * i, 0};
            int rsub[2] = {n_row_rd, n_sys_rt};
            int rbig[2] = {n_row_rt, n_sys_rt};

            MPI_Type_create_subarray(2, rbig, rsub, rstart, MPI_ORDER_C, MPI_DOUBLE, &ddtype_BS[i]);
            MPI_Type_commit(&ddtype_BS[i]);
        }

        count_send.assign(size, 1);
        displ_send.assign(size, 0);
        count_recv.assign(size, 1);
        displ_recv.assign(size, 0);
        return;
    }

    void PTDMAPlanManyRHS::destroy() {
        for (int i = 0; i < size; ++i) {
            MPI_Type_free(&ddtype_FS[i]);
            MPI_Type_free(&ddtype_BS[i]);
        }

        ddtype_FS.clear();
        ddtype_BS.clear();
        count_send.clear();
        displ_send.clear();
        count_recv.clear();
        displ_recv.clear();

        A_rd.clear(); B_rd.clear(); C_rd.clear(); D_rd.clear();
        A_rt.clear(); B_rt.clear(); C_rt.clear(); D_rt.clear();
        return;
    }    

    static void forwardManyRHS(double* A, const double* B, double* C, double* D,
                               const int n_row, const int n_sys) {
        A[0] /= B[0];
        C[0] /= B[0];
        A[1] /= B[1];
        C[1] /= B[1];

        for (int j = 0; j < n_sys; j++) {
            D[j] /= B[0];
            D[n_sys + j] /= B[1];
        }

        for (int i = 2; i < n_row; i++) {
            double r = 1.0 / (B[i] - A[i] * C[i - 1]);
            int idx = i * n_sys;
            int idx_prev = idx - n_sys;
            for (int j = 0; j < n_sys; j++) {
                D[idx] = r * (D[idx] - A[i] * D[idx_prev]);
                ++idx; ++idx_prev;
            }
            C[i] = r * C[i];
            A[i] = -r * A[i] * A[i - 1];
        }
    }

    static void backwardManyRHS(double* A, double* C, double* D,
                                const int n_row, const int n_sys) {
        for (int i = n_row - 3; i >= 1; i--) {
            int idx = i * n_sys;
            int idx_next = idx + n_sys;
            for (int j = 0; j < n_sys; j++) {
                D[idx] = D[idx] - C[i] * D[idx_next];
                ++idx; ++idx_next;
            }
            A[i] = A[i] - C[i] * A[i + 1];
            C[i] = -C[i] * C[i + 1];
        }
    }

    static void reduceManyRHS(const double* A, const double* C, double* D,
                              double* A0, double* A1, double* C0, double* C1,
                              double* D0, double* D1,
                              const int n_row, const int n_sys) {
        double r = 1.0 / (1.0 - A[1] * C[0]);
        for (int j = 0; j < n_sys; ++j) {
            D0[j] = r * (D[j] - C[0] * D[n_sys + j]);
            D1[j] = D[(n_row - 1) * n_sys + j];
        }
        A0[0] = r * A[0];
        C0[0] = -r * C[0] * C[1];
        A1[0] = A[n_row - 1];
        C1[0] = C[n_row - 1];
    }

    static void reconstructManyRHS(const double* A, const double* C, double* D,
                                   const double* D0, const double* D1,
                                   const int n_row, const int n_sys) {
        for (int j = 0; j < n_sys; ++j) {
            D[j] = D0[j];
            D[(n_row - 1) * n_sys + j] = D1[j];
        }

        for (int i = 1; i < n_row - 1; ++i) {
            int idx = i * n_sys;
            for (int j = 0; j < n_sys; ++j) {
                D[idx] = D[idx] - A[i] * D0[j] - C[i] * D1[j];
                ++idx;
            }
        }
    }

    void PTDMASolverManyRHS::solve(PTDMAPlanManyRHS& plan,
                                   double* A, double* B, double* C, double* D) {

        const int n_row = plan.n_row;
        const int n_sys = plan.n_sys;

        assert(n_row > 2);

        if (plan.size == 1) {
            dispatchTDMASolver<BatchType::ManyRHS>(plan.type, A, B, C, D, n_row, n_sys);
            return;
        }
        forwardManyRHS(A, B, C, D, n_row, n_sys);
        backwardManyRHS(A, C, D, n_row, n_sys);
        reduceManyRHS(A, C, D,
                      plan.A_rd.data(), plan.A_rd.data() + 1,
                      plan.C_rd.data(), plan.C_rd.data() + 1,
                      plan.D_rd.getData(), plan.D_rd.getData() + n_sys,
                      n_row, n_sys);

        // Transpose the reduced system using MPI_Ialltoallw
        MPI_Request request[3];
        MPI_Status statuses[3];

        MPI_Iallgather(plan.A_rd.data(), 2, MPI_DOUBLE, 
                       plan.A_rt.data(), 2, MPI_DOUBLE,
                       plan.comm_ptdma, &request[0]);

        MPI_Iallgather(plan.C_rd.data(), 2, MPI_DOUBLE, plan.C_rt.data(), 2, MPI_DOUBLE,
                        plan.comm_ptdma, &request[1]);

        MPI_Ialltoallw(plan.D_rd.getData(), plan.count_send.data(), plan.displ_send.data(), 
                        plan.ddtype_FS.data(),
                        plan.D_rt.getData(), plan.count_recv.data(), plan.displ_recv.data(), 
                        plan.ddtype_BS.data(),
                        plan.comm_ptdma, &request[2]);

        MPI_Waitall(3, request, statuses);

        // Solve the reduced tridiagonal systems
        dispatchTDMASolver<BatchType::ManyRHS>(plan.type, plan.A_rt.data(), plan.B_rt.data(), plan.C_rt.data(), plan.D_rt.getData(), plan.n_row_rt, plan.n_sys_rt);

        // Transpose solutions back
        MPI_Ialltoallw( plan.D_rt.getData(), plan.count_recv.data(), plan.displ_recv.data(), 
                        plan.ddtype_BS.data(),
                        plan.D_rd.getData(), plan.count_send.data(), plan.displ_send.data(), 
                        plan.ddtype_FS.data(),
                        plan.comm_ptdma, &request[0]);
        MPI_Waitall(1, request, statuses);

        // Update full solution from reduced solution
        reconstructManyRHS(A, C, D,
                                    plan.D_rd.getData(), 
                                    plan.D_rd.getData() + n_sys,
                                    n_row, n_sys);
    }


    template <TDMAType tdma_type, BatchType batch_type>
    void batchSolver(double* A, double* B, double* C, double* D, int n_row, int n_sys) {
        if constexpr (batch_type == BatchType::Single) {
            if constexpr (tdma_type == TDMAType::Standard)
                TDMASolver::single(A, B, C, D, n_row);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                TDMASolver::singleCyclic(A, B, C, D, n_row);
        }
        else if constexpr (batch_type == BatchType::Many) {
            if constexpr (tdma_type == TDMAType::Standard)
                TDMASolver::many(A, B, C, D, n_row, n_sys);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                TDMASolver::manyCyclic(A, B, C, D, n_row, n_sys);
        }
        else if constexpr (batch_type == BatchType::ManyRHS) {
            if constexpr (tdma_type == TDMAType::Standard)
                TDMASolver::manyRHS(A, B, C, D, n_row, n_sys);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                TDMASolver::manyRHSCyclic(A, B, C, D, n_row, n_sys);
        }
    }

    template <BatchType batch_type>
    void dispatchTDMASolver(TDMAType type, double* A, double* B, double* C, double* D, int n_row, int n_sys) {
        switch (type) {
            case TDMAType::Standard:
                batchSolver<TDMAType::Standard, batch_type>(A, B, C, D, n_row, n_sys);
                break;
            case TDMAType::Cyclic:
                batchSolver<TDMAType::Cyclic, batch_type>(A, B, C, D, n_row, n_sys);
                break;
            default:
                throw std::invalid_argument("Unknown TDMAType");
        }
    }

    template void batchSolver<TDMAType::Standard, BatchType::Single>(
        double*, double*, double*, double*, int, int);

    template void batchSolver<TDMAType::Cyclic, BatchType::Single>(
        double*, double*, double*, double*, int, int);

    template void batchSolver<TDMAType::Standard, BatchType::Many>(
        double*, double*, double*, double*, int, int);

    template void batchSolver<TDMAType::Cyclic, BatchType::Many>(
        double*, double*, double*, double*, int, int);

    template void batchSolver<TDMAType::Standard, BatchType::ManyRHS>(
        double*, double*, double*, double*, int, int);

    template void batchSolver<TDMAType::Cyclic, BatchType::ManyRHS>(
        double*, double*, double*, double*, int, int);


};    
 // namespace PaScaL_TDMA
