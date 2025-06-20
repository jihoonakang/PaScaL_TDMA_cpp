#pragma once

#include <vector>
#include <mpi.h>
#include "dimArray.hpp"
#include "util.hpp"
#include <cuda_runtime.h>

namespace cuPaScaL_TDMA {

    enum class TDMAType {
        Standard,
        Cyclic
    };

    enum class BatchType {
        Single,
        Many,
        ManyRHS
    };

    class cuPTDMAPlanBase {
    
    protected:
        MPI_Comm comm_ptdma = MPI_COMM_NULL;
        int rank = 0;
        int size = 1;

        std::vector<int> count_send, displ_send;
        std::vector<int> count_recv, displ_recv;
        TDMAType type;

    public:
        virtual ~cuPTDMAPlanBase() = default;

        virtual void create(int n_row_, int ny_sys_, int nz_sys_, MPI_Comm comm_ptdma_, 
                            TDMAType type_);
        virtual void create(int n_row_, int ny_sys_, int nz_sys_, MPI_Comm comm_ptdma_);

        virtual void destroy() = 0;
        inline int getMPIRank() const { return rank; };
        inline int getMPISize() const { return size; };
    };

    class cuPTDMAPlanMany : public cuPTDMAPlanBase {

    friend class cuPTDMASolverMany;

    private:
        dim3 threads, blocks, blocks_rt, blocks_alltoall;

        int n_row = 0;
        int n_sys = 0;
        int ny_sys = 0;
        int nz_sys = 0;

        int n_row_rd = 0;
        int n_row_rt = 0;
        int n_sys_rd = 0;
        int n_sys_rt = 0;

        int nz_sys_rd = 0;
        int nz_sys_rt = 0;

        double *d_a_rd = nullptr, *d_b_rd = nullptr, *d_c_rd = nullptr, *d_d_rd = nullptr;
        double *d_a_rt = nullptr, *d_b_rt = nullptr, *d_c_rt = nullptr, *d_d_rt = nullptr;

    public:
        using cuPTDMAPlanBase::create;
        void create(int n_row_, int ny_sys_, int nz_sys_, MPI_Comm comm_ptdma_, 
                    TDMAType type_) override;
        void destroy() override;

        inline int getRowSize() const { return n_row; };
        inline int getSysSize() const { return n_sys; };
        inline int getNySysSize() const { return ny_sys; };
        inline int getNzSysSize() const { return nz_sys; };
    };

    class cuPTDMASolverMany {
    public:
        static void transpose_slab_yz_to_xy(const cuPTDMAPlanMany& plan,
                                const double* slab_yz,
                                double* slab_xy);
        static void transpose_slab_xy_to_yz(const cuPTDMAPlanMany& plan,
                                const double* slab_xy,
                                double* slab_yz);
        static void cuSolve(cuPTDMAPlanMany& plan,
                          double* A, double* B, double* C, double* D);

        // Inline wrapper
        static inline void cuSolve(cuPTDMAPlanMany& plan, 
                                 std::vector<double>& A, std::vector<double>& B, 
                                 std::vector<double>& C, std::vector<double>& D)
        { cuSolve(plan, A.data(), B.data(), C.data(), D.data()); }
    };

    template <BatchType batch_type>
    void cuDispatchTDMASolver(TDMAType type, double* A, double* B, double* C, double* D, int n_row, int ny, int nz);

    template <TDMAType tdma_type, BatchType batch_type>
    void cuBatchSolver(double* A, double* B, double* C, double* D, int n_row, int ny, int nz);

    extern template void cuBatchSolver<TDMAType::Standard, BatchType::Single>(double*, double*, double*, double*, int, int, int);
    extern template void cuBatchSolver<TDMAType::Cyclic, BatchType::Single>(double*, double*, double*, double*, int, int, int);
    extern template void cuBatchSolver<TDMAType::Standard, BatchType::Many>(double*, double*, double*, double*, int, int, int);
    extern template void cuBatchSolver<TDMAType::Cyclic, BatchType::Many>(double*, double*, double*, double*, int, int, int);
    extern template void cuBatchSolver<TDMAType::Standard, BatchType::ManyRHS>(double*, double*, double*, double*, int, int, int);
    extern template void cuBatchSolver<TDMAType::Cyclic, BatchType::ManyRHS>(double*, double*, double*, double*, int, int, int);
} // namespace cuPaScaL_TDMA