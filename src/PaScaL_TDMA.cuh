#pragma once

#include <vector>
#include <mpi.h>
#include "dimArray.hpp"
#include "util.hpp"
#include "PaScaL_TDMA.hpp"
#include <cuda_runtime.h>

namespace PaScaL_TDMA {

    class cuPTDMAPlanMany : public PTDMAPlanBase {

    friend class cuPTDMASolverMany;

    private:
        int n_sys = 0;
        int n_row = 0;
        int n_sys_rt = 0;
        int n_row_rt = 0;

        std::vector<MPI_Datatype> ddtype_FS, ddtype_BS;

        double *d_A_rd = nullptr, *d_B_rd = nullptr, *d_C_rd = nullptr, *d_D_rd = nullptr;
        double *d_A_rt = nullptr, *d_B_rt = nullptr, *d_C_rt = nullptr, *d_D_rt = nullptr;

    public:
        using PTDMAPlanBase::create;
        void create(int n_row_, int n_sys_, MPI_Comm comm_ptdma_, 
                    TDMAType type_) override;
        void destroy() override;

        inline int getRowSize() const { return n_row; };
        inline int getSysSize() const { return n_sys; };
    };

    class cuPTDMASolverMany {
    public:
        static void cuSolve(cuPTDMAPlanMany& plan,
                          double* A, double* B, double* C, double* D);

        // Inline wrapper
        static inline void cuSolve(cuPTDMAPlanMany& plan, 
                                 std::vector<double>& A, std::vector<double>& B, 
                                 std::vector<double>& C, std::vector<double>& D)
        { cuSolve(plan, A.data(), B.data(), C.data(), D.data()); }
    };

    // template <BatchType batch_type>
    // void dispatchTDMASolver(TDMAType type, double* A, double* B, double* C, double* D, int n_row, int n_sys);

    // template <TDMAType tdma_type, BatchType batch_type>
    // void batchSolver(double* A, double* B, double* C, double* D, int n_row, int n_sys);

    // extern template void batchSolver<TDMAType::Standard, BatchType::Single>(double*, double*, double*, double*, int, int);
    // extern template void batchSolver<TDMAType::Cyclic, BatchType::Single>(double*, double*, double*, double*, int, int);
    // extern template void batchSolver<TDMAType::Standard, BatchType::Many>(double*, double*, double*, double*, int, int);
    // extern template void batchSolver<TDMAType::Cyclic, BatchType::Many>(double*, double*, double*, double*, int, int);
    // extern template void batchSolver<TDMAType::Standard, BatchType::ManyRHS>(double*, double*, double*, double*, int, int);
    // extern template void batchSolver<TDMAType::Cyclic, BatchType::ManyRHS>(double*, double*, double*, double*, int, int);

} // namespace PaScaL_TDMA