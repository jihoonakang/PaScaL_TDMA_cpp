#pragma once

#include <vector>
#include <mpi.h>
#include "dimArray.hpp"
#include "util.hpp"

namespace PaScaL_TDMA {

    enum class TDMAType {
        Standard,
        Cyclic
    };

    enum class BatchType {
        Single,
        Many,
        ManyRHS
    };

    class PTDMAPlanBase {
    
    protected:
        MPI_Comm comm_ptdma = MPI_COMM_NULL;
        int rank = 0;
        int size = 1;

        std::vector<int> count_send, displ_send;
        std::vector<int> count_recv, displ_recv;
        TDMAType type;

    public:
        virtual ~PTDMAPlanBase() = default;

        virtual void create(int myrank_, int nprocs_) {};
        virtual void destroy() = 0;
        inline const int getMPIRank() const { return rank; };
        inline const int getMPISize() const { return size; };
    };

    // ──────────────────────────────────────────────────────────────

    class PTDMAPlanSingle : public PTDMAPlanBase {

    friend class PTDMASolverSingle;

    private:
        int n_row = 0;
        int n_row_rt = 0;
        int root_rank = 0;

        std::vector<double> A_rd, B_rd, C_rd, D_rd;
        std::vector<double> A_rt, B_rt, C_rt, D_rt;

    public:
        void create(int n_row_, MPI_Comm comm_ptdma_, int root_rank_, 
                    TDMAType type_);
        void destroy() override;

        inline const int getRowSize() const { return n_row; };
        inline const int getRootRank() const { return root_rank; };
    };

    // ──────────────────────────────────────────────────────────────

    class PTDMAPlanMany : public PTDMAPlanBase {

    friend class PTDMASolverMany;

    private:
        int n_sys = 0;
        int n_row = 0;
        int n_sys_rt = 0;
        int n_row_rt = 0;

        std::vector<MPI_Datatype> ddtype_FS, ddtype_BS;
        dimArray<double> A_rd, B_rd, C_rd, D_rd;
        dimArray<double> A_rt, B_rt, C_rt, D_rt;

    public:
        void create(int n_row_, int n_sys_, MPI_Comm comm_ptdma_, 
                    TDMAType type_);
        void destroy() override;

        inline const int getRowSize() const { return n_row; };
        inline const int getSysSize() const { return n_sys; };
    };

    // ──────────────────────────────────────────────────────────────

    class PTDMAPlanManyRHS : public PTDMAPlanBase {

    friend class PTDMASolverManyRHS;

    private:
        int n_sys = 0;
        int n_row = 0;
        int n_sys_rt = 0;
        int n_row_rt = 0;

        std::vector<MPI_Datatype> ddtype_FS, ddtype_BS;

        std::vector<double> A_rd, B_rd, C_rd;
        dimArray<double> D_rd;

        std::vector<double> A_rt, B_rt, C_rt;
        dimArray<double> D_rt;

    public:
        void create(int n_row_, int n_sys_, MPI_Comm comm_ptdma_, 
                    TDMAType type_);
        void destroy() override;

        inline const int getRowSize() const { return n_row; };
        inline const int getSysSize() const { return n_sys; };
    };

    // ──────────────────────────────────────────────────────────────

    class PTDMAPlanManyThreadTeam : public PTDMAPlanBase {

    public:
        int n_sys = 0;
        int n_row = 0;
        int n_sys_rt = 0;
        int n_row_rt = 0;

        void create(int n_row_, int n_sys_, MPI_Comm comm_ptdma_);
        void destroy() override;
    };

    class PTDMASolverSingle {
    public:
        static void solve(PTDMAPlanSingle& plan, 
                          double* A, double* B, double* C, double* D);

        // Inline wrapper: defined here
        static inline void solve(PTDMAPlanSingle& plan, 
                                 std::vector<double>& A, std::vector<double>& B, 
                                 std::vector<double>& C, std::vector<double>& D) 
        { solve(plan, A.data(), B.data(), C.data(), D.data()); }

    };

    class PTDMASolverMany {
    public:
        static void solve(PTDMAPlanMany& plan,
                          double* A, double* B, double* C, double* D);

        // Inline wrapper
        static inline void solve(PTDMAPlanMany& plan, 
                                 dimArray<double>& A, dimArray<double>& B, 
                                 dimArray<double>& C, dimArray<double>& D)
        { solve(plan, A.getData(), B.getData(), C.getData(), D.getData()); }

        static inline void solve(PTDMAPlanMany& plan, 
                                 std::vector<double>& A, std::vector<double>& B, 
                                 std::vector<double>& C, std::vector<double>& D)
        { solve(plan, A.data(), B.data(), C.data(), D.data()); }
    };

    class PTDMASolverManyRHS {
    public:
        static void solve(PTDMAPlanManyRHS& plan, 
                          double* A, double* B, double* C, double* D);

        // Inline wrapper
        static inline void solve(PTDMAPlanManyRHS& plan, 
                                 std::vector<double>& A, std::vector<double>& B, 
                                 std::vector<double>& C, dimArray<double>& D)
        { solve(plan, A.data(), B.data(), C.data(), D.getData()); }

        static inline void solve(PTDMAPlanManyRHS& plan, 
                                 std::vector<double>& A, std::vector<double>& B, 
                                 std::vector<double>& C, std::vector<double>& D)
        { solve(plan, A.data(), B.data(), C.data(), D.data()); }
    };


    template <BatchType batch_type>
    void dispatchTDMASolver(TDMAType type, double* A, double* B, double* C, double* D, int n_row, int n_sys);

    template <TDMAType tdma_type, BatchType batch_type>
    void batchSolver(double* A, double* B, double* C, double* D, int n_row, int n_sys);

    extern template void batchSolver<TDMAType::Standard, BatchType::Single>(double*, double*, double*, double*, int, int);
    extern template void batchSolver<TDMAType::Cyclic, BatchType::Single>(double*, double*, double*, double*, int, int);
    extern template void batchSolver<TDMAType::Standard, BatchType::Many>(double*, double*, double*, double*, int, int);
    extern template void batchSolver<TDMAType::Cyclic, BatchType::Many>(double*, double*, double*, double*, int, int);
    extern template void batchSolver<TDMAType::Standard, BatchType::ManyRHS>(double*, double*, double*, double*, int, int);
    extern template void batchSolver<TDMAType::Cyclic, BatchType::ManyRHS>(double*, double*, double*, double*, int, int);


} // namespace PaScaL_TDMA