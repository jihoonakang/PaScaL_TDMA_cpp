/**
 * @file PaScaL_TDMA.cuh
 * @brief CUDA planner and solver dispatch interfaces for PaScaL_TDMA.
 *
 * Declares GPU-specific plan and solver classes for batched TDMA
 * systems over a multi-dimensional MPI+CUDA environment.
 */

#pragma once

#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include "dimArray.hpp"
#include "util.hpp"

namespace cuPaScaL_TDMA {

/// Type of TDMA solver: standard or cyclic (periodic)
enum class TDMAType { Standard, Cyclic };

/// Batch execution mode: single system, many systems, or many RHS vectors
enum class BatchType { Single, Many, ManyRHS };

/**
 * @class cuPTDMAPlanBase
 * @brief Base GPU plan holding MPI communicator and solve type.
 */
class cuPTDMAPlanBase {

protected:
    MPI_Comm comm_ptdma_ = MPI_COMM_NULL;   ///< Sub-communicator
    int rank_ = 0;                          ///< Rank in comm_ptdma_
    int size_ = 1;                          ///< Size of comm_ptdma_
    TDMAType type_;                         ///< Solver type

public:
    virtual ~cuPTDMAPlanBase() = default;

    /**
     * @brief Initialize plan dimensions and MPI communicator.
     * @param n_row    Number of rows per system.
     * @param ny_sys   Number of systems in Y dim.
     * @param nz_sys   Number of systems in Z dim.
     * @param comm_p   Parent MPI communicator.
     * @param type     Solver type.
     */
    virtual void create(int n_row, int ny_sys, int nz_sys, MPI_Comm comm_ptdma, 
                        TDMAType type);

    /// @brief Release any CUDA or MPI resources
    virtual void destroy() = 0;
    inline int getMPIRank() const noexcept { return rank_; };
    inline int getMPISize() const noexcept { return size_; };
};

/**
 * @class cuPTDMAPlanMany
 * @brief GPU plan for many independent TDMA systems.
 */class cuPTDMAPlanMany : public cuPTDMAPlanBase {

    friend class cuPTDMASolverMany;

    template<typename PlanType>
    friend void transposeSlabYZtoXY(const PlanType&, const double*, double*);
    template<typename PlanType>
    friend void transposeSlabXYtoYZ(const PlanType&, const double*, double*);

private:
    dim3 threads_{};         ///< CUDA block dimensions
    dim3 blocks_{};          ///< CUDA grid dimensions
    dim3 blocks_rt_{};       ///< Grid for reduced solve
    dim3 blocks_alltoall_{}; ///< Grid for all-to-all transpose

    int n_row_     = 0;      ///< Rows per system
    int n_sys_     = 0;      ///< Systems count
    int ny_sys_    = 0;      ///< Y-dimension count
    int nz_sys_    = 0;      ///< Z-dimension count

    int n_row_rd_  = 0;      ///< Rows in reduced system
    int n_row_rt_  = 0;      ///< Total reduced rows
    int n_sys_rd_  = 0;      ///< Systems in reduced system
    int n_sys_rt_  = 0;      ///< Total reduced systems

    int nz_sys_rd_ = 0;      ///< Z-systems in reduced
    int nz_sys_rt_ = 0;      ///< Total Z reduced systems

    double *a_rd_d_ = nullptr, *b_rd_d_ = nullptr;  ///< reduced system
    double *c_rd_d_ = nullptr, *d_rd_d_ = nullptr;  ///< reduced system
    double *a_rt_d_ = nullptr, *b_rt_d_ = nullptr;  ///< transposed system
    double *c_rt_d_ = nullptr, *d_rt_d_ = nullptr;  ///< transposed system

public:
    using cuPTDMAPlanBase::create;
    void create(int n_row_, int ny_sys_, int nz_sys_, MPI_Comm comm_ptdma_, 
                TDMAType type_) override;
    void destroy() override;

    inline int getRowSize() const noexcept { return n_row_; };
    inline int getSysSize() const noexcept { return n_sys_; };
};

/**
 * @class cuPTDMASolverMany
 * @brief GPU solver for many independent TDMA systems.
 */
class cuPTDMASolverMany {
public:
    // /**
    //  * @brief Transpose slab from (Y,Z) layout to (X,Y)
    //  */
    // static void transposeSlabYZtoXY(const cuPTDMAPlanMany& plan,
    //                         const double* slab_yz,
    //                         double* slab_xy);
    // /**
    //  * @brief Transpose slab from (X,Y) back to (Y,Z)
    //  */
    // static void transposeSlabXYtoYZ(const cuPTDMAPlanMany& plan,
    //                         const double* slab_xy,
    //                         double* slab_yz);
    /**
     * @brief Launch CUDA kernels to solve all systems
     */
    static void cuSolve(cuPTDMAPlanMany& plan,
                        double* A, double* B, double* C, double* D);

    /**
     * @brief Inline wrapper accepting std::vector references
     */
    static inline void cuSolve(cuPTDMAPlanMany& plan, 
                                std::vector<double>& A, std::vector<double>& B, 
                                std::vector<double>& C, std::vector<double>& D)
    { cuSolve(plan, A.data(), B.data(), C.data(), D.data()); }
};

/**
 * @class cuPTDMAPlanManyRHS
 * @brief GPU plan for many RHS batched TDMA with shared diagonals.
 */
class cuPTDMAPlanManyRHS : public cuPTDMAPlanBase {

    friend class cuPTDMASolverManyRHS;

    template<typename PlanType>
    friend void transposeSlabYZtoXY(const PlanType&, const double*, double*);
    template<typename PlanType>
    friend void transposeSlabXYtoYZ(const PlanType&, const double*, double*);

private:
    dim3 threads_{};         ///< CUDA block dimensions
    dim3 blocks_{};          ///< CUDA grid dimensions
    dim3 blocks_rt_{};       ///< Grid for reduced solve
    dim3 blocks_alltoall_{}; ///< Grid for all-to-all transpose

    int n_row_     = 0;      ///< Rows per system
    int n_sys_     = 0;      ///< Systems count
    int ny_sys_    = 0;      ///< Y-dimension count
    int nz_sys_    = 0;      ///< Z-dimension count

    int n_row_rd_  = 0;      ///< Rows in reduced system
    int n_row_rt_  = 0;      ///< Total reduced rows
    int n_sys_rd_  = 0;      ///< Systems in reduced system
    int n_sys_rt_  = 0;      ///< Total reduced systems

    int nz_sys_rd_ = 0;      ///< Z-systems in reduced
    int nz_sys_rt_ = 0;      ///< Total Z reduced systems

    double *a_rd_d_ = nullptr, *b_rd_d_ = nullptr;  ///< reduced system
    double *c_rd_d_ = nullptr, *d_rd_d_ = nullptr;  ///< reduced system
    double *a_rt_d_ = nullptr, *b_rt_d_ = nullptr;  ///< transposed system
    double *c_rt_d_ = nullptr, *d_rt_d_ = nullptr;  ///< transposed system

public:
    using cuPTDMAPlanBase::create;
    void create(int n_row, int ny_sys, int nz_sys, MPI_Comm comm_ptdma, 
                TDMAType type) override;
    void destroy() override;

    inline int getRowSize() const { return n_row_; };
    inline int getSysSize() const { return n_sys_; };
};

/**
 * @class cuPTDMASolverManyRHS
 * @brief GPU solver for ManyRHS batched TDMA systems.
 */
class cuPTDMASolverManyRHS {
public:
    // /**
    //  * @brief Transpose slab from (Y,Z) layout to (X,Y)
    //  */
    // static void transposeSlabYZtoXY(const cuPTDMAPlanManyRHS& plan,
    //                         const double* slab_yz,
    //                         double* slab_xy);
    // /**
    //  * @brief Transpose slab from (X,Y) back to (Y,Z)
    //  */
    // static void transposeSlabXYtoYZ(const cuPTDMAPlanManyRHS& plan,
    //                         const double* slab_xy,
    //                         double* slab_yz);
    /**
     * @brief Gather shared coefficients across ranks via MPI.
     */
    static void allGather(const cuPTDMAPlanManyRHS& plan, 
                            const double* coef_rd,
                            double* coef_rt);
    /**
     * @brief Launch CUDA kernels to solve systems with many RHS.
     */
    static void cuSolve(cuPTDMAPlanManyRHS& plan,
                        double* a, double* b, double* c, double* d);

    /**
     * @brief Inline wrapper accepting std::vector references
     */
    static inline void cuSolve(cuPTDMAPlanManyRHS& plan, 
                                std::vector<double>& a, std::vector<double>& b, 
                                std::vector<double>& c, std::vector<double>& d)
    { cuSolve(plan, a.data(), b.data(), c.data(), d.data()); }
};

    //------------------------------------------------------------------------------
    // CUDA dispatch templates
    //------------------------------------------------------------------------------

    /**
    * @brief Dispatch to correct CUDA batch solver variant.
    */
    template <BatchType batch_type>
    void cuDispatchTDMASolver(TDMAType type, 
                              double* a, double* b, double* c, double* d, 
                              int n_row, int ny, int nz);

    /**
    * @brief CUDA batch solver: specialization for Standard/Cyclic × Many/ManyRHS.
    */
    template <TDMAType tdma_type, BatchType batch_type>
    void cuBatchSolver(double* a, double* b, double* c, double* d, 
                       int n_row, int ny, int nz);

    // explicit instantiations
    extern template void cuBatchSolver<TDMAType::Standard, BatchType::Many>(
        double*, double*, double*, double*, int, int, int);
    extern template void cuBatchSolver<TDMAType::Cyclic, BatchType::Many>(
        double*, double*, double*, double*, int, int, int);
    extern template void cuBatchSolver<TDMAType::Standard, BatchType::ManyRHS>(
        double*, double*, double*, double*, int, int, int);
    extern template void cuBatchSolver<TDMAType::Cyclic, BatchType::ManyRHS>(
        double*, double*, double*, double*, int, int, int);

} // namespace cuPaScaL_TDMA