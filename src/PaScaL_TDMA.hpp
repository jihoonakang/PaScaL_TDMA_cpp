/**
 * @file PaScaL_TDMA.hpp
 * @brief High-level planner and solver dispatch for PaScaL_TDMA library.
 *
 * Defines plan classes for different TDMA batch types and solver wrappers
 * that execute on CPU or GPU backends across an MPI communicator.
 */

#pragma once

#include <vector>
#include <mpi.h>
#include "dimArray.hpp"
#include "util.hpp"

namespace PaScaL_TDMA {

/// Type of TDMA solver: standard or cyclic (periodic)
enum class TDMAType { Standard, Cyclic };

/// Batch execution mode: single system, many systems, or many RHS vectors
enum class BatchType { Single, Many, ManyRHS };

// ───────────────── Plans ──────────────────────────────────────────────
/**
 * @class PTDMAPlanBase
 * @brief Base class for TDMA execution plans managing MPI exchange metadata.
 */
class PTDMAPlanBase {

protected:
    MPI_Comm comm_ptdma_ = MPI_COMM_NULL;   ///< Sub-communicator for TDMA
    int rank_ = 0;                          ///< Rank in comm_ptdma
    int size_ = 1;                          ///< Size of comm_ptdma

    std::vector<int> count_send_;           ///< Send counts for reduced system
    std::vector<int> displ_send_;           ///< Send displs for reduced system
    std::vector<int> count_recv_;           ///< Recv counts for transpose
    std::vector<int> displ_recv_;           ///< Recv displs for transpose
    TDMAType type_;                         ///< Solver type

public:
    virtual ~PTDMAPlanBase() = default;

    /**
     * @brief Initialize plan for single or cyclic single system.
     * @param n_row         Number of rows per system.
     * @param comm_ptdma    Parent MPI communicator.
     * @param type          TDMA solver type.
     */
    virtual void create(int n_row, MPI_Comm comm_ptdma, TDMAType type);

    /**
     * @brief Initialize plan for batched systems with system count.
     * @param n_row         Number of rows per system.
     * @param n_sys         Number of systems or RHS.
     * @param comm_ptdma    Parent MPI communicator.
     * @param type          TDMA solver type.
     */
    virtual void create(int n_row, int n_sys, MPI_Comm comm_ptdma, 
                        TDMAType type);

    /// Destroy plan and free MPI datatypes
    virtual void destroy() = 0;

    /// Get MPI rank in TDMA communicator
    inline int getMPIRank() const noexcept { return rank_; };
    /// Get MPI size in TDMA communicator
    inline int getMPISize() const noexcept { return size_; };
};

/**
 * @class PTDMAPlanSingle
 * @brief Plan for single-system TDMA across multiple ranks.
 */
class PTDMAPlanSingle : public PTDMAPlanBase {

    friend class PTDMASolverSingle;

private:
    int n_row_    = 0;      ///< Local rows per rank
    int n_row_rt_ = 0;      ///< Local rows after gathering reduced systems
    int root_rank_ = 0;     ///< Root rank for gather/scatter

    std::vector<double> a_rd_, b_rd_, c_rd_, d_rd_; ///< For reduces system
    std::vector<double> a_rt_, b_rt_, c_rt_, d_rt_; ///< For transposed system

public:
    using PTDMAPlanBase::create;
    void create(int n_row, MPI_Comm comm_ptdma, TDMAType type) override;
    void destroy() override;

    inline int getRowSize() const noexcept { return n_row_; }
    inline int getRootRank() const noexcept { return root_rank_; }
};

/**
 * @class PTDMAPlanMany
 * @brief Plan for multiple independent TDMA systems.
 */
class PTDMAPlanMany : public PTDMAPlanBase {

    friend class PTDMASolverMany;

private:
    int n_row_ = 0;     ///< Local rows per system
    int n_sys_ = 0;     ///< Number of systems
    int n_row_rt_ = 0;   ///< Local rows after transpose
    int n_sys_rt_ = 0;   ///< Number of systems after transpose

    std::vector<MPI_Datatype> ddtype_rd_;   ///< MPI types for reduced system
    std::vector<MPI_Datatype> ddtype_rt_;   ///< MPI types for transposed system

    dimArray<double> a_rd_, b_rd_, c_rd_, d_rd_;    ///< For reduces system
    dimArray<double> a_rt_, b_rt_, c_rt_, d_rt_;    ///< For transposed system

public:
    using PTDMAPlanBase::create;
    void create(int n_row, int n_sys, MPI_Comm comm_ptdma, 
                TDMAType type) override;
    void destroy() override;

    inline int getRowSize() const noexcept { return n_row_; };
    inline int getSysSize() const noexcept { return n_sys_; };
};

/**
 * @class PTDMAPlanManyRHS
 * @brief Plan for batched RHS TDMA with shared diagonals.
 */
class PTDMAPlanManyRHS : public PTDMAPlanBase {

    friend class PTDMASolverManyRHS;

private:
    int n_row_ = 0;     ///< Local rows per system
    int n_sys_ = 0;     ///< Number of systems
    int n_row_rt_ = 0;   ///< Local rows after transpose
    int n_sys_rt_ = 0;   ///< Number of systems after transpose

    std::vector<MPI_Datatype> ddtype_rd_;   ///< MPI types for reduced system
    std::vector<MPI_Datatype> ddtype_rt_;   ///< MPI types for transposed system

    std::vector<double> a_rd_, b_rd_, c_rd_;   ///< For reduces system
    dimArray<double> d_rd_;                  ///< For reduces system, RHS

    std::vector<double> a_rt_, b_rt_, c_rt_;   ///< For transposed system
    dimArray<double> d_rt_;                  ///< For transposed system, RHS

public:
    using PTDMAPlanBase::create;
    void create(int n_row, int n_sys, MPI_Comm comm_ptdma, 
                TDMAType type) override;
    void destroy() override;

    inline int getRowSize() const { return n_row_; };
    inline int getSysSize() const { return n_sys_; };
};

// ───────────────── Solvers ──────────────────────────────────────────────

/**
 * @class PTDMASolverSingle
 * @brief Executes single-system TDMA based on plan.
 */
class PTDMASolverSingle {
public:
    static void solve(PTDMAPlanSingle& plan, 
                      double* a, double* b, double* c, double* d);

    // Inline wrapper: defined here
    static inline void solve(PTDMAPlanSingle& plan, 
                             std::vector<double>& a, std::vector<double>& b, 
                             std::vector<double>& c, std::vector<double>& d) 
    { solve(plan, a.data(), b.data(), c.data(), d.data()); }

};

/**
 * @class PTDMASolverMany
 * @brief Executes many independent TDMA systems based on plan.
 */
class PTDMASolverMany {
public:
    static void solve(PTDMAPlanMany& plan,
                      double* a, double* b, double* c, double* d);

    // Inline wrapper
    static inline void solve(PTDMAPlanMany& plan, 
                             dimArray<double>& a, dimArray<double>& b, 
                             dimArray<double>& c, dimArray<double>& d)
    { solve(plan, a.getData(), b.getData(), c.getData(), d.getData()); }

    static inline void solve(PTDMAPlanMany& plan, 
                             std::vector<double>& a, std::vector<double>& b, 
                             std::vector<double>& c, std::vector<double>& d)
    { solve(plan, a.data(), b.data(), c.data(), d.data()); }
};

/**
 * @class PTDMASolverManyRHS
 * @brief Executes RHS-batched TDMA systems based on plan.
 */
class PTDMASolverManyRHS {
public:
    static void solve(PTDMAPlanManyRHS& plan, 
                      double* a, double* b, double* c, double* d);

    // Inline wrapper
    static inline void solve(PTDMAPlanManyRHS& plan, 
                             std::vector<double>& a, std::vector<double>& b, 
                             std::vector<double>& c, dimArray<double>& d)
    { solve(plan, a.data(), b.data(), c.data(), d.getData()); }

    static inline void solve(PTDMAPlanManyRHS& plan, 
                             std::vector<double>& a, std::vector<double>& b, 
                             std::vector<double>& c, std::vector<double>& d)
    { solve(plan, a.data(), b.data(), c.data(), d.data()); }
};

// ───────────────── Dispatch Helpers ─────────────────────────────────────

    /// Dispatch solver based on batch and tdma type
    template <BatchType batch_type>
    void dispatchTDMASolver(TDMAType type, 
                            double* a, double* b, double* c, double* d, 
                            int n_row, int n_sys);

    template <TDMAType tdma_type, BatchType batch_type>
    void batchSolver(double* a, double* b, double* c, double* d, 
                     int n_row, int n_sys);

    /// Explicit instantiations for combinations
    extern template void batchSolver<TDMAType::Standard, BatchType::Single>(
        double*, double*, double*, double*, int, int);

    extern template void batchSolver<TDMAType::Cyclic, BatchType::Single>(
        double*, double*, double*, double*, int, int);

    extern template void batchSolver<TDMAType::Standard, BatchType::Many>(
        double*, double*, double*, double*, int, int);

    extern template void batchSolver<TDMAType::Cyclic, BatchType::Many>(
        double*, double*, double*, double*, int, int);

    extern template void batchSolver<TDMAType::Standard, BatchType::ManyRHS>(
        double*, double*, double*, double*, int, int);

    extern template void batchSolver<TDMAType::Cyclic, BatchType::ManyRHS>(
        double*, double*, double*, double*, int, int);

} // namespace PaScaL_TDMA