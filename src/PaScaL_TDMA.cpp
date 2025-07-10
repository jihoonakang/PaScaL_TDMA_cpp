/**
 * @file PaScaL_TDMA.cpp
 * @brief Implementation of planner and solver dispatch for PaScaL_TDMA library.
 *
 * Contains definitions of PTDMAPlanBase, PTDMAPlanSingle/Many/ManyRHS,
 * solver methods, and batch solver templates.
 */

#include <vector>
#include <mpi.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include "PaScaL_TDMA.hpp"
#include "TDMASolver.hpp"

namespace PaScaL_TDMA {

    /**
     * @brief Default create throws; must be overridden by derived plans.
     */
    void PTDMAPlanBase::create(int n_row_, MPI_Comm comm_ptdma_, TDMAType type_) {
        throw std::runtime_error("create(int, MPI_Comm, TDMAType) not implemented");
    }

    void PTDMAPlanBase::create(int n_row_, int n_sys_, MPI_Comm comm_ptdma_, TDMAType type_) {
        throw std::runtime_error("create(int, int, MPI_Comm, TDMAType) not implemented");
    }

    /**
     * @brief Create plan for single system across ranks.
     */
    void PTDMAPlanSingle::create(int n_row, MPI_Comm comm_ptdma, TDMAType type) {

        n_row_ = n_row;
        MPI_Comm_dup(comm_ptdma, &comm_ptdma_);
        MPI_Comm_size(comm_ptdma_, &size_);
        MPI_Comm_rank(comm_ptdma_, &rank_);
        type_ = type;

        // two boundary rows
        int n_row_rd = 2;
        n_row_rt_ = n_row_rd * size_;

        // resize buffers for reduced and transposed system
        a_rd_.resize(n_row_rd);
        b_rd_.assign(n_row_rd, 1.0);
        c_rd_.resize(n_row_rd);
        d_rd_.resize(n_row_rd);

        a_rt_.resize(n_row_rt_);
        b_rt_.assign(n_row_rt_, 1.0);
        c_rt_.resize(n_row_rt_);
        d_rt_.resize(n_row_rt_);
    }

    /**
     * @brief Destroy plan, clear buffers.
     */
    void PTDMAPlanSingle::destroy() {

        a_rd_.clear();
        b_rd_.clear();
        c_rd_.clear();
        d_rd_.clear();

        a_rt_.clear();
        b_rt_.clear();
        c_rt_.clear();
        d_rt_.clear();
    }

    /**
     * @brief Forward elimination for single-system TDMA.
     */
    static void forwardSingle(double* a, const double* b, double* c, double* d, 
                              int n_row) {

        double r;

        a[0] /= b[0]; d[0] /= b[0]; c[0] /= b[0];
        a[1] /= b[1]; d[1] /= b[1]; c[1] /= b[1];

        for (int i = 2; i < n_row; i++) {
            r = 1.0 / (b[i] - a[i] * c[i - 1]);
            d[i] = r * (d[i] - a[i] * d[i - 1]);
            c[i] = r * c[i];
            a[i] = -r * a[i] * a[i - 1];
        }
    }

    /**
     * @brief Backward substitution for single-system TDMA.
     */
    static void backwardSingle(double* a, double* c, double* d, int n_row) {

        for (int i = n_row - 3; i >= 1; i--) {
            d[i] -= c[i] * d[i + 1];
            a[i] -= c[i] * a[i + 1];
            c[i] = -c[i] * c[i + 1];
        }

        double r = 1.0 / (1.0 - a[1] * c[0]);
        d[0] = r * (d[0] - c[0] * d[1]);
        a[0] = r * a[0];
        c[0] = -r * c[0] * c[1];
    }

    /**
     * @brief Solve single-system TDMA using plan metadata.
     */
    void PTDMASolverSingle::solve(PTDMAPlanSingle& plan, 
                                  double* a, 
                                  double* b, 
                                  double* c,
                                  double* d) {

        const int n_row = plan.n_row_;
        
        if (n_row <= 2) {
            std::cerr << "Error: n_row must be > 2 in rank : " << plan.rank_ << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // single rank: direct dispatch
        if (plan.size_ == 1) {
            dispatchTDMASolver<BatchType::Single>(plan.type_, a, b, c, d, n_row, 1);
            return;
        }

        // parallel: local elimination
        forwardSingle(a, b, c, d, n_row);
        backwardSingle(a, c, d, n_row);

        // gather boundary data
        plan.a_rd_[0] = a[0]; plan.a_rd_[1] = a[n_row - 1];
        plan.c_rd_[0] = c[0]; plan.c_rd_[1] = c[n_row - 1];
        plan.d_rd_[0] = d[0]; plan.d_rd_[1] = d[n_row - 1];

        MPI_Gather(plan.a_rd_.data(), 2, MPI_DOUBLE, 
                   plan.a_rt_.data(), 2, MPI_DOUBLE, 
                   plan.root_rank_, plan.comm_ptdma_);

        MPI_Gather(plan.c_rd_.data(), 2, MPI_DOUBLE, 
                   plan.c_rt_.data(), 2, MPI_DOUBLE, 
                   plan.root_rank_, plan.comm_ptdma_);

        MPI_Gather(plan.d_rd_.data(), 2, MPI_DOUBLE, 
                   plan.d_rt_.data(), 2, MPI_DOUBLE, 
                   plan.root_rank_, plan.comm_ptdma_);

        if (plan.rank_ == plan.root_rank_) {
            dispatchTDMASolver<BatchType::Single>(plan.type_, 
                    plan.a_rt_.data(), plan.b_rt_.data(), 
                    plan.c_rt_.data(), plan.d_rt_.data(), plan.n_row_rt_, 1);
        }

        MPI_Scatter(plan.d_rt_.data(), 2, MPI_DOUBLE, 
                    plan.d_rd_.data(), 2, MPI_DOUBLE, 
                    plan.root_rank_, plan.comm_ptdma_);

        // reconstruct solution
        d[0] = plan.d_rd_[0];
        d[n_row - 1] = plan.d_rd_[1];

        for (int i = 1; i < n_row - 1; i++) {
            d[i] = d[i] - a[i] * d[0] - c[i] * d[n_row - 1];
        }
        return;
    }

    /**
     * @brief Create plan for many system across ranks.
     */
    void PTDMAPlanMany::create(int n_row, int n_sys, MPI_Comm comm_ptdma, 
                               TDMAType type) {
        
        n_row_ = n_row;
        n_sys_ = n_sys;
        MPI_Comm_dup(comm_ptdma, &comm_ptdma_);
        MPI_Comm_size(comm_ptdma_, &size_);
        MPI_Comm_rank(comm_ptdma_, &rank_);
        type_ = type;

        // We only need the first and last row for the reduced system
        const int n_sys_rd = n_sys_;
        const int n_row_rd = 2;
    
        n_sys_rt_ = Util::para_range_n(1, n_sys_rd, size_, rank_);
        n_row_rt_ = n_row_rd * size_;
    
        // Gather all local n_sys_rt_ to build subarray types
        std::vector<int> n_sys_rt_array(size_);
        MPI_Allgather(&n_sys_rt_, 1, MPI_INT, n_sys_rt_array.data(), 1, MPI_INT,
                      comm_ptdma_);
        
        // Resize buffers for reduced system (two rows × n_sys)
        a_rd_.resize(n_row_rd, n_sys_rd);
        b_rd_.assign(n_row_rd, n_sys_rd, 1.0);
        c_rd_.resize(n_row_rd, n_sys_rd);
        d_rd_.resize(n_row_rd, n_sys_rd);
    
        // Resize buffers for transposed system (n_row_rt × n_sys_rt)
        a_rt_.resize(n_row_rt_, n_sys_rt_);
        b_rt_.assign(n_row_rt_, n_sys_rt_, 1.0);
        c_rt_.resize(n_row_rt_, n_sys_rt_);
        d_rt_.resize(n_row_rt_, n_sys_rt_);
    
        // Build noncontiguous MPI types for the forward/backward exchange
        ddtype_rd_.resize(size_);
        ddtype_rt_.resize(size_);
    
        for (int i = 0; i < size_; ++i) {
            int bigsize[2] = {n_row_rd, n_sys_rd};
            int subsize[2] = {n_row_rd, n_sys_rt_array[i]};
            int start[2] = {0, std::accumulate(n_sys_rt_array.begin(), 
                            n_sys_rt_array.begin() + i, 0)};
    
            MPI_Type_create_subarray(2, bigsize, subsize, start, 
                                     MPI_ORDER_C, MPI_DOUBLE, &ddtype_rd_[i]);
            MPI_Type_commit(&ddtype_rd_[i]);
    
            int rstart[2] = {n_row_rd * i, 0};
            int rsub[2] = {n_row_rd, n_sys_rt_};
            int rbig[2] = {n_row_rt_, n_sys_rt_};
    
            MPI_Type_create_subarray(2, rbig, rsub, rstart, 
                                     MPI_ORDER_C, MPI_DOUBLE, &ddtype_rt_[i]);
            MPI_Type_commit(&ddtype_rt_[i]);
        }
    
        // Simple 1-to-1 counts/displacements for Ialltoallw
        count_send_.assign(size_, 1);
        displ_send_.assign(size_, 0);
        count_recv_.assign(size_, 1);
        displ_recv_.assign(size_, 0);

        return;
    }

    /**
     * @brief Destroy plan for many systems.
     */
    void PTDMAPlanMany::destroy() {
        for (auto &t : ddtype_rd_) MPI_Type_free(&t);
        for (auto &t : ddtype_rt_) MPI_Type_free(&t);
    
        ddtype_rd_.clear();
        ddtype_rt_.clear();
        count_send_.clear();
        displ_send_.clear();
        count_recv_.clear();
        displ_recv_.clear();
    
        a_rd_.clear(); b_rd_.clear(); c_rd_.clear(); d_rd_.clear();
        a_rt_.clear(); b_rt_.clear(); c_rt_.clear(); d_rt_.clear();

        return;
    }    

    /**
     * @brief Forward elimination for batched TDMA (many systems).
     */
    static void forwardMany(double* a, const double* b, 
                            double* c, double* d,
                            int n_row, int n_sys) {

        // Normalize the first two boundary rows
        for (int j = 0; j < n_sys; j++) {
            for (int i = 0; i < 2; i++) {
                int idx = i * n_sys + j;
                double inv = 1.0 / b[idx];
                a[idx] *= inv;
                c[idx] *= inv;
                d[idx] *= inv;
            }
            // a[j] /= b[j];d[j] /= b[j]; c[j] /= b[j];
            // a[j + n_sys] /= b[j + n_sys]; 
            // d[j + n_sys] /= b[j + n_sys];
            // c[j + n_sys] /= b[j + n_sys];
        }

        // Eliminate interior rows
        for (int i = 2; i < n_row; ++i) {
            int idx = i * n_sys;
            int idx_prev = idx - n_sys;
            for (int j = 0; j < n_sys; ++j) {
                double r = 1.0 / (b[idx] - a[idx] * c[idx_prev]);
                d[idx] = r * (d[idx] - a[idx] * d[idx_prev]);
                c[idx] = r * c[idx];
                a[idx] = -r * a[idx] * a[idx_prev];
                ++idx; ++idx_prev;
            }
        }
    }

    /**
     * @brief Backward substitution for batched TDMA (many systems).
     */
    static void backwardMany(double* a, double* c, double* d,
                             int n_row, int n_sys) {

        for (int i = n_row - 3; i >= 1; --i) {
            int idx = i * n_sys;
            int idx_next = idx + n_sys;
            for (int j = 0; j < n_sys; ++j) {
                d[idx] -= c[idx] * d[idx_next];
                a[idx] -= c[idx] * a[idx_next];
                c[idx] = -c[idx] * c[idx_next];
                ++idx; ++idx_next;
            }
        }
    }

    /**
    * @brief Reduce boundary data into two vectors (D0/D1, A0/A1, C0/C1).
    */
    static void reduceMany(const double* a, const double* c, const double* d,
                           double* a0, double* a1, 
                           double* c0, double* c1,
                           double* d0, double* d1,
                           int n_row, int n_sys) {

        for (int j = 0; j < n_sys; ++j) {
            double r = 1.0 / (1.0 - a[j + n_sys] * c[j]);
            d0[j] = r * (d[j] - c[j] * d[j + n_sys]);
            a0[j] = r * a[j];
            c0[j] = -r * c[j] * c[j + n_sys];
            a1[j] = a[(n_row - 1) * n_sys + j];
            c1[j] = c[(n_row - 1) * n_sys + j];
            d1[j] = d[(n_row - 1) * n_sys + j];
        }
    }
    /**
    * @brief Reconstruct the full solution from reduced boundary data.
    */
    static void reconstructMany(const double* a, const double* c, double* d,
                                const double* d0, const double* d1,
                                int n_row, int n_sys) {

        for (int j = 0; j < n_sys; ++j) {
            d[0 * n_sys + j]           = d0[j];
            d[(n_row - 1) * n_sys + j] = d1[j];
        }

        for (int i = 1; i < n_row - 1; ++i) {
            int idx = i * n_sys;
            for (int j = 0; j < n_sys; ++j) {
                d[idx] = d[idx] - a[idx] * d0[j] - c[idx] * d1[j];
                ++idx;
            }
        }
    }

    /**
    * @brief Solve many independent tridiagonal systems using plan metadata.
    */
    void PTDMASolverMany::solve(PTDMAPlanMany& plan,
                                double* a,double* b, double* c, double* d) {

        const int n_row = plan.n_row_;
        const int n_sys = plan.n_sys_;

        assert(n_row > 2);

        if (plan.size_ == 1) {
            dispatchTDMASolver<BatchType::Many>(plan.type_, a, b, c, d, n_row, n_sys);
            return;
        }

        // local elimination
        forwardMany(a, b, c, d, n_row, n_sys);
        backwardMany(a, c, d, n_row, n_sys);

        // reduce to two-row system
        reduceMany(a, c, d,
                   plan.a_rd_.getData(), plan.a_rd_.getData() + n_sys,
                   plan.c_rd_.getData(), plan.c_rd_.getData() + n_sys,
                   plan.d_rd_.getData(), plan.d_rd_.getData() + n_sys,
                   n_row, n_sys);

        // transpose across ranks (non-blocking)
        MPI_Request request[3];
        MPI_Status statuses[3];

        MPI_Ialltoallw(plan.a_rd_.getData(), plan.count_send_.data(), 
                       plan.displ_send_.data(), plan.ddtype_rd_.data(),
                       plan.a_rt_.getData(), plan.count_recv_.data(), 
                       plan.displ_recv_.data(), plan.ddtype_rt_.data(),
                       plan.comm_ptdma_, &request[0]);

        MPI_Ialltoallw(plan.c_rd_.getData(), plan.count_send_.data(), 
                       plan.displ_send_.data(), plan.ddtype_rd_.data(),
                       plan.c_rt_.getData(), plan.count_recv_.data(), 
                       plan.displ_recv_.data(), plan.ddtype_rt_.data(),
                       plan.comm_ptdma_, &request[1]);

        MPI_Ialltoallw(plan.d_rd_.getData(), plan.count_send_.data(), 
                       plan.displ_send_.data(), plan.ddtype_rd_.data(),
                       plan.d_rt_.getData(), plan.count_recv_.data(), 
                       plan.displ_recv_.data(), plan.ddtype_rt_.data(),
                       plan.comm_ptdma_, &request[2]);

        MPI_Waitall(3, request, statuses);

        // Solve the reduced tridiagonal systems
        dispatchTDMASolver<BatchType::Many>(plan.type_, plan.a_rt_.getData(), 
            plan.b_rt_.getData(), plan.c_rt_.getData(), plan.d_rt_.getData(), 
            plan.n_row_rt_, plan.n_sys_rt_);

        // transpose back and reconstruct
        MPI_Ialltoallw(plan.d_rt_.getData(), plan.count_recv_.data(), 
                       plan.displ_recv_.data(), plan.ddtype_rt_.data(),
                       plan.d_rd_.getData(), plan.count_send_.data(), 
                       plan.displ_send_.data(), plan.ddtype_rd_.data(),
                       plan.comm_ptdma_, &request[0]);

        MPI_Waitall(1, request, statuses);

        reconstructMany(a, c, d, plan.d_rd_.getData(), 
                        plan.d_rd_.getData() + n_sys, n_row, n_sys);
    }

    /**
    * @brief Create plan for many-RHS TDMA with shared diagonals.
    */
    void PTDMAPlanManyRHS::create(int n_row, int n_sys, MPI_Comm comm_ptdma, 
                                  TDMAType type) {
        n_row_ = n_row;
        n_sys_ = n_sys;
        type_ = type;

        MPI_Comm_dup(comm_ptdma, &comm_ptdma_);
        MPI_Comm_size(comm_ptdma, &size_);
        MPI_Comm_rank(comm_ptdma, &rank_);

        int n_sys_rd = n_sys_;
        int n_row_rd = 2;
        std::vector<int> n_sys_rt_array(size_);

        // Compute local and global problem dimensions
        n_sys_rt_ = Util::para_range_n(1, n_sys_rd, size_, rank_);
        n_row_rt_ = n_row_rd * size_;

        MPI_Allgather(&n_sys_rt_, 1, MPI_INT, 
                      n_sys_rt_array.data(), 1, MPI_INT, comm_ptdma_);
       
        a_rd_.resize(n_row_rd);
        b_rd_.assign(n_row_rd, 1);
        c_rd_.resize(n_row_rd);
        d_rd_.resize(n_row_rd, n_sys_rd);

        a_rt_.resize(n_row_rt_);
        b_rt_.assign(n_row_rt_, 1);
        c_rt_.resize(n_row_rt_);
        d_rt_.resize(n_row_rt_, n_sys_rt_);

        ddtype_rd_.resize(size_);
        ddtype_rt_.resize(size_);

        for (int i = 0; i < size_; ++i) {
            int bigsize[2] = {n_row_rd, n_sys_rd};
            int subsize[2] = {n_row_rd, n_sys_rt_array[i]};
            int start[2] = {0, std::accumulate(n_sys_rt_array.begin(), 
                            n_sys_rt_array.begin() + i, 0)};

            MPI_Type_create_subarray(2, bigsize, subsize, start, 
                                     MPI_ORDER_C, MPI_DOUBLE, &ddtype_rd_[i]);
            MPI_Type_commit(&ddtype_rd_[i]);

            int rstart[2] = {n_row_rd * i, 0};
            int rsub[2] = {n_row_rd, n_sys_rt_};
            int rbig[2] = {n_row_rt_, n_sys_rt_};

            MPI_Type_create_subarray(2, rbig, rsub, rstart, 
                                     MPI_ORDER_C, MPI_DOUBLE, &ddtype_rt_[i]);
            MPI_Type_commit(&ddtype_rt_[i]);
        }

        count_send_.assign(size_, 1);
        displ_send_.assign(size_, 0);
        count_recv_.assign(size_, 1);
        displ_recv_.assign(size_, 0);
        return;
    }


    /**
    * @brief Destroy ManyRHS plan, free resources.
    */
    void PTDMAPlanManyRHS::destroy() {
        for (auto &t : ddtype_rd_) MPI_Type_free(&t);
        for (auto &t : ddtype_rt_) MPI_Type_free(&t);

        ddtype_rd_.clear();
        ddtype_rt_.clear();
        count_send_.clear();
        displ_send_.clear();
        count_recv_.clear();
        displ_recv_.clear();

        a_rd_.clear(); b_rd_.clear(); c_rd_.clear(); d_rd_.clear();
        a_rt_.clear(); b_rt_.clear(); c_rt_.clear(); d_rt_.clear();
        return;
    }    

    static void forwardManyRHS(double* a, const double* b, double* c, double* d,
                               const int n_row, const int n_sys) {
        a[0] /= b[0];
        c[0] /= b[0];
        a[1] /= b[1];
        c[1] /= b[1];

        for (int j = 0; j < n_sys; j++) {
            d[j] /= b[0];
            d[n_sys + j] /= b[1];
        }

        for (int i = 2; i < n_row; i++) {
            double r = 1.0 / (b[i] - a[i] * c[i - 1]);
            int idx = i * n_sys;
            int idx_prev = idx - n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] = r * (d[idx] - a[i] * d[idx_prev]);
                ++idx; ++idx_prev;
            }
            c[i] = r * c[i];
            a[i] = -r * a[i] * a[i - 1];
        }
    }

    static void backwardManyRHS(double* a, double* c, double* d,
                                const int n_row, const int n_sys) {
        for (int i = n_row - 3; i >= 1; i--) {
            int idx = i * n_sys;
            int idx_next = idx + n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] = d[idx] - c[i] * d[idx_next];
                ++idx; ++idx_next;
            }
            a[i] = a[i] - c[i] * a[i + 1];
            c[i] = -c[i] * c[i + 1];
        }
    }

    static void reduceManyRHS(const double* a, const double* c, double* d,
                              double* a0, double* a1, double* c0, double* c1,
                              double* d0, double* d1,
                              const int n_row, const int n_sys) {
        double r = 1.0 / (1.0 - a[1] * c[0]);
        for (int j = 0; j < n_sys; ++j) {
            d0[j] = r * (d[j] - c[0] * d[n_sys + j]);
            d1[j] = d[(n_row - 1) * n_sys + j];
        }
        a0[0] = r * a[0];
        c0[0] = -r * c[0] * c[1];
        a1[0] = a[n_row - 1];
        c1[0] = c[n_row - 1];
    }

    static void reconstructManyRHS(const double* a, const double* c, double* d,
                                   const double* d0, const double* d1,
                                   const int n_row, const int n_sys) {
        for (int j = 0; j < n_sys; ++j) {
            d[j] = d0[j];
            d[(n_row - 1) * n_sys + j] = d1[j];
        }

        for (int i = 1; i < n_row - 1; ++i) {
            int idx = i * n_sys;
            for (int j = 0; j < n_sys; ++j) {
                d[idx] = d[idx] - a[i] * d0[j] - c[i] * d1[j];
                ++idx;
            }
        }
    }

    /**
    * @brief Solve many-RHS TDMA using plan metadata.
    */
    void PTDMASolverManyRHS::solve(PTDMAPlanManyRHS& plan,
                                   double* a, double* b, double* c, double* d) {

        const int n_row = plan.n_row_;
        const int n_sys = plan.n_sys_;

        assert(n_row > 2);

        if (plan.size_ == 1) {
            dispatchTDMASolver<BatchType::ManyRHS>(plan.type_, a, b, c, d, n_row, n_sys);
            return;
        }

        // Forward/backward on each RHS
        forwardManyRHS(a, b, c, d, n_row, n_sys);
        backwardManyRHS(a, c, d, n_row, n_sys);

        // Reduce to boundary vectors
        reduceManyRHS(a, c, d,
                      plan.a_rd_.data(), plan.a_rd_.data() + 1,
                      plan.c_rd_.data(), plan.c_rd_.data() + 1,
                      plan.d_rd_.getData(), plan.d_rd_.getData() + n_sys,
                      n_row, n_sys);

        // Transpose the reduced system using MPI_Ialltoallw
        MPI_Request request[3];
        MPI_Status statuses[3];

        MPI_Iallgather(plan.a_rd_.data(), 2, MPI_DOUBLE, 
                       plan.a_rt_.data(), 2, MPI_DOUBLE,
                       plan.comm_ptdma_, &request[0]);

        MPI_Iallgather(plan.c_rd_.data(), 2, MPI_DOUBLE, 
                       plan.c_rt_.data(), 2, MPI_DOUBLE,
                       plan.comm_ptdma_, &request[1]);

        MPI_Ialltoallw(plan.d_rd_.getData(), plan.count_send_.data(), 
                       plan.displ_send_.data(), plan.ddtype_rd_.data(),
                       plan.d_rt_.getData(), plan.count_recv_.data(), 
                       plan.displ_recv_.data(), plan.ddtype_rt_.data(),
                       plan.comm_ptdma_, &request[2]);

        MPI_Waitall(3, request, statuses);

        // Solve the reduced tridiagonal systems
        dispatchTDMASolver<BatchType::ManyRHS>(plan.type_, plan.a_rt_.data(), 
            plan.b_rt_.data(), plan.c_rt_.data(), plan.d_rt_.getData(), 
            plan.n_row_rt_, plan.n_sys_rt_);

        // Transpose solutions back
        MPI_Ialltoallw( plan.d_rt_.getData(), plan.count_recv_.data(), 
                        plan.displ_recv_.data(), plan.ddtype_rt_.data(),
                        plan.d_rd_.getData(), plan.count_send_.data(), 
                        plan.displ_send_.data(), plan.ddtype_rd_.data(),
                        plan.comm_ptdma_, &request[0]);
        MPI_Waitall(1, request, statuses);

        // Update full solution from reduced solution
        reconstructManyRHS(a, c, d,
                           plan.d_rd_.getData(), 
                           plan.d_rd_.getData() + n_sys,
                           n_row, n_sys);
    }

    template <TDMAType tdma_type, BatchType batch_type>
    void batchSolver(double* a, double* b, double* c, double* d, int n_row, int n_sys) {
        if constexpr (batch_type == BatchType::Single) {
            if constexpr (tdma_type == TDMAType::Standard)
                TDMASolver::single(a, b, c, d, n_row);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                TDMASolver::singleCyclic(a, b, c, d, n_row);
        }
        else if constexpr (batch_type == BatchType::Many) {
            if constexpr (tdma_type == TDMAType::Standard)
                TDMASolver::many(a, b, c, d, n_row, n_sys);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                TDMASolver::manyCyclic(a, b, c, d, n_row, n_sys);
        }
        else if constexpr (batch_type == BatchType::ManyRHS) {
            if constexpr (tdma_type == TDMAType::Standard)
                TDMASolver::manyRHS(a, b, c, d, n_row, n_sys);
            else if constexpr (tdma_type == TDMAType::Cyclic)
                TDMASolver::manyRHSCyclic(a, b, c, d, n_row, n_sys);
        }
    }

    template <BatchType batch_type>
    void dispatchTDMASolver(TDMAType type, double* a, double* b, double* c, double* d, 
                            int n_row, int n_sys) {
        switch (type) {
            case TDMAType::Standard:
                batchSolver<TDMAType::Standard, batch_type>(a, b, c, d, n_row, n_sys);
                break;
            case TDMAType::Cyclic:
                batchSolver<TDMAType::Cyclic, batch_type>(a, b, c, d, n_row, n_sys);
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
