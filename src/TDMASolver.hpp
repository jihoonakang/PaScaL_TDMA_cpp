/**
 * @file TDMASolver.hpp
 * @brief Solver routines for tridiagonal and cyclic tridiagonal systems.
 *
 * Implements the Thomas algorithm and its variants for single,
 * cyclic, and batched systems using std::vector and dimArray.
 */
#pragma once

#include <vector>
#include <cassert>
#include "dimArray.hpp"

/**
 * @class TDMASolver
 * @brief Static methods to solve tridiagonal linear systems.
 *
 * All methods perform in-place modifications on C and D arrays (or dimArray).
 */
class TDMASolver {
public:

    /**
     * @brief Solve a single tridiagonal system (Thomas algorithm).
     * @param a Lower-diagonal (size n).
     * @param b Diagonal (size n).
     * @param c Upper-diagonal (size n), overwritten with modified coefficients.
     * @param d Right-hand side (size n), overwritten with solution.
     * @param n System dimension.
     */
     static void single(const std::vector<double>& a, 
                        const std::vector<double>& b,
                        std::vector<double>& c, 
                        std::vector<double>& d, 
                        const int n) {

        single(a.data(), b.data(), c.data(), d.data(), n);
    }

    /**
     * @brief Solve a periodic tridiagonal system (cyclic Thomas algorithm).
     * @param a Lower-diagonal (size n).
     * @param b Diagonal (size n).
     * @param c Upper-diagonal (size n).
     * @param d Right-hand side (size n), overwritten with solution.
     * @param n System dimension (>2).
     */
    static void singleCyclic(const std::vector<double>& a, 
                             const std::vector<double>& b,
                             std::vector<double>& c,
                             std::vector<double>& d, 
                             const int n) {

        singleCyclic(a.data(), b.data(), c.data(), d.data(), n);
    }

    /**
     * @brief Solve multiple independent tridiagonal systems.
     * @param a Lower-diagonal (n_row × n_sys), row-major.
     * @param b Diagonal (n_row × n_sys).
     * @param c Upper-diagonal (n_row × n_sys), overwritten.
     * @param d RHS (n_row × n_sys), overwritten with solutions.
     * @param n_row Number of rows per system.
     * @param n_sys Number of independent systems.
     */
    static void many(const dimArray<double>& a,
                     const dimArray<double>& b,
                     dimArray<double>& c,
                     dimArray<double>& d,
                     const int n_row, 
                     const int n_sys) {

        many(a.getData(), b.getData(), c.getData(), d.getData(), n_row, n_sys);
    }

    /**
     * @brief Solve multiple cyclic tridiagonal systems.
     * @param a Lower-diagonal (n_row × n_sys).
     * @param b Diagonal (n_row × n_sys).
     * @param c Upper-diagonal (n_row × n_sys).
     * @param d RHS (n_row × n_sys), overwritten.
     * @param n_row Rows per system.
     * @param n_sys Number of systems.
     */
    static void manyCyclic(dimArray<double>& a,
                           dimArray<double>& b,
                           dimArray<double>& c,
                           dimArray<double>& d,
                           const int n_row, const 
                           int n_sys) {
    
        manyCyclic(a.getData(), b.getData(), c.getData(), d.getData(), 
                   n_row, n_sys);
    }

    /**
     * @brief Solve batched systems with common diagonals.
     * @param a Lower-diagonal (size n).
     * @param b Diagonal (size n).
     * @param c Upper-diagonal (size n), overwritten.
     * @param d RHS stored in dimArray (n_row × n_sys), overwritten.
     * @param n_row Number of rows.
     * @param n_sys Number of RHS vectors.
     */    
    static void manyRHS(const std::vector<double>& a,
                              const std::vector<double>& b,
                              std::vector<double>& c,
                              dimArray<double>& d,
                              const int n_row, 
                              const int n_sys) {

        manyRHS(a.data(), b.data(), c.data(), d.getData(), n_row, n_sys);
    }

    /**
     * @brief Solve batched cyclic systems with common diagonals.
     * @param a Lower-diagonal (size n).
     * @param b Diagonal (size n).
     * @param c Upper-diagonal (size n), overwritten.
     * @param d RHS stored in dimArray (n_row × n_sys), overwritten.
     * @param n_row Number of rows.
     * @param n_sys Number of RHS vectors.
     */
    static void manyRHSCyclic(const std::vector<double>& a,
                              const std::vector<double>& b,
                              std::vector<double>& c,
                              dimArray<double>& d,
                              const int n_row, 
                              const int n_sys) {
    
        manyRHSCyclic(a.data(), b.data(), c.data(), d.getData(), n_row, n_sys);
    }

// Pointer-based TDMASolver (For zero-copy version)

    // 1. Single tridiagonal system
    static void single(const double* a, 
                       const double* b, 
                       double* c, 
                       double* d, 
                       const int n) {
        c[0] /= b[0];
        d[0] /= b[0];
        for (int i = 1; i < n; i++) {
            double m = 1.0 / (b[i] - a[i] * c[i - 1]);
            d[i] = m * (d[i] - a[i] * d[i - 1]);
            c[i] *= m;
        }
        for (int i = n - 2; i >= 0; --i) {
            d[i] -= c[i] * d[i + 1];
        }
    }

    // 2. Cyclic tridiagonal system
    static void singleCyclic(const double* a, 
                             const double* b, 
                             double* c, 
                             double* d, 
                             const int n) {
        assert(n > 2);
        double* e = new double[n]();

        e[1] = -a[1];
        e[n - 1] = -c[n - 1];

        d[1] /= b[1];
        e[1] /= b[1];
        c[1] /= b[1];

        for (int i = 2; i < n; i++) {
            double r = 1.0 / (b[i] - a[i] * c[i - 1]);
            d[i] = r * (d[i] - a[i] * d[i - 1]);
            e[i] = r * (e[i] - a[i] * e[i - 1]);
            c[i] *= r;
        }

        for (int i = n - 2; i >= 1; --i) {
            d[i] -= c[i] * d[i + 1];
            e[i] -= c[i] * e[i + 1];
        }

        d[0] = (d[0] - a[0] * d[n - 1] - c[0] * d[1]) /
               (b[0] + a[0] * e[n - 1] + c[0] * e[1]);

        for (int i = 1; i < n; i++) {
            d[i] += d[0] * e[i];
        }

        delete[] e;
    }

    // 3. Multiple systems (2D: n_row x n_sys)
    static void many(const double* a, 
                     const double* b, 
                     double* c, 
                     double* d, 
                     const int n_row, 
                     const int n_sys) {

        for (int j = 0; j < n_sys; j++) {
            d[j] /= b[j];
            c[j] /= b[j];
        }

        for (int i = 1; i < n_row; i++) {
            int idx = i * n_sys;
            int idx_prev = idx - n_sys;
            for (int j = 0; j < n_sys; j++) {
                double r = 1.0 / (b[idx] - a[idx] * c[idx_prev]);
                d[idx] = r * (d[idx] - a[idx] * d[idx_prev]);
                c[idx] = r * c[idx];
                idx++;
                idx_prev++;
            }
        }

        for (int i = n_row - 2; i >= 0; --i) {
            int idx = i * n_sys;
            int idx_next = idx + n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] -= c[idx] * d[idx_next];
                idx++;
                idx_next++;
            }
        }
    }

    // 4. Multiple cyclic systems (2D: n_row x n_sys)
    static void manyCyclic(const double* a, 
                           const double* b, 
                           double* c, 
                           double* d, 
                           const int n_row, 
                           const int n_sys) {

        double* e = new double[n_row * n_sys]();
        double* rr = new double[n_sys];

        for (int j = 0; j < n_sys; j++) {
            e[n_sys + j] = -a[n_sys + j];
            e[(n_row - 1) * n_sys + j] = -c[(n_row - 1) * n_sys + j];
        }

        for (int j = 0; j < n_sys; j++) {
            d[n_sys + j] /= b[n_sys + j];
            e[n_sys + j] /= b[n_sys + j];
            c[n_sys + j] /= b[n_sys + j];
        }

        for (int i = 2; i < n_row; i++) {
            int idx = i * n_sys;
            int idx_prev = idx - n_sys;
            for (int j = 0; j < n_sys; j++) {
                rr[j] = 1.0 / (b[idx] - a[idx] * c[idx_prev]);
                d[idx] = rr[j] * (d[idx] - a[idx] * d[idx_prev]);
                e[idx] = rr[j] * (e[idx] - a[idx] * e[idx_prev]);
                c[idx] *= rr[j];
                idx++;
                idx_prev++;
            }
        }

        for (int i = n_row - 2; i >= 1; --i) {
            int idx = i * n_sys;
            int idx_next = idx + n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] -= c[idx] * d[idx_next];
                e[idx] -= c[idx] * e[idx_next];
                idx++;
                idx_next++;
            }
        }

        for (int j = 0; j < n_sys; j++) {
            d[j] = (d[j] - a[j] * d[(n_row - 1) * n_sys + j] - c[j] * d[j + n_sys]) /
                   (b[j] + a[j] * e[(n_row - 1) * n_sys + j] + c[j] * e[j + n_sys]);
        }

        for (int i = 1; i < n_row; i++) {
            int idx = i * n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] += d[0 * n_sys + j] * e[idx];
                idx++;
            }
        }

        delete[] e;
        delete[] rr;
    }

    // Solves multiple independent tridiagonal systems stored row-wise
    static void manyRHS(const double* a,
                        const double* b,
                        double* c,
                        double* d,
                        const int n_row,
                        const int n_sys) {

        for (int j = 0; j < n_sys; j++) {
            d[j] /= b[0];
        }
        c[0] /= b[0];

        for (int i = 1; i < n_row; i++) {
            double r = 1.0 / (b[i] - a[i] * c[i - 1]);
            int idx = i * n_sys;
            int prev = idx - n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] = r * (d[idx] - a[i] * d[prev]);
                idx++;
                prev++;
            }
            c[i] = r * c[i];
        }

        for (int i = n_row - 2; i >= 0; i--) {
            int idx = i * n_sys;
            int next = idx + n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] -= c[i] * d[next];
                idx++;
                next++;
            }
        }
    }

    // Solves multiple independent tridiagonal systems stored row-wise
    static void manyRHSCyclic(const double* a,
                              const double* b,
                              double* c,
                              double* d,
                              const int n_row,
                              const int n_sys) {

        double* e = new double[n_row];
        std::fill(e, e + n_row, 0.0);

        e[1] = -a[1];
        e[n_row - 1] = -c[n_row - 1];

        for (int j = 0; j < n_sys; j++) {
            d[n_sys + j] /= b[1];
        }
        e[1] /= b[1];
        c[1] /= b[1];

        for (int i = 2; i < n_row; i++) {
            double rr = 1.0 / (b[i] - a[i] * c[i - 1]);
            int idx = i * n_sys;
            int prev = idx - n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] = rr * (d[idx] - a[i] * d[prev]);
                idx++;
                prev++;
            }
            e[i] = rr * (e[i] - a[i] * e[i - 1]);
            c[i] = rr * c[i];
        }

        for (int i = n_row - 2; i >= 1; i--) {
            int idx = i * n_sys;
            int next = idx + n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] -= c[i] * d[next];
                idx++;
                next++;
            }
            e[i] -= c[i] * e[i + 1];
        }

        for (int j = 0; j < n_sys; j++) {
            d[j] = (d[j] - a[0] * d[(n_row - 1) * n_sys + j] - c[0] * d[n_sys + j]) /
                   (b[0] + a[0] * e[n_row - 1] + c[0] * e[1]);
        }

        for (int i = 1; i < n_row; i++) {
            int idx = i * n_sys;
            for (int j = 0; j < n_sys; j++) {
                d[idx] += d[j] * e[i];
                idx++;
            }
        }

        delete[] e;
    }

};
