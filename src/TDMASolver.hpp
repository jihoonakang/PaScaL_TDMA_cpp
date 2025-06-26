#include <vector>
#include "dimArray.hpp"

#pragma once

class TDMASolver {
public:
    // Basic Thomas Algorithm for a single tridiagonal system
    static void single(const std::vector<double>& A, 
                       const std::vector<double>& B,
                       std::vector<double>& C, 
                       std::vector<double>& D, 
                       const int n) {

        C[0] /= B[0];
        D[0] /= B[0];

        for (int i = 1; i < n; i++) {
            double m = 1.0 / (B[i] - A[i] * C[i - 1]);
            D[i] = m * (D[i] - A[i] * D[i - 1]);
            C[i] *= m;
        }

        for (int i = n - 2; i >= 0; i--) {
            D[i] -= C[i] * D[i + 1];
        }
    }

    // Thomas Algorithm for a cyclic tridiagonal system
    static void singleCyclic(const std::vector<double>& A, 
                             const std::vector<double>& B,
                             std::vector<double>& C,
                             std::vector<double>& D, 
                             const int n) {

        std::vector<double> e(n, 0);

        assert(n > 2);

        e[1] = - A[1];
        e[n - 1] = - C[n - 1];

        D[1] /= B[1];
        e[1] /= B[1];
        C[1] /= B[1];

        for (int i = 2; i < n; i++) {
            double rr = 1.0 / (B[i] - A[i] * C[i - 1]);
            D[i] = rr * (D[i] - A[i] * D[i - 1]);
            e[i] = rr * (e[i] - A[i] * e[i - 1]);
            C[i] *= rr;
        }

        for (int i = n - 2; i >= 1; i--) {
            D[i] -= C[i] * D[i + 1];
            e[i] -= C[i] * e[i + 1];
        }

        D[0] =  (D[0] - A[0] * D[n - 1] - C[0] * D[1]) / 
                (B[0] + A[0] * e[n - 1] + C[0] * e[1]);

        for (int i = 1; i < n; i++) {
            D[i] += D[0] * e[i];
        }
    }

    // Solves multiple independent tridiagonal systems stored row-wise
    static void many(const dimArray<double>& a,
                     const dimArray<double>& b,
                     dimArray<double>& c,
                     dimArray<double>& d,
                     const int n_row, 
                     const int n_sys) {
    
        for (int j = 0; j < n_sys; j++) {
            d(0, j) /= b(0, j);
            c(0, j) /= b(0, j);
        }

        double r = 0.0;
    
        for (int i = 1; i < n_row; i++) {
            for (int j = 0; j < n_sys; j++) {
                r = 1.0 / (b(i, j) - a(i, j) * c(i - 1, j));
                d(i, j) = r * (d(i, j) - a(i, j) * d(i - 1, j));
                c(i, j) = r * c(i, j);
            }
        }
    
        for (int i = n_row - 2; i >= 0; i--) {
            for (int j = 0; j < n_sys; j++) {
                d(i, j) -= c(i, j) * d(i + 1, j);
            }
        }
    }

    // Solves multiple independent cyclic tridiagonal systems
    static void manyCyclic(dimArray<double>& a,
                           dimArray<double>& b,
                           dimArray<double>& c,
                           dimArray<double>& d,
                           const int n_row, const 
                           int n_sys) {
    
        dimArray<double> e(n_row, n_sys);
        std::vector<double> rr(n_sys);

        for (int j = 0; j < n_sys; j++) {
            e(1, j) = -a(1, j);
            e(n_row - 1, j) = -c(n_row - 1, j);
        }        
    
        for (int j = 0; j < n_sys; j++) {
            d(1, j) /= b(1, j);
            e(1, j) /= b(1, j);
            c(1, j) /= b(1, j);
        }
    
        for (int i = 2; i < n_row; i++) {
            for (int j = 0; j < n_sys; j++) {
                rr[j] = 1.0 / (b(i, j) - a(i, j) * c(i - 1, j));
                d(i, j) = rr[j] * (d(i, j) - a(i, j) * d(i - 1, j));
                e(i, j) = rr[j] * (e(i, j) - a(i, j) * e(i - 1, j));
                c(i, j) = rr[j] * c(i, j);
            }
        }
    
        for (int i = n_row - 2; i >= 1; i--) {
            for (int j = 0; j < n_sys; j++) {
                d(i, j) -= c(i, j) * d(i + 1, j);
                e(i, j) -= c(i, j) * e(i + 1, j);
            }
        }
    
        for (int j = 0; j < n_sys; j++) {
            d(0, j) = (d(0, j) - a(0, j) * d(n_row - 1, j) - c(0, j) * d(1, j)) /
                      (b(0, j) + a(0, j) * e(n_row - 1, j) + c(0, j) * e(1, j));
        }
    
        for (int i = 1; i < n_row; i++) {
            for (int j = 0; j < n_sys; j++) {
                d(i, j) += d(0, j) * e(i, j);
            }
        }
    }

    // Solves multiple independent tridiagonal systems stored row-wise
    static void manyRHS(const std::vector<double>& a,
                              const std::vector<double>& b,
                              std::vector<double>& c,
                              dimArray<double>& d,
                              const int n_row, 
                              const int n_sys) {
    
        for (int j = 0; j < n_sys; j++) {
            d(0, j) /= b[0];
        }
        c[0] /= b[0];

        double r = 0.0;
    
        for (int i = 1; i < n_row; i++) {
            r = 1.0 / (b[i] - a[i] * c[i - 1]);
            for (int j = 0; j < n_sys; j++) {
                d(i, j) = r * (d(i, j) - a[i] * d(i - 1, j));
            }
            c[i] = r * c[i];
        }
    
        for (int i = n_row - 2; i >= 0; i--) {
            for (int j = 0; j < n_sys; j++) {
                d(i, j) -= c[i] * d(i + 1, j);
            }
        }
    }
    // Solves multiple independent tridiagonal systems stored row-wise
    static void manyRHSCyclic(const std::vector<double>& a,
                              const std::vector<double>& b,
                              std::vector<double>& c,
                              dimArray<double>& d,
                              const int n_row, 
                              const int n_sys) {
    
        std::vector<double> e(n_row, 0.0);

        e[1] = -a[1];
        e[n_row - 1] = -c[n_row - 1];
    
        for (int j = 0; j < n_sys; j++) {
            d(1, j) /= b[1];
        }
        e[1] /= b[1];
        c[1] /= b[1];

        for (int i = 2; i < n_row; i++) {
            double rr = 1.0 / (b[i] - a[i] * c[i - 1]);
            for (int j = 0; j < n_sys; j++) {
                d(i, j) = rr * (d(i, j) - a[i] * d(i - 1, j));
            }
            e[i] = rr * (e[i] - a[i] * e[i - 1]);
            c[i] = rr * c[i];
        }
    
        for (int i = n_row - 2; i >= 1; i--) {
            for (int j = 0; j < n_sys; j++) {
                d(i, j) -= c[i] * d(i + 1, j);
            }
            e[i] -= c[i] * e[i + 1];
        }
    
        for (int j = 0; j < n_sys; j++) {
            d(0, j) = (d(0, j) - a[0] * d(n_row - 1, j) - c[0] * d(1, j)) /
                         (b[0] + a[0] * e[n_row - 1] + c[0] * e[1]);
        }
    
        for (int i = 1; i < n_row; i++) {
            for (int j = 0; j < n_sys; j++) {
                d(i, j) += d(0, j) * e[i];
            }
        }
    }

// Pointer-based TDMASolver (For zero-copy version)

    static void single(const double* A, 
                       const double* B, 
                       double* C, 
                       double* D, 
                       const int n) {
        C[0] /= B[0];
        D[0] /= B[0];
        for (int i = 1; i < n; i++) {
            double m = 1.0 / (B[i] - A[i] * C[i - 1]);
            D[i] = m * (D[i] - A[i] * D[i - 1]);
            C[i] *= m;
        }
        for (int i = n - 2; i >= 0; --i) {
            D[i] -= C[i] * D[i + 1];
        }
    }

    // 2. Cyclic tridiagonal system
    static void singleCyclic(const double* A, 
                             const double* B, 
                             double* C, 
                             double* D, 
                             const int n) {
        assert(n > 2);
        double* e = new double[n]();

        e[1] = -A[1];
        e[n - 1] = -C[n - 1];

        D[1] /= B[1];
        e[1] /= B[1];
        C[1] /= B[1];

        for (int i = 2; i < n; i++) {
            double r = 1.0 / (B[i] - A[i] * C[i - 1]);
            D[i] = r * (D[i] - A[i] * D[i - 1]);
            e[i] = r * (e[i] - A[i] * e[i - 1]);
            C[i] *= r;
        }

        for (int i = n - 2; i >= 1; --i) {
            D[i] -= C[i] * D[i + 1];
            e[i] -= C[i] * e[i + 1];
        }

        D[0] = (D[0] - A[0] * D[n - 1] - C[0] * D[1]) /
               (B[0] + A[0] * e[n - 1] + C[0] * e[1]);

        for (int i = 1; i < n; i++) {
            D[i] += D[0] * e[i];
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
