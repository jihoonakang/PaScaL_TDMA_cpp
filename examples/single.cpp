/**
 * @file    single.cpp
 * @brief   Example program for solving a single tridiagonal system using PaScaL_TDMA.
 * @details This example demonstrates how to create a parallel TDMA plan, scatter the input RHS data
 *          across MPI ranks, solve the system in parallel, and gather the solution back to the root
 *          process. The program also computes and prints the RMS error compared to the exact solution.
 *
 * Usage:
 *      mpirun -n <num_procs> ./single_example <n> <type:standard|cyclic>
 *      - n    : Number of unknowns (10 <= n <= 10,000 recommended)
 *      - type : TDMA solver type ("standard" or "cyclic")
 */

#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>
#include <string>
#include <stdexcept>
#include <iostream>

#include "pascal_tdma.hpp"

constexpr double a_diag  = 10.0;
constexpr double a_upper = -1.0;
constexpr double a_lower = -1.0;
constexpr int ROOT = 0;

using namespace PaScaL_TDMA;

void generateRHS(std::vector<double>& d, std::vector<double>& x, int n, TDMAType type);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    if (argc != 3) {
        throw std::runtime_error("Usage: single_example <n> <type:standard|cyclic>");
    }

    const int n = std::stoi(argv[1]);
    if (n < 10 || n > 10000) {
        throw std::runtime_error("Recommendation: 10 <= n <= 10,000");
    }

    const std::string type_str = argv[2];
    TDMAType type;
    if (type_str == "standard") {
        type = TDMAType::Standard;
    } else if (type_str == "cyclic") {
        type = TDMAType::Cyclic;
    } else {
        throw std::invalid_argument("Invalid TDMA type. Use 'standard' or 'cyclic'");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool is_root = (rank == ROOT);

    int n_sub = Util::paraRangeN(1, n, size, rank);

    std::vector<int> cnt(size), disp(size);
    MPI_Gather(&n_sub, 1, MPI_INT, cnt.data(), 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if (is_root) {
        disp[0] = 0;
        for (int i = 1; i < size; i++) disp[i] = disp[i - 1] + cnt[i - 1];
    }

    std::vector<double> d, x;
    if (is_root) generateRHS(d, x, n, type);

    std::vector<double> d_sub(n_sub, 0.0), x_sub(n_sub, 0.0);
    MPI_Scatterv(d.data(), cnt.data(), disp.data(), MPI_DOUBLE, d_sub.data(), n_sub, MPI_DOUBLE, ROOT,
                 MPI_COMM_WORLD);
    MPI_Scatterv(x.data(), cnt.data(), disp.data(), MPI_DOUBLE, x_sub.data(), n_sub, MPI_DOUBLE, ROOT,
                 MPI_COMM_WORLD);

    std::vector<double> a_sub(n_sub, a_lower), b_sub(n_sub, a_diag), c_sub(n_sub, a_upper);

    PTDMAPlanSingle plan;
    plan.create(n_sub, MPI_COMM_WORLD, type);
    PTDMASolverSingle::solve(plan, a_sub, b_sub, c_sub, d_sub);
    plan.destroy();

    MPI_Gatherv(d_sub.data(), n_sub, MPI_DOUBLE, d.data(), cnt.data(), disp.data(), MPI_DOUBLE, ROOT,
                MPI_COMM_WORLD);

    if (is_root) {
        std::vector<double> error(n);
        for (int i = 0; i < n; i++) error[i] = d[i] - x[i];
        std::cout << "Avg. RMS error = " << std::sqrt(Util::norm2(error) / n) << std::endl;
    }

    MPI_Finalize();
    return 0;
}

/**
 * @brief Generate the right-hand side (RHS) and exact solution for a tridiagonal system.
 */
void generateRHS(std::vector<double>& d, std::vector<double>& x, int n, TDMAType type) {

    std::vector<double> a(n, a_lower), b(n, a_diag), c(n, a_upper);
    x.resize(n);
    d.resize(n);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < n; i++) x[i] = dis(gen);

    if (type == TDMAType::Cyclic) {
        d[0] = a[0] * x[n - 1] + b[0] * x[0] + c[0] * x[1];
        for (int i = 1; i < n - 1; i++) d[i] = a[i] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1];
        d[n - 1] = a[n - 1] * x[n - 2] + b[n - 1] * x[n - 1] + c[n - 1] * x[0];
    } else {
        d[0] = b[0] * x[0] + c[0] * x[1];
        for (int i = 1; i < n - 1; i++) d[i] = a[i] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1];
        d[n - 1] = a[n - 1] * x[n - 2] + b[n - 1] * x[n - 1];
    }
}