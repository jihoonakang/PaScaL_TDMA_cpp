/**
 * @file    many.cpp
 * @brief   Example program for solving multiple tridiagonal systems in 2D using PaScaL_TDMA.
 * @details This example demonstrates solving tridiagonal systems in both x and y directions using
 *          a 2D MPI process topology. The program distributes RHS data, performs two sequential TDMA
 *          solves (x- then y-direction), and collects the solution at the root rank for error analysis.
 *
 * Usage:
 *      mpirun -n <num_procs> ./many_example <nx> <ny>
 *      - nx : Number of unknowns in x-direction (10 ≤ nx ≤ 10,000)
 *      - ny : Number of unknowns in y-direction (10 ≤ ny ≤ 10,000)
 */

#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>
#include <string>
#include <stdexcept>
#include <iostream>

#include "pascal_tdma.hpp"
#include "comm_layout_2d.hpp"

constexpr double a_diag = 10.0;
constexpr double a_upper = -1.0;
constexpr double a_lower = -1.0;

using namespace PaScaL_TDMA;

void generate_rhs(DimArray<double>& d, DimArray<double>& x, int nx, int ny);
void distribute_rhs(DimArray<double>& d_sub, const DimArray<double>& d, const DomainLayout2D& dom,
                    const CommLayout2D& topo);
void collect_solution(DimArray<double>& d, const DimArray<double>& d_sub, const DomainLayout2D& dom,
                      const CommLayout2D& topo);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    if (argc != 3) throw std::runtime_error("Usage: many_example <nx> <ny>");

    const int nx = std::stoi(argv[1]);
    const int ny = std::stoi(argv[2]);
    if (nx < 10 || nx > 10000) throw std::runtime_error("Recommendation: 10 <= nx <= 10,000");
    if (ny < 10 || ny > 10000) throw std::runtime_error("Recommendation: 10 <= ny <= 10,000");

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    bool is_root = (rank == 0);

    int dims[2] = {0, 0};
    int period[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    CommLayout2D topo(dims, period);
    DomainLayout2D dom(nx, ny, topo);

    const int nx_sub = dom.getParDimX();
    const int ny_sub = dom.getParDimY();
    topo.buildCommBufferInfo(nx_sub, ny_sub);

    DimArray<double> d, x;
    if (is_root) generate_rhs(d, x, nx, ny);

    DimArray<double> d_sub(nx_sub, ny_sub), x_sub(nx_sub, ny_sub);
    distribute_rhs(d_sub, d, dom, topo);

    // Solve in x-direction
    DimArray<double> a_sub, b_sub, c_sub;
    a_sub.assign(nx_sub, ny_sub, a_lower);
    b_sub.assign(nx_sub, ny_sub, a_diag);
    c_sub.assign(nx_sub, ny_sub, a_upper);

    PTDMAPlanMany px_many;
    px_many.create(nx_sub, ny_sub, topo.getCommX(), TDMAType::Standard);
    PTDMASolverMany::solve(px_many, a_sub, b_sub, c_sub, d_sub);
    px_many.destroy();

    // Solve in y-direction
    DimArray<double> d_sub_tr(ny_sub, nx_sub);
    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++) d_sub_tr(j, i) = d_sub(i, j);

    a_sub.assign(ny_sub, nx_sub, a_lower);
    b_sub.assign(ny_sub, nx_sub, a_diag);
    c_sub.assign(ny_sub, nx_sub, a_upper);

    PTDMAPlanMany py_many;
    py_many.create(ny_sub, nx_sub, topo.getCommY(), TDMAType::Cyclic);
    PTDMASolverMany::solve(py_many, a_sub, b_sub, c_sub, d_sub_tr);
    py_many.destroy();

    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++) d_sub(i, j) = d_sub_tr(j, i);

    collect_solution(d, d_sub, dom, topo);

    if (is_root) {
        double norm2 = 0.0;
        for (int i = 0; i < nx * ny; i++) norm2 += std::pow(d.getVector()[i] - x.getVector()[i], 2);
        std::cout << "Avg. RMS error = " << std::sqrt(norm2 / nx / ny) << std::endl;
    }

    MPI_Finalize();
    return 0;
}

void generate_rhs(DimArray<double>& d, DimArray<double>& x, int nx, int ny) {
    DimArray<double> a, b, c, y(nx, ny);
    a.assign(nx, ny, a_lower);
    b.assign(nx, ny, a_diag);
    c.assign(nx, ny, a_upper);
    x.resize(nx, ny);
    d.resize(nx, ny);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) x(i, j) = dis(gen);

    // y = A_j * x
    for (int i = 0; i < nx; i++) {
        y(i, 0) = a(i, 0) * x(i, ny - 1) + b(i, 0) * x(i, 0) + c(i, 0) * x(i, 1);
        for (int j = 1; j < ny - 1; j++)
            y(i, j) = a(i, j) * x(i, j - 1) + b(i, j) * x(i, j) + c(i, j) * x(i, j + 1);
        y(i, ny - 1) = a(i, ny - 1) * x(i, ny - 2) + b(i, ny - 1) * x(i, ny - 1) + c(i, ny - 1) * x(i, 0);
    }

    // d = A_i * y
    for (int j = 0; j < ny; j++) d(0, j) = b(0, j) * y(0, j) + c(0, j) * y(1, j);
    for (int i = 1; i < nx - 1; i++)
        for (int j = 0; j < ny; j++)
            d(i, j) = a(i, j) * y(i - 1, j) + b(i, j) * y(i, j) + c(i, j) * y(i + 1, j);
    for (int j = 0; j < ny; j++)
        d(nx - 1, j) = a(nx - 1, j) * y(nx - 2, j) + b(nx - 1, j) * y(nx - 1, j);
}

void distribute_rhs(DimArray<double>& d_sub, const DimArray<double>& d, const DomainLayout2D& dom,
                    const CommLayout2D& topo) {
    std::vector<double> d_blk;
    const auto& cnt_x = topo.getCountX();
    const auto& cnt_y = topo.getCountY();
    const auto& cnt_all = topo.getCountAll();
    const auto& disp_x = topo.getDisplX();
    const auto& disp_y = topo.getDisplY();
    const auto& disp_all = topo.getDisplAll();
    const int npx = topo.getSizeX();
    const int npy = topo.getSizeY();
    const int rank = topo.getRank();

    const int nx_sub = dom.getParDimX();
    const int ny_sub = dom.getParDimY();
    const int n_sub = dom.getParDimXY();
    const int nxy = dom.getDimXY();

    if (rank == 0) {
        int idx = 0;
        d_blk.resize(nxy);
        for (int iblk = 0; iblk < npx; iblk++)
            for (int jblk = 0; jblk < npy; jblk++)
                for (int i = 0; i < cnt_x[iblk]; i++)
                    for (int j = 0; j < cnt_y[jblk]; j++)
                        d_blk[idx++] = d(i + disp_x[iblk], j + disp_y[jblk]);
    }

    std::vector<double> recv_blk(n_sub);
    MPI_Scatterv(d_blk.data(), cnt_all.data(), disp_all.data(), MPI_DOUBLE, recv_blk.data(), n_sub,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++) d_sub(i, j) = recv_blk[i * ny_sub + j];
}

void collect_solution(DimArray<double>& d, const DimArray<double>& d_sub, const DomainLayout2D& dom,
                      const CommLayout2D& topo) {
    const auto& cnt_x = topo.getCountX();
    const auto& cnt_y = topo.getCountY();
    const auto& cnt_all = topo.getCountAll();
    const auto& disp_x = topo.getDisplX();
    const auto& disp_y = topo.getDisplY();
    const auto& disp_all = topo.getDisplAll();
    const int npx = topo.getSizeX();
    const int npy = topo.getSizeY();
    const int rank = topo.getRank();

    const int nx_sub = dom.getParDimX();
    const int ny_sub = dom.getParDimY();
    const int n_sub = dom.getParDimXY();
    const int nxy = dom.getDimXY();

    std::vector<double> d_blk_recv;
    if (rank == 0) d_blk_recv.resize(nxy);

    std::vector<double> d_blk_send(n_sub);
    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++) d_blk_send[j + i * ny_sub] = d_sub(i, j);

    MPI_Gatherv(d_blk_send.data(), n_sub, MPI_DOUBLE, d_blk_recv.data(), cnt_all.data(),
                disp_all.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int idx = 0;
        for (int iblk = 0; iblk < npx; iblk++)
            for (int jblk = 0; jblk < npy; jblk++)
                for (int i = 0; i < cnt_x[iblk]; i++)
                    for (int j = 0; j < cnt_y[jblk]; j++)
                        d(i + disp_x[iblk], j + disp_y[jblk]) = d_blk_recv[idx++];
    }
}