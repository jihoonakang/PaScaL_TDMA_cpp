/**
 * @file    many_rhs.cpp
 * @brief   Example program for solving multiple RHS tridiagonal systems in 3D using PaScaL_TDMA.
 * @details This example demonstrates solving tridiagonal systems sequentially in x, y, and z directions
 *          using a 2D MPI process topology. The RHS is distributed, solved in three directions, and the
 *          solution is collected at the root rank for error analysis.
 *
 * Usage:
 *      mpirun -n <num_procs> ./many_rhs_example <nx> <ny> <nz>
 *      - nx : Number of unknowns in x-direction (10 ≤ nx ≤ 10,000)
 *      - ny : Number of unknowns in y-direction (10 ≤ ny ≤ 10,000)
 *      - nz : Number of unknowns in z-direction (10 ≤ nz ≤ 10,000)
 */

#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>
#include <string>
#include <stdexcept>
#include <iostream>

#include "tdma_solver.hpp"
#include "pascal_tdma.hpp"
#include "comm_layout_2d.hpp"

constexpr double a_diag = 10.0;
constexpr double a_upper = -1.0;
constexpr double a_lower = -1.0;

using namespace PaScaL_TDMA;

void generate_rhs(DimArray<double>& D, DimArray<double>& X, int nx, int ny, int nz);
void distribute_rhs(DimArray<double>& d_sub, const DimArray<double>& d, const DomainLayout3D& dom,
                    const CommLayout2D& topo);
void collect_solution(DimArray<double>& d, const DimArray<double>& d_sub, const DomainLayout3D& dom,
                      const CommLayout2D& topo);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    if (argc != 4) throw std::runtime_error("Usage: many_rhs_example <nx> <ny> <nz>");

    const int nx = std::stoi(argv[1]);
    const int ny = std::stoi(argv[2]);
    const int nz = std::stoi(argv[3]);
    if (nx < 10 || nx > 10000) throw std::runtime_error("Recommendation: 10 <= nx <= 10,000");
    if (ny < 10 || ny > 10000) throw std::runtime_error("Recommendation: 10 <= ny <= 10,000");
    if (nz < 10 || nz > 10000) throw std::runtime_error("Recommendation: 10 <= nz <= 10,000");

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    bool is_root = (rank == 0);

    int dims[2] = {0, 0};
    int period[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    CommLayout2D topo(dims, period);
    DomainLayout3D dom(nx, ny, nz, topo);

    const int nx_sub = dom.getParDimX();
    const int ny_sub = dom.getParDimY();
    topo.buildCommBufferInfo(nx_sub, ny_sub, nz);

    DimArray<double> d, x;
    if (is_root) {
        d.resize(nx, ny, nz);
        x.resize(nx, ny, nz);
        generate_rhs(d, x, nx, ny, nz);
    }

    DimArray<double> d_sub(nx_sub, ny_sub, nz);
    distribute_rhs(d_sub, d, dom, topo);

    // Solve in x-direction
    std::vector ax(nx_sub, a_lower), bx(nx_sub, a_diag), cx(nx_sub, a_upper);

    PTDMAPlanManyRHS px_many;
    px_many.create(nx_sub, ny_sub * nz, topo.getCommX(), TDMAType::Cyclic);
    d_sub.convert2D(nx_sub, ny_sub * nz);
    PTDMASolverManyRHS::solve(px_many, ax, bx, cx, d_sub);
    px_many.destroy();

    // Solve in y-direction
    std::vector ay(ny_sub, a_lower), by(ny_sub, a_diag), cy(ny_sub, a_upper);
    DimArray<double> d_sub_tr(ny_sub, nx_sub, nz);

    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            for (int k = 0; k < nz; k++)
                d_sub_tr(j, i, k) = d_sub(i, j, k);

    PTDMAPlanManyRHS py_many;
    py_many.create(ny_sub, nx_sub * nz, topo.getCommY(), TDMAType::Standard);
    d_sub_tr.convert2D(ny_sub, nx_sub * nz);
    PTDMASolverManyRHS::solve(py_many, ay, by, cy, d_sub_tr);
    py_many.destroy();

    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            for (int k = 0; k < nz; k++)
                d_sub(i, j, k) = d_sub_tr(j, i, k);

    // Solve in z-direction
    std::vector az(nz, a_lower), bz(nz, a_diag), cz(nz, a_upper);
    d_sub_tr.resize(nz, nx_sub, ny_sub);

    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            for (int k = 0; k < nz; k++)
                d_sub_tr(k, i, j) = d_sub(i, j, k);

    d_sub_tr.convert2D(nz, nx_sub * ny_sub);
    TDMASolver::manyRHS(az, bz, cz, d_sub_tr, nz, nx_sub * ny_sub);

    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            for (int k = 0; k < nz; k++)
                d_sub(i, j, k) = d_sub_tr(k, i, j);

    collect_solution(d, d_sub, dom, topo);

    if (is_root) {
        double norm2 = 0.0;
        for (int i = 0; i < nx * ny * nz; i++)
            norm2 += std::pow(d.getVector()[i] - x.getVector()[i], 2);
        std::cout << "Avg. RMS error = " << std::sqrt(norm2 / nx / ny / nz) << std::endl;
    }

    MPI_Finalize();
    return 0;
}

void generate_rhs(DimArray<double>& D, DimArray<double>& X, int nx, int ny, int nz) {
    const std::vector<double> ax(nx, a_lower), bx(nx, a_diag), cx(nx, a_upper);
    const std::vector<double> ay(ny, a_lower), by(ny, a_diag), cy(ny, a_upper);
    const std::vector<double> az(nz, a_lower), bz(nz, a_diag), cz(nz, a_upper);

    DimArray<double> y(nx, ny, nz), z(nx, ny, nz);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                X(i, j, k) = dis(gen);

    // y = A_k * x
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            y(i, j, 0) = bz[0] * X(i, j, 0) + cz[0] * X(i, j, 1);
            for (int k = 1; k < nz - 1; k++)
                y(i, j, k) = az[k] * X(i, j, k - 1) + bx[k] * X(i, j, k) + cz[k] * X(i, j, k + 1);
            y(i, j, nz - 1) = az[nz - 1] * X(i, j, nz - 2) + bz[nz - 1] * X(i, j, nz - 1);
        }
    }

    // z = A_j * y
    for (int i = 0; i < nx; i++) {
        for (int k = 0; k < nz; k++)
            z(i, 0, k) = by[0] * y(i, 0, k) + cy[0] * y(i, 1, k);
        for (int j = 1; j < ny - 1; j++)
            for (int k = 0; k < nz; k++)
                z(i, j, k) = ay[j] * y(i, j - 1, k) + by[j] * y(i, j, k) + cy[j] * y(i, j + 1, k);
        for (int k = 0; k < nz; k++)
            z(i, ny - 1, k) = ay[ny - 1] * y(i, ny - 2, k) + by[ny - 1] * y(i, ny - 1, k);
    }

    // D = A_i * z (cyclic)
    for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
            D(0, j, k) = ax[0] * z(nx - 1, j, k) + bx[0] * z(0, j, k) + cx[0] * z(1, j, k);

    for (int i = 1; i < nx - 1; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                D(i, j, k) = ax[i] * z(i - 1, j, k) + bx[i] * z(i, j, k) + cx[i] * z(i + 1, j, k);

    for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
            D(nx - 1, j, k) = ax[nx - 1] * z(nx - 2, j, k) + bx[nx - 1] * z(nx - 1, j, k) +
                              cx[nx - 1] * z(0, j, k);
}

void distribute_rhs(DimArray<double>& d_sub, const DimArray<double>& d, const DomainLayout3D& dom,
                    const CommLayout2D& topo) {
    std::vector<double> d_blk;
    const auto cnt_x = topo.getCountX();
    const auto cnt_y = topo.getCountY();
    const auto cnt_all = topo.getCountAll();
    const auto disp_x = topo.getDisplX();
    const auto disp_y = topo.getDisplY();
    const auto disp_all = topo.getDisplAll();
    const int npx = topo.getSizeX();
    const int npy = topo.getSizeY();
    const int rank = topo.getRank();

    const int nx_sub = dom.getParDimX();
    const int ny_sub = dom.getParDimY();
    const int nz_sub = dom.getParDimZ();
    const int n_sub = dom.getParDimXYZ();
    const int nxyz = dom.getDimXYZ();

    if (rank == 0) {
        int idx = 0;
        d_blk.resize(nxyz);
        for (int iblk = 0; iblk < npx; iblk++)
            for (int jblk = 0; jblk < npy; jblk++)
                for (int i = 0; i < cnt_x[iblk]; i++)
                    for (int j = 0; j < cnt_y[jblk]; j++)
                        for (int k = 0; k < nz_sub; k++)
                            d_blk[idx++] = d(i + disp_x[iblk], j + disp_y[jblk], k);
    }

    std::vector<double> recv_blk(n_sub);
    MPI_Scatterv(d_blk.data(), cnt_all.data(), disp_all.data(), MPI_DOUBLE, recv_blk.data(), n_sub,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            for (int k = 0; k < nz_sub; k++)
                d_sub(i, j, k) = recv_blk[i * ny_sub * nz_sub + j * nz_sub + k];
}

void collect_solution(DimArray<double>& d, const DimArray<double>& d_sub, const DomainLayout3D& dom,
                      const CommLayout2D& topo) {
    std::vector<double> d_blk;
    const auto cnt_x = topo.getCountX();
    const auto cnt_y = topo.getCountY();
    const auto cnt_all = topo.getCountAll();
    const auto disp_x = topo.getDisplX();
    const auto disp_y = topo.getDisplY();
    const auto disp_all = topo.getDisplAll();
    const int npx = topo.getSizeX();
    const int npy = topo.getSizeX();
    const int rank = topo.getRank();

    const int nx_sub = dom.getParDimX();
    const int ny_sub = dom.getParDimY();
    const int nz_sub = dom.getParDimZ();
    const int n_sub = dom.getParDimXYZ();
    const int nxyz = dom.getDimXYZ();

    std::vector<double> d_blk_recv;
    if (rank == 0) d_blk_recv.resize(nxyz);

    std::vector<double> d_blk_send(n_sub);
    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            for (int k = 0; k < nz_sub; k++)
                d_blk_send[i * ny_sub * nz_sub + j * nz_sub + k] = d_sub(i, j, k);

    MPI_Gatherv(d_blk_send.data(), n_sub, MPI_DOUBLE, d_blk_recv.data(), cnt_all.data(),
                disp_all.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int idx = 0;
        for (int iblk = 0; iblk < npx; iblk++)
            for (int jblk = 0; jblk < npy; jblk++)
                for (int i = 0; i < cnt_x[iblk]; i++)
                    for (int j = 0; j < cnt_y[jblk]; j++)
                        for (int k = 0; k < nz_sub; k++)
                            d(i + disp_x[iblk], j + disp_y[jblk], k) = d_blk_recv[idx++];
    }
}