#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>
#include <string>
#include <stdexcept>

#include "TDMASolver.hpp"
#include "PaScaL_TDMA.hpp"
#include "commLayout2D.hpp"

constexpr double tolerance = 1e-12;
constexpr double a_diag = 10.0;
constexpr double a_upper = -1.0;
constexpr double a_lower = -1.0;

using namespace PaScaL_TDMA;

void generateRHS(dimArray<double>& D, dimArray<double>& X, 
                 int Nx, int Ny, int Nz);

void distributeRHS(dimArray<double>& d_sub, const dimArray<double>& d,
                    const DomainLayout3D& dom, const CommLayout2D& topo);

TEST(PaScaL_TDMA_many, Solve) {

    // Read from global argc/argv (GoogleTest doesn't pass arguments to TEST directly)
    extern int g_argc;
    extern char** g_argv;

    if (g_argc != 4)
        throw std::runtime_error("Usage: testMany <nx> <ny> <nz>");

    const int nx = std::stoi(g_argv[1]);
    const int ny = std::stoi(g_argv[2]);
    const int nz = std::stoi(g_argv[3]);
    ::testing::Test::RecordProperty("nx", nx);
    ::testing::Test::RecordProperty("ny", ny);
    ::testing::Test::RecordProperty("nz", nz);

    if (nx < 10 || nx > 10000)
        throw std::runtime_error("Recommendation of 10 <= nx <= 10,000");

    if (ny < 10 || ny > 10000)
        throw std::runtime_error("Recommendation of 10 <= ny <= 10,000");

    if (nz < 10 || nz > 10000)
        throw std::runtime_error("Recommendation of 10 <= nz <= 10,000");

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

    dimArray<double> D, X;
    if (is_root) {
        D.resize(nx, ny, nz);
        X.resize(nx, ny, nz);
        generateRHS(D, X, nx, ny, nz);
    }

    dimArray<double> d_sub(nx_sub, ny_sub, nz), x_sub(nx_sub, ny_sub, nz);
    distributeRHS(d_sub, D, dom, topo);
    distributeRHS(x_sub, X, dom, topo);

// Solve in x-direction
    std::vector ax(nx_sub, a_lower);
    std::vector bx(nx_sub, a_diag);
    std::vector cx(nx_sub, a_upper);

    PTDMAPlanManyRHS px_many;
    px_many.create(nx_sub, ny_sub * nz, topo.getCommX(), TDMAType::Cyclic);
    d_sub.convert2D(nx_sub, ny_sub * nz);
    PTDMASolverManyRHS::solve(px_many, ax, bx, cx, d_sub);
    px_many.destroy();

    // Solve in y-direction
    std::vector ay(ny_sub, a_lower);
    std::vector by(ny_sub, a_diag);
    std::vector cy(ny_sub, a_upper);

    dimArray<double> d_sub_tr(ny_sub, nx_sub, nz);

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
    std::vector az(nz, a_lower);
    std::vector bz(nz, a_diag);
    std::vector cz(nz, a_upper);

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

    for (int i = 0; i < nx_sub; i++) {
        for (int j = 0; j < ny_sub; j++)
            for (int k = 0; k < nz; k++)
                EXPECT_NEAR(d_sub(i, j, k), x_sub(i, j, k), tolerance) << 
                    "Mismatch at (i,j,k) = (" <<i <<", "<<j<<", "<<k<<" )";
    }
}

void generateRHS(dimArray<double>& D, dimArray<double>& X, 
                 int Nx, int Ny, int Nz) {

    const std::vector<double> ax(Nx, a_lower);
    const std::vector<double> bx(Nx, a_diag);
    const std::vector<double> cx(Nx, a_upper);

    const std::vector<double> ay(Ny, a_lower);
    const std::vector<double> by(Ny, a_diag);
    const std::vector<double> cy(Ny, a_upper);

    const std::vector<double> az(Nz, a_lower);
    const std::vector<double> bz(Nz, a_diag);
    const std::vector<double> cz(Nz, a_upper);

    dimArray<double> y(Nx, Ny, Nz);
    dimArray<double> z(Nx, Ny, Nz);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < Nx; i++) 
        for (int j = 0; j < Ny; j++) 
            for (int k = 0; k < Nz; k++)
                X(i, j, k) = dis(gen);

    // y = A_k * x
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            y(i, j, 0) = bz[0] * X(i, j, 0) + cz[0] * X(i, j, 1);

            for (int k = 1; k < Nz - 1; k++)
                y(i, j, k)  = az[k] * X(i, j, k - 1) + bz[k] * X(i, j, k) 
                            + cz[k] * X(i, j, k + 1);

            y(i, j, Nz - 1) = az[Nz - 1] * X(i, j, Nz - 2) 
                            + bz[Nz - 1] * X(i, j, Nz - 1);
        }
    }

    // z = A_j * y
    for (int i = 0; i < Nx; i++) {
        for (int k = 0; k < Nz; k++)
            z(i, 0, k) = by[0] * y(i, 0, k) + cy[0] * y(i, 1, k);

        for (int j = 1; j < Ny - 1; j++) {
            for (int k = 0; k < Nz; k++) {
                z(i, j, k)  = ay[j] * y(i, j - 1, k) + by[j] * y(i, j, k) 
                            + cy[j] * y(i, j + 1, k);
            }
        }
        for (int k = 0; k < Nz; k++)
            z(i, Ny - 1, k) = ay[Ny - 1] * y(i, Ny - 2, k) 
                            + by[Ny - 1] * y(i, Ny - 1, k);
    }

    // d = A_i * z
    // Cyclic
    for (int j = 0; j < Ny; j++) 
        for (int k = 0; k < Nz; k++) 
            D(0, j, k)  = ax[0] * z(Nx - 1, j, k) 
                        + bx[0] * z(0, j, k) 
                        + cx[0] * z(1, j, k);

    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                D(i, j, k)  = ax[i] * z(i - 1, j, k) 
                            + bx[i] * z(i, j, k) 
                            + cx[i] * z(i + 1, j, k);
            }
        }
    }

    // Cyclic
    for (int j = 0; j < Ny; j++)
        for (int k = 0; k < Nz; k++) 
            D(Nx - 1, j, k) = ax[Nx - 1] * z(Nx - 2, j, k) 
                            + bx[Nx - 1] * z(Nx - 1, j, k)
                            + cx[Nx - 1] * z(0, j, k);
}

void distributeRHS(dimArray<double>& d_sub, const dimArray<double>& d,
                    const DomainLayout3D& dom, const CommLayout2D& topo) {

    std::vector<double> d_blk;
    const std::vector<int> cnt_x = topo.getCountX();
    const std::vector<int> cnt_y = topo.getCountY();
    const std::vector<int> cnt_all = topo.getCountAll();
    const std::vector<int> disp_x = topo.getDisplX();
    const std::vector<int> disp_y = topo.getDisplY();
    const std::vector<int> disp_all = topo.getDisplAll();
    const int npx = topo.getSizeX();
    const int npy = topo.getSizeY();
    const int myrank = topo.getRank();

    const int nx_sub = dom.getParDimX();
    const int ny_sub = dom.getParDimY();
    const int nz_sub = dom.getParDimZ();
    const int n_sub = dom.getParDimXYZ();
    const int nxyz = dom.getDimXYZ();

    if (myrank == 0) {
        int idx = 0;
        d_blk.resize(nxyz);
        for (int iblk = 0; iblk < npx; iblk++) {
            for (int jblk = 0; jblk < npy; jblk++) {
                for (int i = 0; i < cnt_x[iblk]; i++) {
                    for (int j = 0; j < cnt_y[jblk]; j++) {
                        for (int k = 0; k < nz_sub; k++) {
                            d_blk[idx] = d(i + disp_x[iblk], j + disp_y[jblk], k);
                            idx++;
                        }
                    }
                }
            }
        }
    }

    std::vector<double> recv_blk(n_sub);
    MPI_Scatterv(d_blk.data(), cnt_all.data(), disp_all.data(),MPI_DOUBLE,
                 recv_blk.data(), n_sub, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            for (int k = 0; k < nz_sub; k++)
                d_sub(i, j, k) = recv_blk[i * ny_sub * nz_sub + j * nz_sub + k];
}


int g_argc;
char** g_argv;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    g_argc = argc;
    g_argv = argv;
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
