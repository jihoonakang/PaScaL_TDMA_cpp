/**
 * @file test_many.cpp
 * @brief Unit test for distributed 2D TDMA (multiple right-hand sides) using MPI and GoogleTest.
 *
 * This test solves a 2D problem with TDMA in both x and y directions, using distributed arrays.
 * The system is built such that the analytical solution is known, and the result is verified.
 */
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>
#include <string>
#include <stdexcept>

#include "pascal_tdma.hpp"
#include "comm_layout_2d.hpp"

constexpr double tolerance = 1e-12;
constexpr double a_diag = 10.0;
constexpr double a_upper = -1.0;
constexpr double a_lower = -1.0;

using namespace PaScaL_TDMA;

/**
 * @brief Generate right-hand side (RHS) and exact solution arrays for the test system.
 *
 * This function constructs a 2D grid solution, computes its image under two sweeps
 * of tridiagonal systems (cyclic in y-direction), and sets the right-hand side accordingly.
 *
 * @param[out] D The generated RHS 2D array (Nx × Ny)
 * @param[out] X The exact solution 2D array (Nx × Ny)
 * @param Nx     Number of grid points in x direction
 * @param Ny     Number of grid points in y direction
 */
void generateRHS(DimArray<double>& D, DimArray<double>& X, int Nx, int Ny) {

    DimArray<double> A, B, C;
    DimArray<double> Y(Nx, Ny);

    A.assign(Nx, Ny, a_lower);
    B.assign(Nx, Ny, a_diag);
    C.assign(Nx, Ny, a_upper);
    X.resize(Nx, Ny);
    D.resize(Nx, Ny);

    std::random_device rd;
    std::mt19937 gen(0); // fixed seed for reproducibility
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < Nx; i++) 
        for (int j = 0; j < Ny; j++) X(i, j) = dis(gen);

    // y = A_j * x
    for (int i = 0; i < Nx; i++) {

        // cyclic
        Y(i, 0) = A(i, 0) * X(i, Ny - 1) + B(i, 0) * X(i, 0) + C(i, 0) * X(i, 1);

        for (int j = 1; j < Ny - 1; j++)
            Y(i, j) = A(i, j) * X(i, j - 1) + B(i, j) * X(i, j) + C(i, j) * X(i, j + 1);

        // cyclic
        Y(i, Ny - 1) = A(i, Ny - 1) * X(i, Ny - 2) + B(i, Ny - 1) * X(i, Ny - 1) 
                     + C(i, Ny - 1) * X(i, 0);
    }

    // d = A_i * y
    for (int j = 0; j < Ny; j++) 
        D(0, j) = B(0, j) * Y(0, j) + C(0, j) * Y(1, j);

    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 0; j < Ny; j++) {
            D(i, j) = A(i, j) * Y(i - 1, j) + B(i, j) * Y(i, j) + C(i, j) * Y(i + 1, j);
        }
    }

    for (int j = 0; j < Ny; j++)
        D(Nx - 1, j) = A(Nx - 1, j) * Y(Nx - 2, j) + B(Nx - 1, j) * Y(Nx - 1, j);
}

/**
 * @brief Distributes a global 2D array into process-local blocks.
 *
 * Uses MPI_Scatterv to distribute the data to each process.
 *
 * @param[out] d_sub Local block of the distributed array (output)
 * @param[in]  d     Global array on the root process (input)
 * @param[in]  dom   Domain decomposition information
 * @param[in]  topo  2D communication topology
 */
void distributeRHS(DimArray<double>& d_sub, const DimArray<double>& d,
                    const DomainLayout2D& dom, const CommLayout2D& topo) {

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
    const int n_sub = dom.getParDimXY();
    const int nxy = dom.getDimXY();

    // Only root process prepares contiguous data for MPI_Scatterv
    if (myrank == 0) {
        int idx = 0;
        d_blk.resize(nxy);
        for (int iblk = 0; iblk < npx; iblk++) {
            for (int jblk = 0; jblk < npy; jblk++) {
                for (int i = 0; i < cnt_x[iblk]; i++) {
                    for (int j = 0; j < cnt_y[jblk]; j++) {
                        d_blk[idx] = d(i + disp_x[iblk], j + disp_y[jblk]);
                        idx++;
                    }
                }
            }
        }
    }

    // Scatter the blocks to each process
    std::vector<double> recv_blk(n_sub);
    MPI_Scatterv(d_blk.data(), cnt_all.data(), disp_all.data(),MPI_DOUBLE,
                 recv_blk.data(), n_sub, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rebuild local 2D array from received block
    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            d_sub(i, j) = recv_blk[i * ny_sub + j];
}

/**
 * @test
 * @brief Distributed 2D TDMA test: solves in x- then y-direction and checks solution.
 *
 * This test reads grid size from command line, sets up 2D domain decomposition,
 * distributes the RHS, solves sequentially along x and y, and compares result with the exact solution.
 *
 * Command-line arguments:
 * - nx: grid size in x direction
 * - ny: grid size in y direction
 */
TEST(PaScaL_TDMA_many, Solve) {

    // Use global argc/argv as GoogleTest does not pass them directly
    extern int g_argc;
    extern char** g_argv;

    if (g_argc != 3)
        throw std::runtime_error("Usage: testMany <nx> <ny>");

    const int nx = std::stoi(g_argv[1]);
    const int ny = std::stoi(g_argv[2]);
    ::testing::Test::RecordProperty("nx", nx);
    ::testing::Test::RecordProperty("ny", ny);

    if (nx < 10 || nx > 10000)
        throw std::runtime_error("Recommendation of 10 <= nx <= 10,000");

    if (ny < 10 || ny > 10000)
        throw std::runtime_error("Recommendation of 10 <= ny <= 10,000");

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    bool is_root = (rank == 0);

    // Create a 2D process grid for domain decomposition
    int dims[2] = {0, 0};
    int period[2] = {0, 0};

    MPI_Dims_create(size, 2, dims);

    CommLayout2D topo(dims, period);
    DomainLayout2D dom(nx, ny, topo);

    const int nx_sub = dom.getParDimX();
    const int ny_sub = dom.getParDimY();

    // Build communication buffer information for 2D topology
    topo.buildCommBufferInfo(nx_sub, ny_sub);

    // Prepare and distribute RHS and exact solution
    DimArray<double> D, X;
    if (is_root)
        generateRHS(D, X, nx, ny);

    DimArray<double> d_sub(nx_sub, ny_sub), x_sub(nx_sub, ny_sub);
    distributeRHS(d_sub, D, dom, topo);
    distributeRHS(x_sub, X, dom, topo);

    // =====[ Solve in x-direction ]=====
    DimArray<double> a_sub, b_sub, c_sub;
    a_sub.assign(nx_sub, ny_sub, a_lower);
    b_sub.assign(nx_sub, ny_sub, a_diag);
    c_sub.assign(nx_sub, ny_sub, a_upper);

    PTDMAPlanMany px_many;
    px_many.create(nx_sub, ny_sub, topo.getCommX(), TDMAType::Standard);
    PTDMASolverMany::solve(px_many, a_sub, b_sub, c_sub, d_sub);
    px_many.destroy();

    // =====[ Solve in y-direction ]=====
    DimArray<double> d_sub_tr(ny_sub, nx_sub);

     // Transpose for y-direction sweep (required by solver interface)
    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            d_sub_tr(j, i) = d_sub(i, j);

    a_sub.assign(ny_sub, nx_sub, a_lower);
    b_sub.assign(ny_sub, nx_sub, a_diag);
    c_sub.assign(ny_sub, nx_sub, a_upper);

    PTDMAPlanMany py_many;
    py_many.create(ny_sub, nx_sub, topo.getCommY(), TDMAType::Cyclic);
    PTDMASolverMany::solve(py_many, a_sub, b_sub, c_sub, d_sub_tr);
    py_many.destroy();

    // Transpose back to original layout
    for (int i = 0; i < nx_sub; i++)
        for (int j = 0; j < ny_sub; j++)
            d_sub(i, j) = d_sub_tr(j, i);

    // =====[ Solution Verification ]=====
    for (int i = 0; i < nx_sub; i++) {
        for (int j = 0; j < ny_sub; j++)
            EXPECT_NEAR(d_sub(i, j), x_sub(i, j), tolerance) << 
                "Mismatch at (i,j) = ( " << i << ", " << j << " )";
    }
}

int g_argc;
char** g_argv;

/**
 * @brief Main entry point for MPI+GoogleTest runs.
 *
 * Initializes MPI, passes arguments to tests, runs all tests, and finalizes MPI.
 */
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    g_argc = argc;
    g_argv = argv;
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
