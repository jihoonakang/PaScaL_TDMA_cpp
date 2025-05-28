#include <vector>
#include <cmath>
#include <random>
#include <mpi.h>
#include <string>
#include <stdexcept>

#include "PaScaL_TDMA.hpp"

constexpr double a_diag = 10.0;
constexpr double a_upper = -1.0;
constexpr double a_lower = -1.0;
constexpr int root = 0;

using namespace PaScaL_TDMA;

void generateRHS(std::vector<double>& D, std::vector<double>& X, 
                 int n, TDMAType type);

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    if (argc != 3) {
        throw std::runtime_error("Usage: single_example <n> <type:standard|cyclic>");
    }

    const int n = std::stoi(argv[1]);

    if (n < 10 || n > 10000) {
        throw std::runtime_error("Recommendation of 10 <= n <= 10,000");
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
    const bool is_root = (rank == root);

    int n_sub = Util::para_range_n(1, n, size, rank);

    std::vector<int> cnt(size), disp(size);
    MPI_Gather(&n_sub, 1, MPI_INT, cnt.data(), 1, MPI_INT, root, MPI_COMM_WORLD);

    if (is_root) {
        disp[0] = 0;
        for (int i = 1; i < size; i++)
            disp[i] = disp[i - 1] + cnt[i - 1];
    }

    std::vector<double> d, x;
    if (is_root) {
        generateRHS(d, x, n, type);
    }

    std::vector<double> d_sub(n_sub, 0.0), x_sub(n_sub, 0.0);

    MPI_Scatterv(d.data(), cnt.data(), disp.data(), MPI_DOUBLE,
                 d_sub.data(), n_sub, MPI_DOUBLE, root, MPI_COMM_WORLD);

    MPI_Scatterv(x.data(), cnt.data(), disp.data(), MPI_DOUBLE,
                 x_sub.data(), n_sub, MPI_DOUBLE, root, MPI_COMM_WORLD);

    std::vector<double> a_sub(n_sub, a_lower);
    std::vector<double> b_sub(n_sub, a_diag);
    std::vector<double> c_sub(n_sub, a_upper);

    PTDMAPlanSingle plan;
    plan.create(n_sub, MPI_COMM_WORLD, root, type);
    PTDMASolverSingle::solve(plan, a_sub, b_sub, c_sub, d_sub);
    plan.destroy();

    MPI_Gatherv(d_sub.data(), n_sub, MPI_DOUBLE, d.data(), cnt.data(), disp.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (is_root) {
        std::vector<double> error(n);
        for (int i = 0; i < n; ++i)
            error[i] = d[i] - x[i];

        std::cout << "Avg. RMS error = " << sqrt(Util::norm2(error) / n) << std::endl;
    }

    MPI_Finalize();
    return 0;
}

void generateRHS(std::vector<double>& D, std::vector<double>& X, int n, TDMAType type) {

    std::vector<double> A(n, a_lower);
    std::vector<double> B(n, a_diag);
    std::vector<double> C(n, a_upper);
    X.resize(n);
    D.resize(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < n; i++) X[i] = dis(gen);

    if (type == TDMAType::Cyclic) {
        D[0] = A[0] * X[n - 1] + B[0] * X[0] + C[0] * X[1];
        for (int i = 1; i < n - 1; i++)
            D[i] = A[i] * X[i - 1] + B[i] * X[i] + C[i] * X[i + 1];
        D[n - 1] = A[n - 1] * X[n - 2] + B[n - 1] * X[n - 1] + C[n - 1] * X[0];
    } else if (type == TDMAType::Standard) {
        D[0] = B[0] * X[0] + C[0] * X[1];
        for (int i = 1; i < n - 1; i++)
            D[i] = A[i] * X[i - 1] + B[i] * X[i] + C[i] * X[i + 1];
        D[n - 1] = A[n - 1] * X[n - 2] + B[n - 1] * X[n - 1];
    } else {
        throw std::invalid_argument("Invalid TDMA type. Use 'standard' or 'cyclic'");
    }

}
