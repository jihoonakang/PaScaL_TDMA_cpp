#pragma once
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <mpi.h>

#define ROOT_RANK 0

class GlobalParams {
public:
    // Constants
    static constexpr double PI = 3.14159265358979323846;

    // Physical parameters
    const double Pr = 5.0;
    const double Ra = 200.0;

    // Iteration steps
    int Tmax;

    // Grid sizes
    int nx, ny, nz;
    int nxm, nym, nzm;
    int nxp, nyp, nzp;

    // Time step
    double dt, dtStart = 5e-3, tStart = 0.0;

    // Physical domain size
    const double lx = 1.0, ly = 1.0, lz = 1.0;
    double dx, dy, dz;

    // Boundary conditions
    const double theta_cold = -1.0;
    double theta_hot;
    const double alphaG = 1.0;
    double nu, Ct;

    // MPI topology
    std::array<int, 3> np_dim = {0, 0, 0};

    // Constructor
    GlobalParams(const std::string& input_file, bool is_root) {
        loadParameters(input_file, is_root);
        computeDerivedParameters();
    }

private:
    void loadParameters(const std::string& input_file, bool is_root) {

        if (is_root) {
            std::ifstream file(input_file);
            if (!file.is_open())
                throw std::runtime_error("Could not open input file: " + input_file);

            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);

            // Parse key-value pairs
                size_t eq_pos = line.find('=');
                if (eq_pos != std::string::npos) {
                    std::string key = line.substr(0, eq_pos);
                    std::string value = line.substr(eq_pos + 1);

                    // Trim spaces
                    key.erase(remove_if(key.begin(), key.end(), isspace), key.end());
                    value.erase(remove_if(value.begin(), value.end(), isspace), value.end());

                    if (key == "nx") nx = std::stoi(value);
                    else if (key == "ny") ny = std::stoi(value);
                    else if (key == "nz") nz = std::stoi(value);
                    else if (key == "npx") np_dim[0] = std::stoi(value);
                    else if (key == "npy") np_dim[1] = std::stoi(value);
                    else if (key == "npz") np_dim[2] = std::stoi(value);
                    else if (key == "Tmax") Tmax = std::stoi(value);
                }
            }

            file.close();
        }
        MPI_Bcast(&nx, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
        MPI_Bcast(&ny, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
        MPI_Bcast(&nz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
        MPI_Bcast(&Tmax, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
        MPI_Bcast(np_dim.data(), 3, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
    }

    void computeDerivedParameters() {
        // Increase grid counts
        nx += 1; ny += 1; nz += 1;
        nxm = nx - 1; nym = ny - 1; nzm = nz - 1;
        nxp = nx + 1; nyp = ny + 1; nzp = nz + 1;

        // Boundary-related
        theta_hot = 2.0 + theta_cold;
        nu = 1.0 / std::sqrt(Ra / (alphaG * Pr * std::pow(ly, 3) * (theta_hot - theta_cold)));
        Ct = nu / Pr;

        // Derived grid spacing
        dx = lx / static_cast<double>(nxm);
        dy = ly / static_cast<double>(nym);
        dz = lz / static_cast<double>(nzm);
    }
};