#pragma once
#include <vector>
#include <array>
#include <mpi.h>
#include "globalParams.hpp"
#include "dimArray.hpp"
#include "commLayout3D.hpp"
#include "util.hpp"

class DomainLayout3D {

    int nx;
    int ny;
    int nz;
    int n_all;
    int nx_sub;
    int ny_sub;
    int nz_sub;
    int n_sub;
    int ista, iend, jsta, jend, ksta, kend;

    double dx, dy, dz;

    // Coordinates and spacing
    std::vector<double> x_sub, y_sub, z_sub;
    std::vector<double> dmx_sub, dmy_sub, dmz_sub;

    // Boundary condition fields
    dimArray<double> thetaBC3_sub, thetaBC4_sub;

    // Flags
    std::vector<int> jmbc_index, jpbc_index;

    // MPI datatypes for ghostcell communication
    MPI_Datatype send_E, recv_W, send_W, recv_E;
    MPI_Datatype send_N, recv_S, send_S, recv_N;
    MPI_Datatype send_F, recv_B, send_B, recv_F;

    // MPI type creation
    void createGhostCellMPITypes();

public:
    DomainLayout3D( const int nx_, const int ny_, const int nz_,
                    const CommLayout3D& topo);
    ~DomainLayout3D() {};

    void cleanupDomain();

    // Ghost update
    void updateGhostCells(dimArray<double>& theta_sub, const CommLayout3D& comm) const;
    void assignMesh(const CommLayout3D& comm, const GlobalParams& params);

    // Initialization of theta_sub
    void initializeField(dimArray<double>& theta_sub,
                         const CommLayout3D& topo,
                         const GlobalParams& params);

    void assignBoundaries(const dimArray<double>& theta_sub, 
                          const CommLayout3D& topo,
                          const GlobalParams& params);

    const int getDimX() const { return nx; }
    const int getDimY() const { return ny; }
    const int getDimZ() const { return nz; }
    const int getDimXYZ() const { return n_all; }

    const int getParDimX() const { return nx_sub; }
    const int getParDimY() const { return ny_sub; }
    const int getParDimZ() const { return nz_sub; }
    const int getParDimXYZ() const { return n_sub; }

    const std::vector<int> getLowerBoundaryFlags() const { return jmbc_index; }
    const std::vector<int> getUpperBoundaryFlags() const { return jpbc_index; }

    const dimArray<double> getLowerBoundaryValues() const { return thetaBC3_sub; }
    const dimArray<double> getUpperBoundaryValues() const { return thetaBC4_sub; }

    const std::vector<double> getDMX() const { return dmx_sub; }
    const std::vector<double> getDMY() const { return dmy_sub; }
    const std::vector<double> getDMZ() const { return dmz_sub; }
};

