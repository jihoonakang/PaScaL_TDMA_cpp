#include "domainLayout3D.hpp"


DomainLayout3D::DomainLayout3D( const int nx_, const int ny_, const int nz_,
                const CommLayout3D& topo)
                : nx(nx_), ny(ny_), nz(nz_) {

    n_all = nx * ny * nz;

    nx_sub = Util::para_range(nx - 1, topo.getSizeX(), topo.getRankX(), ista, iend) + 1;
    ny_sub = Util::para_range(ny - 1, topo.getSizeY(), topo.getRankY(), jsta, jend) + 1;
    nz_sub = Util::para_range(nz - 1, topo.getSizeZ(), topo.getRankZ(), ksta, kend) + 1;

    n_sub = nx_sub * ny_sub * nz_sub;

    x_sub.resize(nx_sub + 1);
    y_sub.resize(ny_sub + 1);
    z_sub.resize(nz_sub + 1);
    dmx_sub.resize(nx_sub + 1);
    dmy_sub.resize(ny_sub + 1);
    dmz_sub.resize(nz_sub + 1);

    thetaBC3_sub.resize(nx_sub + 1, nz_sub + 1);
    thetaBC4_sub.resize(nx_sub + 1, nz_sub + 1);

    // Initialize to 1 (grid exists)
    jmbc_index.resize(ny_sub + 1, 1);
    jpbc_index.resize(ny_sub + 1, 1);

    createGhostCellMPITypes();
}

void DomainLayout3D::cleanupDomain()
{
    x_sub.clear();
    y_sub.clear();
    z_sub.clear();
    dmx_sub.clear();
    dmy_sub.clear();
    dmz_sub.clear();

    thetaBC3_sub.clear();
    thetaBC4_sub.clear();

    jmbc_index.clear();
    jpbc_index.clear();

    MPI_Type_free(&send_E);
    MPI_Type_free(&send_W);
    MPI_Type_free(&send_N);
    MPI_Type_free(&send_S);
    MPI_Type_free(&send_F);
    MPI_Type_free(&send_B);
    MPI_Type_free(&recv_E);
    MPI_Type_free(&recv_W);
    MPI_Type_free(&recv_N);
    MPI_Type_free(&recv_S);
    MPI_Type_free(&recv_F);
    MPI_Type_free(&recv_B);
}

// MPI type creation
void DomainLayout3D::createGhostCellMPITypes() {

    const int sizes[3] = {nx_sub + 1, ny_sub + 1, nz_sub + 1};
    int subsizes[3];
    int starts[3];

    // X-direction (East, West)
    subsizes[0] = 1;
    subsizes[1] = ny_sub + 1;
    subsizes[2] = nz_sub + 1;

    // East send
    starts[0] = nx_sub - 1; starts[1] = 0; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &send_E);
    MPI_Type_commit(&send_E);

    // West receive
    starts[0] = 0; starts[1] = 0; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &recv_W);
    MPI_Type_commit(&recv_W);

    // West send
    starts[0] = 1; starts[1] = 0; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &send_W);
    MPI_Type_commit(&send_W);

    // East receive
    starts[0] = nx_sub; starts[1] = 0; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &recv_E);
    MPI_Type_commit(&recv_E);

    // Y-direction (North, South)
    subsizes[0] = nx_sub + 1;
    subsizes[1] = 1;
    subsizes[2] = nz_sub + 1;

    // North send
    starts[0] = 0; starts[1] = ny_sub - 1; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &send_N);
    MPI_Type_commit(&send_N);

    // South receive
    starts[0] = 0; starts[1] = 0; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &recv_S);
    MPI_Type_commit(&recv_S);

    // South send
    starts[0] = 0; starts[1] = 1; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &send_S);
    MPI_Type_commit(&send_S);

    // North receive
    starts[0] = 0; starts[1] = ny_sub; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &recv_N);
    MPI_Type_commit(&recv_N);

    // Z-direction (Forth, Back)
    subsizes[0] = nx_sub + 1;
    subsizes[1] = ny_sub + 1;
    subsizes[2] = 1;

    // Forth send
    starts[0] = 0; starts[1] = 0; starts[2] = nz_sub - 1;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &send_F);
    MPI_Type_commit(&send_F);

    // Back receive
    starts[0] = 0; starts[1] = 0; starts[2] = 0;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &recv_B);
    MPI_Type_commit(&recv_B);

    // Back send
    starts[0] = 0; starts[1] = 0; starts[2] = 1;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &send_B);
    MPI_Type_commit(&send_B);

    // Forth receive
    starts[0] = 0; starts[1] = 0; starts[2] = nz_sub;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &recv_F);
    MPI_Type_commit(&recv_F);
}

// Ghost update
void DomainLayout3D::updateGhostCells(dimArray<double>& theta_sub, 
                                      const CommLayout3D& comm) const {

    MPI_Request requests[12];

    // X-direction communication
    MPI_Isend(theta_sub.getData(), 1, send_E, comm.getRankNextX(), 111, comm.getCommX(), &requests[0]);
    MPI_Irecv(theta_sub.getData(), 1, recv_W, comm.getRankPrevX(), 111, comm.getCommX(), &requests[1]);

    MPI_Isend(theta_sub.getData(), 1, send_W, comm.getRankPrevX(), 222, comm.getCommX(), &requests[2]);
    MPI_Irecv(theta_sub.getData(), 1, recv_E, comm.getRankNextX(), 222, comm.getCommX(), &requests[3]);

    // Y-direction communication
    MPI_Isend(theta_sub.getData(), 1, send_N, comm.getRankNextY(), 111, comm.getCommY(), &requests[4]);
    MPI_Irecv(theta_sub.getData(), 1, recv_S, comm.getRankPrevY(), 111, comm.getCommY(), &requests[5]);

    MPI_Isend(theta_sub.getData(), 1, send_S, comm.getRankPrevY(), 222, comm.getCommY(), &requests[6]);
    MPI_Irecv(theta_sub.getData(), 1, recv_N, comm.getRankNextY(), 222, comm.getCommY(), &requests[7]);

    // Z-direction communication
    MPI_Isend(theta_sub.getData(), 1, send_F, comm.getRankNextZ(), 111, comm.getCommZ(), &requests[8]);
    MPI_Irecv(theta_sub.getData(), 1, recv_B, comm.getRankPrevZ(), 111, comm.getCommZ(), &requests[9]);

    MPI_Isend(theta_sub.getData(), 1, send_B, comm.getRankPrevZ(), 222, comm.getCommZ(), &requests[10]);
    MPI_Irecv(theta_sub.getData(), 1, recv_F, comm.getRankNextZ(), 222, comm.getCommZ(), &requests[11]);

    MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);
}

void DomainLayout3D::assignMesh(const CommLayout3D& comm, 
                                const GlobalParams& params) {

    const double dx = params.dx;
    const double dy = params.dy;
    const double dz = params.dz;

    for (int i = 0; i <= nx_sub; ++i) {
        x_sub[i] = static_cast<double>(i - 1 + ista - 1) * dx;
        dmx_sub[i] = dx;
    }

    for (int j = 0; j <= ny_sub; ++j) {
        y_sub[j] = static_cast<double>(j + jsta - 1) * dy;
        dmy_sub[j] = dy;
    }

    for (int k = 0; k <= nz_sub; ++k) {
        z_sub[k] = static_cast<double>(k - 1 + ksta - 1) * dz;
        dmz_sub[k] = dz;
    }

    if (comm.getRankX() == 0) dmx_sub[0] = dx;
    if (comm.getRankX() == comm.getSizeX() - 1) dmx_sub[nx_sub] = dx;

    if (comm.getRankY() == 0) dmy_sub[0] = dy / 2.0;
    if (comm.getRankY() ==  comm.getSizeY() - 1) dmy_sub[ny_sub] = dy / 2.0;

    if (comm.getRankZ() == 0) dmz_sub[0] = dz;
    if (comm.getRankZ() ==  comm.getSizeZ() - 1) dmz_sub[nz_sub] = dz;
}

// Initialization of theta_sub
void DomainLayout3D::initializeField(dimArray<double>& theta_sub,
                                     const CommLayout3D& topo,
                                     const GlobalParams& params) {

    const double PI = GlobalParams::PI;
    for (int k = 0; k <= nz_sub; ++k) {
        for (int j = 0; j <= ny_sub; ++j) {
            for (int i = 0; i <= nx_sub; ++i) {
                theta_sub(i, j, k) =
                    (params.theta_cold - params.theta_hot) / params.ly * y_sub[j] + params.theta_hot +
                    std::sin(4.0 * PI / params.lx * x_sub[i]) *
                    std::sin(4.0 * PI / params.lz * z_sub[k]) *
                    std::sin(4.0 * PI / params.ly * y_sub[j]);
            }
        }
    }

    if (topo.getRankY() == 0) {
        for (int k = 0; k <= nz_sub; ++k) 
            for (int i = 0; i <= nx_sub; ++i) 
                    theta_sub(i, 0, k) = params.theta_hot;
    }
    if (topo.getRankY() == topo.getSizeY() - 1) {
        for (int k = 0; k <= nz_sub; ++k) 
            for (int i = 0; i <= nx_sub; ++i) 
                theta_sub(i, ny_sub, k) = params.theta_cold;
    }
}

// Boundary condition assignment
void DomainLayout3D::assignBoundaries(const dimArray<double>& theta_sub,
                                      const CommLayout3D& topo,
                                      const GlobalParams& params) {

    for (int k = 0; k <= nz_sub; ++k) {
        for (int i = 0; i <= nx_sub; ++i) {
            thetaBC3_sub(i, k) = theta_sub(i, 0, k);
            thetaBC4_sub(i, k) = theta_sub(i, ny_sub, k);
        }
    }

    if (topo.getRankY() == 0) {

        // Lower boundary (j = 0 becomes invalid)
        jmbc_index[1] = 0;

        for (int k = 0; k <= nz_sub; ++k)
                for (int i = 0; i <= nx_sub; ++i)
                    thetaBC3_sub(i, k) = params.theta_hot;
    }

    if (topo.getRankY() == topo.getSizeY() - 1) {

        // Upper boundary (j = ny_sub - 1 becomes invalid)
        jpbc_index[ny_sub - 1] = 0;

        for (int k = 0; k <= nz_sub; ++k)
                for (int i = 0; i <= nx_sub; ++i)
                    thetaBC4_sub(i, k) = params.theta_cold;
    }

}

 // namespace DomainLayout3D

