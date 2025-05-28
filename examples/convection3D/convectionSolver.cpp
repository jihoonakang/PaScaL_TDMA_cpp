#include "convectionSolver.hpp"

void ConvectionSolver::solveThetaMany(dimArray<double>& theta,
                    const DomainLayout3D& dom3D,
                    const CommLayout3D& com3D,
                    const GlobalParams& params) {

    int myrank, ierr;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double t_curr = params.tStart;
    double dt = params.dtStart;
    double Tmax = params.Tmax;

    const int nx_sub = dom3D.getParDimX();
    const int ny_sub = dom3D.getParDimY();
    const int nz_sub = dom3D.getParDimZ();

    const double dx = params.dx;
    const double dy = params.dy;
    const double dz = params.dz;

    const std::vector<double> dmx_sub = dom3D.getDMX();
    const std::vector<double> dmy_sub = dom3D.getDMY();
    const std::vector<double> dmz_sub = dom3D.getDMZ();

    const double Ct = params.Ct;

    const std::vector<int> jpbc_index = dom3D.getUpperBoundaryFlags();
    const std::vector<int> jmbc_index = dom3D.getLowerBoundaryFlags();

    const dimArray<double> thetaBC3_sub = dom3D.getLowerBoundaryValues();
    const dimArray<double> thetaBC4_sub = dom3D.getUpperBoundaryValues();

    dimArray<double> rhs(nx_sub + 1, ny_sub + 1, nz_sub + 1);

    for (int time_step = 1; time_step <= Tmax; ++time_step) {
        t_curr += dt;
        if (myrank == 0)
            std::cout << "[Main] Current time step = " << time_step << std::endl;

        // Allocate RHS
        rhs.assign(nx_sub + 1, ny_sub + 1, nz_sub + 1, 0.0);

        for (int i = 1; i < nx_sub; ++i) {
            int ip = i + 1, im = i - 1;
            for (int j = 1; j < ny_sub; ++j) {
                int jp = j + 1, jm = j - 1;
                int jep = jpbc_index[j], jem = jmbc_index[j];

                for (int k = 1; k < nz_sub; ++k) {
                    int kp = k + 1, km = k - 1;

                    // Spatial derivatives
                    double dedx1 = (theta(i, j, k) - theta(im,j, k)) / dmx_sub[i];
                    double dedx2 = (theta(ip,j, k) - theta(i, j, k)) / dmx_sub[ip];
                    double dedy3 = (theta(i, j, k) - theta(i, jm,k)) / dmy_sub[j];
                    double dedy4 = (theta(i, jp,k) - theta(i, j, k)) / dmy_sub[jp];
                    double dedz5 = (theta(i, j, k) - theta(i, j,km)) / dmz_sub[k];
                    double dedz6 = (theta(i, j,kp) - theta(i, j, k)) / dmz_sub[kp];

                    double visc1 = (dedx2 - dedx1) / dx;
                    double visc2 = (dedy4 - dedy3) / dy;
                    double visc3 = (dedz6 - dedz5) / dz;
                    double viscous = 0.5 * Ct * (visc1 + visc2 + visc3);

                    // Y-direction boundary term
                    double ebc_dn = 0.5 * Ct / dy / dmy_sub[j] * thetaBC3_sub(i, k);
                    double ebc_up = 0.5 * Ct / dy / dmy_sub[jp] * thetaBC4_sub(i, k);
                    double ebc = (1 - jem) * ebc_dn + (1 - jep) * ebc_up;

                    // Diffusion terms for each direction
                    double eAPI = -0.5 * Ct / dx / dmx_sub[ip];
                    double eAMI = -0.5 * Ct / dx / dmx_sub[i];
                    double eACI = 0.5 * Ct / dx * (1.0 / dmx_sub[ip] + 1.0 / dmx_sub[i]);

                    double eAPK = -0.5 * Ct / dz / dmz_sub[kp];
                    double eAMK = -0.5 * Ct / dz / dmz_sub[k];
                    double eACK = 0.5 * Ct / dz * (1.0 / dmz_sub[kp] + 1.0 / dmz_sub[k]);

                    double eAPJ = -0.5 * Ct / dy / dmy_sub[jp] * jep;
                    double eAMJ = -0.5 * Ct / dy / dmy_sub[j] * jem;
                    double eACJ = 0.5 * Ct / dy * (1.0 / dmy_sub[jp] + 1.0 / dmy_sub[j]);

                    double eRHS = eAPK * theta(i, j,kp) + eACK * theta(i, j, k) + eAMK * theta(i, j, km)
                                + eAPJ * theta(i, jp,k) + eACJ * theta(i, j, k) + eAMJ * theta(i, jm,k)
                                + eAPI * theta(ip,j, k) + eACI * theta(i, j, k) + eAMI * theta(im,j, k);

                    rhs(i, j, k) = theta(i, j, k) / dt + viscous + ebc - (theta(i, j, k) / dt + eRHS);
                }
            }
        }

        // -----------------------
        // TDMA solves in z, y, x
        // -----------------------
        // z-direction (i-j planes)

        dimArray<double> ap(nz_sub - 1, ny_sub - 1); //(n_row, n_sys)
        dimArray<double> am(nz_sub - 1, ny_sub - 1);
        dimArray<double> ac(nz_sub - 1, ny_sub - 1);
        dimArray<double> ad(nz_sub - 1, ny_sub - 1);

        PaScaL_TDMA::PTDMAPlanMany pz;
        pz.create(nz_sub - 1, ny_sub - 1, com3D.getCommZ(), PaScaL_TDMA::TDMAType::Cyclic);

        for (int i = 1; i < nx_sub; i++) {
            for (int j = 1; j < ny_sub; j++) {
                for (int k = 1; k < nz_sub; k++) {
                    int kp = k + 1;
                    ap(k - 1, j - 1) = -0.5 * Ct / dz / dmz_sub[kp] * dt;
                    am(k - 1, j - 1) = -0.5 * Ct / dz / dmz_sub[k ] * dt;
                    ac(k - 1, j - 1) = (0.5 * Ct / dz * (1.0 / dmz_sub[kp] + 1.0 / dmz_sub[k])) * dt + 1.0;
                    ad(k - 1, j - 1) = rhs(i, j, k) * dt;
                }
            }

            PaScaL_TDMA::PTDMASolverMany::solve(pz, am, ac, ap, ad);

            for (int j = 1; j < ny_sub; j++)
                for (int k = 1; k < nz_sub; k++)
                    rhs(i, j, k) = ad(k - 1, j - 1);
        }
        pz.destroy();

        // y-direction (i-k planes)

        ap.resize(ny_sub - 1, nz_sub - 1);
        am.resize(ny_sub - 1, nz_sub - 1);
        ac.resize(ny_sub - 1, nz_sub - 1);
        ad.resize(ny_sub - 1, nz_sub - 1);
    
        PaScaL_TDMA::PTDMAPlanMany py;
        py.create(ny_sub - 1, nz_sub - 1, com3D.getCommY(), PaScaL_TDMA::TDMAType::Standard);
    
        for (int i = 1; i < nx_sub; i++) {
            for (int j = 1; j < ny_sub; j++) {
                int jp = j + 1, jm = j - 1;
                int jep = jpbc_index[j], jem = jmbc_index[j];
                for (int k = 1; k < nz_sub; k++) {
    
                    ap(j - 1, k - 1) = -0.5 * Ct / dy / dmy_sub[jp] * static_cast<double>(jep) * dt;
                    am(j - 1, k - 1) = -0.5 * Ct / dy / dmy_sub[j ] * static_cast<double>(jem) * dt;
                    ac(j - 1, k - 1) = (0.5 * Ct / dy * (1.0 / dmy_sub[jp] + 1.0 / dmy_sub[j])) * dt + 1.0;
                    ad(j - 1, k - 1) = rhs(i, j, k);
                }
            }
    
            PaScaL_TDMA::PTDMASolverMany::solve(py, am, ac, ap, ad);
    
            for (int j = 1; j < ny_sub; j++)
                for (int k = 1; k < nz_sub; k++)
                    rhs(i, j, k) = ad(j - 1, k - 1);
        }
    
        py.destroy();

        // x-direction (j-k planes)
        ap.resize(nx_sub - 1, nz_sub - 1);
        am.resize(nx_sub - 1, nz_sub - 1);
        ac.resize(nx_sub - 1, nz_sub - 1);
        ad.resize(nx_sub - 1, nz_sub - 1);
    
        PaScaL_TDMA::PTDMAPlanMany px;
        px.create(nx_sub - 1, nz_sub - 1, com3D.getCommX(), PaScaL_TDMA::TDMAType::Cyclic);
    
        for (int j = 1; j < ny_sub; j++) {
            for (int k = 1; k < nz_sub; k++) {
                for (int i = 1; i < nx_sub; i++) {
                    int ip = i + 1, im = i - 1;
                    ap(i - 1, k - 1) = -0.5 * Ct / dx / dmx_sub[ip] * dt;
                    am(i - 1, k - 1) = -0.5 * Ct / dx / dmx_sub[i ] * dt;
                    ac(i - 1, k - 1) = (0.5 * Ct / dx * (1.0 / dmx_sub[ip] + 1.0 / dmx_sub[i])) * dt + 1.0;
                    ad(i - 1, k - 1) = rhs(i, j, k);
                }
            }
    
            PaScaL_TDMA::PTDMASolverMany::solve(px, am, ac, ap, ad);
    
            for (int i = 1; i < nx_sub; i++)
                for (int k = 1; k < nz_sub; k++)
                    theta(i, j, k) += ad(i - 1, k - 1);  // theta 갱신
        }
    
        px.destroy();

        dom3D.updateGhostCells(theta, com3D);
    }
}