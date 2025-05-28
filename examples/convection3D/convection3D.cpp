#include <string>
#include <iterator>
#include "mpi.h"
#include "convectionSolver.hpp"

using namespace PaScaL_TDMA;

template<class T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& values)
{
	copy( begin(values), end(values), std::ostream_iterator<T>(stream, ", ") );
	return stream;
}

int main(int argc, char** argv) {

    int myrank, nprocs;
    bool is_root = false;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (myrank == ROOT_RANK) is_root = true;

    int dims[3] = {0, 0, 0};
    int period[3] = {0, 0, 0};

    MPI_Dims_create(nprocs, 3, dims);
    if (argc < 2) {
        std::cout << "Input file is not specifed. Abort run" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::string filename = argv[argc-1];

    GlobalParams param(filename, is_root);
    CommLayout3D comm(dims, period);
    DomainLayout3D dom(param.nx, param.ny, param.nz, comm);
    dimArray<double> theta_sub(dom.getParDimX() + 1, dom.getParDimY() + 1, dom.getParDimZ() + 1);

    comm.print_info();
    dom.assignMesh(comm, param);
    dom.initializeField(theta_sub, comm, param);
    dom.assignBoundaries(theta_sub, comm, param);

    dom.updateGhostCells(theta_sub, comm);

    ConvectionSolver::solveThetaMany(theta_sub, dom, comm, param);

    dom.cleanupDomain();

    MPI_Finalize();

    return 0;
}