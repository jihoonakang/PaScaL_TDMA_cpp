#include "cuPaScaL_TDMA_C_interface.hpp"
#include "PaScaL_TDMA.cuh"
#include <mpi.h>

using namespace cuPaScaL_TDMA;

extern "C" {

void cu_pascal_tdma_plan_many_create(void** handle,
                                     int n_row, int ny_sys, int nz_sys,
                                     int myrank, int nprocs,
                                     MPI_Fint mpi_comm_f, int tdma_type) {
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_f);
    auto* plan = new cuPTDMAPlanMany();
    TDMAType type = (tdma_type == 0) ? TDMAType::Standard : TDMAType::Cyclic;
    plan->create(n_row, ny_sys, nz_sys, comm, type);
    *handle = static_cast<void*>(plan);
}

void cu_pascal_tdma_many_solve(void* handle,
                                double* a, double* b, double* c, double* d) {
    auto* plan = static_cast<cuPTDMAPlanMany*>(handle);
    cuPTDMASolverMany::cuSolve(*plan, a, b, c, d);
}

void cu_pascal_tdma_plan_many_destroy(void* handle) {
    auto* plan = static_cast<cuPTDMAPlanMany*>(handle);
    plan->destroy();
    delete plan;
    handle = nullptr;
}

}