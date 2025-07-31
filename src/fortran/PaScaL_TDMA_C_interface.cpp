#include "pascal_tdma_c_interface.hpp"
#include "pascal_tdma.hpp"
#include <mpi.h>

using namespace PaScaL_TDMA;

extern "C" {

void pascal_tdma_plan_single_create(void** handle, int n_row,
                                    int myrank, int nprocs,
                                    MPI_Fint mpi_comm_f, int tdma_type) {
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_f);
    auto* plan = new PTDMAPlanSingle();
    TDMAType type = (tdma_type == 0) ? TDMAType::Standard : TDMAType::Cyclic;
    plan->create(n_row, comm, type);
    *handle = static_cast<void*>(plan);
}

void pascal_tdma_single_solve(void* handle,
                              double* a, double* b, double* c, double* d) {
    auto* plan = static_cast<PTDMAPlanSingle*>(handle);
    PTDMASolverSingle::solve(*plan, a, b, c, d);
}

void pascal_tdma_plan_single_destroy(void* handle) {
    auto* plan = static_cast<PTDMAPlanSingle*>(handle);
    plan->destroy();
    delete plan;
    handle = nullptr;
}

void pascal_tdma_plan_many_create(void** handle, int n_row, int n_sys,
                                    int myrank, int nprocs,
                                    MPI_Fint mpi_comm_f, int tdma_type) {
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_f);
    auto* plan = new PTDMAPlanMany();
    TDMAType type = (tdma_type == 0) ? TDMAType::Standard : TDMAType::Cyclic;
    plan->create(n_row, n_sys, comm, type);
    *handle = static_cast<void*>(plan);
}

void pascal_tdma_many_solve(void* handle,
                              double* a, double* b, double* c, double* d) {
    auto* plan = static_cast<PTDMAPlanMany*>(handle);
    PTDMASolverMany::solve(*plan, a, b, c, d);
}

void pascal_tdma_plan_many_destroy(void* handle) {
    auto* plan = static_cast<PTDMAPlanMany*>(handle);
    plan->destroy();
    delete plan;
    handle = nullptr;
}

void pascal_tdma_plan_many_rhs_create(void** handle, int n_row, int n_sys,
                                    int myrank, int nprocs,
                                    MPI_Fint mpi_comm_f, int tdma_type) {
    MPI_Comm comm = MPI_Comm_f2c(mpi_comm_f);
    auto* plan = new PTDMAPlanManyRHS();
    TDMAType type = (tdma_type == 0) ? TDMAType::Standard : TDMAType::Cyclic;
    plan->create(n_row, n_sys, comm, type);
    *handle = static_cast<void*>(plan);
}

void pascal_tdma_many_rhs_solve(void* handle,
                              double* a, double* b, double* c, double* d) {
    auto* plan = static_cast<PTDMAPlanManyRHS*>(handle);
    PTDMASolverManyRHS::solve(*plan, a, b, c, d);
}

void pascal_tdma_plan_many_rhs_destroy(void* handle) {
    auto* plan = static_cast<PTDMAPlanManyRHS*>(handle);
    plan->destroy();
    delete plan;
    handle = nullptr;
}

}
