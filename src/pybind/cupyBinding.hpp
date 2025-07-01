#pragma once

#include <pybind11/pybind11.h>
#include <mpi.h>
#include "../PaScaL_TDMA.cuh"

namespace py = pybind11;
namespace cuTDMA = cuPaScaL_TDMA;

inline double* get_cupy_ptr(py::object arr) {
    py::dict iface = arr.attr("__cuda_array_interface__");
    auto data = iface["data"].cast<std::pair<size_t, bool>>();
    return reinterpret_cast<double*>(data.first);
}

PYBIND11_MODULE(PaScaL_TDMA_cuda_pybind, m) {
    m.doc() = "CUDA version of PaScaL_TDMA with CuPy support";

    py::class_<cuTDMA::cuPTDMAPlanMany>(m, "cuPTDMAPlanMany")
        .def(py::init<>())
        .def("create", [](cuTDMA::cuPTDMAPlanMany& plan, int n_row, int ny_sys, int nz_sys, int comm, bool cyclic){
            auto type = cyclic ? cuTDMA::TDMAType::Cyclic : cuTDMA::TDMAType::Standard;
            plan.create(n_row, ny_sys, nz_sys, MPI_Comm_f2c(comm), type);
        },
        py::arg("n_row"),
        py::arg("ny_sys"),
        py::arg("nz_sys"),
        py::arg("communicator"),
        py::arg("cyclic"),
        "Create plan_many_cuda")
        .def("destroy", &cuTDMA::cuPTDMAPlanMany::destroy);

    py::class_<cuTDMA::cuPTDMAPlanManyRHS>(m, "cuPTDMAPlanManyRHS")
        .def(py::init<>())
        .def("create", [](cuTDMA::cuPTDMAPlanManyRHS& plan, int n_row, int ny_sys, int nz_sys, int comm, bool cyclic){
            auto type = cyclic ? cuTDMA::TDMAType::Cyclic : cuTDMA::TDMAType::Standard;
            plan.create(n_row, ny_sys, nz_sys, MPI_Comm_f2c(comm), type);
        },
        py::arg("n_row"),
        py::arg("ny_sys"),
        py::arg("nz_sys"),
        py::arg("communicator"),
        py::arg("cyclic"),
        "Create plan_manyRHS_cuda" )
        .def("destroy", &cuTDMA::cuPTDMAPlanManyRHS::destroy);

    m.def("cuSolveMany", [](cuTDMA::cuPTDMAPlanMany& plan, py::object A, py::object B, py::object C, py::object D) {
        cuTDMA::cuPTDMASolverMany::cuSolve(plan,
            get_cupy_ptr(A),
            get_cupy_ptr(B),
            get_cupy_ptr(C),
            get_cupy_ptr(D));
        },
        py::arg("plan"),
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("D"),
    "Solve many tridiagonal systems");

    m.def("cuSolveManyRHS", [](cuTDMA::cuPTDMAPlanManyRHS& plan, py::object A, py::object B, py::object C, py::object D) {
        cuTDMA::cuPTDMASolverManyRHS::cuSolve(plan,
            get_cupy_ptr(A),
            get_cupy_ptr(B),
            get_cupy_ptr(C),
            get_cupy_ptr(D));
        },
        py::arg("plan"),
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("D"),
    "Solve many tridiagonal systems");
}