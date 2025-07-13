/**
 * @file pybind_cuda.hpp
 * @brief pybind11 bindings for the CUDA PaScaL_TDMA solver using CuPy.
 *
 * Exposes GPU-based plan creation and solver functions to Python,
 * operating on CuPy arrays via the CUDA Array Interface.
 */
#pragma once

#include <mpi.h>
#include <pybind11/pybind11.h>
#include "../PaScaL_TDMA.cuh"

namespace py = pybind11;
namespace cuTDMA = cuPaScaL_TDMA;

/**
 * @brief Extract raw device pointer from a CuPy array.
 *
 * Reads the __cuda_array_interface__ dictionary to obtain
 * the device pointer address.
 *
 * @param arr Python object exposing CUDA array interface.
 * @return Pointer to double data on the GPU.
 */
inline double* get_cupy_ptr(py::object arr) {
    py::dict iface = arr.attr("__cuda_array_interface__");
    auto data = iface["data"].cast<std::pair<size_t, bool>>();
    return reinterpret_cast<double*>(data.first);
}

/**
 * @brief Initialize the CUDA-enabled Python module.
 *
 * Exports plan classes and solver functions for GPU tridiagonal systems.
 */
PYBIND11_MODULE(PaScaL_TDMA_cuda_pybind, m) {
    m.doc() = R"doc(
        CUDA-based PaScaL_TDMA tridiagonal solver bindings.

        Plan classes (cuPTDMAPlanMany, cuPTDMAPlanManyRHS) manage GPU buffers;
        solve functions operate in-place on CuPy arrays.
    )doc";

    // ----------------------------------------------------------------
    // cuPTDMAPlanMany: plan for many systems on GPU
    // ----------------------------------------------------------------
    /**
     * @class cuPTDMAPlanMany
     * @brief GPU plan for solving multiple tridiagonal systems in YZ slabs.
     */
    py::class_<cuTDMA::cuPTDMAPlanMany>(m, "cuPTDMAPlanMany")
        .def(py::init<>())
        .def("create", []   (cuTDMA::cuPTDMAPlanMany& plan, 
                            int n_row, int ny_sys, int nz_sys, 
                            int comm, bool cyclic){

            auto type = cyclic ? cuTDMA::TDMAType::Cyclic 
                               : cuTDMA::TDMAType::Standard;
            plan.create(n_row, ny_sys, nz_sys, MPI_Comm_f2c(comm), type);

        },
        py::arg("n_row"),
        py::arg("ny_sys"),
        py::arg("nz_sys"),
        py::arg("communicator"),
        py::arg("cyclic") = false,
            "Initialize GPU plan for many tridiagonal systems.")
        .def("destroy", &cuTDMA::cuPTDMAPlanMany::destroy);

    // ----------------------------------------------------------------
    // cuPTDMAPlanManyRHS: plan for many RHS vectors on GPU
    // ----------------------------------------------------------------
    /**
     * @class cuPTDMAPlanManyRHS
     * @brief GPU plan for solving many RHS vectors with shared diagonals.
     */
    py::class_<cuTDMA::cuPTDMAPlanManyRHS>(m, "cuPTDMAPlanManyRHS")
        .def(py::init<>())
        .def("create", []   (cuTDMA::cuPTDMAPlanManyRHS& plan, 
                            int n_row, int ny_sys, int nz_sys, 
                            int comm, bool cyclic){

            auto type = cyclic ? cuTDMA::TDMAType::Cyclic 
                               : cuTDMA::TDMAType::Standard;
            plan.create(n_row, ny_sys, nz_sys, MPI_Comm_f2c(comm), type);

        },
        py::arg("n_row"),
        py::arg("ny_sys"),
        py::arg("nz_sys"),
        py::arg("communicator"),
        py::arg("cyclic") = false,
            "Initialize GPU plan for many-RHS tridiagonal solve.")
        .def("destroy", &cuTDMA::cuPTDMAPlanManyRHS::destroy);

    // ----------------------------------------------------------------
    // cuSolveMany: solver for GPU many systems
    // ----------------------------------------------------------------
    /**
     * @brief Solve multiple tridiagonal systems on GPU.
     *
     * @param plan GPU plan instance.
     * @param A    CuPy array of sub-diagonal coefficients.
     * @param B    CuPy array of diagonal coefficients.
     * @param C    CuPy array of super-diagonal coefficients.
     * @param D    CuPy array of RHS values, overwritten with solutions.
     */
    m.def("cuSolveMany", [](cuTDMA::cuPTDMAPlanMany& plan, 
                            py::object A, py::object B, 
                            py::object C, py::object D) {

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

    // ----------------------------------------------------------------
    // cuSolveManyRHS: solver for GPU many RHS vectors
    // ----------------------------------------------------------------
    /**
     * @brief Solve many RHS tridiagonal systems on GPU.
     *
     * @param plan GPU plan instance.
     * @param A    CuPy array of sub-diagonal coefficients.
     * @param B    CuPy array of diagonal coefficients.
     * @param C    CuPy array of super-diagonal coefficients.
     * @param D    CuPy array of RHS values, overwritten with solutions.
     */
    m.def("cuSolveManyRHS", [] (cuTDMA::cuPTDMAPlanManyRHS& plan, 
                                py::object A, py::object B,
                                py::object C, py::object D) {
                                    
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