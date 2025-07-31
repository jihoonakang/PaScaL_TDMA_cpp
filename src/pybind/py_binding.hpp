/**
 * @file pybind.hpp
 * @brief pybind11 bindings for the CPU PaScaL_TDMA solver.
 *
 * Exposes plan objects and solver functions to Python,
 * operating on NumPy arrays for standard and cyclic tridiagonal systems.
 */
#include "mpi.h"
#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>
#include "pascal_tdma.hpp"

namespace py = pybind11;
namespace PaScaL_TDMA {

/**
 * @brief Python module for PaScaL_TDMA CPU solver.
 */
PYBIND11_MODULE(PaScaL_TDMA_pybind, m){

    m.doc() = R"doc(
        CPU-based PaScaL_TDMA tridiagonal solver bindings.

        Plan classes (Single, Many, ManyRHS) manage MPI decomposition;
        solve functions operate in-place on NumPy arrays.
    )doc";

    // ----------------------------------------------------------------
    // PTDMAPlanSingle: single-system tridiagonal plan
    // ----------------------------------------------------------------
    /**
     * @class PTDMAPlanSingle
     * @brief MPI-aware plan for one tridiagonal system (standard or cyclic).
     */
    py::class_<PTDMAPlanSingle>(m, "PTDMAPlanSingle")
        .def(py::init())
        .def("create",
            [](PTDMAPlanSingle &plan, int n_row, int comm_fhandle, bool cyclic){
                auto type = cyclic ? TDMAType::Cyclic : TDMAType::Standard;
                plan.create(n_row, MPI_Comm_f2c(static_cast<MPI_Fint>(comm_fhandle)), type);
            },
            py::arg("n_row"),
            py::arg("comm_fhandle"),
            py::arg("cyclic") = false,
            R"doc(
            Initialize plan for a single tridiagonal system.

            @param n_rows Number of equations (rows).
            @param comm_fhandle MPI communicator handle (Fortran integer).
            @param cyclic   True for cyclic TDMA, false for standard.
            )doc")
        .def("destroy",
            [](PTDMAPlanSingle& plan) {
                plan.destroy();
            },
            "Destroy plan_single");

    // ----------------------------------------------------------------
    // PTDMAPlanMany: multiple independent systems plan
    // ----------------------------------------------------------------
    /**
     * @class PTDMAPlanMany
     * @brief MPI-aware plan for many independent tridiagonal systems.
     */
    py::class_<PTDMAPlanMany>(m, "PTDMAPlanMany")
        .def(py::init())
        .def("create",
            [](PTDMAPlanMany &plan, int n_row, int n_sys, int comm_fhandle, bool cyclic){
                auto type = cyclic ? TDMAType::Cyclic : TDMAType::Standard;
                plan.create(n_row, n_sys, MPI_Comm_f2c(static_cast<MPI_Fint>(comm_fhandle)), type);
            },
            py::arg("n_row"),
            py::arg("n_sys"),
            py::arg("comm_fhandle"),
            py::arg("cyclic") = false,
            R"doc(
            Initialize plan for many tridiagonal system.s

            @param n_row Number of equations (rows).
            @param n_sys Number of system (columns).
            @param comm_fhandle MPI communicator handle (Fortran integer).
            @param cyclic   True for cyclic TDMA, false for standard.
            )doc")
        .def("destroy",
            [](PTDMAPlanMany& plan) {
                plan.destroy();
            },
            "Destroy plan_many");

    // ----------------------------------------------------------------
    // PTDMAPlanManyRHS: many RHS vectors with common diagonals
    // ----------------------------------------------------------------
    /**
     * @class PTDMAPlanManyRHS
     * @brief MPI-aware plan for many RHS solves sharing sub-/super-diagonals.
     */
    py::class_<PTDMAPlanManyRHS>(m, "PTDMAPlanManyRHS")
        .def(py::init())
        .def("create",
            [](PTDMAPlanManyRHS &plan, int n_row, int n_sys, int comm_fhandle, bool cyclic){
                auto type = cyclic ? TDMAType::Cyclic : TDMAType::Standard;
                plan.create(n_row, n_sys, MPI_Comm_f2c(static_cast<MPI_Fint>(comm_fhandle)), type);
            },
            py::arg("n_row"),
            py::arg("n_sys"),
            py::arg("comm_fhandle"),
            py::arg("cyclic") = false,
            R"doc(
            Initialize plan for many tridiagonal system.s

            @param n_row Number of equations (rows).
            @param n_sys Number of system (columns).
            @param comm_fhandle MPI communicator handle (Fortran integer).
            @param cyclic   True for cyclic TDMA, false for standard.
            )doc")
        .def("destroy",
            &PTDMAPlanManyRHS::destroy,
            "Destroy plan_many_RHS");

    /**
     * @brief Solve a single tridiagonal system in-place.
     *
     * @param plan Reference to a created PTDMAPlanSingle.
     * @param A    NumPy array (size n_rows) of lower-diagonal entries.
     * @param B    NumPy array (size n_rows) of diagonal entries.
     * @param C    NumPy array (size n_rows) of upper-diagonal entries.
     * @param D    NumPy array (size n_rows) of right-hand side, overwritten with solution.
     */
    m.def("solveSingle", [](PTDMAPlanSingle& plan,
            py::array_t<double, py::array::c_style | py::array::forcecast> A, 
            py::array_t<double, py::array::c_style | py::array::forcecast> B, 
            py::array_t<double, py::array::c_style | py::array::forcecast> C, 
            py::array_t<double, py::array::c_style | py::array::forcecast> D) {

                double* a = static_cast<double*>(A.request().ptr);
                double* b = static_cast<double*>(B.request().ptr);
                double* c = static_cast<double*>(C.request().ptr);
                double* d = static_cast<double*>(D.request().ptr);

                PTDMASolverSingle::solve(plan, a, b, c, d);
            },
            py::arg("plan"),
            py::arg("A"),
            py::arg("B"),
            py::arg("C"),
            py::arg("D"),
        "Solve a single tridiagonal system");

    /**
     * @brief Solve many tridiagonal systems in-place.
     *
     * @param plan Reference to a created PTDMAPlanSingle.
     * @param A    NumPy array (size n_rows * n_sys) of lower-diagonal entries.
     * @param B    NumPy array (size n_rows * n_sys) of diagonal entries.
     * @param C    NumPy array (size n_rows * n_sys) of upper-diagonal entries.
     * @param D    NumPy array (size n_rows * n_sys) of RHS, overwritten with solution.
     */
    m.def("solveMany", [](PTDMAPlanMany& plan,
            py::array_t<double, py::array::c_style | py::array::forcecast> A,
            py::array_t<double, py::array::c_style | py::array::forcecast> B,
            py::array_t<double, py::array::c_style | py::array::forcecast> C,
            py::array_t<double, py::array::c_style | py::array::forcecast> D) {

                double* a = static_cast<double*>(A.request().ptr);
                double* b = static_cast<double*>(B.request().ptr);
                double* c = static_cast<double*>(C.request().ptr);
                double* d = static_cast<double*>(D.request().ptr);

                PTDMASolverMany::solve(plan, a, b, c, d);
            },
            py::arg("plan"),
            py::arg("A"),
            py::arg("B"),
            py::arg("C"),
            py::arg("D"),
        "Solve many tridiagonal systems");

    /**
     * @brief Solve batched tridiagonal systems with many RHS in-place.
     *
     * @param plan Reference to a created PTDMAPlanSingle.
     * @param A    NumPy array (size n_rows) of lower-diagonal entries.
     * @param B    NumPy array (size n_rows) of diagonal entries.
     * @param C    NumPy array (size n_rows) of upper-diagonal entries.
     * @param D    NumPy array (size n_rows * n_sys) of RHS, overwritten with solution.
     */
    m.def("solveManyRHS", [] (PTDMAPlanManyRHS& plan,
            py::array_t<double, py::array::c_style | py::array::forcecast> A,
            py::array_t<double, py::array::c_style | py::array::forcecast> B,
            py::array_t<double, py::array::c_style | py::array::forcecast> C,
            py::array_t<double, py::array::c_style | py::array::forcecast> D) {

                double* a = static_cast<double*>(A.request().ptr);
                double* b = static_cast<double*>(B.request().ptr);
                double* c = static_cast<double*>(C.request().ptr);
                double* d = static_cast<double*>(D.request().ptr);

                PTDMASolverManyRHS::solve(plan, a, b, c, d);
            },
            py::arg("plan"),
            py::arg("A"),
            py::arg("B"),
            py::arg("C"),
            py::arg("D"),
        "Solve many RHS tridiagonal system");

}
}