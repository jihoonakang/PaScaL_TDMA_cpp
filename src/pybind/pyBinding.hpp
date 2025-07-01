#include "mpi.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "PaScaL_TDMA.hpp"


template<class T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& values)
{
	copy( begin(values), end(values), std::ostream_iterator<T>(stream, ", ") );
	return stream;
}

namespace py = pybind11;
namespace PaScaL_TDMA {

PYBIND11_MODULE(PaScaL_TDMA_pybind, m){

    m.doc() = "C++ CPU version of PaScaL_TDMA with NumPy support";

    // Binding for a single trigonal system
    // Plan binding: PTDMAPlanSingle

    py::class_<PTDMAPlanSingle>(m, "PTDMAPlanSingle")
        .def(py::init())
        .def("create",
            [](PTDMAPlanSingle &plan, int n_row_, int comm_fhandle_, bool cyclic_){
                auto type = cyclic_ ? TDMAType::Cyclic : TDMAType::Standard;
                plan.create(n_row_, MPI_Comm_f2c(static_cast<MPI_Fint>(comm_fhandle_)), type);
            },
            py::arg("n_row"),
            py::arg("communicator"),
            py::arg("cyclic"),
            "Create plan_single")
        .def("destroy",
            [](PTDMAPlanSingle& plan) {
                plan.destroy();
            },
            "Destroy plan_single");

    // Binding for many trigonal systems
    // Plan binding: PTDMAPlanMany

    py::class_<PTDMAPlanMany>(m, "PTDMAPlanMany")
        .def(py::init())
        .def("create",
            [](PTDMAPlanMany &plan, int n_row_, int n_sys_, int comm_fhandle_, bool cyclic_){
                auto type = cyclic_ ? TDMAType::Cyclic : TDMAType::Standard;
                plan.create(n_row_, n_sys_, MPI_Comm_f2c(static_cast<MPI_Fint>(comm_fhandle_)), type);
            },
            py::arg("n_row"),
            py::arg("n_sys"),
            py::arg("communicator"),
            py::arg("cyclic"),
            "Create plan_many")
        .def("destroy",
            [](PTDMAPlanMany& plan) {
                plan.destroy();
            },
            "Destroy plan_many");

    // Binding for a trigonal system with many RHS
    // Plan binding: PTDMAPlanManyRHS

    py::class_<PTDMAPlanManyRHS>(m, "PTDMAPlanManyRHS")
        .def(py::init())
        .def("create",
            [](PTDMAPlanManyRHS &plan, int n_row_, int n_sys_, int comm_fhandle_, bool cyclic_){
                auto type = cyclic_ ? TDMAType::Cyclic : TDMAType::Standard;
                plan.create(n_row_, n_sys_, MPI_Comm_f2c(static_cast<MPI_Fint>(comm_fhandle_)), type);
            },
            py::arg("n_row"),
            py::arg("n_sys"),
            py::arg("communicator"),
            py::arg("cyclic"),
            "Create plan_many_RHS")
        .def("destroy",
            &PTDMAPlanManyRHS::destroy,
            "Destroy plan_many_RHS");

        // Solver function bindings (not classes)

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