!===================================================================================================
!> @file        fortran_interface.f90
!> @brief       Fortran interface for the PaScaL_TDMA C API
!> @details     This module provides Fortran bindings to the C interface functions of the
!>              PaScaL_TDMA library. It allows Fortran applications to create, solve, and destroy
!>              execution plans for single and multiple tridiagonal systems, with or without
!>              multiple right-hand sides (RHS).
!>
!>              The interface connects Fortran code with the C functions using the `bind(C)`
!>              attribute and `iso_c_binding` module to ensure correct interoperability between
!>              Fortran and C/C++.
!>
!> @note        All procedures are direct wrappers of the C API defined in
!>              `pascal_tdma_c_interface.hpp`.
!===================================================================================================

module fortran_interface
    use iso_c_binding
    implicit none

    interface

        !-------------------------------------------------------------------------------------------
        !> @brief   Create a plan for a single tridiagonal system.
        !> @param   handle    [out] Pointer to the created plan (C pointer)
        !> @param   n_row     [in]  Number of rows in the local partition
        !> @param   myrank    [in]  Rank ID in the communicator
        !> @param   nprocs    [in]  Number of MPI processes in the communicator
        !> @param   mpi_comm  [in]  MPI communicator (Fortran integer handle)
        !> @param   tdma_type [in]  Type of TDMA solver (0: standard, 1: cyclic)
        !--------------------------------------------------------------------------------------------
        subroutine pascal_tdma_plan_single_create(handle, n_row, myrank, nprocs, mpi_comm, &
                                                  tdma_type) bind(C)
            import :: c_ptr, c_int
            type(c_ptr) :: handle
            integer(c_int), value :: n_row, myrank, nprocs, mpi_comm, tdma_type
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Solve a single tridiagonal system using the provided plan.
        !> @param   handle [in]     Plan handle (C pointer)
        !> @param   a,b,c  [in]     Lower, diagonal, and upper coefficient arrays
        !> @param   d      [inout]  Right-hand side array, overwritten with solution
        !-------------------------------------------------------------------------------------------
        subroutine pascal_tdma_single_solve(handle, a, b, c, d) bind(C)
            import :: c_ptr, c_double
            type(c_ptr), value :: handle
            real(c_double) :: a(*), b(*), c(*), d(*)
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Destroy a single TDMA plan and release allocated resources.
        !> @param   handle [in] Plan handle (C pointer)
        !-------------------------------------------------------------------------------------------
        subroutine pascal_tdma_plan_single_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Create a plan for solving multiple tridiagonal systems.
        !> @param   handle    [out] Pointer to the created plan (C pointer)
        !> @param   n_row     [in]  Number of rows per system in the local partition
        !> @param   n_sys     [in]  Number of tridiagonal systems
        !> @param   myrank    [in]  Rank ID in the communicator
        !> @param   nprocs    [in]  Number of MPI processes in the communicator
        !> @param   mpi_comm  [in]  MPI communicator (Fortran integer handle)
        !> @param   tdma_type [in]  Type of TDMA solver (0: standard, 1: cyclic)
        !-------------------------------------------------------------------------------------------
        subroutine pascal_tdma_plan_many_create(handle, n_row, n_sys, myrank, nprocs, mpi_comm, &
                                                tdma_type) bind(C)
            import :: c_ptr, c_int
            type(c_ptr) :: handle
            integer(c_int), value :: n_row, n_sys, myrank, nprocs, mpi_comm, tdma_type
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Solve multiple tridiagonal systems using the provided plan.
        !> @param   handle [in]     Plan handle (C pointer)
        !> @param   a,b,c  [in]     Lower, diagonal, and upper coefficient arrays
        !> @param   d      [inout]  Right-hand side arrays, overwritten with solutions
        !-------------------------------------------------------------------------------------------
        subroutine pascal_tdma_many_solve(handle, a, b, c, d) bind(C)
            import :: c_ptr, c_double
            type(c_ptr), value :: handle
            real(c_double) :: a(*), b(*), c(*), d(*)
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Destroy a multiple TDMA plan and release allocated resources.
        !> @param   handle [in] Plan handle (C pointer)
        !-------------------------------------------------------------------------------------------
        subroutine pascal_tdma_plan_many_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Create a plan for multiple TDMA systems with multiple RHS partitions.
        !> @param   handle    [out] Pointer to the created plan (C pointer)
        !> @param   n_row     [in]  Number of rows per system in the local partition
        !> @param   n_sys     [in]  Number of right-hand sides
        !> @param   myrank    [in]  Rank ID in the communicator
        !> @param   nprocs    [in]  Number of MPI processes in the communicator
        !> @param   mpi_comm  [in]  MPI communicator (Fortran integer handle)
        !> @param   tdma_type [in]  Type of TDMA solver (0: standard, 1: cyclic)
        !-------------------------------------------------------------------------------------------
        subroutine pascal_tdma_plan_many_RHS_create(handle, n_row, n_sys, myrank, nprocs, &
                                                    mpi_comm, tdma_type) bind(C)
            import :: c_ptr, c_int
            type(c_ptr) :: handle
            integer(c_int), value :: n_row, n_sys, myrank, nprocs, mpi_comm, tdma_type
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Solve multiple TDMA systems with multiple RHS partitions.
        !> @param   handle [in]     Plan handle (C pointer)
        !> @param   a,b,c  [in]     Lower, diagonal, and upper coefficient arrays
        !> @param   d      [inout]  RHS arrays, overwritten with solutions
        !-------------------------------------------------------------------------------------------
        subroutine pascal_tdma_many_RHS_solve(handle, a, b, c, d) bind(C)
            import :: c_ptr, c_double
            type(c_ptr), value :: handle
            real(c_double) :: a(*), b(*), c(*), d(*)
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Destroy a multiple TDMA (RHS) plan and release allocated resources.
        !> @param   handle [in] Plan handle (C pointer)
        !-------------------------------------------------------------------------------------------
        subroutine pascal_tdma_plan_many_RHS_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

    end interface

end module fortran_interface