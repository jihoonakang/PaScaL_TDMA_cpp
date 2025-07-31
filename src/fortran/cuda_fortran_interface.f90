!===================================================================================================
!> @file        cuda_fortran_interface.f90
!> @brief       Fortran interface for the CUDA-based PaScaL_TDMA C API
!> @details     This module provides Fortran bindings to the CUDA C interface functions of the
!>              PaScaL_TDMA library. It enables Fortran applications to allocate/free GPU memory,
!>              transfer data between host and device, and create, solve, and destroy execution
!>              plans for multiple tridiagonal systems on GPU.
!>
!>              All procedures are wrappers around the C API defined in
!>              `cuda_pascal_tdma_c_interface.hpp`. The interface uses `bind(C)` and `iso_c_binding`
!>              to ensure interoperability.
!===================================================================================================

module cuda_fortran_interface
    use iso_c_binding
    implicit none

    interface

        !-------------------------------------------------------------------------------------------
        !> @brief   Allocate GPU memory.
        !> @param   ptr   [out] C pointer to allocated device memory
        !> @param   bytes [in]  Number of bytes to allocate
        !-------------------------------------------------------------------------------------------
        subroutine cuda_malloc(ptr, bytes) bind(C, name="cuda_malloc")
            import :: c_ptr, c_size_t
            type(c_ptr) :: ptr
            integer(c_size_t), value :: bytes
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Free GPU memory.
        !> @param   ptr [in]  C pointer to device memory to free
        !-------------------------------------------------------------------------------------------
        subroutine cuda_free(ptr) bind(C, name="cuda_free")
            import :: c_ptr
            type(c_ptr), value :: ptr
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Copy data from host to device.
        !> @param   dst [out] Device pointer (destination)
        !> @param   src [in]  Host array (source)
        !> @param   n   [in]  Number of elements to copy (double precision)
        !-------------------------------------------------------------------------------------------
        subroutine cuda_memcpy_h2d(dst, src, n) bind(C, name="cuda_memcpy_h2d")
            import :: c_ptr, c_double, c_int
            type(c_ptr), value :: dst
            real(c_double) :: src(*)
            integer(c_int), value :: n
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Copy data from device to host.
        !> @param   dst [out] Host array (destination)
        !> @param   src [in]  Device pointer (source)
        !> @param   n   [in]  Number of elements to copy (double precision)
        !-------------------------------------------------------------------------------------------
        subroutine cuda_memcpy_d2h(dst, src, n) bind(C, name="cuda_memcpy_d2h")
            import :: c_ptr, c_double, c_int
            real(c_double) :: dst(*)
            type(c_ptr), value :: src
            integer(c_int), value :: n
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Create a plan for solving multiple tridiagonal systems on GPU.
        !> @param   handle    [out] Pointer to the created plan (C pointer)
        !> @param   n_row     [in]  Number of rows per system
        !> @param   ny_sys    [in]  Number of systems in y-dimension
        !> @param   nz_sys    [in]  Number of systems in z-dimension
        !> @param   myrank    [in]  MPI rank
        !> @param   nprocs    [in]  Number of MPI processes
        !> @param   mpi_comm  [in]  MPI communicator (Fortran integer handle)
        !> @param   tdma_type [in]  Type of TDMA solver (0=standard, 1=cyclic)
        !-------------------------------------------------------------------------------------------
        subroutine cu_pascal_tdma_plan_many_create(handle, n_row, ny_sys, nz_sys, myrank, nprocs, &
                                                   mpi_comm, tdma_type) bind(C)
            import :: c_ptr, c_int
            type(c_ptr) :: handle
            integer(c_int), value :: n_row, ny_sys, nz_sys, myrank, nprocs, mpi_comm, tdma_type
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Solve multiple tridiagonal systems using the provided CUDA plan.
        !> @param   handle [in]    Plan handle (C pointer)
        !> @param   a,b,c  [in]    Sub-diagonal, diagonal, and super-diagonal coefficient arrays
        !>                        (device pointers)
        !> @param   d      [inout] RHS arrays (device pointers), overwritten with solutions
        !-------------------------------------------------------------------------------------------
        subroutine cu_pascal_tdma_many_solve(handle, a, b, c, d) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr), value :: a, b, c, d
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Destroy a multiple TDMA plan and release GPU resources.
        !> @param   handle [in] Plan handle (C pointer)
        !-------------------------------------------------------------------------------------------
        subroutine cu_pascal_tdma_plan_many_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Create a plan for multiple TDMA systems with multiple RHS partitions on GPU.
        !> @param   handle    [out] Pointer to the created plan (C pointer)
        !> @param   n_row     [in]  Number of rows per system
        !> @param   ny_sys    [in]  Number of systems in y-dimension
        !> @param   nz_sys    [in]  Number of systems in z-dimension
        !> @param   myrank    [in]  MPI rank
        !> @param   nprocs    [in]  Number of MPI processes
        !> @param   mpi_comm  [in]  MPI communicator (Fortran integer handle)
        !> @param   tdma_type [in]  Type of TDMA solver (0=standard, 1=cyclic)
        !-------------------------------------------------------------------------------------------
        subroutine cu_pascal_tdma_plan_many_rhs_create(handle, n_row, ny_sys, nz_sys, &
                                                       myrank, nprocs, mpi_comm, tdma_type) bind(C)
            import :: c_ptr, c_int
            type(c_ptr) :: handle
            integer(c_int), value :: n_row, ny_sys, nz_sys, myrank, nprocs, mpi_comm, tdma_type
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Solve multiple TDMA systems with multiple RHS vectors using the provided plan.
        !> @param   handle [in]    Plan handle (C pointer)
        !> @param   a,b,c  [in]    Sub-diagonal, diagonal, and super-diagonal coefficient arrays
        !>                        (device pointers)
        !> @param   d      [inout] RHS arrays (device pointers), overwritten with solutions
        !-------------------------------------------------------------------------------------------
        subroutine cu_pascal_tdma_many_rhs_solve(handle, a, b, c, d) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
            type(c_ptr), value :: a, b, c, d
        end subroutine

        !-------------------------------------------------------------------------------------------
        !> @brief   Destroy a multiple TDMA (RHS) plan and release GPU resources.
        !> @param   handle [in] Plan handle (C pointer)
        !-------------------------------------------------------------------------------------------
        subroutine cu_pascal_tdma_plan_many_rhs_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

    end interface

end module cuda_fortran_interface