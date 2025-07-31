!===================================================================================================
!> @file        single.f90
!> @brief       Example program for solving a single tridiagonal system using PaScaL_TDMA in Fortran.
!> @details     This example demonstrates distributed TDMA solving in MPI using the Fortran interface
!>              to the PaScaL_TDMA library. It partitions the problem, scatters/gathers data, and
!>              compares the computed solution with the known exact solution.
!===================================================================================================

program main

    use mpi
    use iso_c_binding
    use fortran_interface

    implicit none

    integer, parameter :: N = 100
    integer :: nprocs, myrank, ierr, i, n_sub
    integer :: para_range_n
    logical :: is_root = .false.

    double precision, allocatable :: d(:), x(:)
    double precision, allocatable :: a_sub(:), b_sub(:), c_sub(:), d_sub(:), x_sub(:)
    integer, allocatable :: cnt(:), disp(:)
    type(c_ptr) :: px_single  ! Plan handle for a single tridiagonal system

    call MPI_Init(ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

    if (myrank == 0) is_root = .true.

    n_sub = para_range_n(1, N, nprocs, myrank)
    call build_cnt_disp_array()

    if (is_root) then
        allocate(d(N), x(N))
        d(:) = 0.0
        x(:) = 0.0
        call build_global_coeff_array()
    else
        allocate(d(0), x(0))
    end if

    call build_local_coeff_array()

    ! Scatter RHS vector to all ranks
    call MPI_Scatterv(d, cnt, disp, MPI_DOUBLE_PRECISION, d_sub, n_sub, MPI_DOUBLE_PRECISION, 0, &
                      MPI_COMM_WORLD, ierr)

    ! Solve the TDMA system
    call pascal_tdma_plan_single_create(px_single, n_sub, myrank, nprocs, MPI_COMM_WORLD, 0)
    call pascal_tdma_single_solve(px_single, a_sub, b_sub, c_sub, d_sub)
    call pascal_tdma_plan_single_destroy(px_single)

    ! Gather solution and compute norm2 error
    call MPI_Gatherv(d_sub, n_sub, MPI_DOUBLE_PRECISION, d, cnt, disp, MPI_DOUBLE_PRECISION, 0, &
                     MPI_COMM_WORLD, ierr)

    if (is_root) print *, "Avg. norm2 error = ", norm2(d - x) / N

    call dealloc_all()
    call MPI_Finalize(ierr)

contains
    !-----------------------------------------------------------------------------------------------
    !> @brief Build arrays for scatter/gather counts and displacements.
    !-----------------------------------------------------------------------------------------------
    subroutine build_cnt_disp_array

        allocate(cnt(nprocs), disp(nprocs))
        cnt(:) = 0
        disp(:) = 0

        call MPI_Gather(n_sub, 1, MPI_INTEGER, cnt, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

        if (myrank == 0) then
            disp(1) = 0
            do i = 2, size(cnt)
                disp(i) = disp(i - 1) + cnt(i - 1)
            end do
        end if

    end subroutine build_cnt_disp_array

    !-----------------------------------------------------------------------------------------------
    !> @brief Allocate and initialize local coefficient arrays.
    !-----------------------------------------------------------------------------------------------
    subroutine build_local_coeff_array

        allocate(a_sub(n_sub), b_sub(n_sub), c_sub(n_sub), d_sub(n_sub), x_sub(n_sub))

        a_sub(:) = 1.0
        b_sub(:) = 2.0
        c_sub(:) = 1.0
        d_sub(:) = 0.0
        x_sub(:) = 0.0

    end subroutine build_local_coeff_array

    !-----------------------------------------------------------------------------------------------
    !> @brief Generate global coefficient arrays and compute RHS vector on rank 0.
    !-----------------------------------------------------------------------------------------------
    subroutine build_global_coeff_array

        double precision, allocatable :: a(:), b(:), c(:)

        allocate(a(N), b(N), c(N))

        a(:) = 1.0
        b(:) = 2.0
        c(:) = 1.0

        call random_number(x)

        d(1) = b(1) * x(1) + c(1) * x(2)
        do i = 2, N - 1
            d(i) = a(i) * x(i - 1) + b(i) * x(i) + c(i) * x(i + 1)
        end do
        d(N) = a(N) * x(N - 1) + b(N) * x(N)
        deallocate(a, b, c)

    end subroutine build_global_coeff_array

    !-----------------------------------------------------------------------------------------------
    !> @brief Deallocate all allocated arrays.
    !-----------------------------------------------------------------------------------------------
    subroutine dealloc_all

        deallocate(d, x, cnt, disp, a_sub, b_sub, c_sub, d_sub, x_sub)

    end subroutine dealloc_all

end program main

!===================================================================================================
!> @brief Compute the number of elements for a given rank based on the global range partition.
!===================================================================================================
integer function para_range_n(n1, n2, nprocs, myrank) result(n)

    implicit none

    integer, intent(in) :: n1, n2, nprocs, myrank
    integer :: remainder

    n = int((n2 - n1 + 1) / nprocs)
    remainder = mod(n2 - n1 + 1, nprocs)
    if (remainder > myrank) n = n + 1
    
end function para_range_n