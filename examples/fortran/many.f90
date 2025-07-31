!===================================================================================================
!> @file        many.f90
!> @brief       Example program for solving multiple tridiagonal systems in 2D using PaScaL_TDMA.
!> @details     This example distributes a 2D domain across MPI processes, builds subcommunicators
!>              for x- and y-directions, and solves the tridiagonal systems in both directions using
!>              the PaScaL_TDMA library. It computes the average norm2 error compared to the 
!>              reference solution generated on the root process.
!===================================================================================================

program main

    use mpi
    use iso_c_binding
    use fortran_interface
    use mpi_topology_2d

    implicit none

    integer :: nx = 100, ny = 100
    integer :: nx_sub, ny_sub, n_sub
    integer :: nprocs, myrank, ierr
    integer :: npx
    integer :: para_range_n
    integer, allocatable :: cnt_x(:), disp_x(:), cnt_y(:), disp_y(:), cnt_all(:), disp_all(:)
    logical :: is_root = .false.

    double precision, allocatable :: d(:,:), x(:,:)
    double precision, allocatable :: a_sub(:,:), b_sub(:,:), c_sub(:,:), d_sub(:,:), d_sub_tr(:,:)

    type(c_ptr) :: px_many, py_many   !< Plans for multiple tridiagonal systems

    call MPI_Init(ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

    if (myrank == 0) is_root = .true.

    call MPI_Dims_create(nprocs, 2, np_dim, ierr)
    period = [.false., .false.]
    call mpi_topology_make()

    npx = np_dim(0)
    nx_sub = para_range_n(1, nx, comm_1d_x%nprocs, comm_1d_x%myrank)
    ny_sub = para_range_n(1, ny, comm_1d_y%nprocs, comm_1d_y%myrank)
    n_sub = nx_sub * ny_sub

    call build_comm_info_array()

    ! Allocate and build reference solution on root
    if (is_root) then
        allocate(d(nx, ny), x(nx, ny))
        call build_global_coeff_array()
    end if

    call distribute_rhs_array()

    ! Solve in y-direction
    allocate(a_sub(nx_sub, ny_sub), b_sub(nx_sub, ny_sub), c_sub(nx_sub, ny_sub))
    a_sub = 1.0; b_sub = 2.0; c_sub = 1.0

    call pascal_tdma_plan_many_create(py_many, ny_sub, nx_sub, comm_1d_y%myrank, comm_1d_y%nprocs, &
                                      comm_1d_y%mpi_comm, 0)
    call pascal_tdma_many_solve(py_many, a_sub, b_sub, c_sub, d_sub)
    call pascal_tdma_plan_many_destroy(py_many)

    ! Solve in x-direction
    a_sub = 1.0; b_sub = 2.0; c_sub = 1.0
    allocate(d_sub_tr(ny_sub, nx_sub))
    d_sub_tr = transpose(d_sub)

    call pascal_tdma_plan_many_create(px_many, nx_sub, ny_sub, comm_1d_x%myrank, comm_1d_x%nprocs, &
                                      comm_1d_x%mpi_comm, 0)
    call pascal_tdma_many_solve(px_many, a_sub, b_sub, c_sub, d_sub_tr)
    call pascal_tdma_plan_many_destroy(px_many)

    d_sub = transpose(d_sub_tr)
    deallocate(d_sub_tr)

    call collect_solution_array()

    if (is_root) print *, "Avg. norm2 error = ", norm2(d - x) / nx / ny

    call dealloc_all()
    call mpi_topology_clean()
    call MPI_Finalize(ierr)

contains
    !-----------------------------------------------------------------------------------------------
    !> @brief Build count and displacement arrays for scatter/gather operations.
    !-----------------------------------------------------------------------------------------------
    subroutine build_comm_info_array()

        integer :: i

        allocate(cnt_x(np_dim(0)), cnt_y(np_dim(1)), cnt_all(nprocs))
        allocate(disp_x(np_dim(0)), disp_y(np_dim(1)), disp_all(nprocs))

        call MPI_Allgather(nx_sub, 1, MPI_INTEGER, cnt_x, 1, MPI_INTEGER, comm_1d_x%mpi_comm, ierr)
        call MPI_Allgather(ny_sub, 1, MPI_INTEGER, cnt_y, 1, MPI_INTEGER, comm_1d_y%mpi_comm, ierr)
        call MPI_Allgather(n_sub,  1, MPI_INTEGER, cnt_all, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)

        disp_x(1) = 0
        do i = 2, size(cnt_x)
            disp_x(i) = disp_x(i - 1) + cnt_x(i - 1)
        end do

        disp_y(1) = 0
        do i = 2, size(cnt_y)
            disp_y(i) = disp_y(i - 1) + cnt_y(i - 1)
        end do

        disp_all(1) = 0
        do i = 2, nprocs
            disp_all(i) = disp_all(i - 1) + cnt_all(i - 1)
        end do

    end subroutine build_comm_info_array

    !-----------------------------------------------------------------------------------------------
    !> @brief Build reference coefficients and right-hand side arrays on the root process.
    !-----------------------------------------------------------------------------------------------
    subroutine build_global_coeff_array()

        integer :: i, j
        double precision, allocatable :: a(:,:), b(:,:), c(:,:), y(:,:)

        allocate(a(nx, ny), b(nx, ny), c(nx, ny), y(nx, ny))
        a = 1.0; b = 2.0; c = 1.0; y = 0.0

        call random_number(x)

        ! y = A_x * x
        do j = 1, ny
            y(1, j) = b(1, j) * x(1, j) + c(1, j) * x(2, j)
            do i = 2, nx - 1
                y(i, j) = a(i, j) * x(i - 1, j) + b(i, j) * x(i, j) + c(i, j) * x(i + 1, j)
            end do
            y(nx, j) = a(nx, j) * x(nx - 1, j) + b(nx, j) * x(nx, j)
        end do

        ! d = A_y * y
        do i = 1, nx
            d(i, 1) = b(i, 1) * y(i, 1) + c(i, 1) * y(i, 2)
        end do
        do j = 2, ny - 1
            do i = 1, nx
                d(i, j) = a(i, j) * y(i, j - 1) + b(i, j) * y(i, j) + c(i, j) * y(i, j + 1)
            end do
        end do
        do i = 1, nx
            d(i, ny) = a(i, ny) * y(i, ny - 1) + b(i, ny) * y(i, ny)
        end do

        deallocate(a, b, c, y)

    end subroutine build_global_coeff_array

    !-----------------------------------------------------------------------------------------------
    !> @brief Distribute the right-hand side array to all processes.
    !-----------------------------------------------------------------------------------------------
    subroutine distribute_rhs_array()

        integer :: i, j, iblk
        double precision, allocatable :: d_blk(:)

        if (is_root) then
            allocate(d_blk(nx * ny))
            do iblk = 1, npx
                do j = 1, ny
                    do i = 1, cnt_x(iblk)
                        d_blk(i + (j - 1) * cnt_x(iblk) + disp_x(iblk) * ny) = &
                            d(i + disp_x(iblk), j)
                    end do
                end do
            end do
        else
            allocate(d_blk(0))
        end if

        allocate(d_sub(nx_sub, ny_sub))
        call MPI_Scatterv(d_blk, cnt_all, disp_all, MPI_DOUBLE_PRECISION, d_sub, n_sub, &
                          MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
        deallocate(d_blk)

    end subroutine distribute_rhs_array

    !-----------------------------------------------------------------------------------------------
    !> @brief Collect the solution from all processes to the root process.
    !-----------------------------------------------------------------------------------------------
    subroutine collect_solution_array()

        integer :: i, j, iblk
        double precision, allocatable :: d_blk(:)

        if (is_root) then
            allocate(d_blk(nx * ny))
        else
            allocate(d_blk(0))
        end if

        call MPI_Gatherv(d_sub, n_sub, MPI_DOUBLE_PRECISION, d_blk, cnt_all, disp_all, &
                         MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)

        if (is_root) then
            do iblk = 1, npx
                do j = 1, ny
                    do i = 1, cnt_x(iblk)
                        d(i + disp_x(iblk), j) = &
                            d_blk(i + (j - 1) * cnt_x(iblk) + disp_x(iblk) * ny)
                    end do
                end do
            end do
        end if

        deallocate(d_blk)

    end subroutine collect_solution_array

    !-----------------------------------------------------------------------------------------------
    !> @brief Deallocate all allocated arrays.
    !-----------------------------------------------------------------------------------------------
    subroutine dealloc_all()
        if (is_root) then
            deallocate(d, x)
        end if
        deallocate(a_sub, b_sub, c_sub, d_sub)
        deallocate(cnt_x, disp_x, cnt_y, disp_y, cnt_all, disp_all)
    end subroutine dealloc_all
end program main

!---------------------------------------------------------------------------------------------------
!> @brief Compute the number of elements assigned to each process in a 1D partitioning.
!---------------------------------------------------------------------------------------------------
integer function para_range_n(n1, n2, nprocs, myrank) result(n)

    implicit none

    integer, intent(in) :: n1, n2, nprocs, myrank
    integer :: remainder

    n = int((n2 - n1 + 1) / nprocs)
    remainder = mod(n2 - n1 + 1, nprocs)
    if (remainder > myrank) n = n + 1

end function para_range_n