!===================================================================================================
!> @file        cuda_many.f90
!> @brief       Example program for solving multiple tridiagonal systems on GPU using CuPaScaL_TDMA.
!> @details     This example demonstrates how to use both the CPU (PaScaL_TDMA) and GPU 
!>              (CuPaScaL_TDMA) solvers to compute solutions of multiple tridiagonal systems.
!>              It allocates memory on both host and device, performs computation on both solvers,
!>              and compares the solutions to compute the average error.
!===================================================================================================

program cuda_many
    use mpi
    use iso_c_binding
    use cuda_fortran_interface      !< CUDA memory allocation and copy interface
    use fortran_interface           !< Fortran-C interface for PaScaL_TDMA
    implicit none

    integer, parameter :: nx = 4, ny = 40, nz = 160
    integer, parameter :: N = nx * ny * nz

    integer :: ierr, rank, size, i
    real(c_double) :: error

    !> Host-side arrays
    real(c_double), allocatable :: a_h(:), b_h(:), c_h(:), d_h(:), d_h_out(:)

    !> Device pointers for GPU arrays
    type(c_ptr) :: a_d, b_d, c_d, d_d

    !> Solver plans for CPU and GPU
    type(c_ptr) :: plan_cpu, plan_gpu

    !-----------------------------------------------------------------------------------------------
    ! Initialize MPI
    !-----------------------------------------------------------------------------------------------
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

    !-----------------------------------------------------------------------------------------------
    ! Allocate and initialize host arrays
    !-----------------------------------------------------------------------------------------------
    allocate(a_h(N), b_h(N), c_h(N), d_h(N), d_h_out(N))
    a_h = -1.0d0
    b_h =  4.0d0
    c_h = -1.0d0
    do i = 1, N
        d_h(i) = sin(real(i-1, c_double))
    end do

    !-----------------------------------------------------------------------------------------------
    ! Allocate GPU memory and copy data from host to device
    !-----------------------------------------------------------------------------------------------
    call cuda_malloc(a_d, N * c_sizeof(0.0d0))  ! double precision = 8 bytes
    call cuda_malloc(b_d, N * c_sizeof(0.0d0))
    call cuda_malloc(c_d, N * c_sizeof(0.0d0))
    call cuda_malloc(d_d, N * c_sizeof(0.0d0))

    call cuda_memcpy_h2d(a_d, a_h, N)
    call cuda_memcpy_h2d(b_d, b_h, N)
    call cuda_memcpy_h2d(c_d, c_h, N)
    call cuda_memcpy_h2d(d_d, d_h, N)

    !-----------------------------------------------------------------------------------------------
    ! Solve using CPU solver (PaScaL_TDMA)
    !-----------------------------------------------------------------------------------------------
    call pascal_tdma_plan_many_create(plan_cpu, nx, ny*nz, rank, size, MPI_COMM_WORLD, 0)
    call pascal_tdma_many_solve(plan_cpu, a_h, b_h, c_h, d_h)
    call pascal_tdma_plan_many_destroy(plan_cpu)

    !-----------------------------------------------------------------------------------------------
    ! Solve using GPU solver (CuPaScaL_TDMA)
    !-----------------------------------------------------------------------------------------------
    call cu_pascal_tdma_plan_many_create(plan_gpu, nx, ny, nz, rank, size, MPI_COMM_WORLD, 0)
    call cu_pascal_tdma_many_solve(plan_gpu, a_d, b_d, c_d, d_d)
    call cu_pascal_tdma_plan_many_destroy(plan_gpu)

    !-----------------------------------------------------------------------------------------------
    ! Copy GPU result back to host
    !-----------------------------------------------------------------------------------------------
    call cuda_memcpy_d2h(d_h_out, d_d, N)

    !-----------------------------------------------------------------------------------------------
    ! Compute and print average error
    !-----------------------------------------------------------------------------------------------
    if (rank == 0) then
        print *, "Avg. norm2 error = ", norm2((d_h - d_h_out) / N)
    end if

    !-----------------------------------------------------------------------------------------------
    ! Free GPU memory
    !-----------------------------------------------------------------------------------------------
    call cuda_free(a_d)
    call cuda_free(b_d)
    call cuda_free(c_d)
    call cuda_free(d_d)

    ! Finalize MPI
    call MPI_Finalize(ierr)

end program cuda_many