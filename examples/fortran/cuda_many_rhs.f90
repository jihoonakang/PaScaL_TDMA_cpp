!===================================================================================================
!> @file        cu_many_rhs.f90
!> @brief       GPU-accelerated example for solving multiple RHS (Many RHS) TDMA systems.
!> @details     This program demonstrates how to solve a tridiagonal system with multiple RHS
!>              using CuPaScaL_TDMA on GPU and compares the results with the CPU solver.
!===================================================================================================

program cuda_many_rhs

    use mpi
    use iso_c_binding
    use cuda_fortran_interface       ! C bindings for CuPTDMAPlanManyRHS
    use fortran_interface            ! Fortran interface for PTDMAPlanManyRHS

    implicit none

    integer :: ierr, rank, size
    integer, parameter :: nx=4, ny=40, nz=160
    integer, parameter :: N = nx*ny*nz
    integer :: i
    real(c_double) :: error

    ! Host arrays
    real(c_double), allocatable :: a_h(:), b_h(:), c_h(:), d_h(:), d_h_out(:)

    ! Device pointers
    type(c_ptr) :: a_d, b_d, c_d, d_d

    ! TDMA plan handles
    type(c_ptr) :: plan_cpu, plan_gpu

    !------------------------------------------------------------------------------------------------
    ! Initialize MPI
    !------------------------------------------------------------------------------------------------
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

    !------------------------------------------------------------------------------------------------
    ! Allocate and initialize host arrays
    !------------------------------------------------------------------------------------------------
    allocate(a_h(nx), b_h(nx), c_h(nx), d_h(N), d_h_out(N))
    a_h = -1.0d0
    b_h = 10.0d0
    c_h = -1.0d0

    do i = 1, N
        d_h(i) = sin(real(i-1, c_double))
    end do

    !------------------------------------------------------------------------------------------------
    ! Allocate GPU memory
    !------------------------------------------------------------------------------------------------
    call cuda_malloc(a_d, nx * c_sizeof(0.0d0))
    call cuda_malloc(b_d, nx * c_sizeof(0.0d0))
    call cuda_malloc(c_d, nx * c_sizeof(0.0d0))
    call cuda_malloc(d_d, N  * c_sizeof(0.0d0))

    !------------------------------------------------------------------------------------------------
    ! Copy data from host to device
    !------------------------------------------------------------------------------------------------
    call cuda_memcpy_h2d(a_d, a_h, nx)
    call cuda_memcpy_h2d(b_d, b_h, nx)
    call cuda_memcpy_h2d(c_d, c_h, nx)
    call cuda_memcpy_h2d(d_d, d_h, N)

    !------------------------------------------------------------------------------------------------
    ! CPU-based Many RHS TDMA solve
    !------------------------------------------------------------------------------------------------
    call pascal_tdma_plan_many_rhs_create(plan_cpu, nx, ny*nz, rank, size, MPI_COMM_WORLD, 0)
    call pascal_tdma_many_rhs_solve(plan_cpu, a_h, b_h, c_h, d_h)
    call pascal_tdma_plan_many_rhs_destroy(plan_cpu)

    !------------------------------------------------------------------------------------------------
    ! GPU-based Many RHS TDMA solve
    !------------------------------------------------------------------------------------------------
    call cu_pascal_tdma_plan_many_rhs_create(plan_gpu, nx, ny, nz, rank, size, MPI_COMM_WORLD, 0)
    call cu_pascal_tdma_many_rhs_solve(plan_gpu, a_d, b_d, c_d, d_d)
    call cu_pascal_tdma_plan_many_rhs_destroy(plan_gpu)

    !------------------------------------------------------------------------------------------------
    ! Copy solution back to host
    !------------------------------------------------------------------------------------------------
    call cuda_memcpy_d2h(d_h_out, d_d, N)

    !------------------------------------------------------------------------------------------------
    ! Compute RMS error
    !------------------------------------------------------------------------------------------------
    if (rank == 0) then
        print *, "Avg. norm2 error = ", norm2((d_h - d_h_out) / N)
    end if

    !------------------------------------------------------------------------------------------------
    ! Free GPU memory and finalize MPI
    !------------------------------------------------------------------------------------------------
    call cuda_free(a_d)
    call cuda_free(b_d)
    call cuda_free(c_d)
    call cuda_free(d_d)

    call MPI_Finalize(ierr)

end program cuda_many_rhs