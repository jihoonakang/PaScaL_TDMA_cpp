program cuMany
    use mpi
    use iso_c_binding
    use cuda_fortran_interface   ! CuPTDMAPlanMany 관련 C 바인딩 인터페이스
    use fortran_interface     ! PTDMAPlanMany 관련 CPU 인터페이스
    implicit none

    integer :: ierr, rank, size
    integer, parameter :: nx=4, ny=40, nz=160
    integer, parameter :: N = nx*ny*nz
    integer :: i
    real(c_double) :: error

    real(c_double), allocatable :: a_h(:), b_h(:), c_h(:), d_h(:), d_h_out(:)
    type(c_ptr) :: a_d, b_d, c_d, d_d  ! GPU 포인터 (CUDA malloc)
    type(c_ptr) :: plan_cpu, plan_gpu

    ! MPI 초기화
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

    ! 호스트 배열 할당 및 초기화
    allocate(a_h(N), b_h(N), c_h(N), d_h(N), d_h_out(N))
    a_h = -1.0d0
    b_h =  4.0d0
    c_h = -1.0d0
    do i = 1, N
        d_h(i) = sin(real(i-1, c_double))
    end do

    ! ===== GPU 메모리 할당 및 복사 =====
    call cuda_malloc(a_d, N*8_8)  ! double=8 bytes
    call cuda_malloc(b_d, N*8_8)
    call cuda_malloc(c_d, N*8_8)
    call cuda_malloc(d_d, N*8_8)

    call cuda_memcpy_h2d(a_d, a_h, N)
    call cuda_memcpy_h2d(b_d, b_h, N)
    call cuda_memcpy_h2d(c_d, c_h, N)
    call cuda_memcpy_h2d(d_d, d_h, N)

    ! ===== CPU solver =====
    call pascal_tdma_plan_many_create(plan_cpu, nx, ny*nz, rank, size, MPI_COMM_WORLD, 0)
    call pascal_tdma_many_solve(plan_cpu, a_h, b_h, c_h, d_h)
    call pascal_tdma_plan_many_destroy(plan_cpu)

    ! ===== GPU solver =====
    call cu_pascal_tdma_plan_many_create(plan_gpu, nx, ny, nz, rank, size, MPI_COMM_WORLD, 0)
    call cu_pascal_tdma_many_solve(plan_gpu, a_d, b_d, c_d, d_d)
    call cu_pascal_tdma_plan_many_destroy(plan_gpu)

    ! 결과 복사
    call cuda_memcpy_d2h(d_h_out, d_d, N)

    ! 오차 계산
    if (rank == 0) then
        error = 0.0d0
        do i = 1, N
            error = error + abs(d_h(i) - d_h_out(i))
        end do
        print *, "Avg. norm2 error = ", error / N
    end if

    ! GPU 메모리 해제
    call cuda_free(a_d)
    call cuda_free(b_d)
    call cuda_free(c_d)
    call cuda_free(d_d)

    call MPI_Finalize(ierr)

end program cuMany