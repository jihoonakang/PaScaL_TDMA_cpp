module cuFortranInterface
    use iso_c_binding
    implicit none

    interface

        subroutine cuda_malloc(ptr, bytes) bind(C, name="cuda_malloc")
            use iso_c_binding
            type(c_ptr) :: ptr
            integer(c_size_t), value :: bytes
        end subroutine

        subroutine cuda_free(ptr) bind(C, name="cuda_free")
            use iso_c_binding
            type(c_ptr), value :: ptr
        end subroutine

        subroutine cuda_memcpy_h2d(dst, src, n) bind(C, name="cuda_memcpy_h2d")
            use iso_c_binding
            type(c_ptr), value :: dst
            real(c_double) :: src(*)
            integer(c_int), value :: n
        end subroutine

        subroutine cuda_memcpy_d2h(dst, src, n) bind(C, name="cuda_memcpy_d2h")
            use iso_c_binding
            real(c_double) :: dst(*)
            type(c_ptr), value :: src
            integer(c_int), value :: n
        end subroutine

        subroutine cu_pascal_tdma_plan_many_create(handle, n_row, ny_sys, nz_sys, myrank, nprocs, mpi_comm, tdma_type) bind(C)
            import :: c_ptr, c_int
            type(c_ptr) :: handle
            integer(c_int), value :: n_row, ny_sys, nz_sys, myrank, nprocs, mpi_comm, tdma_type
        end subroutine

        subroutine cu_pascal_tdma_many_solve(handle, a, b, c, d) bind(C)
            import :: c_ptr, c_double
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: d
        end subroutine

        subroutine cu_pascal_tdma_plan_many_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

        subroutine cu_pascal_tdma_plan_many_rhs_create(handle, n_row, ny_sys, nz_sys, myrank, nprocs, mpi_comm, tdma_type) bind(C)
            import :: c_ptr, c_int
            type(c_ptr) :: handle
            integer(c_int), value :: n_row, ny_sys, nz_sys, myrank, nprocs, mpi_comm, tdma_type
        end subroutine

        subroutine cu_pascal_tdma_many_rhs_solve(handle, a, b, c, d) bind(C)
            import :: c_ptr, c_double
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: d
        end subroutine

        subroutine cu_pascal_tdma_plan_many_rhs_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

    end interface

end module cuFortranInterface