module fortran_interface
  	use iso_c_binding
  	implicit none

  	interface

        subroutine PaScaL_TDMA_plan_single_create(handle, n_row, myrank, nprocs, mpi_comm, tdma_type) bind(C)
      	    import :: c_ptr, c_int
      	    type(c_ptr) :: handle
      	    integer(c_int), value :: n_row, myrank, nprocs, mpi_comm, tdma_type
    	end subroutine

        subroutine PaScaL_TDMA_single_solve(handle, a, b, c, d) bind(C)
	        import :: c_ptr, c_double
            type(c_ptr), value :: handle
            real(c_double) :: a(*), b(*), c(*), d(*)
        end subroutine

        subroutine PaScaL_TDMA_plan_single_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

        subroutine PaScaL_TDMA_plan_many_create(handle, n_row, n_sys, myrank, nprocs, mpi_comm, tdma_type) bind(C)
      	    import :: c_ptr, c_int
      	    type(c_ptr) :: handle
      	    integer(c_int), value :: n_row, n_sys, myrank, nprocs, mpi_comm, tdma_type
    	end subroutine

        subroutine PaScaL_TDMA_many_solve(handle, a, b, c, d) bind(C)
	        import :: c_ptr, c_double
            type(c_ptr), value :: handle
            real(c_double) :: a(*), b(*), c(*), d(*)
        end subroutine

        subroutine PaScaL_TDMA_plan_many_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

        subroutine PaScaL_TDMA_plan_many_RHS_create(handle, n_row, n_sys, myrank, nprocs, mpi_comm, tdma_type) bind(C)
      	    import :: c_ptr, c_int
      	    type(c_ptr) :: handle
      	    integer(c_int), value :: n_row, n_sys, myrank, nprocs, mpi_comm, tdma_type
    	end subroutine

        subroutine PaScaL_TDMA_many_RHS_solve(handle, a, b, c, d) bind(C)
	        import :: c_ptr, c_double
            type(c_ptr), value :: handle
            real(c_double) :: a(*), b(*), c(*), d(*)
        end subroutine

        subroutine PaScaL_TDMA_plan_many_RHS_destroy(handle) bind(C)
            import :: c_ptr
            type(c_ptr), value :: handle
        end subroutine

    end interface

end module fortran_interface