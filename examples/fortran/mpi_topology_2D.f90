!===================================================================================================
!> @file        mpi_topology_2d.f90
!> @brief       Module for creating a 2D Cartesian MPI topology and its subcommunicators.
!> @details     This module provides functionality to create and destroy a 2D Cartesian topology for
!>              MPI processes. It defines two 1D subcommunicators (x- and y-directions) and stores
!>              rank, size, and neighbor information for each subcommunicator.
!===================================================================================================

module mpi_topology_2d

    use mpi
    
    implicit none

    !-----------------------------------------------------------------------------------------------
    !> @brief Global communicator for Cartesian topology
    integer, public :: mpi_world_cart

    !> @brief Number of MPI processes in each dimension (2D topology)
    integer, public :: np_dim(0:1)

    !> @brief Periodicity in each dimension
    logical, public :: period(0:1)

    !-----------------------------------------------------------------------------------------------
    !> @brief Type for holding information about a 1D Cartesian subcommunicator.
    type, public :: cart_comm_1d
        integer :: myrank       !< Rank in the current communicator
        integer :: nprocs       !< Number of processes in the communicator
        integer :: west_rank    !< Rank of the previous neighbor
        integer :: east_rank    !< Rank of the next neighbor
        integer :: mpi_comm     !< MPI communicator handle
    end type cart_comm_1d

    !-----------------------------------------------------------------------------------------------
    !> @brief Subcommunicator information in x- and y-directions
    type(cart_comm_1d), public :: comm_1d_x
    type(cart_comm_1d), public :: comm_1d_y

    private
    public :: mpi_topology_make, mpi_topology_clean

contains
    !-----------------------------------------------------------------------------------------------
    !> @brief Destroy the communicators for the Cartesian topology and subcommunicators.
    !-----------------------------------------------------------------------------------------------
    subroutine mpi_topology_clean()

        implicit none

        integer :: ierr

        call MPI_Comm_free(comm_1d_x%mpi_comm, ierr)
        call MPI_Comm_free(comm_1d_y%mpi_comm, ierr)
        call MPI_Comm_free(mpi_world_cart, ierr)

    end subroutine mpi_topology_clean

    !-----------------------------------------------------------------------------------------------
    !> @brief Create a 2D Cartesian topology and its 1D subcommunicators.
    !-----------------------------------------------------------------------------------------------
    subroutine mpi_topology_make()

        implicit none

        logical :: remain(0:1)
        integer :: ierr

        ! Create the 2D Cartesian topology
        call MPI_Cart_create(MPI_COMM_WORLD, 2, np_dim, period, .false., mpi_world_cart, ierr)

        ! Create x-direction subcommunicator
        remain = [ .true., .false. ]
        call MPI_Cart_sub(mpi_world_cart, remain, comm_1d_x%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_x%mpi_comm, comm_1d_x%myrank, ierr)
        call MPI_Comm_size(comm_1d_x%mpi_comm, comm_1d_x%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_x%mpi_comm, 0, 1, comm_1d_x%west_rank, comm_1d_x%east_rank, ierr)

        ! Create y-direction subcommunicator
        remain = [ .false., .true. ]
        call MPI_Cart_sub(mpi_world_cart, remain, comm_1d_y%mpi_comm, ierr)
        call MPI_Comm_rank(comm_1d_y%mpi_comm, comm_1d_y%myrank, ierr)
        call MPI_Comm_size(comm_1d_y%mpi_comm, comm_1d_y%nprocs, ierr)
        call MPI_Cart_shift(comm_1d_y%mpi_comm, 0, 1, comm_1d_y%west_rank, comm_1d_y%east_rank, ierr)

    end subroutine mpi_topology_make

end module mpi_topology_2d