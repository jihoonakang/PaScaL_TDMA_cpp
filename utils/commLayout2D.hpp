/**
 * @file CommLayout2D.hpp
 * @brief Defines 2D MPI communicator topology and related layout functions.
 *
 * Provides classes to create Cartesian sub-communicators in X/Y directions,
 * compute communication buffer extents and manage global domain partitioning
 * in 2D and 3D.
 */

#pragma once

#include <mpi.h>
#include <iostream>
#include <vector>

/** Number of dimensions for the Cartesian topology. */
static constexpr int DIM = 2;

/**
 * @class SubCommunicator
 * @brief Encapsulates an MPI Cartesian sub-communicator.
 *
 * Creates a communicator by dropping one coordinate from the parent
 * Cartesian communicator.
 */
class SubCommunicator {

    MPI_Comm comm_ = MPI_COMM_NULL;  ///< Sub-communicator handle
    int rank_ = 0;                   ///< Rank within sub-communicator
    int size_ = 1;                   ///< Size of sub-communicator

public:
    /// Default constructs an empty sub-communicator.
    SubCommunicator() noexcept = default;

    /**
     * @brief Initialize from a parent Cartesian communicator.
     * @param parent_comm Parent MPI Cartesian communicator.
     * @param remain_dims Boolean mask of dimensions to keep.
     */
    void initialize(MPI_Comm parent_comm, const int remain_dims[2]) {
        MPI_Cart_sub(parent_comm, remain_dims, &comm_);
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }
   
    /// Returns the rank in this sub-communicator.
    const int getRank() const noexcept { return rank_; }
    /// Returns the size of this sub-communicator.
    const int getSize() const noexcept { return size_; }
    /// Returns the MPI_Comm handle.
    const MPI_Comm& getComm() const noexcept { return comm_; }
};

/**
 * @class CommLayout2D
 * @brief Manages a 2D Cartesian communicator and buffer layout info.
 *
 * Constructs a 2D Cartesian topology over MPI_COMM_WORLD, creates
 * X- and Y-direction sub-communicators, and computes counts and
 * displacements for buffer exchanges.
 */
class CommLayout2D {
private:
    int rank_ = 0, size_ = 1;           ///< Rank/size in comm_cart_
    MPI_Comm comm_cart_;                ///< Cartesian communicator (2D)
    int west_, east_, south_, north_;   ///< Neighbor ranks

    int coords_[DIM] = {0,0};           ///< Coordinates in the grid
    int dims[DIM];
    int periods[DIM];

    SubCommunicator comm_x_;            ///< Sub-communicator for X-direction
    SubCommunicator comm_y_;            ///< Sub-communicator for Y-direction

    std::vector<int> count_x_, displ_x_;        ///< Count/displ. along X
    std::vector<int> count_y_, displ_y_;        ///< Count/displ. along Y
    std::vector<int> count_all_, displ_all_;    ///< Global counts/displacements

public:
    /**
     * @brief Constructs 2D topology with given dimensions and periodicity.
     * @param dims Array of grid dimensions [nx, ny].
     * @param periods Array of periodic flags [px, py].
     */
    CommLayout2D(const int dims[DIM], const int periods[DIM]) {

        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        // Create Cartesian communicator
        MPI_Cart_create(MPI_COMM_WORLD, DIM, dims, periods, 0, &comm_cart_);
        MPI_Cart_coords(comm_cart_, rank_, DIM, coords_);

        // Get neighbor ranks in both directions
        MPI_Cart_shift(comm_cart_, 0, 1, &south_, &north_);
        MPI_Cart_shift(comm_cart_, 1, 1, &west_, &east_);

        // Initialize 1D sub-communicators
        const int remain_dims_x[DIM] = {1, 0};
        const int remain_dims_y[DIM] = {0, 1};
        comm_x_.initialize(comm_cart_, remain_dims_x);
        comm_y_.initialize(comm_cart_, remain_dims_y);

        // Allocate buffers
        count_x_.resize(dims[0]); displ_x_.resize(dims[0]);
        count_y_.resize(dims[1]); displ_y_.resize(dims[1]);
        count_all_.resize(size_); displ_all_.resize(size_);
    }

    /**
     * @brief Builds counts and displacements for exchange buffers.
     * @param nx_sub Local sub-domain size in X.
     * @param ny_sub Local sub-domain size in Y.
     * @param nz     Domain size in Z (default=1 for 2D).
     */
    void buildCommBufferInfo(int nx_sub, int ny_sub, int nz = 1) noexcept {

        const int n_sub = nx_sub * ny_sub * nz;

        // Gather per-dimension counts across each sub-communicator
        MPI_Allgather(&nx_sub, 1, MPI_INT, count_x_.data(), 1, MPI_INT, 
                        comm_x_.getComm());
        MPI_Allgather(&ny_sub, 1, MPI_INT, count_y_.data(), 1, MPI_INT, 
                        comm_y_.getComm());
        MPI_Allgather(&n_sub,  1, MPI_INT, count_all_.data(), 1, MPI_INT, 
                        MPI_COMM_WORLD);
    
        // Compute zero-based displacements instead of partial_sum
        displ_x_[0] = 0;
        for (int i = 1; i < count_x_.size(); i++)
            displ_x_[i] = displ_x_[i - 1] + count_x_[i - 1];
    
        displ_y_[0] = 0;
        for (int i = 1; i < count_y_.size(); i++)
            displ_y_[i] = displ_y_[i - 1] + count_y_[i - 1];
    
        displ_all_[0] = 0;
        for (int i = 1; i < count_all_.size(); i++)
            displ_all_[i] = displ_all_[i - 1] + count_all_[i - 1];
    }

    /// Returns the MPI rank in the 2D communicator.
    int getRank() const noexcept { return rank_; }
    /// Returns the size of the 2D communicator.
    int getSize() const noexcept { return size_; }
    /// Returns the Cartesian communicator handle.
    MPI_Comm getComm() const noexcept { return comm_cart_; }

    /// X-direction sub-communicator rank.
    int getRankX() const noexcept { return comm_x_.getRank(); }
    /// X-direction sub-communicator size.
    int getSizeX() const noexcept { return comm_x_.getSize(); }
    /// X-direction sub-communicator handle.
    MPI_Comm getCommX() const noexcept { return comm_x_.getComm(); }

    /// Y-direction sub-communicator rank.
    int getRankY() const noexcept { return comm_y_.getRank(); }
    /// Y-direction sub-communicator size.
    int getSizeY() const noexcept { return comm_y_.getSize(); }
    /// Y-direction sub-communicator handle.
    MPI_Comm getCommY() const noexcept { return comm_y_.getComm(); }

    /// Counts for each sub-domain in X.
    const std::vector<int>& getCountX() const noexcept { return count_x_; }
    /// Displacements for each sub-domain in X.
    const std::vector<int>& getDisplX() const noexcept { return displ_x_; }
    /// Counts for each sub-domain in Y.
    const std::vector<int>& getCountY() const noexcept { return count_y_; }
    /// Displacements for each sub-domain in Y.
    const std::vector<int>& getDisplY() const noexcept { return displ_y_; }
    /// Global counts per process.
    const std::vector<int>& getCountAll() const noexcept { return count_all_; }
    /// Global displacements per process.
    const std::vector<int>& getDisplAll() const noexcept { return displ_all_; }


    void print_info() const noexcept {

        for (size_t i = 0; i < size_; i++) {
            if (i == rank_) {
                std::cout << "Rank " << rank_ << " coords=(" << coords_[0]
                          << "," << coords_[1] << ") N=" << north_
                          << " S=" << south_ << " E=" << east_
                          << " W=" << west_ << std::endl;
                std::cout << "Rank " << rank_ 
                          << " Rank comm_x = " << comm_x_.getRank() 
                          << ", Rank comm_y = " << comm_y_.getRank() << std::endl;
            }
        }
    }
};

/**
 * @class DomainLayout2D
 * @brief Represents global 2D domain and its local partition.
 */
class DomainLayout2D {
private:
    int dim_x_;
    int dim_y_;
    int dim_xy_;
    int par_dim_x_;
    int par_dim_y_;
    int par_dim_xy_;

public:
    /**
     * @brief Compute global and per-rank dimensions in 2D.
     * @param dim_x Global X dimension.
     * @param dim_y Global Y dimension.
     * @param topo Communicator layout to query rank/grid info.
     */
    DomainLayout2D(const int dim_x, const int dim_y, const CommLayout2D& topo)
                    : dim_x_(dim_x), dim_y_(dim_y) {

        dim_xy_ = dim_x_ * dim_y_;

        par_dim_x_ = Util::para_range_n(1, dim_x_, topo.getSizeX(), topo.getRankX());
        par_dim_y_ = Util::para_range_n(1, dim_y_, topo.getSizeY(), topo.getRankY());
        par_dim_xy_ = par_dim_x_ * par_dim_y_;

    }

    /// Global X size.
    int getDimX() const noexcept { return dim_x_; }
    /// Global Y size.
    int getDimY() const noexcept { return dim_y_; }
    /// Global element count.
    int getDimXY() const noexcept { return dim_xy_; }
    /// Local X size.
    int getParDimX() const noexcept { return par_dim_x_; }
    /// Local Y size.
    int getParDimY() const noexcept { return par_dim_y_; }
    /// Local element count.
    int getParDimXY() const noexcept { return par_dim_xy_; }
};

/**
 * @class DomainLayout3D
 * @brief Represents global 3D domain and its local partition.
 */
class DomainLayout3D {
    int dim_x_;
    int dim_y_;
    int dim_z_;
    int dim_xyz_;
    int par_dim_x_;
    int par_dim_y_;
    int par_dim_z_;
    int par_dim_xyz_;

public:
    DomainLayout3D(const int dim_x, const int dim_y, const int dim_z, 
                   const CommLayout2D& topo)
                    : dim_x_(dim_x), dim_y_(dim_y), dim_z_(dim_z) {

        dim_xyz_ = dim_x_ * dim_y_ * dim_z_;

        par_dim_x_ = Util::para_range_n(1, dim_x_, topo.getSizeX(), topo.getRankX());
        par_dim_y_ = Util::para_range_n(1, dim_y_, topo.getSizeY(), topo.getRankY());
        par_dim_z_ = dim_z_;     // no decomposition in Z dimension
        par_dim_xyz_ = par_dim_x_ * par_dim_y_ * par_dim_z_;

    }

    /// Global X size.
    int getDimX() const noexcept { return dim_x_; }
    /// Global Y size.
    int getDimY() const noexcept { return dim_y_; }
    /// Global Z size.
    int getDimZ() const noexcept { return dim_z_; }
    /// Global element count.
    int getDimXYZ() const noexcept { return dim_xyz_; }
    /// Local X size.
    int getParDimX() const noexcept { return par_dim_x_; }
    /// Local Y size.
    int getParDimY() const noexcept { return par_dim_y_; }
    /// Local Z size.
    int getParDimZ() const noexcept { return par_dim_z_; }
    /// Local element count.
    int getParDimXYZ() const noexcept { return par_dim_xyz_; }
};