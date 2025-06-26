#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

#define DIM 2

#pragma once

class SubCommunicator {

    MPI_Comm comm;
    int rank;
    int size;

public:
    SubCommunicator() : comm(MPI_COMM_NULL), rank(0), size(1) {}

    void initialize(MPI_Comm parent_comm, const int remain_dims[2]) {
        MPI_Cart_sub(parent_comm, remain_dims, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }
   
    const int getRank() const { return rank; }
    const int getSize() const { return size; }
    const MPI_Comm& getComm() const { return comm; }
};

class CommLayout2D {
    int rank;
    int size;
    MPI_Comm comm_cart;
    SubCommunicator comm_x;
    SubCommunicator comm_y;

    std::vector<int> count_x, displ_x;
    std::vector<int> count_y, displ_y;
    std::vector<int> count_all, displ_all;

    int dims[DIM];
    int coords[DIM];
    int periods[DIM];

    int west, east, south, north;

public:
    CommLayout2D(const int dims_[DIM], const int periods_[DIM]) {

        for (size_t i = 0; i < DIM; i++) {
            dims[i] = dims_[i];
            periods[i] = periods_[i];
        }

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        MPI_Cart_create(MPI_COMM_WORLD, DIM, dims, periods, 0, &comm_cart);
        MPI_Cart_coords(comm_cart, rank, DIM, coords);

        MPI_Cart_shift(comm_cart, 0, 1, &south, &north);
        MPI_Cart_shift(comm_cart, 1, 1, &west, &east);

        const int remain_dims_x[DIM] = {1, 0};
        const int remain_dims_y[DIM] = {0, 1};
        comm_x.initialize(comm_cart, remain_dims_x);
        comm_y.initialize(comm_cart, remain_dims_y);

        count_x.resize(dims[0]);
        displ_x.resize(dims[0]);
        count_y.resize(dims[1]);
        displ_y.resize(dims[1]);
        count_all.resize(size);
        displ_all.resize(size);
    }

    void buildCommBufferInfo(const int nx_sub, const int ny_sub) {

        const int n_sub = nx_sub * ny_sub;

        MPI_Allgather(&nx_sub, 1, MPI_INT, count_x.data(), 1, MPI_INT, comm_x.getComm());
        MPI_Allgather(&ny_sub, 1, MPI_INT, count_y.data(), 1, MPI_INT, comm_y.getComm());
        MPI_Allgather(&n_sub,  1, MPI_INT, count_all.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
        displ_x[0] = 0;
        for (int i = 1; i < count_x.size(); i++)
            displ_x[i] = displ_x[i - 1] + count_x[i - 1];
    
        displ_y[0] = 0;
        for (int i = 1; i < count_y.size(); i++)
            displ_y[i] = displ_y[i - 1] + count_y[i - 1];
    
        displ_all[0] = 0;
        for (int i = 1; i < count_all.size(); i++)
            displ_all[i] = displ_all[i - 1] + count_all[i - 1];
    
    }

    void buildCommBufferInfo(const int nx_sub, const int ny_sub, const int nz) {

        const int n_sub = nx_sub * ny_sub * nz;

        MPI_Allgather(&nx_sub, 1, MPI_INT, count_x.data(), 1, MPI_INT, comm_x.getComm());
        MPI_Allgather(&ny_sub, 1, MPI_INT, count_y.data(), 1, MPI_INT, comm_y.getComm());
        MPI_Allgather(&n_sub,  1, MPI_INT, count_all.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
        displ_x[0] = 0;
        for (int i = 1; i < count_x.size(); i++)
            displ_x[i] = displ_x[i - 1] + count_x[i - 1];
    
        displ_y[0] = 0;
        for (int i = 1; i < count_y.size(); i++)
            displ_y[i] = displ_y[i - 1] + count_y[i - 1];
    
        displ_all[0] = 0;
        for (int i = 1; i < count_all.size(); i++)
            displ_all[i] = displ_all[i - 1] + count_all[i - 1];
    
    }

    const std::vector<int>& getCountX() const { return count_x; }
    const std::vector<int>& getCountY() const { return count_y; }
    const std::vector<int>& getDisplX() const { return displ_x; }
    const std::vector<int>& getDisplY() const { return displ_y; }
    const std::vector<int>& getCountAll() const { return count_all; }
    const std::vector<int>& getDisplAll() const { return displ_all; }

    const int getRank() const { return rank; }
    const int getSize() const { return size; }
    const MPI_Comm& getComm() const { return comm_cart; }

    const int getRankX() const { return comm_x.getRank(); }
    const int getSizeX() const { return comm_x.getSize(); }
    const MPI_Comm& getCommX() const { return comm_x.getComm(); }

    const int getRankY() const { return comm_y.getRank(); }
    const int getSizeY() const { return comm_y.getSize(); }
    const MPI_Comm& getCommY() const { return comm_y.getComm(); }


    void print_info() const {

        for (size_t i = 0; i < size; i++) {
            if (i == rank) {
                std::cout << "Rank " << rank << " is at (px, py) = (" << coords[0] << ", " << coords[1] << ")"
                        << ", N: " << north << ", S: " << south
                        << ", E: " << east << ", W: " << west << std::endl;
                std::cout << "Rank " << rank << " Rank comm_x = " << comm_x.getRank() 
                        << ", Rank comm_y = " << comm_y.getRank()  << std::endl;
            }
        }
    }
};


class DomainLayout2D {
    int dim_x;
    int dim_y;
    int dim_xy;
    int par_dim_x;
    int par_dim_y;
    int par_dim_xy;

public:
    DomainLayout2D(const int dim_x_, const int dim_y_, const CommLayout2D& topo)
                    : dim_x(dim_x_), dim_y(dim_y_) {

        dim_xy = dim_x * dim_y;

        par_dim_x = Util::para_range_n(1, dim_x, topo.getSizeX(), topo.getRankX());
        par_dim_y = Util::para_range_n(1, dim_y, topo.getSizeY(), topo.getRankY());
        par_dim_xy = par_dim_x * par_dim_y;

    }

    const int getDimX() const { return dim_x; }
    const int getDimY() const { return dim_y; }
    const int getDimXY() const { return dim_xy; }

    const int getParDimX() const { return par_dim_x; }
    const int getParDimY() const { return par_dim_y; }
    const int getParDimXY() const { return par_dim_xy; }
};


class DomainLayout3D {
    int dim_x;
    int dim_y;
    int dim_z;
    int dim_xyz;
    int par_dim_x;
    int par_dim_y;
    int par_dim_z;
    int par_dim_xyz;

public:
    DomainLayout3D(const int dim_x_, const int dim_y_, const int dim_z_, const CommLayout2D& topo)
                    : dim_x(dim_x_), dim_y(dim_y_), dim_z(dim_z_) {

        dim_xyz = dim_x * dim_y * dim_z;

        par_dim_x = Util::para_range_n(1, dim_x, topo.getSizeX(), topo.getRankX());
        par_dim_y = Util::para_range_n(1, dim_y, topo.getSizeY(), topo.getRankY());
        par_dim_z = dim_z;
        par_dim_xyz = par_dim_x * par_dim_y * par_dim_z;

    }

    const int getDimX() const { return dim_x; }
    const int getDimY() const { return dim_y; }
    const int getDimZ() const { return dim_z; }
    const int getDimXYZ() const { return dim_xyz; }

    const int getParDimX() const { return par_dim_x; }
    const int getParDimY() const { return par_dim_y; }
    const int getParDimZ() const { return par_dim_z; }
    const int getParDimXYZ() const { return par_dim_xyz; }
};