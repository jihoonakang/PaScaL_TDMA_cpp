#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

#pragma once

#define DIM 3

class SubCommunicator {

    MPI_Comm comm;
    int rank;
    int size;
    int rank_next;
    int rank_prev;

public:
    SubCommunicator() : comm(MPI_COMM_NULL), rank(0), size(1) {}

    void initialize(MPI_Comm parent_comm, const int remain_dims[2]) {
        MPI_Cart_sub(parent_comm, remain_dims, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        MPI_Cart_shift(comm, 0, 1, &rank_prev, &rank_next);

    }
   
    const int getRank() const { return rank; }
    const int getRankNext() const { return rank_next; }
    const int getRankPrev() const { return rank_prev; }
    const int getSize() const { return size; }
    const MPI_Comm& getComm() const { return comm; }
};

class CommLayout3D {
    int rank;
    int size;
    MPI_Comm comm_cart;
    SubCommunicator comm_x;
    SubCommunicator comm_y;
    SubCommunicator comm_z;

    std::vector<int> count_x, displ_x;
    std::vector<int> count_y, displ_y;
    std::vector<int> count_z, displ_z;
    std::vector<int> count_all, displ_all;

    int dims[DIM];
    int coords[DIM];
    int periods[DIM];

    int north, south, east, west, front, back;

public:
    CommLayout3D(const int dims_[DIM], const int periods_[DIM]) {

        for (size_t i = 0; i < DIM; ++i) {
            dims[i] = dims_[i];
            periods[i] = periods_[i];
        }

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        MPI_Cart_create(MPI_COMM_WORLD, DIM, dims, periods, 0, &comm_cart);
        MPI_Cart_coords(comm_cart, rank, DIM, coords);

        MPI_Cart_shift(comm_cart, 0, 1, &south, &north);
        MPI_Cart_shift(comm_cart, 1, 1, &west, &east);
        MPI_Cart_shift(comm_cart, 2, 1, &back, &front);

        const int remain_dims_x[DIM] = {1, 0, 0};
        const int remain_dims_y[DIM] = {0, 1, 0};
        const int remain_dims_z[DIM] = {0, 0, 1};

        comm_x.initialize(comm_cart, remain_dims_x);
        comm_y.initialize(comm_cart, remain_dims_y);
        comm_z.initialize(comm_cart, remain_dims_z);

        count_x.resize(dims[0]);
        displ_x.resize(dims[0]);
        count_y.resize(dims[1]);
        displ_y.resize(dims[1]);
        count_y.resize(dims[2]);
        displ_y.resize(dims[2]);
        count_all.resize(size);
        displ_all.resize(size);
    }

    void buildCommBufferInfo(const int nx_sub, const int ny_sub, const int nz_sub) {

        const int n_sub = nx_sub * ny_sub * nz_sub;

        MPI_Allgather(&nx_sub, 1, MPI_INT, count_x.data(), 1, MPI_INT, comm_x.getComm());
        MPI_Allgather(&ny_sub, 1, MPI_INT, count_y.data(), 1, MPI_INT, comm_y.getComm());
        MPI_Allgather(&nz_sub, 1, MPI_INT, count_z.data(), 1, MPI_INT, comm_z.getComm());
        MPI_Allgather(&n_sub,  1, MPI_INT, count_all.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
        displ_x[0] = 0;
        for (int i = 1; i < count_x.size(); i++)
            displ_x[i] = displ_x[i - 1] + count_x[i - 1];
    
        displ_y[0] = 0;
        for (int i = 1; i < count_y.size(); i++)
            displ_y[i] = displ_y[i - 1] + count_y[i - 1];
    
        displ_z[0] = 0;
        for (int i = 1; i < count_z.size(); i++)
            displ_z[i] = displ_z[i - 1] + count_z[i - 1];
    
        displ_all[0] = 0;
        for (int i = 1; i < count_all.size(); i++)
            displ_all[i] = displ_all[i - 1] + count_all[i - 1];
    
    }

    const std::vector<int>& getCountX() const { return count_x; }
    const std::vector<int>& getCountY() const { return count_y; }
    const std::vector<int>& getCountZ() const { return count_z; }
    const std::vector<int>& getDisplX() const { return displ_x; }
    const std::vector<int>& getDisplY() const { return displ_y; }
    const std::vector<int>& getDisplZ() const { return displ_z; }
    const std::vector<int>& getCountAll() const { return count_all; }
    const std::vector<int>& getDisplAll() const { return displ_all; }

    const int getSize() const { return size; }
    const int getRank() const { return rank; }
    const MPI_Comm& getComm() const { return comm_cart; }

    const int getSizeX() const { return comm_x.getSize(); }
    const int getRankX() const { return comm_x.getRank(); }
    const int getRankNextX() const { return comm_x.getRankNext(); }
    const int getRankPrevX() const { return comm_x.getRankPrev(); }
    const MPI_Comm& getCommX() const { return comm_x.getComm(); }

    const int getSizeY() const { return comm_y.getSize(); }
    const int getRankY() const { return comm_y.getRank(); }
    const int getRankNextY() const { return comm_y.getRankNext(); }
    const int getRankPrevY() const { return comm_y.getRankPrev(); }
    const MPI_Comm& getCommY() const { return comm_y.getComm(); }

    const int getSizeZ() const { return comm_z.getSize(); }
    const int getRankZ() const { return comm_z.getRank(); }
    const int getRankNextZ() const { return comm_z.getRankNext(); }
    const int getRankPrevZ() const { return comm_z.getRankPrev(); }
    const MPI_Comm& getCommZ() const { return comm_z.getComm(); }

    void print_info() const {

        for (size_t i = 0; i < size; i++) {
            if (i == rank) {
                std::cout << "Rank " << rank << " is at (px, py, pz) = (" 
                          << coords[0] << ", " << coords[1] << ", " << coords[2] 
                          << ", "<< ")"
                          << ", N: " << north << ", S: " << south
                          << ", E: " <<  east << ", W: " << west 
                          << ", F: " << front << ", B: " << back << std::endl;
                std::cout << "Rank " << rank 
                          << ", Rank comm_x = " << comm_x.getRank() 
                          << ", Rank comm_y = " << comm_y.getRank()
                          << ", Rank comm_z = " << comm_z.getRank()
                          << std::endl;
            }
        }
    }
};
