/**
 * @file cudaEnv.hpp
 * @brief Initializes CUDA environment and verifies CUDA-Aware MPI support.
 *
 * Provides functions to check GPU availability, display device information,
 * and test CUDA-Aware MPI via MPI_Bcast and MPI_Alltoall operations.
 */
#pragma once

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/**
 * @class cudaEnv
 * @brief Manages CUDA environment setup and CUDA-Aware MPI detection.
 *
 * All methods are static since the environment is global across MPI processes.
 */
class cudaEnv {
public:
    /**
     * @brief Initialize CUDA and test MPI support.
     *
     * Performs the following steps:
     *   1. Detect CUDA-compatible GPU devices.
     *   2. Print device properties for the root MPI rank.
     *   3. Test CUDA-Aware MPI support via MPI_Bcast on GPU memory.
     *   4. Test CUDA-Aware MPI support via MPI_Alltoall on GPU memory.
     *
     * Aborts MPI if no GPU devices are found. Sets internal flag
     * to false if MPI operations on GPU memory fail.
     */
    static void initialize() {

        int rank = 0;           ///< MPI rank of the current process
        int size = 1;           ///< Total number of MPI ranks
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // 1. GPU device detection
        int device_count = 0;   ///< Number of CUDA-compatible devices
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            if (!rank) {
                std::fprintf(stderr, "[ERROR] No GPU (CUDA) device found.\n");
                std::fprintf(stderr, "[ERROR] Build with -DCUDA=OFF.\n");
            }
            MPI_Abort(MPI_COMM_WORLD, -1);
            exit(EXIT_FAILURE);
        }

        // 2. Print GPU device properties on root rank
        if (!rank) printf("[INFO] %d CUDA device(s) found:\n", device_count);
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            if (!rank) {
                std::printf("  - Device %d: %s | Compute Capability: %d.%d | "
                            "Global Mem: %.2f GB\n",
                            i, prop.name, prop.major, prop.minor, 
                            prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
            }
        }

        // 3. Test CUDA-Aware MPI via MPI_Bcast
        double* gpu_buf = nullptr;      ///< Device buffer for MPI_Bcast
        cudaMalloc(&gpu_buf, sizeof(double) * 10);

        int result = MPI_Bcast(gpu_buf, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (result != MPI_SUCCESS) {
            if (!rank) {
                std::fprintf(stderr, "[ERROR] CUDA-Aware MPI not supported.");
                std::fprintf(stderr, "[ERROR] MPI_Bcast failed on GPU mem.");
                std::fprintf(stderr, "[ERROR] Build with -DCUDA_AWARE_MPI=OFF");
            }
            cudaAwareMPI = false;
        }

        cudaFree(gpu_buf);

        // 4. Test CUDA-Aware MPI via MPI_Alltoall
        double* sendbuf = nullptr;      ///< Device send buffer for Alltoall
        double* recvbuf = nullptr;      ///< Device receive buffer for Alltoall

        cudaMalloc(&sendbuf, size * sizeof(double));
        cudaMalloc(&recvbuf, size * sizeof(double));

        result = MPI_Alltoall(sendbuf, 1, MPI_DOUBLE,
                                  recvbuf, 1, MPI_DOUBLE,
                                  MPI_COMM_WORLD);

        if (result != MPI_SUCCESS) {
            if (!rank) {
                std::fprintf(stderr, "[ERROR] CUDA-Aware MPI not supported.");
                std::fprintf(stderr, "[ERROR] MPI_Alltoall failed on GPU mem.");
                std::fprintf(stderr, "[ERROR] Build with -DCUDA_AWARE_MPI=OFF");
            }
            cudaAwareMPI = false;
        }

        cudaFree(sendbuf);
        cudaFree(recvbuf);
    }

    /// Returns true if CUDA-Aware MPI support was detected.
    static bool isCudaAwareMPI() {
        return cudaAwareMPI;
    }

private:
    /// Flag indicating if CUDA-Aware MPI is supported.
    static inline bool cudaAwareMPI = true;
};