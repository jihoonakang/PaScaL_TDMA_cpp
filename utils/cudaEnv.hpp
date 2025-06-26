#pragma once

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

class cudaEnv {
public:
    // 초기화 함수: GPU 검사 + CUDA-aware MPI 확인
    static void initialize() {

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // 1. GPU 디바이스 확인
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            fprintf(stderr, "[ERROR] No CUDA-compatible GPU device found.\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
            exit(EXIT_FAILURE);
        }

        // 디바이스 정보 출력
        printf("[INFO] %d CUDA device(s) found:\n", device_count);
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            printf("  - Device %d: %s | Compute Capability: %d.%d | Global Mem: %.2f GB\n",
                   i, prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        }

        // 2. CUDA-Aware MPI 테스트 via Bcast
        int buf_size = 10;
        double* gpu_buf = nullptr;
        cudaMalloc(&gpu_buf, sizeof(double) * buf_size);

        int result = MPI_Bcast(gpu_buf, buf_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (result != MPI_SUCCESS) {
            if (rank == 0) {
                fprintf(stderr, "[WARNING] CUDA-Aware MPI not supported. Falling back to host memory.\n");
            }
            cudaAwareMPI = false;
        }

        cudaFree(gpu_buf);

        // 3. CUDA-Aware MPI 테스트 via Alltoall
        double* sendbuf = nullptr;
        double* recvbuf = nullptr;
        size_t count_per_rank = 1;

        cudaMalloc(&sendbuf, size * count_per_rank * sizeof(double));
        cudaMalloc(&recvbuf, size * count_per_rank * sizeof(double));

        // Alltoall: 모든 랭크가 GPU 버퍼에서 서로 데이터 교환
        result = MPI_Alltoall(sendbuf, count_per_rank, MPI_DOUBLE,
                                  recvbuf, count_per_rank, MPI_DOUBLE,
                                  MPI_COMM_WORLD);

        if (result != MPI_SUCCESS) {
            if (rank == 0) {
                fprintf(stderr, "[WARNING] CUDA-Aware MPI not supported (MPI_Alltoall failed on GPU memory).\n");
            }
            cudaAwareMPI = false;
        }

        cudaFree(sendbuf);
        cudaFree(recvbuf);
    }

    // CUDA-aware MPI 지원 여부 확인
    static bool isCudaAwareMPI() {
        return cudaAwareMPI;
    }

private:
    static inline bool cudaAwareMPI = true;
};