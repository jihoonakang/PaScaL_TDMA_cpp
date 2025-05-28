#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "TDMASolver.hpp"
#include "TDMASolver.cuh"

int main() {
    const int nx = 15, ny = 40, nz = 40;
    const int N = nx * ny * nz;

    std::vector<double> h_a(N, -1.0);
    std::vector<double> h_b(N,  4.0);
    std::vector<double> h_c(N, -1.0);
    std::vector<double> h_d(N);

    for (int i = 0; i < N; i++) {
        h_d[i] = std::sin(i);
    }

    // for (int j = 0; j < ny * nz; j++)
    //         h_d[j] = 2.0;
    // for (int i = ny * nz; i < (nx - 1) * ny * nz; i += ny * nz) {
    //     for (int j = 0; j < ny * nz; j++)
    //         h_d[i + j] = 2.0;
    // }
    // int i = (nx - 1) * ny * nz;
    // for (int j = 0; j < ny * nz; j++)
    //     h_d[i + j] = 2.0;

    std::vector<double> h_d_ref = h_d;

    for (int i = 0; i < nx * ny * nz; i+=ny*nz) {
        for (int j = 0; j < ny*nz; j++) std::cout << h_d_ref[i+j] << ' ';
        std::cout<<std::endl;
    }

    // GPU 메모리 할당
    double *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, N * sizeof(double));
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_c, N * sizeof(double));
    cudaMalloc(&d_d, N * sizeof(double));

    cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // CPU 참조 해 계산
    TDMASolver::manyCyclic(h_a.data(), h_b.data(), h_c.data(), h_d_ref.data(), nx, ny*nz);

    for (int i = 0; i < nx * ny * nz; i+=ny*nz) {
        for (int j = 0; j < ny*nz; j++) std::cout << h_d_ref[i+j] << ' ';
        std::cout<<std::endl;
    }

    // 커널 호출
    cuTDMASolver::cuManyCyclic(d_a, d_b, d_c, d_d, nx, ny, nz);

    // 결과 복사 및 비교
    std::vector<double> h_d_out(N);
    cudaMemcpy(h_d_out.data(), d_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nx * ny * nz; i+=ny*nz) {
        for (int j = 0; j < ny*nz; j++) std::cout << h_d_out[i+j] << ' ';
        std::cout<<std::endl;
    }

    double error = 0.0;
    for (int i = 0; i < N; ++i) {
        error += std::abs(h_d_ref[i] - h_d_out[i]);
    }

    std::cout << "Total error: " << error << std::endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    return 0;
}