# examples/python/example_cuda_usage.py

import cupy as cp
import numpy as np
import PaScaL_TDMA_cuda_pybind as cuTdma
import PaScaL_TDMA_pybind as Tdma

from mpi4py import MPI

a_diag = 10.0
a_upper = -1.0
a_lower = -1.0
root = 0

def generate_3d_rhs(a, b, c, nx, ny, nz, tdma_type):
    X = cp.random.rand(nx, ny, nz)
    # X = np.full((nx, ny, nz), 1.0)
    D = np.zeros((nx, ny, nz))

    for k in range(nz):
        for j in range(ny):  # 열을 기준으로 반복
            if tdma_type == 'cyclic':
                D[0, j, k] = a[0] * X[-1, j, k] + b[0] * X[0, j, k] + c[0] * X[1, j, k]
                for i in range(1, nx - 1):
                    D[i, j, k] = a[i] * X[i - 1, j, k] + b[i] * X[i, j, k] + c[i] * X[i + 1, j, k]
                D[-1, j, k] = a[-1] * X[-2, j, k] + b[-1] * X[-1, j, k] + c[-1] * X[0, j, k]
            elif tdma_type == 'standard':
                D[0, j, k] = b[0] * X[0, j, k] + c[0] * X[1, j, k]
                for i in range(1, nx - 1):
                    D[i, j, k] = a[i] * X[i - 1, j, k] + b[i] * X[i, j, k] + c[i] * X[i + 1, j, k]
                D[-1, j, k] = a[-1] * X[-2, j, k] + b[-1] * X[-1, j, k]
            else:
                raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    return D, X

def main(nx, ny, nz, type_str):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if type_str not in ["standard", "cyclic"]:
        raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    is_cyclic = (type_str == "cyclic")

    a = np.full(nx, a_lower)
    b = np.full(nx, a_diag)
    c = np.full(nx, a_upper)

    D, X = generate_3d_rhs(a, b, c, nx, ny, nz, type_str)

    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.asarray(c)
    D_gpu = cp.asarray(D)

    comm = MPI.COMM_WORLD

    plan = Tdma.PTDMAPlanManyRHS()
    plan.create(nx, ny * nz, comm.py2f(), cyclic=is_cyclic)
    Tdma.solveManyRHS(plan, a, b, c, D.reshape(nx, -1))
    plan.destroy()


    plan = cuTdma.cuPTDMAPlanManyRHS()
    plan.create(nx, ny, nz, comm.py2f(), cyclic=is_cyclic)
    cuTdma.cuSolveManyRHS(plan, a_gpu, b_gpu, c_gpu, D_gpu)
    plan.destroy()

    D_gpu_result = cp.asnumpy(D_gpu)

    rmse = np.sqrt(np.mean((D - D_gpu_result) ** 2))

    print(f"RMSE between CPU and GPU results: {rmse:.6e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: mpirun -n <num_procs> python many.py <nx> <ny> <nz> <standard|cyclic>")
        sys.exit(1)

    # ----------------------
    # GPU 정보 출력
    # ----------------------
    device = cp.cuda.Device(0)   # GPU ID 0 사용
    device.use()                 # 해당 GPU에 현재 context 바인딩

    props = cp.cuda.runtime.getDeviceProperties(device.id)
    print("Using GPU Device ID:", device.id)
    print("GPU Name:", props['name'])

    print("Total Memory (bytes):", device.mem_info[1])
    print("Free Memory (bytes):", device.mem_info[0])

    nx = int(sys.argv[1])
    ny = int(sys.argv[2])
    nz = int(sys.argv[3])
    type_str = sys.argv[4]
    main(nx, ny, nz, type_str)