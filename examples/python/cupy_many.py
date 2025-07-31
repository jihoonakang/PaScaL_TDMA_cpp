import cupy as cp
import numpy as np
import PaScaL_TDMA_cuda_pybind as cuTdma
import PaScaL_TDMA_pybind as Tdma
from mpi4py import MPI

# Tridiagonal coefficients
a_diag = 10.0
a_upper = -1.0
a_lower = -1.0
root = 0


def generate_3d_rhs(a: np.ndarray, b: np.ndarray, c: np.ndarray, nx: int, ny: int, nz: int, tdma_type: str):
    """
    Generate the right-hand side (RHS) tensor and reference solution for a 3D tridiagonal
    system.

    :param a: Lower diagonal coefficients of shape (nx, ny, nz).
    :type a: numpy.ndarray
    :param b: Main diagonal coefficients of shape (nx, ny, nz).
    :type b: numpy.ndarray
    :param c: Upper diagonal coefficients of shape (nx, ny, nz).
    :type c: numpy.ndarray
    :param nx: Number of rows in the 3D grid.
    :type nx: int
    :param ny: Number of columns in the 3D grid.
    :type ny: int
    :param nz: Number of depth layers in the 3D grid.
    :type nz: int
    :param tdma_type: TDMA type, must be ``"standard"`` or ``"cyclic"``.
    :type tdma_type: str
    :return: Tuple ``(D, X)`` where ``D`` is the RHS tensor and ``X`` is the reference solution tensor.
    :rtype: tuple[numpy.ndarray, cupy.ndarray]
    :raises ValueError: If ``tdma_type`` is invalid.
    """
    X = cp.random.rand(nx, ny, nz)
    D = np.zeros((nx, ny, nz))

    for k in range(nz):
        for j in range(ny):
            if tdma_type == "cyclic":
                D[0, j, k] = a[0, j, k] * X[-1, j, k] + b[0, j, k] * X[0, j, k] + c[0, j, k] * X[1, j, k]
                for i in range(1, nx - 1):
                    D[i, j, k] = a[i, j, k] * X[i - 1, j, k] + b[i, j, k] * X[i, j, k] + c[i, j, k] * X[i + 1, j, k]
                D[-1, j, k] = a[-1, j, k] * X[-2, j, k] + b[-1, j, k] * X[-1, j, k] + c[-1, j, k] * X[0, j, k]
            elif tdma_type == "standard":
                D[0, j, k] = b[0, j, k] * X[0, j, k] + c[0, j, k] * X[1, j, k]
                for i in range(1, nx - 1):
                    D[i, j, k] = a[i, j, k] * X[i - 1, j, k] + b[i, j, k] * X[i, j, k] + c[i, j, k] * X[i + 1, j, k]
                D[-1, j, k] = a[-1, j, k] * X[-2, j, k] + b[-1, j, k] * X[-1, j, k]
            else:
                raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    return D, X


def main(nx: int, ny: int, nz: int, type_str: str):
    """
    Solve a 3D tridiagonal system using both CPU and GPU solvers (PaScaL_TDMA and CuPaScaL_TDMA)
    and compute the RMSE between CPU and GPU results.

    :param nx: Number of rows in the 3D grid.
    :type nx: int
    :param ny: Number of columns in the 3D grid.
    :type ny: int
    :param nz: Number of depth layers in the 3D grid.
    :type nz: int
    :param type_str: TDMA type, must be ``"standard"`` or ``"cyclic"``.
    :type type_str: str
    :raises ValueError: If ``type_str`` is invalid.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if type_str not in ["standard", "cyclic"]:
        raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    is_cyclic = type_str == "cyclic"

    # Display GPU info
    device = cp.cuda.Device(0)
    device.use()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    print("MPI Rank: ", rank)
    print("  Using GPU Device ID:", device.id)
    print("  GPU Name:", props["name"])
    print("  Total Memory (bytes):", device.mem_info[1])
    print("  Free Memory (bytes):", device.mem_info[0])

    # Create coefficient arrays
    a = np.full((nx, ny, nz), a_lower)
    b = np.full((nx, ny, nz), a_diag)
    c = np.full((nx, ny, nz), a_upper)

    # Generate RHS and reference solution
    D, X = generate_3d_rhs(a, b, c, nx, ny, nz, type_str)

    # Move data to GPU
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.asarray(c)
    D_gpu = cp.asarray(D)

    # CPU solve
    plan = Tdma.PTDMAPlanMany()
    plan.create(nx, ny * nz, comm.py2f(), cyclic=is_cyclic)
    Tdma.solveMany(plan, a, b, c, D.reshape(nx, -1))
    plan.destroy()

    # GPU solve
    plan = cuTdma.CuPTDMAPlanMany()
    plan.create(nx, ny, nz, comm.py2f(), cyclic=is_cyclic)
    cuTdma.cuSolveMany(plan, a_gpu, b_gpu, c_gpu, D_gpu)
    plan.destroy()

    # Compare results
    D_gpu_result = cp.asnumpy(D_gpu)
    rmse = np.sqrt(np.mean((D - D_gpu_result) ** 2))
    print(f"Rank {rank:3d}, RMSE between CPU and GPU results: {rmse:.6e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: mpirun -n <num_procs> python cuda_many.py <nx> <ny> <nz> <standard|cyclic>")
        sys.exit(1)

    nx = int(sys.argv[1])
    ny = int(sys.argv[2])
    nz = int(sys.argv[3])
    type_str = sys.argv[4]

    main(nx, ny, nz, type_str)