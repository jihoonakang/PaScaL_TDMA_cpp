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
    Generate the right-hand side (RHS) tensor and reference solution for a 3D multi-RHS tridiagonal
    system.

    :param a: Lower diagonal coefficients of length ``nx``.
    :type a: numpy.ndarray
    :param b: Main diagonal coefficients of length ``nx``.
    :type b: numpy.ndarray
    :param c: Upper diagonal coefficients of length ``nx``.
    :type c: numpy.ndarray
    :param nx: Number of grid points in the x-direction.
    :type nx: int
    :param ny: Number of grid points in the y-direction.
    :type ny: int
    :param nz: Number of grid points in the z-direction.
    :type nz: int
    :param tdma_type: TDMA type, must be ``"standard"`` or ``"cyclic"``.
    :type tdma_type: str
    :return: Tuple ``(D, X)`` where ``D`` is the RHS array and ``X`` is the reference solution array.
    :rtype: tuple[numpy.ndarray, cupy.ndarray]
    :raises ValueError: If ``tdma_type`` is invalid.
    """
    X = cp.random.rand(nx, ny, nz)
    D = np.zeros((nx, ny, nz))

    for k in range(nz):
        for j in range(ny):
            if tdma_type == "cyclic":
                D[0, j, k] = a[0] * X[-1, j, k] + b[0] * X[0, j, k] + c[0] * X[1, j, k]
                for i in range(1, nx - 1):
                    D[i, j, k] = a[i] * X[i - 1, j, k] + b[i] * X[i, j, k] + c[i] * X[i + 1, j, k]
                D[-1, j, k] = a[-1] * X[-2, j, k] + b[-1] * X[-1, j, k] + c[-1] * X[0, j, k]
            elif tdma_type == "standard":
                D[0, j, k] = b[0] * X[0, j, k] + c[0] * X[1, j, k]
                for i in range(1, nx - 1):
                    D[i, j, k] = a[i] * X[i - 1, j, k] + b[i] * X[i, j, k] + c[i] * X[i + 1, j, k]
                D[-1, j, k] = a[-1] * X[-2, j, k] + b[-1] * X[-1, j, k]
            else:
                raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    return D, X


def main(nx: int, ny: int, nz: int, type_str: str):
    """
    Solve a 3D multi-RHS tridiagonal system using both CPU and GPU solvers
    (PaScaL_TDMA and CuPaScaL_TDMA) and compute the RMSE between the results.

    :param nx: Number of grid points in the x-direction.
    :type nx: int
    :param ny: Number of grid points in the y-direction.
    :type ny: int
    :param nz: Number of grid points in the z-direction.
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


    # Create 1D coefficient arrays
    a = np.full(nx, a_lower)
    b = np.full(nx, a_diag)
    c = np.full(nx, a_upper)

    # Generate RHS and reference solution
    D, X = generate_3d_rhs(a, b, c, nx, ny, nz, type_str)

    # Move data to GPU
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.asarray(c)
    D_gpu = cp.asarray(D)

    # CPU solve
    plan = Tdma.PTDMAPlanManyRHS()
    plan.create(nx, ny * nz, comm.py2f(), cyclic=is_cyclic)
    Tdma.solveManyRHS(plan, a, b, c, D.reshape(nx, -1))
    plan.destroy()

    # GPU solve
    plan = cuTdma.CuPTDMAPlanManyRHS()
    plan.create(nx, ny, nz, comm.py2f(), cyclic=is_cyclic)
    cuTdma.cuSolveManyRHS(plan, a_gpu, b_gpu, c_gpu, D_gpu)
    plan.destroy()

    # Compare results
    D_gpu_result = cp.asnumpy(D_gpu)
    rmse = np.sqrt(np.mean((D - D_gpu_result) ** 2))
    print(f"Rank {rank:3d}, RMSE between CPU and GPU results: {rmse:.6e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: mpirun -n <num_procs> python cuda_many_rhs.py <nx> <ny> <nz> <standard|cyclic>")
        sys.exit(1)

    nx = int(sys.argv[1])
    ny = int(sys.argv[2])
    nz = int(sys.argv[3])
    type_str = sys.argv[4]
    main(nx, ny, nz, type_str)