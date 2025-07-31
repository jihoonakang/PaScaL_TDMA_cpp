import numpy as np
from mpi4py import MPI
import PaScaL_TDMA_pybind as Tdma

# Constants for tridiagonal coefficients
a_diag = 10.0
a_upper = -1.0
a_lower = -1.0
root = 0


def generate_2d_rhs(nx: int, ny: int, tdma_type: str):
    """
    Generate the right-hand side (RHS) matrix and reference solution for a 2D tridiagonal
    system with multiple right-hand sides.

    :param nx: Number of rows in the 2D grid.
    :type nx: int
    :param ny: Number of columns in the 2D grid.
    :type ny: int
    :param tdma_type: Type of TDMA system. Must be either ``"standard"`` or ``"cyclic"``.
    :type tdma_type: str
    :return: A tuple ``(D, X)`` where ``D`` is the RHS matrix and ``X`` is the reference solution matrix.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    :raises ValueError: If ``tdma_type`` is not ``"standard"`` or ``"cyclic"``.
    """
    A = np.full(nx, a_lower)
    B = np.full(nx, a_diag)
    C = np.full(nx, a_upper)
    X = np.random.rand(nx, ny)
    D = np.zeros((nx, ny))

    for j in range(ny):
        if tdma_type == "cyclic":
            D[0, j] = A[0] * X[-1, j] + B[0] * X[0, j] + C[0] * X[1, j]
            for i in range(1, nx - 1):
                D[i, j] = A[i] * X[i - 1, j] + B[i] * X[i, j] + C[i] * X[i + 1, j]
            D[-1, j] = A[-1] * X[-2, j] + B[-1] * X[-1, j] + C[-1] * X[0, j]
        elif tdma_type == "standard":
            D[0, j] = B[0] * X[0, j] + C[0] * X[1, j]
            for i in range(1, nx - 1):
                D[i, j] = A[i] * X[i - 1, j] + B[i] * X[i, j] + C[i] * X[i + 1, j]
            D[-1, j] = A[-1] * X[-2, j] + B[-1] * X[-1, j]
        else:
            raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    return D, X


def main(nx: int, ny: int, type_str: str):
    """
    Main function for solving multiple tridiagonal systems with multiple right-hand sides
    in 2D using MPI and PaScaL_TDMA.

    Each MPI process is assigned a subset of rows, and each column in those rows is solved
    independently using the TDMA solver for many RHS.

    :param nx: Number of rows in the 2D grid.
    :type nx: int
    :param ny: Number of columns (number of RHS vectors).
    :type ny: int
    :param type_str: Type of TDMA system. Must be either ``"standard"`` or ``"cyclic"``.
    :type type_str: str
    :raises ValueError: If ``type_str`` is not ``"standard"`` or ``"cyclic"``.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if type_str not in ["standard", "cyclic"]:
        raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    is_cyclic = type_str == "cyclic"

    # Determine number of rows per process
    quotient, remainder = divmod(nx, size)
    nx_sub = quotient + 1 if rank < remainder else quotient

    counts = comm.gather(nx_sub, root=root)
    displs = None
    counts_col = None
    displs_col = None

    if rank == root:
        counts = np.array(counts)
        displs = np.array([sum(counts[:i]) for i in range(size)])
        counts_col = tuple(counts * ny)
        displs_col = tuple(displs * ny)

    # Generate RHS on the root process
    D = X = None
    if rank == root:
        D, X = generate_2d_rhs(nx, ny, type_str)

    # Scatter the rows to each process
    d_sub = np.zeros((nx_sub, ny))
    x_sub = np.zeros((nx_sub, ny))
    comm.Scatterv([D, counts_col, displs_col, MPI.DOUBLE], d_sub, root)
    comm.Scatterv([X, counts_col, displs_col, MPI.DOUBLE], x_sub, root)

    # Solve each column independently
    plan = Tdma.PTDMAPlanManyRHS()
    plan.create(nx_sub, ny, comm.py2f(), is_cyclic)
    a = np.full(nx_sub, a_lower)
    b = np.full(nx_sub, a_diag)
    c = np.full(nx_sub, a_upper)
    Tdma.solveManyRHS(plan, a, b, c, d_sub)
    plan.destroy()

    # Gather results
    D_solved = None
    if rank == root:
        D_solved = np.zeros((nx, ny))
    comm.Gatherv(d_sub, [D_solved, counts_col, displs_col, MPI.DOUBLE], root)

    if rank == root:
        error = D_solved - X
        rms_error = np.sqrt(np.sum(error**2) / (nx * ny))
        print("Avg. RMS error =", rms_error)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: mpirun -n <num_procs> python manyRHS.py <nx> <ny> <standard|cyclic>")
        sys.exit(1)

    nx = int(sys.argv[1])
    ny = int(sys.argv[2])
    type_str = sys.argv[3]

    main(nx, ny, type_str)