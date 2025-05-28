import numpy as np
from mpi4py import MPI
from PaScaL_TDMA_pybind import PTDMAPlanManyRHS, PTDMASolverManyRHS

a_diag = 10.0
a_upper = -1.0
a_lower = -1.0
root = 0

def generate_2d_rhs(nx, ny, tdma_type):
    A = np.full(nx, a_lower)
    B = np.full(nx, a_diag)
    C = np.full(nx, a_upper)
    X = np.random.rand(nx, ny)
    D = np.zeros((nx, ny))

    for j in range(ny):  # 열을 기준으로 반복
        if tdma_type == 'cyclic':
            D[0, j] = A[0] * X[-1, j] + B[0] * X[0, j] + C[0] * X[1, j]
            for i in range(1, nx - 1):
                D[i, j] = A[i] * X[i - 1, j] + B[i] * X[i, j] + C[i] * X[i + 1, j]
            D[-1, j] = A[-1] * X[-2, j] + B[-1] * X[-1, j] + C[-1] * X[0, j]
        elif tdma_type == 'standard':
            D[0, j] = B[0] * X[0, j] + C[0] * X[1, j]
            for i in range(1, nx - 1):
                D[i, j] = A[i] * X[i - 1, j] + B[i] * X[i, j] + C[i] * X[i + 1, j]
            D[-1, j] = A[-1] * X[-2, j] + B[-1] * X[-1, j]
        else:
            raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    return D, X

def main(nx, ny, type_str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if type_str not in ["standard", "cyclic"]:
        raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    is_cyclic = (type_str == "cyclic")

    # 분산 처리할 행(row) 수 결정
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

    # Root가 RHS 생성
    D = X = None
    if rank == root:
        D, X = generate_2d_rhs(nx, ny, type_str)

    # Scatter row-wise
    d_sub = np.zeros((nx_sub, ny))
    x_sub = np.zeros((nx_sub, ny))
    comm.Scatterv([D, counts_col, displs_col, MPI.DOUBLE], d_sub, root)
    comm.Scatterv([X, counts_col, displs_col, MPI.DOUBLE], x_sub, root)

    # # Solve 각 row independently
    plan = PTDMAPlanManyRHS()
    plan.create(nx_sub, ny, comm.py2f(), is_cyclic)
    a = np.full(nx_sub, a_lower)
    b = np.full(nx_sub, a_diag)
    c = np.full(nx_sub, a_upper)
    PTDMASolverManyRHS.solve(plan, a, b, c, d_sub)
    plan.destroy()

    # Gather solved D
    D_solved = None
    if rank == root:
        D_solved = np.zeros((nx, ny))
    comm.Gatherv(d_sub, [D_solved, counts_col, displs_col, MPI.DOUBLE], root)

    if rank == root:
        error = D_solved - X
        rms_error = np.sqrt(np.sum(error ** 2) / (nx * ny))
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