import numpy as np
from mpi4py import MPI
from PaScaL_TDMA_pybind import PTDMAPlanSingle, PTDMASolverSingle

# 상수 정의
a_diag = 10.0
a_upper = -1.0
a_lower = -1.0
root = 0

def generate_rhs(n, tdma_type):
    A = np.full(n, a_lower)
    B = np.full(n, a_diag)
    C = np.full(n, a_upper)
    X = np.random.rand(n)
    D = np.zeros(n)

    if tdma_type == 'cyclic':
        D[0] = A[0] * X[-1] + B[0] * X[0] + C[0] * X[1]
        for i in range(1, n - 1):
            D[i] = A[i] * X[i - 1] + B[i] * X[i] + C[i] * X[i + 1]
        D[-1] = A[-1] * X[-2] + B[-1] * X[-1] + C[-1] * X[0]
    elif tdma_type == 'standard':
        D[0] = B[0] * X[0] + C[0] * X[1]
        for i in range(1, n - 1):
            D[i] = A[i] * X[i - 1] + B[i] * X[i] + C[i] * X[i + 1]
        D[-1] = A[-1] * X[-2] + B[-1] * X[-1]
    else:
        raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")

    return D, X

def main(n, type_str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if type_str not in ["standard", "cyclic"]:
        raise ValueError("Invalid TDMA type. Use 'standard' or 'cyclic'")
    is_cyclic = (type_str == "cyclic")

    # Work distribution
    quotient, remainder = divmod(n, size)
    n_sub = quotient + 1 if rank < remainder else quotient
    counts = comm.gather(n_sub, root=root)
    displs = None
    if rank == root:
        displs = [sum(counts[:i]) for i in range(size)]

    # Generate RHS
    D = X = None
    if rank == root:
        D, X = generate_rhs(n, type_str)

    d_sub = np.zeros(n_sub)
    x_sub = np.zeros(n_sub)
    comm.Scatterv([D, counts, displs, MPI.DOUBLE], d_sub, root=root)
    comm.Scatterv([X, counts, displs, MPI.DOUBLE], x_sub, root=root)

    # Prepare a, b, c vectors
    a_sub = np.full(n_sub, a_lower)
    b_sub = np.full(n_sub, a_diag)
    c_sub = np.full(n_sub, a_upper)

    # Create and solve
    plan = PTDMAPlanSingle()
    plan.create(n_sub, comm.py2f(), root, is_cyclic)
    PTDMASolverSingle.solve(plan, a_sub, b_sub, c_sub, d_sub)
    plan.destroy()

    # Gather result
    D_solved = None
    if rank == root:
        D_solved = np.empty(n)
    comm.Gatherv(d_sub, [D_solved, counts, displs, MPI.DOUBLE], root=root)

    if rank == root:
        error = D_solved - X
        rms_error = np.sqrt(np.sum(error ** 2) / n)
        print("Avg. RMS error =", rms_error)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: mpirun -n <num_procs> python single.py <n> <standard|cyclic>")
        sys.exit(1)
    n = int(sys.argv[1])
    type_str = sys.argv[2]
    if not (10 <= n <= 10000):
        raise ValueError("Recommendation of 10 <= n <= 10,000")
    main(n, type_str)