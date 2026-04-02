'''Sample solution for CSC336H1S Assignment 2, Question 3.'''

# It would be fine to use numpy's ndarray instead of python list
# Using numpy would have made the coding much easier, though this solution
# uses python only.

import pytest    # for checking floats
import copy      # for making copies of matrices and vectors
import random    # for making random matrices, vectors and dimensions

# The following backsolve function is used by each of GE_no_pivot(),
#   GE_partial_pivot() and GE_complete_pivot().

def backsolve(U: list[list[float]], b: list[float], 
              rp: list[int] = None, cp: list[int] = None) -> list[float]:
    """Return the solution to the upper triangular system Ux = b that
    may make use of the row pivot vector rp and the column pivot vector cp.
    """
   
    # Function that uses backward substitution to solve an 
    # upper triangular linear system. 
   
    n = len(U)
   
    # use rp = [0, 1, 2, ..., n-1] when not given
    if not rp:
       rp = list(range(n))

    # use cp = [0, 1, 2, ..., n-1] when not given
    if not cp:
       cp = list(range(n))
   
    x = [0] * n
    for j in range(n - 1, -1, -1):
       x[cp[j]] = b[rp[j]] / U[rp[j]][cp[j]]
       for i in range(j):
           b[rp[i]] = b[rp[i]] - U[rp[i]][cp[j]] * x[cp[j]]

    return x


def GE_no_pivot(A: list[list[float]], b: list[float]) -> list[float]:
    """Return the solution x to the linear system Ax = b computed using
    Gaussian elimination without pivoting.

    >>> # example from class notes 31 Jan 2025
    >>> A = [[2, 4, -2, 6], [4, 14, 0, 14], [-1, 10, 7, 4], [-2, -1, 0, 2]]
    >>> b = [4, 18, 15, -6]
    >>> x = GE_no_pivot(A, b)
    >>> x == pytest.approx([1, 2, 0, -1])
    True
    """

    # Function that implements Gaussian elimination with no pivoting
    # to solve a linear system Ax = b for x.
    #
    # Steps:
    #
    #     Use Gaussian elimination to factor A into LU.
    #     Solve Ly = b for y
    #     Solve Ux = y for x
    #
    #     Since this method does not have to return L and U, it is easiest
    #     to just update b along with A, and then solve the left over
    #     upper-triangular system for x at the end.
    
    n = len(A)
   
    # Use Gaussian elimination to reduce A to upper-triangular form.
    # Apply the same elimination steps to b to update it.
    # Elements below the main diagonal are explicitly set to 0 so that 
    # the resulting matrix is upper triangular - otherwise rounding errors in
    # the computations # may leave an element as simply near 0.
   
    for i in range(n - 1):
        # eliminate the entries below A[i][i]
        for j in range(i + 1, n):
            multiplier = A[j][i] / A[i][i]
            for k in range(i + 1, n):
                A[j][k] = A[j][k] - multiplier * A[i][k]
            b[j] = b[j] - multiplier * b[i]
            A[j][i] = 0.0
   
    # Solve the resulting upper triangular system for x.
    x = backsolve(A, b)

    return x


# GE with partial (row) pivoting

def GE_partial_pivot(A: list[list[float]], b: list[float]) -> list[float]:
    """Return the solution x to the linear system Ax = b computed using
    Gaussian elimination with partial (row) pivoting.

    >>> # example from class notes 31 Jan 2025
    >>> A = [[2, 4, -2, 6], [4, 14, 0, 14], [-1, 10, 7, 4], [-2, -1, 0, 2]]
    >>> b = [4, 18, 15, -6]
    >>> x = GE_partial_pivot(A, b)
    >>> x == pytest.approx([1, 2, 0, -1])
    True
    """
   
    # This algorithm differs from the basic GE algorithm in that before each
    # elimination step is performed, the entries in the column below the pivot
    # are searched for the largest entry.   The row corresponding to the 
    # largest entry is used as the basis for the next elimination step.

    n = len(A)

    # Set up the vector rp to keep track of the row interchanges.  Initialize
    # it to be such that rp[i] = i.  This corresponds to no pivoting.   The
    # effect of row pivoting is that all references to row i of the matrix
    # should be done indirectly through the rp vector.   That is, refer to
    # element A[rp[i]][j] instead of element A[i][j].
    rp = list(range(n))    # [0, 1, 2, ..., n-1]

    for i in range(n - 1):
        
        # Determine the pivot row
        pivotRow = i
        maxValue = abs(A[rp[pivotRow]][i])
        for j in range(i + 1, n):
            if abs(A[rp[j]][i]) > maxValue:
                maxValue = abs(A[rp[j]][i])
                pivotRow = j
        
        # Swap rp[i] and rp[pivotRow] to implement the pivoting.
        rp[i], rp[pivotRow] = rp[pivotRow], rp[i]
         
        # Eliminate the entries below A[rp[i]][i]
        for j in range(i + 1, n):
            multiplier = A[rp[j]][i] / A[rp[i]][i]
            for k in range(i + 1, n):
                A[rp[j]][k] = A[rp[j]][k] - multiplier * A[rp[i]][k]
            b[rp[j]] = b[rp[j]] - multiplier * b[rp[i]]
            A[rp[j]][i] = 0.0

    # Solve the resulting upper triangular system for x.
    x = backsolve(A, b, rp)

    return x
 

# GE with complete pivoting

def GE_complete_pivot(A: list[list[float]], b: list[float]) -> list[float]:
    """Return the solution x to the linear system Ax = b computed using
    Gaussian elimination with complete pivoting.

    >>> # example from class notes 31 Jan 2025
    >>> A = [[2, 4, -2, 6], [4, 14, 0, 14], [-1, 10, 7, 4], [-2, -1, 0, 2]]
    >>> b = [4, 18, 15, -6]
    >>> x = GE_complete_pivot(A, b)
    >>> x == pytest.approx([1, 2, 0, -1])
    True
    """

    # This algorithm differs from the basic GE algorithm in that before each
    # elimination step is performed, the entries in the whole uneliminated
    # submatrix are searched for the largest entry.   The row and column indices
    # corresponding to the largest entry is then used as the basis for the next
    # elimination step.

    n = len(A)

    # Set up the vector rp to keep track of the row interchanges.
    # Set up the vector cp to keep track of the column interchanges.
    # Initialize them to be such that rp[i] = cp[i] = i.  This corresponds
    # to no pivoting.   The effect of row pivoting is that all references to
    # row i of the matrix should be done indirectly through the rp vector.
    # And similarly for column pivoting. That is, refer to element 
    # A[rp[i]][cp[j]] instead of element A[i][j].
    rp = list(range(n))
    cp = list(range(n))

    for i in range(n - 1):

        # Determine the pivot row and column by finding the largest 
        # element in the submatrix that has not yet been eliminated.

        maxValue = abs(A[rp[i]][cp[i]])
        pivotRow = i
        pivotColumn = i
        for j in range(i, n):
            for k in range(i, n):
                if abs(A[rp[j]][cp[k]]) > maxValue:
                    maxValue = abs(A[rp[j]][cp[k]])
                    pivotRow = j
                    pivotColumn = k

        # Swap rp[i] with rp[pivotRow] and cp[i] with cp[pivotColumn] to
        # implement the pivoting.
        rp[i], rp[pivotRow] = rp[pivotRow], rp[i]
        cp[i], cp[pivotColumn] = cp[pivotColumn], cp[i]

        # Eliminate the entries below A[rp[i]][cp[i]]
        for j in range(i + 1, n):
            multiplier = A[rp[j]][cp[i]] / A[rp[i]][cp[i]]
            for k in range(i + 1, n):
                A[rp[j]][cp[k]] = A[rp[j]][cp[k]] - multiplier * A[rp[i]][cp[k]]
            b[rp[j]] = b[rp[j]] - multiplier * b[rp[i]]
            A[rp[j]][cp[i]] = 0.0

    # Solve the resulting upper triangular system for x.
    x = backsolve(A, b, rp, cp)

    return x


def gen_random_problem(n: int) \
    -> tuple[list[list[float]], list[float], list[float]]:
    """Return a tuple of A, x, b for which Ax = b for use in testing
    for part (b). Here A is a random n x n matrix, x has components x_i
    with x_i = (-1)^{i+1} and b = A * x.

    Note: modified to generate random x_i instead of (-1)^{i+1}
    """

    # generate random n x n matrix A
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(random.random())
        A.append(row)

    # generate solution n-vector x with x_i = (-1)^{i+1}
    # Note: modified to generate random x_i not (-1)^{i+1}
    x = []
    # x_i = 1.0
    for i in range(n):
        # x_i = - x_i
        # x.append(x_i)
        x.append(random.random())

    # now determine b = A*x so that x solves Ax = b
    b = mat_vec_prod(A, x)

    return (A, x, b)

def gen_cp_2pt7_problem(n: int) \
    -> tuple[list[list[float]], list[float], list[float]]:
    """Return a tuple of A, x, b for which Ax = b for use in testing
    for part (c). Here A is a n x n matrix of the form suggested by Heath
    Computer Problem 2.7, x has components x_i with x_i = (-1)^{i+1} and 
    b = A * x.

    Note: modified to generate random x_i instead of (-1)^{i+1}
    """

    # Generate the coefficient matrix. First make a 0 matrix then fill it in.
    A = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0.0)
        A.append(row)

    for i in range(n):
        for j in range(n):
            if i > j:
                A[i][j] = -1.0
            elif i == j:
                A[i][j] = 1.0
            elif j == n - 1 :
                A[i][j] = 1.0

    # generate solution n-vector x with x_i = (-1)^{i+1}
    # Note: modified to generate random x_i not (-1)^{i+1}
    x = []
    # x_i = 1.0
    for i in range(n):
        # x_i = - x_i
        # x.append(x_i)
        x.append(random.random())

    # now determine b = A*x so that x solves Ax = b
    b = mat_vec_prod(A, x)

    return (A, x, b)

# Helper functions for basic linear algebra operations

def inf_norm(v: list[float]) -> float:
    """Return the infinity norm of the vector v.
    """

    result = abs(v[0])
    for i in range(1, len(v)):
        if abs(v[i]) > result:
            result = abs(v[i])
    return result


def mat_vec_prod(A: list[list[float]], w: list[float]) -> list[float]:
    """Return the matrix-vector product z = A * w.
    """

    n = len(A)
    z = []
    for i in range(n):
        z_i = 0.0
        for j in range(n):
            z_i = z_i + A[i][j] * w[j]
        z.append(z_i)
    return z


def vec_diff(a: list[float], b: list[float]) -> list[float]:
    """Return the vector difference c = a - b.
    """

    n = len(a)
    c = []
    for i in range(n):
        c_i = a[i] - b[i]
        c.append(c_i)
    return c


def print_table_header(title: str) -> None:
    """Print header for results table with given title.
    """

    print(title)
    print('          |   NO PIVOTING     |   ROW PIVOTING    | COMPLETE PIVOTING')
    print('          | relative relative | relative relative | relative relative')
    print('dimension | residual error    | residual error    | residual error')
    print('=====================================================================')


def print_results(A: list[list[float]], x_true: list[float], b: list[float]) \
        -> None:
    """Print relative residual and relative error results for solving the
    problem Ax = b (having true solution x_true) using the solvers
    GE_no_pivot(), GE_partial_pivot() and GE_complete_pivot().
    """

    # Solve the problem Ax = b using each of the three solvers developed
    #  in part (a).   Since the solvers change the input data, make copies
    #  before calling them.

    Acopy = copy.deepcopy(A)
    bcopy = copy.deepcopy(b)
    xnp = GE_no_pivot(Acopy, bcopy)

    Acopy = copy.deepcopy(A)
    bcopy = copy.deepcopy(b)
    xpp = GE_partial_pivot(Acopy, bcopy)

    Acopy = copy.deepcopy(A)
    bcopy = copy.deepcopy(b)
    xcp = GE_complete_pivot(Acopy, bcopy)

    #  Now determine the relative residuals and relative errors in the 
    #  computed solutions.

    norm_b = inf_norm(b)
    norm_x_true = inf_norm(x_true)

    rrnp = inf_norm(vec_diff(b,  mat_vec_prod(A, xnp))) / norm_b
    renp = inf_norm(vec_diff(xnp, x_true)) / norm_x_true

    rrpp = inf_norm(vec_diff(b,  mat_vec_prod(A, xpp))) / norm_b
    repp = inf_norm(vec_diff(xpp, x_true)) / norm_x_true

    rrcp = inf_norm(vec_diff(b,  mat_vec_prod(A, xcp))) / norm_b
    recp = inf_norm(vec_diff(xcp, x_true)) / norm_x_true

    print(
    '{0:4d}       {1:5.2e}  {2:5.2e}  {3:5.2e}  {4:5.2e}  {5:5.2e}  {6:5.2e}  '\
        .format(len(b), rrnp, renp, rrpp, repp, rrcp, recp))

def q_3_b() -> None:
    """Produce numerical results required for Q 3 (b).
    """

    # Test the functions for random matrices of 8 different dimensions.
   
    print_table_header('\n\nRandom matrix results : \n')
    for i in range(8):

        # Determine a dimension for the problem between 1 and 100.

        n = random.randint(1, 100)

        # Generate the coefficient matrix, solution and corresponding
        # right hand side.

        A, x_true, b = gen_random_problem(n)

        print_results(A, x_true, b)


def q_3_c() -> None:
    """Produce numerical results required for Q 3 (c).
    """

    # Test the functions for random matrices of 8 different dimensions.
   
    print_table_header('\n\nHeath Computer Problem 2.7 matrix results : \n')
    for i in range(8):

        # Determine a dimension for the problem between 1 and 100.

        n = random.randint(1, 100)

        # Generate the coefficient matrix, solution and corresponding
        # right hand side.

        A, x_true, b = gen_cp_2pt7_problem(n)

        print_results(A, x_true, b)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    q_3_b()
    q_3_c()

