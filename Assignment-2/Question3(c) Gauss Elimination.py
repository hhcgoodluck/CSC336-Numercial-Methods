import numpy as np

def compute_errors(A, b, x_computed, x_true):
    """Compute relative error and absolute residual using the infinity norm."""
    relative_error = np.linalg.norm(x_computed - x_true, ord=np.inf) / np.linalg.norm(x_true, ord=np.inf)
    residual = np.linalg.norm(A @ x_computed - b, ord=np.inf)
    return relative_error, residual

def gaussian_elimination_partial_pivoting(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    A = A.astype(float)
    b = b.astype(float)

    for k in range(n - 1):
        max_row = np.argmax(np.abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

def gaussian_elimination_complete_pivoting(A, b):
    """Solve Ax = b using Gaussian elimination with complete pivoting."""
    n = len(A)
    A = A.astype(float)
    b = b.astype(float)

    row_swaps = np.arange(n)
    col_swaps = np.arange(n)

    for k in range(n - 1):
        max_index = np.unravel_index(np.argmax(np.abs(A[k:, k:])), A[k:, k:].shape)
        max_row, max_col = max_index[0] + k, max_index[1] + k

        A[[k, max_row]] = A[[max_row, k]]
        A[:, [k, max_col]] = A[:, [max_col, k]]
        b[k], b[max_row] = b[max_row], b[k]

        row_swaps[k], row_swaps[max_row] = row_swaps[max_row], row_swaps[k]
        col_swaps[k], col_swaps[max_col] = col_swaps[max_col], col_swaps[k]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    x_reordered = np.zeros(n)
    for i in range(n):
        x_reordered[col_swaps[i]] = x[i]

    return x_reordered

# Define the given non-random matrix A and vector b
A = np.array([
    [1, 0, 0, 0, 1],
    [-1, 1, 0, 0, 1],
    [-1, -1, 1, 0, 1],
    [-1, -1, -1, 1, 1],
    [-1, -1, -1, -1, 1]
])

x_true = np.array([1, -1, 1, -1, 1])
b = A @ x_true  # Compute the right-hand side

# Solve using partial pivoting
x_partial = gaussian_elimination_partial_pivoting(A.copy(), b.copy())
rel_error_partial, residual_partial = compute_errors(A, b, x_partial, x_true)

# Solve using complete pivoting
x_complete = gaussian_elimination_complete_pivoting(A.copy(), b.copy())
rel_error_complete, residual_complete = compute_errors(A, b, x_complete, x_true)

# Display results
print("===== Comparison Summary =====")
print(f"Partial Pivoting:   Solution = {x_partial}")
print(f"                    Relative Error = {rel_error_partial:.2e}, Residual = {residual_partial:.2e}\n")

print(f"Complete Pivoting:  Solution = {x_complete}")
print(f"                    Relative Error = {rel_error_complete:.2e}, Residual = {residual_complete:.2e}")
