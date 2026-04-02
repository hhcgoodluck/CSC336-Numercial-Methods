# Question3: 我们比较不同的高斯消元方法（无主元、部分主元、完全主元）并评估它们的误差和稳定性使用
# 矩阵范数使用:行和范数(∞-norm)计算误差，以更准确地衡量不同方法的数值稳定性

# 整个代码设计分为 四个核心部分：
# Step1: 生成随机线性系统（保证唯一解）
# Step2: 误差与残差计算（使用 ∞-norm）
# Step3: 实现三种高斯消元方法（无主元、部分主元、完全主元）
# Step4: 实验与误差对比（运行不同方法并分析结果）并且绘制误差曲线（观察误差随 n 变化）

import numpy as np
import matplotlib.pyplot as plt


### STEP 1: 实现三种 Gaussian 消元方法 ###
def gaussian_elimination_no_pivoting(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b using Gaussian elimination without pivoting."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)

    # Forward elimination
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x.tolist()


def gaussian_elimination_partial_pivoting(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)

    # Forward elimination with row swapping
    for k in range(n - 1):
        max_row = np.argmax(np.abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x.tolist()


def gaussian_elimination_complete_pivoting(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b using Gaussian elimination with complete pivoting using pivot vectors."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)

    row_perm = np.arange(n)  # Row permutation
    col_perm = np.arange(n)  # Column permutation

    # Forward elimination with complete pivoting
    for k in range(n - 1):
        max_index = np.unravel_index(np.argmax(np.abs(A[k:, k:])), A[k:, k:].shape)
        max_row, max_col = max_index[0] + k, max_index[1] + k

        A[[k, max_row]] = A[[max_row, k]]
        A[:, [k, max_col]] = A[:, [max_col, k]]
        b[k], b[max_row] = b[max_row], b[k]

        row_perm[k], row_perm[max_row] = row_perm[max_row], row_perm[k]
        col_perm[k], col_perm[max_col] = col_perm[max_col], col_perm[k]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    # Undo column swaps
    x_final = np.zeros(n)
    for i in range(n):
        x_final[col_perm[i]] = x[i]

    return x_final.tolist()


### STEP 2: 误差计算 ###
def compute_errors(A, b, x_computed, x_true):
    """ Compute numerical error and residual using the infinity norm """

    # Compute relative error (avoid division by zero using a small epsilon)
    error = np.linalg.norm(x_computed - x_true, ord=np.inf) / (np.linalg.norm(x_true, ord=np.inf) + 1e-15)

    # Compute absolute residual
    residual = np.linalg.norm(A @ x_computed - b, ord=np.inf)

    return error, residual


### STEP 3: 生成随机系统并测试 ###
def generate_random_system(n: int):
    """ Generate a random linear system Ax = b with a unique solution """
    A = np.random.rand(n, n)
    x_true = np.array([(-1) ** (i + 1) for i in range(n)])
    b = A @ x_true
    return A.tolist(), b.tolist(), x_true.tolist()


### STEP 4: 运行测试并绘制误差曲线 ###
def test_gaussian_methods():
    """ Test Gaussian elimination methods on a toy 3×3 system """
    A = [[2.0, -1.0, 1.0],
         [3.0, 3.0, 9.0],
         [3.0, 3.0, 5.0]]
    b = [8.0, 27.0, 21.0]

    print("\n===== Testing on Toy 3×3 System =====")
    print(f"A = {A}")
    print(f"b = {b}\n")

    methods = {
        "No Pivoting": gaussian_elimination_no_pivoting,
        "Partial Pivoting": gaussian_elimination_partial_pivoting,
        "Complete Pivoting": gaussian_elimination_complete_pivoting
    }

    for name, method in methods.items():
        x_computed = method(A, b)
        print(f"{name}: Solution = {x_computed}")


def compare_methods_experiment():
    """ Compare Gaussian elimination methods on increasing system sizes and plot results """
    n_values = [5, 30, 200, 500, 1000]
    errors = {m: [] for m in ["No Pivoting", "Partial Pivoting", "Complete Pivoting"]}
    residuals = {m: [] for m in ["No Pivoting", "Partial Pivoting", "Complete Pivoting"]}
    methods = {
        "No Pivoting": gaussian_elimination_no_pivoting,
        "Partial Pivoting": gaussian_elimination_partial_pivoting,
        "Complete Pivoting": gaussian_elimination_complete_pivoting
    }
    for n in n_values:
        A, b, x_true = generate_random_system(n)
        A = np.array(A)
        b = np.array(b)
        x_true = np.array(x_true)
        for name, method in methods.items():
            x_computed = np.array(method(A.tolist(), b.tolist()))
            error, residual = compute_errors(A, b, x_computed, x_true)
            errors[name].append(error)
            residuals[name].append(residual)
    for metric, title in zip([errors, residuals], ["Error", "Residual"]):
        plt.figure(figsize=(8, 5))
        for method, values in metric.items():
            plt.plot(n_values, values, 'o-', label=method)
        plt.xlabel("Matrix Size (n)")
        plt.ylabel(f"Infinity Norm {title}")
        plt.title(f"{title} Comparison of Gaussian Elimination Methods")
        plt.legend()
        plt.yscale("log")
        plt.grid()
        plt.show()


if __name__ == '__main__':
    test_gaussian_methods()
    compare_methods_experiment()
