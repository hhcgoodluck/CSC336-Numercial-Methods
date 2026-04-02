import numpy as np
import pandas as pd
import scipy.linalg


def generate_system(n):
    """ Generate the Pascal matrix A, the exact solution vector x*, and the right-hand side vector b. """

    # Generate Pascal matrix A (生成 Pascal 矩阵 A)
    A = scipy.linalg.pascal(n)

    # Generate exact solution vector x* (random values) (生成随机精确解向量 x*（随机数))
    x_star = np.random.rand(n, 1)

    # Compute the right-hand side vector (计算右端项向量 b = A * x_star)
    b = A @ x_star

    return A, x_star, b


def solve_and_evaluate(n):
    """ Solve Ax = b and compute errors, condition number, and determinant. """
    A, x_star, b = generate_system(n)

    # 使用 SciPy 求解 Ax = b
    x = scipy.linalg.solve(A, b)

    # 计算相对误差 ||x - x*|| / ||x*||
    # Compute relative error
    relative_error = np.linalg.norm(x - x_star) / np.linalg.norm(x_star)

    # 计算残差 r = b - Ax & 相对残差 ||r|| / ||b||
    # Compute residual and relative residual
    residual = b - A @ x
    relative_residual = np.linalg.norm(residual) / np.linalg.norm(b)

    # 计算矩阵的条件数和行列式
    # Compute condition number and determinant
    cond_A = np.linalg.cond(A)
    det_A = np.linalg.det(A)

    return n, relative_error, relative_residual, cond_A, det_A


def run_experiment():
    """
    Run the experiment, gradually increasing n, computing and printing errors,
    residuals, condition number, and determinant until the relative error exceeds 1.
    """
    results = []  # Store results for different values of n
    n = 1         # Start with n=1

    while True:
        n, rel_err, rel_res, cond_A, det_A = solve_and_evaluate(n)
        results.append([n, rel_err, rel_res, cond_A, det_A])

        # Stop experiment if relative error exceeds 1
        if rel_err > 1:
            break

        n += 1  # Increase n and continue

    # Convert results to a Pandas DataFrame and display as a table
    df = pd.DataFrame(results, columns=["n", "Relative Error", "Relative Residual", "Condition Number cond(A)",
                                        "Determinant det(A)"])

    print("\n Experiment Results:\n")
    print(df.to_string(index=False))

    return df

if __name__ == "__main__":
    run_experiment()
