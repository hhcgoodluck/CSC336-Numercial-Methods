import numpy as np
import pandas as pd
from scipy.linalg import hilbert, inv, invhilbert

""" 希尔伯特矩阵（病态矩阵）分析 """
def compute_hilbert_analysis():

    # 定义 n 的范围
    n_values = range(2, 13)

    # 计算机的浮点数机器精度
    eps_mach = np.finfo(float).eps / 2

    # 存储计算结果
    results = []

    # 遍历 n 计算相应的值
    for n in n_values:

        # 生成 Hilbert 矩阵
        H_n = hilbert(n)

        # 数值方法求逆矩阵
        H_inv_numeric = inv(H_n)

        # 解析计算的精确逆矩阵
        H_inv_exact = invhilbert(n)

        # 计算相对误差
        relative_error = np.linalg.norm(H_inv_numeric - H_inv_exact, 2) / np.linalg.norm(H_inv_exact, 2)

        # 计算 Hilbert 矩阵的条件数
        cond_H = np.linalg.cond(H_n, 2)

        # 计算误差与条件数的比值
        error_ratio = relative_error / cond_H

        # 存储结果
        results.append([n, relative_error, cond_H, error_ratio, eps_mach])

    # 转换为 Pandas DataFrame 并打印
    df = pd.DataFrame(results, columns=["n", "Relative Error", "Condition Number", "Error/Cond Ratio", "Machine Epsilon"])
    print(df)

# 运行函数
if __name__ == "__main__":
    compute_hilbert_analysis()
