'''Sample solution for CSC336H1S Assignment 2, Question 4.'''

import time
import random

import numpy as np

# regular Python 3 functions

def make_random_matrix(n: int) -> list[list[float]]:
    """Return a n by n matrix stored in a Python nested list, with each
    row in the outermost list.
    """

    matrix = [] 
    for row in range(n):
        matrix.append([ 0 ] * n)    # add a row of 0's, to be filled in
        for col in range(n):
            matrix[row][col] = random.random()
    return matrix


def matrix_multiply(A: list[list[float]], B: list[list[float]]) ->  \
                                                 list[list[float]]:
    """Return the matrix-matrix product of A times B.

    Precondition: len(A[0]) == len(B)
                  each sublist of A contains the same number of floats
                  each sublist of B contains the same number of floats
    """

    n = len(A)
    product = []
    for row in range(n):
        product.append([ 0 ] * n)    # add a row of 0's, to be filled in
        for col in range(n):
            element = 0
            for k in range(n):
                element += A[row][k] * B[k][col]
            product[row][col] = element
    return product


def time_python_multiply_experiment(n: int) -> float:
    """Return the time required to multiply two random n x n matrices that
    are stored as Python lists of lists of floats.
    """

    # construct the random matrices 
    A = make_random_matrix(n)
    B = make_random_matrix(n)

    # time the matrix multiply operation
    start_time = time.perf_counter()
    C = matrix_multiply(A, B)
    stop_time = time.perf_counter()

    # determine time to multiply and return 
    return stop_time - start_time

# start of functions that use NumPy

def print_numpy_product() -> None:
    """Print the NumPy matrix-matrix product of the 3-by-3 identity matrix
    with the matrix whose (i,j)th element is equal to i + j.

    Reading the output produced by this function will allow you to confirm
    that you are multiplying matrices correctly in NumPy.
    """

    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print('I = \n', I)
    M = np.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    print('M = \n', M)

    # print the matrix-matrix product of I into M by using a single
    # numpy operator.  
    # Students: Write one line of code below this comment.

    # Q4(e)
    print('product = \n', I @ M)


def time_numpy_multiply_experiment(n: int) -> float:
    """Return the time required to multiply two random n x n matrices that
    are stored as NumPy ndarrays and using the NumPy matrix-matrix multiply
    operator.
    """

    # Q4(f)
    # construct the random matrices 
    # Students: enter your code here
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # time the NumPy matrix multiply operation
    # Students: enter your code here
    start_time = time.perf_counter()
    C = A @ B
    stop_time = time.perf_counter()

    # Students: modify this return statement to return the time required
    #   by the NumPy matrix multiply operation instead of the number 42
    # return 42
    return stop_time - start_time

# end of functions that use NumPy

if __name__ == '__main__':

    # Students: you do not need to modify the __main__ code though you
    #        may want to revise the final print statement to produce
        
    # revise the print_numpy_product function so that it prints the
    # correct matrix before proceeding
    print_numpy_product()

    multiply_times = []

    # determine the time to multiply two n x n matrices
    n = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for i in range(len(n)):
        multiply_times.append([time_python_multiply_experiment(n[i])])
                  
    # now repeat the experiment using operations on numpy
    for i in range(len(n)):
        multiply_times[i].append(time_numpy_multiply_experiment(n[i]))
    
    # display the timing results
    print('\n\n  Time to multiply two n x n matrices \n')
    print('       using\n' + 
          '       basic     using\n' +
          '   n   Python    NumPy\n' +
          '----  --------  ---------')
    for i in range(len(n)):
        print('{0:4d}  {1:5.2e}  {2:5.2e} '.format(n[i], multiply_times[i][0],
                                                         multiply_times[i][1]))
