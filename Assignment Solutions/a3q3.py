'''Sample solution for CSC336H1S Assignment 3, Question 3'''

import numpy
import scipy

def generateData(n: int) -> tuple:
    """Return a tuple that includes the n x n Pascal matrix, a random
    n-vector and the their matrix-vector product.
    """
    A = scipy.linalg.pascal(n)
    xstar = numpy.random.rand(n,1)
    b = A @ xstar

    return (A, xstar, b)


def a3q3() -> None:
    """Print a table of results on the accuracy of computing a solution to
    a linear system involving the Pascal matrix by using the scipy.linalg.solve
    function.
    """

    # Solve the given systems   Ax = b  for increasing values of n until the 
    # relative error in the computed solution is greater than 1.

    print('    relative relative')
    print(' n  error    residual cond(A)  det(A)')
    print(' == ======== ======== ======== ========')
    print(' ')

    n = 1
    relativeError = 0
    while relativeError < 1:

        # Generate the system and then solve it.

        A, xstar, b = generateData(n)

        cond_A = numpy.linalg.cond(A, p=2)
        det_A = scipy.linalg.det(A)
        
        x = scipy.linalg.solve(A, b)

        # Find the relative error and relative residual for
        # the computed solution.

        relativeError = scipy.linalg.norm(x - xstar, ord=2) \
                            / scipy.linalg.norm(xstar, ord=2)
        relativeResidual = scipy.linalg.norm(b - A @ x, ord=2) \
                            / scipy.linalg.norm(b, ord=2)

        # Print results for table

        print('{0:2d}  {1:5.2e} {2:5.2e} {3:5.2e} {4:5.2e}'.format(
          n, relativeError, relativeResidual, cond_A, det_A))

        # Repeat with a large dimension problem.

        n = n + 1

if __name__ == '__main__':

    # Print results for n = 3 so we can verify.
    P, xstar, b = generateData(3)
    print('Results for n = 3:\n')
    print('P = ')
    print(P)
    print('xstar = ')
    print(xstar)
    print('b = ')
    print(b)
    print('b - P @ xstar = ')
    print(b - P @ xstar)

    # Generate numerical results as requested.
    a3q3()
