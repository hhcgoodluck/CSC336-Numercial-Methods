'''Sample solution for CSC336H1S Assignment 3, Question 2'''

import numpy
import scipy

def a3q2() -> None:
    """Print a table of results on the accuracy of computing the inverse
    of the Hilbert matrix by using the scipy.linalg.inv function.
    """

    # Print table headings
    print('     relative   condition              machine')
    print('n    error      number      ratio      epsilon')
    print('==   ========   =========   ========   =======')
    print(' ')

    # Calculate the required table of values for the Hilbert matrix.
    eps_mach = numpy.finfo(float).eps / 2
    for n in range(2, 13):

        # Generate the matrices - produces type numpy.ndarray
        H = scipy.linalg.hilbert(n)
        Hinv = scipy.linalg.inv(H)  
        preciseHinv = scipy.linalg.invhilbert(n)

        # Measure the error and condition number and report.
        relativeError = scipy.linalg.norm(Hinv - preciseHinv, ord=2) / \
                           scipy.linalg. norm(preciseHinv, ord=2)
        conditionNumber = numpy.linalg.cond(H, p=2)
        ratio = relativeError / conditionNumber

        print('{0:2d}   {1:5.2e}   {2:5.2e}    {3:5.2e}   {4:5.2e}'.format(
          n, relativeError, conditionNumber, ratio, eps_mach))

if __name__ == '__main__':

    a3q2()
