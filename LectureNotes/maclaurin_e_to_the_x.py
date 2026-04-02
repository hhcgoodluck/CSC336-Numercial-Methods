""" Compute exp(x) using a Maclaurin series. """

import math

def maclaurin(x: float, tolerance: float) -> None:
    """Print an approximation to the value of e^x using the given tolerance.

    Compute the approximation by using the Maclaurin series
         e^x = sum_{i=0}^{Inf} x^i / i!
    and stopping when the partial sums
         S_{n-1} = sum_{i=0}^{n-1} x^i / i! and
         S_{n} = sum_{i=0}^{n} x^i / i! 
    are such that S_{n} and S_{n-1} differ by less than tolerance.
    """

    S_nm1 = 0.0        # an arbitrary starting value different from 1.0 so
    S_n = 1.0          # code will get into the loop; nm1 stands for n minus 1
    n = 0

    while abs(S_n - S_nm1) >= tolerance:
        S_nm1 = S_n
        n = n + 1
        # determine next term and then add it in to partial sum
        i = n
        new_term = math.pow(x, i) / math.factorial(i)
        S_n = S_nm1 + new_term

    print('Series converged. x = ', x, ' num terms = ', n + 1, ' Sum = ', S_n)
    return 

if __name__ == '__main__':
    # experiment with the function
    tolerance = 1.0e-08
    x_values = [0.0, 1.0, 10.0, 20.0, 40.0]
    for x in x_values:
        maclaurin(x, tolerance)

    input('.pause.')
    maclaurin(-30.0, tolerance)
    maclaurin(-40.0, tolerance)
    input('.done.')
