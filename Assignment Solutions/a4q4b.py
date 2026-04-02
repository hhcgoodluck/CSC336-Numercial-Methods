'''Sample solution for CSC336H1S Assignment 4, Question 4.'''

import math
from typing import Callable

def fixedPointIteration(g: Callable[[float], float], xknot: float) -> None:
    """Print the results from applying fixed-point iteration in an attempt
    to find fixed points of x = g(x), starting at x_0 = xknot.
    Stop the iteration when either it has not converged after 30 iterations,
    it has produced an Inf or the change in the iterates is less than 1.0e-10.
    """

    print('\n')
    print(' k    xk                       delta(xk)\n' \
          ' ==   ======================   =========\n')

    k = 0
    xk = xknot
    print('{0:3d}   {1:18.15e}'.format(k, xk))

    xkp1 = g(xk)
    deltax = abs(xkp1 - xk)
    k = k + 1
    xk = xkp1
    print('{0:3d}   {1:18.15e}    {2:5.2e}'.format(k, xk, deltax))
    while k < 30 and abs(xk) < math.inf and deltax > 1.0e-10:
        xkp1 = g(xk)
        deltax = abs(xkp1 - xk)
        k = k + 1
        xk = xkp1
        print('{0:3d}   {1:18.15e}    {2:5.2e}'.format(k, xk, deltax))


def g1(x: float) -> float:
    """Return the value of the first fixed point function."""
    return (x * x + 2) / 3


def g2(x: float) -> float:
    """Return the value of the second fixed point function."""
    return math.sqrt(3 * x - 2)


def g3(x: float) -> float:
    """Return the value of the third fixed point function."""
    return 3 - 2 / x


def g4(x: float) -> float:
    """Return the value of the fourth fixed point function."""
    return (x * x - 2) / (2 * x - 3)


if __name__ == '__main__':
    
    #  Apply fixed-point iteration for the four given functions
    #  starting with x_0 = 4.0. 
    
    print('Results for g_1(x):')
    fixedPointIteration(g1, 4.0);

    print('\n\nResults for g_2(x):')
    fixedPointIteration(g2, 4.0);

    print('\n\nResults for g_3(x):')
    fixedPointIteration(g3, 4.0);

    print('\n\nResults for g_4(x):')
    fixedPointIteration(g4, 4.0);
