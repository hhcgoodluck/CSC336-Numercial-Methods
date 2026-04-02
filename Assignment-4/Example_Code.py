'''Example code for CSC336H1S Assignment 4.'''

# Demonstrates the use of:
#  - numpy.polynomial.polynomial.polyfit and numpy.polynomial.polynomial.polyval
#  - scipy interpolate.CubicSpline
#  - matplotlib

import numpy as np
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Interpolating the 9 data points:
    #
    # t | 0 | 1 | 4 | 9 | 16 | 25 | 36 | 49 | 64
    # --|---|---|---|---|----|----|----|----|---
    # y | 0 | 1 | 2 | 3 |  4 |  5 |  6 |  7 |  8
    #
    # should give an approximation to the square root function.

    # The code below determines the degree 8 polynomial interpolant, plots it
    # and a sampling of the error in the interpolant. Then the cubic spline
    # interpolant is determined.  The spline and a sampling of its error are
    # then plotted.  Separate plots over the interval [0, 1] are also produced.

    # Read on for details.

    # Store the given data points in vectors t and y.
    # Uses the numpy * operator to do componentwise multiplication.
    y = np.linspace(0, 8, 9)
    t = y * y

    # Determine 1001 evenly spaced t-values that will be used when plotting
    # over the interval [0, 64].  Include the original 9 points and then
    # sort so that the t-values are increasing.  Also determine 1001
    # points on the interval [0, 1] for a zoomed in look at that interval.
    t1001 = np.linspace(0, 64, 1001)
    tplot = np.hstack([t, t1001])
    tplot.sort()
    tzoom = np.linspace(0, 1, 1001)

    # Determine the degree 8 polynomial that interpolates the 9 given
    # {(t_i, y_i)} data points.
    p = polyfit(t, y, len(t) - 1)

    # produce the plots for t in [0, 64]

    # plot the data, sqrt(t) and the polynomial p_8(t) in a single figure
    # by sampling sqrt(t) and p_8(t) at the tplot values.
    # Uses the numpy sqrt function to do computations elementwise.

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Visualizing the degree 8 polynomial interpolant of sqrt(t)',
                 fontsize=16)

    # sqrt(t) and p_8(t) vs. t
    axs[0].plot(t, y, 'o', label='data')
    axs[0].plot(tplot, np.sqrt(tplot), '-', label='sqrt(t)')
    axs[0].plot(tplot, polyval(tplot, p), '--', label='p_8(t)')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y')
    axs[0].legend()

    # plot the error |sqrt(t) - p_8(t)| vs. t
    axs[1].plot(tplot, np.abs(np.sqrt(tplot) - polyval(tplot, p)), '-',
                label='|sqrt(t) - p_8(t)|')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('|sqrt(t) - p_8(t)|')
    axs[1].legend()

    # repeat the above plots, zooming in to the interval [0, 1]

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Zooming in to t in [0, 1]', fontsize=16)

    # sqrt(t) and p_8(t) vs. t
    axs[0].plot([0, 1], [0, 1], 'o', label='data')
    axs[0].plot(tzoom, np.sqrt(tzoom), '-', label='sqrt(t)')
    axs[0].plot(tzoom, polyval(tzoom, p), '--', label='p_8(t)')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y')
    axs[0].legend()

    # plot the error |sqrt(t) - p_8(t)| vs. t
    axs[1].plot(tzoom, np.abs(np.sqrt(tzoom) - polyval(tzoom, p)), '-',
                label='|sqrt(t) - p_8(t)|')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('|sqrt(t) - p_8(t)|')
    axs[1].legend()

    # repeat all of the above, but this time using a cubic spline interpolant
    # instead of the degree 8 polynomial interpolant

    # Determine a cubic spline interpolant of the 9 given {(t_i, y_i)}
    # data points.
    cs = CubicSpline(t, y)

    # produce the plots for t in [0, 64]

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Visualizing a cubic spline interpolant of sqrt(t)',
                 fontsize=16)

    # sqrt(t) and cs(t) vs. t
    axs[0].plot(t, y, 'o', label='data')
    axs[0].plot(tplot, np.sqrt(tplot), '-', label='sqrt(t)')
    axs[0].plot(tplot, cs(tplot), '--', label='cubic_spline(t)')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y')
    axs[0].legend()

    # plot the error |sqrt(t) - cubic_spline(t)| vs. t
    axs[1].plot(tplot, np.abs(np.sqrt(tplot) - cs(tplot)), '-',
                label='|sqrt(t) - cubic_spline(t)|')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('|sqrt(t) - cubic_spline(t)|')
    axs[1].legend()

    # repeat the above plots, zooming in to the interval [0, 1]

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle('Zooming in to t in [0, 1]', fontsize=16)

    # sqrt(t) and cs(t) vs. t
    axs[0].plot([0, 1], [0, 1], 'o', label='data')
    axs[0].plot(tzoom, np.sqrt(tzoom), '-', label='sqrt(t)')
    axs[0].plot(tzoom, cs(tzoom), '--', label='cubic_spline(t)')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('y')
    axs[0].legend()

    # plot the error |sqrt(t) - cubic_spline(t)| vs. t
    axs[1].plot(tzoom, np.abs(np.sqrt(tzoom) - cs(tzoom)), '-',
                label='|sqrt(t) - cubic_spline(t)|')
    axs[1].set_yscale('log')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('|sqrt(t) - cubic_spline(t)|')
    axs[1].legend()

    # show the plots!
    plt.show()
