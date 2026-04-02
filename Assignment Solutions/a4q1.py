'''Sample solution for CSC336H1S Assignment 4, Question 1'''

import math
from random import random
import numpy as np
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
import scipy

# Generate the 5 equally spaced points in the interval [0, pi/4]
# and the data points.

m = 5
pi = math.pi
t = np.linspace(0, pi/4, m)
y = np.sin(t)

# Determine the error bound

h = (pi / 4) / (m - 1)
M = 1
errorBound = (M * h ** m) / (4 * m)
print('The error bound is: {0:11.5e}\n'.format(errorBound))

# Construct the interpolating polynomial p_4(t). 

p4 = polyfit(t, y, m - 1)

# Compare p_4(t) to sin(t) at 10 randomly chosen points in [0, pi/4].

print('\n' \
      'ti            sin(ti)       p_4(ti)     |sin(ti)- p_4(ti)|  <= bound\n' \
      '==            =======       =======     ==================  ========')
for _ in range(10):
    ti = (pi / 4) * random()
    yi = math.sin(ti)
    p4_ti = polyval(ti, p4)
    error = abs(p4_ti - yi)
    if error <= errorBound:
        met = 'True'
    else:
        met = 'False'
    print('{0:11.5e}   {1:11.5e}   {2:11.5e}    {3:11.5e}         {4}'.format(
        ti, yi, p4_ti, error, met))
