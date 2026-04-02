'''Sample solution for CSC336H1S Assignment 4, Question 3'''

import math
import numpy as np
from scipy.interpolate import CubicSpline

# Part (a)

# Construct the data set.

t = np.array([0, 0.25, 0.5, 1.25, 2.0])
y = np.exp(-(t * t) / 2)

# y' = -t*y, so y'(0) = 0 and y'(2) = -2exp(-2)

clamped_conditions = ((1, 0.0), (1, -2 * math.exp(-2.0)))

# Construct the clamped cubic spline.

cubic_spline = CubicSpline(t, y, bc_type=clamped_conditions)

# Part (b)

# Sample | S(t) - f(t) | at 501 evenly spaced points in [0,2]
# and report the maximum observed error.

tp = np.linspace(0.0, 2.0, 501)
spline_at_tp = cubic_spline(tp)
y_at_tp = np.exp(-(tp * tp) / 2)
max_observed_error = np.linalg.norm(spline_at_tp - y_at_tp, ord=np.inf)

# class error bound

# We know that
#
# max    | f(t) - S(t) |  <= (5/384) * h^4 * max | f^(4) (t) |
# t in [0,2]                                 t in [0,2]
#
# where h is the maximum step in the t-values in the data set.
#
# Here:
# f(t) = exp(- (t^2) / 2)
#
# f'(t) = -t exp(- (t^2) / 2)
#       = -t f(t)
#
# f''(t) = - f(t) - tf'(t) 
#        = - f(t) - t(-t f(t))
#        = f(t) (t^2 - 1)
#
# f'''(t) = f'(t) (t^2 - 1) + f(t) (2t)  
#         = (-t f(t))(t^2 - 1) + f(t)(2t)  
#         = f(t)(-t^3 + t + 2t)
#         = f(t)(3t - t^3)
#
# f''''(t) = f'(t)(3t - t^3) + f(t)(3 - 3t^2)
#          = (-t f(t))(3t - t^3) + f(t)(3 - 3t^2)
#          = f(t)(-3t^2 + t^4 + 3 - 3t^2)
#          = f(t)(t^4 - 6t^2 + 3)

# Estimate the max value of f'''' by sampling it at 501 points in [0,2]
# and finding the max observation.

fourth_derivs = []
for i in range(len(tp)):
    t = tp[i]
    t_sq = t * t
    t_fourth = t_sq * t_sq
    f_value = math.exp(- t_sq / 2)
    f_iv = f_value * (t_fourth - 6 * t_sq + 3)
    fourth_derivs.append(f_iv)

max_fourth = max(fourth_derivs)

# Compute the value of the "class bound" and compare with the observed bound.

class_bound = (5 / 384) * ( 0.75 ) ** 4 * max_fourth

if class_bound > max_observed_error:
    print('The class bound was met.')
else:
    print('The class bound was broken!')

print('\nclass bound: {0:8.3e}'.format(class_bound))
print('observed bound: {0:8.3e}'.format(max_observed_error))
