import numpy as np
import math
from frenet_spline import SplineFitter
import matplotlib.pyplot as plt
from spline import BSpline, BSplineBasis


fitter = SplineFitter(knot_intervals=10)

pos = np.loadtxt("../pos.csv", delimiter=',')
xdata = pos[:, 0]
zdata = pos[:, 1]
t = np.linspace(0, 0.9, xdata.shape[0])
pitch = -1.999/(1+np.exp(-20*(t-0.45)))+1.999/2
#
_, x_spline = fitter.fitting(t, xdata, degree=3)
_, z_spline = fitter.fitting(t, zdata, degree=3)
_, p_spline = fitter.fitting(t, pitch, degree=3)
#
x_spline_t = [x_spline(t_)[0] for t_ in t/max(t)]
z_spline_t = [z_spline(t_)[0] for t_ in t/max(t)]
p_spline_t = [p_spline(t_)[0] for t_ in t/max(t)]

x_spline_knots, x_spline_degree,  x_spline_coeffs = x_spline.basis.knots, x_spline.basis.degree, x_spline.coeffs
z_spline_knots, z_spline_degree,  z_spline_coeffs = z_spline.basis.knots, z_spline.basis.degree, z_spline.coeffs
p_spline_knots, p_spline_degree,  p_spline_coeffs = p_spline.basis.knots, p_spline.basis.degree, p_spline.coeffs

params = np.zeros((3, len(np.concatenate((np.array([x_spline_degree]), np.squeeze(x_spline_knots), np.squeeze(x_spline_coeffs))))))
params[0, :] = np.concatenate((np.array([x_spline_degree]), np.squeeze(x_spline_knots), np.squeeze(x_spline_coeffs)))
params[1, :] = np.concatenate((np.array([z_spline_degree]), np.squeeze(z_spline_knots), np.squeeze(z_spline_coeffs)))
params[2, :] = np.concatenate((np.array([p_spline_degree]), np.squeeze(p_spline_knots), np.squeeze(p_spline_coeffs)))

np.savetxt("../pos_spline.csv", params, delimiter=',')

# x_part = BSpline(BSplineBasis(params[0, 1:11], int(params[0, 0])), params[0, 18:24])
# z_part = BSpline(BSplineBasis(params[1, 1:11], int(params[1, 0])), params[1, 18:24])
# p_part = BSpline(BSplineBasis(params[2, 1:11], int(params[2, 0])), params[2, 18:24])
#
# # plt.plot(xdata, zdata)
# plt.plot([x_spline(t_)[0] for t_ in x_spline_knots], [z_spline(t_)[0] for t_ in z_spline_knots], '*')
# t_part = t[0:200]
# plt.plot([x_part(t_)[0] for t_ in t_part], [z_part(t_)[0] for t_ in t_part])
# # plt.plot(t, pitch)
# # plt.plot(t, x_spline_t, '*')
# # plt.plot(t, z_spline_t, '*')
# # plt.plot(t, p_spline_t, '*')
# plt.show()

