import pickle

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import os

time = np.linspace(0, 20, 20)
timescale = 1.5
x = np.log(1+np.exp(time*timescale-3))/2  # softplus function
y = 0*time
z = np.arctan(time*timescale-8)/2.5+1.3

t, c, k = interpolate.splrep(time, x, s=0, k=4)
spl_x = interpolate.BSpline(t, c, k, extrapolate=False)
t, c, k = interpolate.splrep(time, y, s=0, k=4)
spl_y = interpolate.BSpline(t, c, k, extrapolate=False)
t, c, k = interpolate.splrep(time, z, s=0, k=4)
spl_z = interpolate.BSpline(t, c, k, extrapolate=False)

if os.path.exists('dummy_traj.pickle'):
    os.remove('dummy_traj.pickle')
with open('dummy_traj.pickle', 'wb') as file:
    pickle.dump([spl_x, spl_y, spl_z], file)

plt.plot(x, z)
plt.plot(time, x)
plt.plot(time, z)
plt.show()
