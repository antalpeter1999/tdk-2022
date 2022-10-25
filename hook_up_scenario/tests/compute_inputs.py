import numpy as np
import sympy as sp
import os
import pickle
import matplotlib.pyplot as plt

def hat(a):
    mat = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return mat


def vee(mat):
    a = sp.zeros(3, 1)
    a[0] = mat[2, 1]
    a[1] = mat[0, 2]
    a[2] = mat[1, 0]
    return a


def traj_coeffs(f0, df0, fT, dfT, T):
    return np.linalg.inv(np.array([[0, 0, 0, 1], [0, 0, 1, 0], [T**3, T**2, T, 1], [3*T**2, 2*T, 1, 0]])) @ \
           np.array([f0, df0, fT, dfT])


def make_traj(coeffs, t):
    return coeffs[0]*t**3 + coeffs[1]*t**2 + coeffs[2]*t + coeffs[3]


if __name__ == "__main__":
    # Main parameters
    m = 1.5  # Quadcopter mass
    mL = 0.1  # Hook mass
    g = 9.81
    L = 1  # Pole length
    J = sp.diag(0.082, 0.085, 0.138)
    e3 = sp.Matrix([0, 0, 1])
    e2 = sp.Matrix([0, 1, 0])

    T = 4  # length of the trajectory in seconds
    # alpha_c = traj_coeffs(0, 0, 0, 0, T)  # load pitch angle
    phi_c = traj_coeffs(0, 0, 0, 0, T)  # drone roll angle
    psi_c = traj_coeffs(0, 0, 0, 0, T)  # drone yaw angle
    x_c = traj_coeffs(-1, 0, 2, 0, T)  # load position
    y_c = traj_coeffs(0, 0, 0, 0, T)  # load position
    z_c = traj_coeffs(0.5, 0, 0.33, 0, T)  # load position

    t = sp.symbols('t')

    # Calculate the flat outputs
    rL = sp.Matrix([make_traj(x_c, t), make_traj(y_c, t), make_traj(z_c, t)])
    # alpha, phi, psi = make_traj(alpha_c, t), make_traj(phi_c, t), make_traj(psi_c, t)
    phi, psi = make_traj(phi_c, t), make_traj(psi_c, t)

    Fc = mL * (- g * e3 - sp.diff(rL, t, 2))
    alpha = sp.atan2(Fc[0]*sp.cos(psi) + Fc[1]*sp.sin(psi), -Fc[2])

    RL = (sp.Matrix([[1, 0, 0], [0, sp.cos(phi), sp.sin(phi)], [0, -sp.sin(phi), sp.cos(phi)]]) *
          sp.Matrix([[sp.cos(alpha), 0, sp.sin(alpha)], [0, 1, 0], [-sp.sin(alpha), 0, sp.cos(alpha)]]) *
          sp.Matrix([[sp.cos(psi), sp.sin(psi), 0], [-sp.sin(psi), sp.cos(psi), 0], [0, 0, 1]])).T
    # Rotation matrix of the pole
    p = - RL[:, 2]
    Re2 = RL[:, 1]

    r = rL - p * L

    Fcp = mL * (- g * e3.T - sp.diff(rL, t, 2).T) @ p
    Fcy = mL * (- g * e3.T - sp.diff(rL, t, 2).T) @ Re2
    # Fc2 = p * Fcp + Re2 * Fcy

    FRe3 = m * (g * e3 + sp.diff(r, t, 2)) - Fc
    F = sp.sqrt(FRe3.T @ FRe3)
    Re3 = FRe3 * F**-1
    Re1 = hat(Re2) @ Re3
    R = sp.zeros(3)
    R[:, 0] = Re1
    R[:, 1] = Re2
    R[:, 2] = Re3
    dR = sp.diff(R, t)
    om = vee(R.T @ dR)
    dom = sp.diff(om, t)
    Mc = L * hat(p) @ Fc
    tau = J @ dom + hat(om) @ J @ om - Mc

    # print([(Mc).subs(t, t0) for t0 in np.linspace(0, 10, 50)])
    F_arr = [F.subs(t, t0) for t0 in np.linspace(0, T, 50*T)]
    tau_arr = [tau.subs(t, t0) for t0 in np.linspace(0, T, 50*T)]
    r_arr = [r.subs(t, t0) for t0 in np.linspace(0, T, 50*T)]
    rL_arr = [rL.subs(t, t0) for t0 in np.linspace(0, T, 50*T)]
    alpha_arr = [tau.subs(t, t0) for t0 in np.linspace(0, T, 100)]

    # plt.plot(np.array(r_arr).astype(np.float64)[:, :, 0])
    # plt.plot(np.array(rL_arr).astype(np.float64)[:, :, 0])
    plt.plot(np.array(alpha_arr).astype(np.float64)[:, :, 0])
    plt.show()
    # print(F_arr)
    # print(tau_arr)
    #
    # if os.path.exists('inputs.pickle'):
    #     os.remove('inputs.pickle')
    # with open('inputs.pickle', 'wb') as file:
    #     pickle.dump([F_arr,
    #                  tau_arr,
    #                  sp.diff(r, t).subs(t, 0),
    #                  R.subs(t, 0),
    #                  om.subs(t, 0),
    #                  alpha.subs(t, 0),
    #                  sp.diff(alpha).subs(t, 0)], file)















#         sp.Matrix([sp.sin(phi) * sp.sin(psi) + sp.cos(phi) * sp.cos(psi) * sp.sin(alpha),
#                    sp.cos(phi) * sp.sin(alpha) * sp.sin(psi) - sp.cos(psi) * sp.sin(phi),
#                    sp.cos(phi) * sp.cos(alpha)])