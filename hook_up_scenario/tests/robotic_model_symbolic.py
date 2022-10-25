import inspect

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import os
import dill


def hat(a):
    mat = sp.Matrix([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return mat


if __name__ == "__main__":
    mb = 1.5
    ml = 0.1
    J = sp.diag(0.082, 0.085, 0.138)
    g = 9.81
    L = 0.4
    t = sp.symbols('t')
    phi, theta, psi, rx, ry, rz, alpha = sp.Function('phi')(t), sp.Function('theta')(t), sp.Function('psi')(t), \
                                         sp.Function('rx')(t), sp.Function('ry')(t), sp.Function('rz')(t), \
                                         sp.Function('alpha')(t)
    dphi, dtheta, dpsi, drx, dry, drz, dalpha = sp.symbols("dphi, dtheta, dpsi, drx, dry, drz, dalpha")
    lam = sp.Matrix([phi, theta, psi])
    r = sp.Matrix([rx, ry, rz])
    q = sp.Matrix([rx, ry, rz, phi, theta, psi, alpha])
    S_phi = sp.sin(phi)
    S_theta = sp.sin(theta)
    S_psi = sp.sin(psi)
    C_phi = sp.cos(phi)
    C_theta = sp.cos(theta)
    C_psi = sp.cos(psi)

    W = sp.Matrix([[1, 0, -S_theta], [0, C_phi, C_theta*S_phi], [0, -S_phi, C_theta*C_phi]])

    R = (sp.Matrix([[1, 0, 0], [0, C_phi, S_phi], [0, -S_phi, C_phi]]) *
         sp.Matrix([[C_theta, 0, -S_theta], [0, 1, 0], [S_theta, 0, C_theta]]) *
         sp.Matrix([[C_psi, S_psi, 0], [-S_psi, C_psi, 0], [0, 0, 1]])).T

    Q = R.T * W

    Jp = sp.Matrix([L*sp.cos(alpha), 0, L*sp.sin(alpha)])
    Reb = sp.Matrix([[sp.cos(alpha), 0, -sp.sin(alpha)], [0, 1, 0], [sp.sin(alpha), 0, sp.cos(alpha)]])  # TODO transpose?
    peb = sp.Matrix([L*sp.sin(alpha), 0, -L*sp.cos(alpha)])

    H = sp.zeros(7)
    H[0:3, 0:3] = (mb + ml) * sp.eye(3)
    H[3:6, 3:6] = Q.T * J * Q + ml * W.T * hat(R * peb).T * hat(R * peb).T * W
    H[6, 6] = ml * L ** 2
    H[0:3, 3:6] = -ml * hat(R * peb) * W
    H[3:6, 0:3] = H[0:3, 3:6].T
    H[0:3, 6] = ml * R * Jp
    H[6, 0:3] = H[0:3, 6].T
    H[3:6, 6] = -ml * W.T * hat(R * peb).T * R * Jp
    H[6, 3:6] = H[3:6, 6].T

    U = mb * g * rz + ml * g * (r + R * peb)[2]

    G = U.diff(q)

    C = sp.zeros(7)
    for i, j in zip(range(7), range(7)):
        C[i, j] = sum([0.5 * (H[i, j].diff(q[k]) + H[i, k].diff(q[j])
                              + H[j, k].diff(q[i])) * q[k].diff(t) for k in range(7)])

    N = sp.Matrix([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1]])
    Rb = sp.eye(7)
    Rb[0:3, 0:3] = R
    Rb[3:6, 3:6] = Q.T
    Xi = Rb * N  # u = Xi * u_quad

    # save to dill file
    # if os.path.exists('hook_model.dill'):
    #     os.remove('hook_model.dill')
    # with open('hook_model.dill', 'wb') as file:
    #     dill.dump([H, C, G, Xi], file)

    # or lambdify and write to Python file
    q_arr = [rx, ry, rz, phi, theta, psi, alpha]
    dq_arr = [q_.diff(t) for q_ in q_arr]
    state_arr = q_arr + dq_arr

    # Write functions to py file
    H_fun = sp.lambdify(state_arr, H, modules="numpy", cse=True)
    H_source = inspect.getsource(H_fun)
    H_args = H_fun.__code__.co_varnames

    replace_vars = {'phi(t)': H_args[3],
                    'theta(t)': H_args[4],
                    'psi(t)': H_args[5],
                    'alpha(t)': H_args[6],
                    '_lambdifygenerated': 'H_fun'}
    # Iterate over all key-value pairs in dictionary
    for key, value in replace_vars.items():
        # Replace key character with value character in string
        H_source = H_source.replace(key, value)

    sub_to = [dphi, dtheta, dpsi, drx, dry, drz, dalpha]

    C = C.subs([(dq_arr_, sub_to_) for dq_arr_, sub_to_ in zip(dq_arr, sub_to)])
    C_fun = sp.lambdify(state_arr, C, modules="numpy", cse=True)
    C_source = inspect.getsource(C_fun)
    C_args = C_fun.__code__.co_varnames
    replace_vars = {'phi(t)': C_args[3],
                    'theta(t)': C_args[4],
                    'psi(t)': C_args[5],
                    'alpha(t)': C_args[6],
                    'drx': C_args[7],
                    'dry': C_args[8],
                    'drz': C_args[9],
                    'dphi': C_args[10],
                    'dtheta': C_args[11],
                    'dpsi': C_args[12],
                    'dalpha': C_args[13],
                    '_lambdifygenerated': 'C_fun'}
    # Iterate over all key-value pairs in dictionary
    for key, value in replace_vars.items():
        # Replace key character with value character in string
        C_source = C_source.replace(key, value)

    G_fun = sp.lambdify(state_arr, G, modules="numpy", cse=True)
    G_source = inspect.getsource(G_fun)
    G_args = G_fun.__code__.co_varnames
    replace_vars = {'phi(t)': G_args[3],
                    'theta(t)': G_args[4],
                    'psi(t)': G_args[5],
                    'alpha(t)': G_args[6],
                    '_lambdifygenerated': 'G_fun'}
    # Iterate over all key-value pairs in dictionary
    for key, value in replace_vars.items():
        # Replace key character with value character in string
        G_source = G_source.replace(key, value)

    Xi_fun = sp.lambdify(state_arr, Xi, modules="numpy", cse=True)
    Xi_source = inspect.getsource(Xi_fun)
    Xi_args = Xi_fun.__code__.co_varnames
    replace_vars = {'phi(t)': Xi_args[3],
                    'theta(t)': Xi_args[4],
                    'psi(t)': Xi_args[5],
                    'alpha(t)': Xi_args[6],
                    '_lambdifygenerated': 'Xi_fun'}
    # Iterate over all key-value pairs in dictionary
    for key, value in replace_vars.items():
        # Replace key character with value character in string
        Xi_source = Xi_source.replace(key, value)

    full_source = '''from numpy import *
    
    
''' + H_source + '\n' + C_source + '\n' + G_source + '\n' + Xi_source

    if os.path.exists('../hook_model.py'):
        os.remove('../hook_model.py')
    with open('../hook_model.py', 'wt') as file:
        file.write(full_source)
