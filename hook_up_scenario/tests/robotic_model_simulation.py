import dill
import pickle
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from hook_model import *

if __name__ == "__main__":
    # t = sp.symbols('t')
    # phi, theta, psi, rx, ry, rz, alpha = sp.Function('phi')(t), sp.Function('theta')(t), sp.Function('psi')(t), \
    #                                      sp.Function('rx')(t), sp.Function('ry')(t), sp.Function('rz')(t), \
    #                                      sp.Function('alpha')(t)

    # initial conditions
    with open('../pickle/inputs.pickle', 'rb') as file:
        datas = pickle.load(file)
        F = np.squeeze(datas[0])
        tau = datas[1]
        r0 = datas[2]
        v0 = datas[3]
        R0 = datas[4]
        om0 = datas[5]
        alpha0 = datas[6]
        dalpha0 = datas[7]

    lam0 = Rotation.from_matrix(R0).as_euler('xyz')

    q0 = np.hstack((r0, lam0, alpha0))
    dq0 = np.hstack((v0, om0, dalpha0))
    # q_arr = [rx, ry, rz, phi, theta, psi, alpha]
    # dq_arr = [q_.diff(t) for q_ in q_arr]
    # state_arr = q_arr + dq_arr
    state_num = np.hstack((q0, dq0))
    state_der = np.zeros(14)

    # load symbolic model TODO: actually we do not need this anymore
    # with open('hook_model.dill', 'rb') as file:
    #     datas = dill.load(file)
    #     H = sp.lambdify(state_arr, datas[0])
    #     C = sp.lambdify(state_arr, datas[1])
    #     G = sp.lambdify(state_arr, datas[2])
    #     Xi = sp.lambdify(state_arr, datas[3])

    # simulation parameters
    episode_length = 4
    simulation_step = 0.001
    control_step = 0.02
    num_steps = int(episode_length/simulation_step)
    cur_step = 0

    q_log = np.zeros((7, num_steps))
    q_log[:, 0] = q0

    for i in range(int(episode_length/control_step)):
        f = np.hstack((F[i], tau[i], 0))
        for j in range(int(control_step/simulation_step)):
            fun_arg = state_num.astype('float64')
            H_num = H_fun(*fun_arg)
            C_num = C_fun(*fun_arg)
            G_num = G_fun(*fun_arg)
            Xi_num = Xi_fun(*fun_arg)
            u = Xi_num @ f

            q_der = state_num[7:]
            q_derder = np.linalg.inv(H_num) @ (u - np.squeeze(G_num) - C_num @ state_num[7:])

            state_der = np.hstack((q_der, q_derder))

            state_num = state_num + simulation_step * state_der

            cur_step = int(i * control_step / simulation_step + j)
            q_log[:, cur_step] = state_num[0:7]
        print('Simulation progress: ' + str(int(100*cur_step/num_steps)) + ' %')

    plt.plot(q_log[0, :])
    plt.show()
