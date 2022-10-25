import numpy as np
from logger import Logger


def hat(a):
    mat = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return mat


def payload_dyn(r, R, rL, dr, om, drL, F, tau):
    m = 1.5  # drone mass
    mL = 0.1  # hook mass
    g = 9.81
    J = np.diag([0.082, 0.085, 0.138])
    M = np.diag([m, m, m, J[0, 0], J[1, 1], J[2, 2], mL, mL, mL])

    e3 = np.array([0, 0, 1])
    e2 = np.array([0, 1, 0])

    f = np.hstack([- m*g*e3 + F*R @ e3, tau - np.cross(om, J @ om), - mL*g*e3])
    p = rL - r
    dp = drL - dr

    A = np.zeros((2, 9))
    A[0, 0:3] = -p
    A[0, 6:9] = p
    A[1, 0:3] = -R @ e2
    A[1, 3:6] = -p @ R @ hat(e2)
    A[1, 6:9] = R @ e2

    b = np.zeros(2)
    b[0] = -dp @ dp
    b[1] = 2*dp @ R @ hat(e2) @ om - p @ R @ hat(om) @ hat(om) @ e2

    mat = A @ np.linalg.inv(np.sqrt(M))
    fc = np.sqrt(M) @ np.linalg.pinv(mat) @ (b - A @ np.linalg.inv(M) @ f)
    return np.linalg.inv(M) @ (f + fc)

if __name__ == "__main__":

    r = np.array([0, 0, 1])
    dr = np.array([1, 1, 0])
    R = np.eye(3)
    om = np.zeros(3)
    rL = np.array([0, 0, -0.0])
    drL = np.zeros(3)

    simtime = 0.0
    simulation_step = 0.001
    episode_length = 3
    logger = Logger(episode_length, simulation_step)

    for i in range(int(episode_length / simulation_step)):
        t = i*simulation_step
        # ctrl = np.array([16, 0.05*np.sin(5*t), 0.05*np.cos(5*t), 0])
        ctrl = np.array([16, 0, 0, 0])
        # dynamics
        accels = payload_dyn(r, R, rL, dr, om, drL, ctrl[0], ctrl[1:4])
        # Euler
        dr = dr + simulation_step*accels[0:3]
        om = om + simulation_step*accels[3:6]
        drL = drL + simulation_step*accels[6:9]
        r = r + simulation_step*dr
        R = R + simulation_step*R*hat(om)
        rL = rL + simulation_step*drL
        state = np.hstack([rL, np.zeros(10)])
        logger.log(timestamp=t, state=state, control=ctrl)

        print(np.linalg.norm(r-rL))

    logger.plot()

    # r, lam, rL, dr, dlam, drL = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    # m = 1.5  # drone mass
    # mL = 0.1  # hook mass
    # J = np.diag([0.082, 0.085, 0.138])
    # M = np.diag([m, m, m, 1, 1, 1, mL, mL, mL])
    # x = np.hstack((r, lam, rL))
    # dx = np.hstack((dr, dlam, drL))
    #
    # t = sp.symbols('t')
    # phi, theta, psi = sp.Function('phi')(t), sp.Function('theta')(t), sp.Function('psi')(t)
    # S_phi = sp.sin(phi)
    # S_theta = sp.sin(theta)
    # S_psi = sp.sin(psi)
    # C_phi = sp.cos(phi)
    # C_theta = sp.cos(theta)
    # C_psi = sp.cos(psi)
    #
    # W = sp.Matrix([[1, 0, -S_theta], [0, C_phi, C_theta*S_phi], [0, -S_phi, C_theta*C_phi]])
    # dW = sp.diff(W, t)
    # ddW = sp.diff(dW, t)
    # R = (sp.Matrix([[1, 0, 0], [0, C_phi, S_phi], [0, -S_phi, C_phi]]) *
    #      sp.Matrix([[C_theta, 0, -S_theta], [0, 1, 0], [S_theta, 0, C_theta]]) *
    #      sp.Matrix([[C_psi, S_psi, 0], [-S_psi, C_psi, 0], [0, 0, 1]])).T
    # dR = sp.diff(R, t)
    # ddR = sp.diff(dR, t)