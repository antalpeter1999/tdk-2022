import numpy as np
import cvxopt as opt
import matplotlib.pyplot as plt
import sympy as sp


def traj_opt_1d():
    '''
    Trajectory optimization method based on Mellinger (2011), by minimizing the snap in a constrained quadratic
    optimization.
    '''
    T = 2
    x0 = 2
    dx0 = 1
    ddx0 = 0
    xf = 1
    dxf = 1
    ddxf = 0

    P = np.zeros((6, 6))
    P[-2:, -2:] = np.array([[24 ** 2 * T, 120 * 24 / 2 * T ** 2],
                            [120 * 24 / 2 * T ** 2, 120 ** 2 / 3 * T ** 3]])
    q = np.zeros(6)
    A = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 2, 0, 0, 0],
                  [1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                  [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                  [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]])
    b = np.array([[x0, dx0, ddx0, xf, dxf, ddxf]]).T

    P = opt.matrix(P)
    q = opt.matrix(q)
    A = opt.matrix(A, tc='d')
    b = opt.matrix(b, tc='d')

    sol = opt.solvers.qp(P, q, A=A, b=b)['x']

    t = np.linspace(0, T, 100)
    x = np.polynomial.Polynomial(np.squeeze(np.array(sol)))
    dx = x.deriv()
    ddx = x.deriv(2)
    plt.plot(x(t))
    plt.plot(dx(t))
    plt.plot(ddx(t))
    plt.show()


def traj_opt(init_pos, final_pos):
    # Compute optimal trajectories for the flat outputs to hook up an object in two sections
    # First section ##################################################################################
    T1 = 5
    x0 = init_pos[0]
    dx0 = 0
    ddx0 = 0
    x1 = -0.3
    dx1 = 0.2
    ddx1 = 0

    y0 = init_pos[1]
    dy0 = 0
    ddy0 = 0
    y1 = 0
    dy1 = 0
    ddy1 = 0

    z0 = init_pos[2]
    dz0 = 0
    ddz0 = 0
    z1 = 0.8
    dz1 = 0
    ddz1 = 0

    # Second section ########################################################################################
    L = 0.4  # Pole length
    T2 = 1
    T3 = 2
    T12 = T1 + T2
    T23 = T2 + T3
    T123 = T1 + T2 + T3
    x2 = 0
    dx2 = 0.2
    ddx2 = 0.2
    x3 = final_pos[0]
    dx3 = 0
    ddx3 = 0
    z2 = z1 - L
    dz2 = 0
    ddz2 = 0
    z3 = final_pos[2]
    dz3 = 0
    ddz3 = 0

    P = np.zeros((42, 42))
    P[4:6, 4:6] = np.array([[24 ** 2 * T1, 120 * 24 / 2 * T1 ** 2],
                            [120 * 24 / 2 * T1 ** 2, 120 ** 2 / 3 * T1 ** 3]])
    P[10:12, 10:12] = P[4:6, 4:6]
    P[16:18, 16:18] = P[4:6, 4:6]
    P[22:24, 22:24] = np.array([[24 ** 2 * (T12 - T1), 120 * 24 / 2 * (T12 ** 2 - T2 ** 2)],
                                [120 * 24 / 2 * (T12 ** 2 - T2 ** 2), 120 ** 2 / 3 * (T12 ** 3 - T2 ** 3)]])
    P[28:30, 28:30] = P[22:24, 22:24]
    P[34:36, 34:36] = np.array([[24 ** 2 * (T123 - T12), 120 * 24 / 2 * (T123 ** 2 - T12 ** 2)],
                                [120 * 24 / 2 * (T123 ** 2 - T12 ** 2), 120 ** 2 / 3 * (T123 ** 3 - T12 ** 3)]])
    P[40:42, 40:42] = P[34:36, 34:36]
    q = np.zeros(42)

    # Add constraints
    # First section
    A, b = None, None
    A, b = add_constraint(A, b, 0, x0, 0)
    A, b = add_constraint(A, b, 1, dx0, 0)
    A, b = add_constraint(A, b, 2, ddx0, 0)
    A, b = add_constraint(A, b, 0, x1, T1)
    A, b = add_constraint(A, b, 1, dx1, T1)
    # A, b = add_constraint(A, b, 2, ddx1, T1)
    A, b = add_constraint(A, b, 3, y0, 0)
    A, b = add_constraint(A, b, 4, dy0, 0)
    A, b = add_constraint(A, b, 5, ddy0, 0)
    A, b = add_constraint(A, b, 3, y1, T1)
    A, b = add_constraint(A, b, 4, dy1, T1)
    A, b = add_constraint(A, b, 5, ddy1, T1)
    A, b = add_constraint(A, b, 6, z0, 0)
    A, b = add_constraint(A, b, 7, dz0, 0)
    A, b = add_constraint(A, b, 8, ddz0, 0)
    # A, b = add_constraint(A, b, 6, z1, T1)
    # A, b = add_constraint(A, b, 7, dz1, T1)
    # A, b = add_constraint(A, b, 8, ddz1, T1)

    # Second section
    A, b = add_constraint(A, b, [0, 9], 0, T1)
    A, b = add_constraint(A, b, [1, 10], 0, T1)
    A, b = add_constraint(A, b, [2, 11], 0, T1)
    A, b = add_constraint(A, b, [6, 12], L, T1)
    A, b = add_constraint(A, b, [7, 13], 0, T1)
    A, b = add_constraint(A, b, [8, 14], 0, T1)
    A, b = add_constraint(A, b, 9, x2, T1 + T2)
    # A, b = add_constraint(A, b, 10, dx2, T1 + T2)
    # A, b = add_constraint(A, b, 11, ddx2, T1 + T2)
    A, b = add_constraint(A, b, 12, z2, T1 + T2)
    A, b = add_constraint(A, b, 13, dz2, T1 + T2)
    A, b = add_constraint(A, b, 14, ddz2, T1 + T2)

    # Third section
    A, b = add_constraint(A, b, [9, 15], 0, T1 + T2)
    A, b = add_constraint(A, b, [10, 16], 0, T1 + T2)
    A, b = add_constraint(A, b, [11, 17], 0, T1 + T2)
    A, b = add_constraint(A, b, [12, 18], 0, T1 + T2)
    A, b = add_constraint(A, b, [13, 19], 0, T1 + T2)
    A, b = add_constraint(A, b, [14, 20], 0, T1 + T2)
    A, b = add_constraint(A, b, 15, x3, T1 + T2 + T3)
    A, b = add_constraint(A, b, 16, dx3, T1 + T2 + T3)
    A, b = add_constraint(A, b, 17, ddx3, T1 + T2 + T3)
    A, b = add_constraint(A, b, 18, z3, T1 + T2 + T3)
    A, b = add_constraint(A, b, 19, dz3, T1 + T2 + T3)
    A, b = add_constraint(A, b, 20, ddz3, T1 + T2 + T3)

    P = opt.matrix(P)
    q = opt.matrix(q)
    A = opt.matrix(A, tc='d')
    b = opt.matrix(b, tc='d')

    traj_sol = opt.solvers.qp(P, q, A=A, b=b, kktsolver='ldl')
    print(traj_sol)
    traj_coeffs = traj_sol['x']
    traj1_coeffs = traj_coeffs[0:18]
    traj2_coeffs = traj_coeffs[18:30]
    traj3_coeffs = traj_coeffs[30:42]

    return traj1_coeffs, traj2_coeffs, traj3_coeffs, T1, T2, T3


def add_constraint(A, b, var_num, rhs, T):
    A_ = np.zeros(42)
    if not isinstance(var_num, list):
        diff_num = var_num % 3
        idx = int((var_num - diff_num) / 3)
        if diff_num == 0:
            A_[6 * idx:6 * (idx + 1)] = np.array([1, T, T ** 2, T ** 3, T ** 4, T ** 5])
        elif diff_num == 1:
            A_[6 * idx:6 * (idx + 1)] = np.array([0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4])
        elif diff_num == 2:
            A_[6 * idx:6 * (idx + 1)] = np.array([0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3])
        else:
            raise NotImplementedError
    else:
        diff_num = var_num[0] % 3
        idx = [int((var_num_ - diff_num) / 3) for var_num_ in var_num]
        if diff_num == 0:
            A_[6 * idx[0]:6 * (idx[0] + 1)] = np.array([1, T, T ** 2, T ** 3, T ** 4, T ** 5])
            A_[6 * idx[1]:6 * (idx[1] + 1)] = -1 * np.array([1, T, T ** 2, T ** 3, T ** 4, T ** 5])
        elif diff_num == 1:
            A_[6 * idx[0]:6 * (idx[0] + 1)] = np.array([0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4])
            A_[6 * idx[1]:6 * (idx[1] + 1)] = -1 * np.array([0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4])
        elif diff_num == 2:
            A_[6 * idx[0]:6 * (idx[0] + 1)] = np.array([0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3])
            A_[6 * idx[1]:6 * (idx[1] + 1)] = -1 * np.array([0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3])
        else:
            raise NotImplementedError
    b_ = rhs
    if A is None:
        A = A_
        b = b_
    else:
        A = np.vstack((A, A_))
        b = np.hstack((b, b_))
    return A, b


def flatness_quadcopter(x_coef, y_coef, z_coef, psi_coef):
    # Compute all state and input variables of the quadrotor from the flat outputs given by the polynomial coefficients

    # Main parameters
    m = 1.5  # Quadcopter mass
    g = 9.81
    J = sp.diag(0.082, 0.085, 0.138)
    zW = sp.Matrix([0, 0, 1])
    yW = sp.Matrix([0, 1, 0])

    t = sp.symbols('t')
    x, y, z, psi = make_traj(x_coef, t), make_traj(y_coef, t), make_traj(z_coef, t), make_traj(psi_coef, t)

    temp1 = sp.Matrix([sp.diff(x, t, 2), sp.diff(y, t, 2), sp.diff(z, t, 2) + g])
    F_m = sp.sqrt(temp1[0] ** 2 + temp1[1] ** 2 + temp1[2] ** 2)
    zB = temp1 / F_m
    F = F_m * m
    xC = sp.Matrix([sp.cos(psi), sp.sin(psi), 0])
    temp2 = hat(zB) @ xC
    yB = temp2 / sp.sqrt(temp2[0] ** 2 + temp2[1] ** 2 + temp2[2] ** 2)
    xB = hat(yB) @ zB

    R = sp.zeros(3)
    R[:, 0] = xB
    R[:, 1] = yB
    R[:, 2] = zB
    da = sp.Matrix([sp.diff(x, t, 3), sp.diff(y, t, 3), sp.diff(z, t, 3)])
    h = m*(da - zB*(zB.T @ da))*F**-1
    p = -h.T @ yB
    q = h.T @ xB
    r = sp.diff(psi, t) * zB[2]
    om = sp.Matrix([p, q, r])
    dda = sp.Matrix([sp.diff(x, t, 4), sp.diff(y, t, 4), sp.diff(z, t, 4)])
    h_acc = m * dda*F**-1 - sp.diff(F, t, 2) * zB*F**-1 -  2 * sp.diff(F, t) * hat(om) * zB * F**-1 - \
            hat(om) * hat(om) * zB
    alpha1 = -h_acc.T @ yB
    alpha2 = h_acc.T @ xB
    alpha3 = sp.diff(psi, t, 2) * zB[2]
    alpha = sp.Matrix([alpha1, alpha2, alpha3])
    tau = J @ alpha + hat(om) @ J @ om
    # print(F)
    # print(R)
    # tau_lam = sp.lambdify(t, tau, modules="numpy", cse=True)
    # import inspect
    # print(inspect.getsource(tau_lam))
    return sp.Matrix([x, y, z]), sp.diff(sp.Matrix([x, y, z]), t), R, om


def flatness_quadcopter_load_2d(x_coef, z_coef):
    # Compute all state and input variables of the quadrotor + payload connected by a rigid pole
    # from the flat outputs given by polynomial coefficients

    # Main parameters
    m = 1.5  # Quadcopter mass
    mL = 0.02  # Hook mass
    g = 9.81
    L = 0.4  # Pole length
    J = 0.085
    e3 = sp.Matrix([0, 1])

    t = sp.symbols('t')
    x, z = make_traj(x_coef, t), make_traj(z_coef, t)

    rL = sp.Matrix([x, z])

    Fc = mL * (- g * e3 - sp.diff(rL, t, 2))
    p = Fc * sp.sqrt(Fc[0] ** 2 + Fc[1] ** 2) ** -1
    alpha = sp.atan2(Fc[0], -Fc[1])

    r = rL - p * L

    FRe3 = m * (g * e3 + sp.diff(r, t, 2)) - Fc
    F = sp.sqrt(FRe3.T @ FRe3)
    Re3 = FRe3 * F ** -1
    phi = sp.atan2(Re3[0], Re3[1])
    om = sp.diff(phi, t)
    dom = sp.diff(om, t)
    tau = J * dom

    # # print([(Mc).subs(t, t0) for t0 in np.linspace(0, 10, 50)])
    # F_arr = [F.subs(t, t0) for t0 in np.linspace(0, T, T * 50)]
    # tau_arr = [np.array([0, tau.subs(t, t0), 0]) for t0 in np.linspace(0, T, T * 50)]
    # # r_arr = [r.subs(t, t0) for t0 in np.linspace(0, T, 100)]
    # # rL_arr = [rL.subs(t, t0) for t0 in np.linspace(0, T, 100)]
    # p_arr = [p.subs(t, t0) for t0 in np.linspace(0, T, 100)]
    # phi_arr = [alpha.subs(t, t0) for t0 in np.linspace(0, T, 100)]
    R = sp.Matrix([[sp.cos(phi), 0, sp.sin(phi)], [0, 1, 0], [-sp.sin(phi), 0, sp.cos(phi)]])
    return sp.Matrix([r[0], 0, r[1]]), sp.diff(sp.Matrix([r[0], 0, r[1]]), t), R, sp.Matrix([0, om, 0])


def make_traj(coef, t):
    return coef[5]*t**5 + coef[4]*t**4 + coef[3]*t**3 + coef[2]*t**2 + coef[1]*t + coef[0]


def hat(a):
    mat = sp.Matrix([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return mat


def get_optimal_traj(init_pos, final_pos):
    # init pos: [-1, 1, 1.2], final pos: [0.6, 0, 1.25]
    traj1_coeffs, traj2_coeffs, traj3_coeffs, T1, T2, T3 = traj_opt(init_pos, final_pos)
    r1, v1, R1, om1 = flatness_quadcopter(traj1_coeffs[0:6], traj1_coeffs[6:12], traj1_coeffs[12:18], np.zeros(6))
    r2, v2, R2, om2 = flatness_quadcopter_load_2d(traj2_coeffs[0:6], traj2_coeffs[6:12])
    r3, v3, R3, om3 = flatness_quadcopter_load_2d(traj3_coeffs[0:6], traj3_coeffs[6:12])
    return {'r1': r1, 'v1': v1, 'R1': R1, 'om1': om1, 'T1': T1, 'r2': r2, 'v2': v2, 'R2': R2, 'om2': om2, 'T2': T2,
            'r3': r3, 'v3': v3, 'R3': R3, 'om3': om3, 'T3': T3}


if __name__ == "__main__":
    init_pos = np.array([-0.5, 1, 1.2])  # np.array([-1, 1, 1.2])
    final_pos = np.array([0.6, 0, 1.25])  # np.array([0.6, 0, 1.25])
    optimal_trajs = get_optimal_traj(init_pos, final_pos)
    r1 = optimal_trajs['r1']
    r2 = optimal_trajs['r2']
    r3 = optimal_trajs['r3']
    v1 = optimal_trajs['v1']
    v2 = optimal_trajs['v2']
    v3 = optimal_trajs['v3']
    t = sp.symbols('t')
    T1 = optimal_trajs['T1']
    T2 = optimal_trajs['T2']
    T3 = optimal_trajs['T3']
    r1_sample = [r1.subs(t, t0) for t0 in np.linspace(0, T1, 100)]
    r2_sample = [r2.subs(t, t0) for t0 in np.linspace(T1, T1 + T2, 100)]
    r3_sample = [r3.subs(t, t0) for t0 in np.linspace(T1 + T2, T1 + T2 + T3, 100)]
    v1_sample = [v1.subs(t, t0) for t0 in np.linspace(0, T1, 100)]
    v2_sample = [v2.subs(t, t0) for t0 in np.linspace(T1, T1 + T2, 100)]
    v3_sample = [v3.subs(t, t0) for t0 in np.linspace(T1 + T2, T1 + T2 + T3, 100)]
    plt.figure()
    ax = plt.axes(projection="3d")
    x1 = [r_[0] for r_ in r1_sample]
    y1 = [r_[1] for r_ in r1_sample]
    z1 = [r_[2] for r_ in r1_sample]
    x2 = [r_[0] for r_ in r2_sample]
    y2 = [r_[1] for r_ in r2_sample]
    z2 = [r_[2] for r_ in r2_sample]
    x3 = [r_[0] for r_ in r3_sample]
    y3 = [r_[1] for r_ in r3_sample]
    z3 = [r_[2] for r_ in r3_sample]
    dx1 = [r_[0] for r_ in v1_sample]
    dy1 = [r_[1] for r_ in v1_sample]
    dz1 = [r_[2] for r_ in v1_sample]
    dx2 = [r_[0] for r_ in v2_sample]
    dy2 = [r_[1] for r_ in v2_sample]
    dz2 = [r_[2] for r_ in v2_sample]
    dx3 = [r_[0] for r_ in v3_sample]
    dy3 = [r_[1] for r_ in v3_sample]
    dz3 = [r_[2] for r_ in v3_sample]

    ax.plot3D(x1, y1, z1)
    ax.plot3D(x2, y2, z2)
    ax.plot3D(x3, y3, z3)
    ax.scatter(0, 0, 0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    # plt.figure()
    # plt.plot(np.linspace(0, T1, 100), dx1)
    # plt.plot(np.linspace(T1, T1 + T2, 100), dx2)
    # plt.plot(np.linspace(T1 + T2, T1+T2+T3, 100), dx3)
    # plt.plot(np.linspace(0, T1, 100), dz1)
    # plt.plot(np.linspace(T1, T1+T2, 100), dz2)
    # plt.plot(np.linspace(T1+T2, T1+T2+T3, 100), dz3)
    # plt.show()
