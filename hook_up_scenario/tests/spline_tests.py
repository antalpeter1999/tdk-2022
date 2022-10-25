import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import cvxopt as opt


def derivative_transformation(t, k, n):
    # Computes a transformation matrix T that can be used to express the
    # derivative of a B-spline by using the same coefficients that the spline has.
    def zerocheck(a):
        return int(a == 0) + a * int(a != 0)

    T = np.zeros((n + 1, n))
    for i in range(n):
        T[i, i] = k / zerocheck(t[i+k]-t[i])
        T[i+1, i] = -k / zerocheck(t[i+k+1]-t[i+1])
    return T


if __name__ == "__main__":
    # Simple spline optimization program
    # Let's have n = 6 coefficients to optimize, uniform knots, and degree k = 3
    n = 25
    k = 7

    x0 = 1
    dx0 = 0
    x1 = 0
    dx1 = 0
    t = np.zeros(n + k + 1)
    t[0:k] = 0 * np.ones(k)
    t[-k:] = 10 * np.ones(k)
    t[k:-k] = np.linspace(0, 10, n + k + 1 - 2 * k)

    c = np.vstack((np.linspace(0, 10, n), np.linspace(-1, -10, n), np.linspace(-5, 5, n)))
    x = np.linspace(0, 10, 20)
    spl = (t, c, k)
    y = si.splev(x, spl)
    print(y)


'''
    # Cost function: minimize snap and the coefficients
    gamma = 0.9
    P1 = np.diag([gamma ** (n-i) for i in range(n)])
    # n_ = 5
    # P1 = np.diag(np.hstack((np.zeros(n_), np.ones(n-n_))))
    T76 = derivative_transformation(t, 7, n)
    T65 = derivative_transformation(t, 6, n+1)
    T54 = derivative_transformation(t, 5, n+2)
    T43 = derivative_transformation(t, 4, n+3)
    P2_eval = np.linspace(0, 10, 100)
    P2_chol = si.BSpline.design_matrix(P2_eval, t, k-4).toarray() @ T43 @ T54 @ T65 @ T76
    P2 = 1e-6 * P2_chol.T @ P2_chol
    P = P1 + P2
    q = 0 * np.array([gamma ** (n-i) for i in range(n)])

    # Equality constraints: initial condition
    A = np.vstack((si.BSpline.design_matrix(0, t, k).toarray(),
                   si.BSpline.design_matrix(0.0001, t, k-1).toarray() @ T76,
                   # si.BSpline.design_matrix(1-0.0001, t, k-1).toarray() @ T76,
                   # si.BSpline.design_matrix(1, t, k).toarray(),
                   si.BSpline.design_matrix(0.0001, t, k-2).toarray() @ T65 @ T76))
    b = np.array([[x0, dx0, 0]]).T

    # Inequality constraints: minimal and maximal velocity
    G_eval = np.linspace(0, 1, n)
    G = np.vstack((si.BSpline.design_matrix(G_eval, t, k-1).toarray() @ T76,
                   -1 * si.BSpline.design_matrix(G_eval, t, k-1).toarray() @ T76,
                   si.BSpline.design_matrix(G_eval, t, k-2).toarray() @ T65 @ T76,
                   -1 * si.BSpline.design_matrix(G_eval, t, k-2).toarray() @ T65 @ T76))
    v_max = 1.5
    a_max = 2.5
    h = np.vstack((np.expand_dims(v_max * np.ones(2 * G_eval.size), 1),
                   np.expand_dims(a_max * np.ones(2 * G_eval.size), 1)))

    P = opt.matrix(P)
    q = opt.matrix(q)
    G = opt.matrix(G, tc='d')
    h = opt.matrix(h, tc='d')
    A = opt.matrix(A, tc='d')
    b = opt.matrix(b, tc='d')

    sol = opt.solvers.qp(P, q, G=G, h=h, A=A, b=b, kktsolver='ldl')
    print(sol)
    c = np.squeeze(np.array(sol['x']))
    print(c)
    spl = si.BSpline(t, c, k)
    pp = si.PPoly.from_spline(spl)
    pp_root = si.PPoly(pp.c, pp.x)
    pp_root.c[-1, :] = pp_root.c[-1, :] - 1e-3
    print(pp_root.roots())
    x = np.linspace(0, 10, 1000)
    plt.figure()
    plt.plot(x, pp(x))
    plt.show(block=False)
    plt.figure()
    plt.plot(x, pp.derivative(2)(x))
    plt.show()
'''