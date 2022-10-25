import numpy as np
import sympy as sp


def hat(a):
    mat = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return mat


# def pinv(A):
#     A.T @ (A @ A.T) ** -1


# def payload_dyn(r, lam, rL, dr, dlam, drL):
if __name__ == "__main__":
    t, m, mL, Jx, Jy, Jz, g, F, taux, tauy, tauz = \
        sp.symbols('t m mL Jx Jy Jz g F taux tauy tauz')
    omx, omy, omz, rx, ry, rz, rLx, rLy, rLz = sp.Function('omx')(t), sp.Function('omy')(t), sp.Function('omz')(t),\
                                               sp.Function('rx')(t), sp.Function('ry')(t), sp.Function('rz')(t), \
                                               sp.Function('rLx')(t), sp.Function('rLy')(t), sp.Function('rLz')(t)
    R = sp.MatrixSymbol('R', 3, 3)
    e3 = sp.Matrix([0, 0, 1])
    e2 = sp.Matrix([0, 1, 0])
    M = sp.diag(m, m, m, Jx, Jy, Jz, mL, mL, mL)
    r = sp.Matrix([rx, ry, rz])
    rL = sp.Matrix([rLx, rLy, rLz])
    om = sp.Matrix([omx, omy, omz])
    tau = sp.Matrix([taux, tauy, tauz])
    J = sp.diag(Jx, Jy, Jz)
    ddx = sp.Matrix([sp.diff(sp.diff(r, t), t), sp.diff(om, t), sp.diff(sp.diff(rL, t), t)])
    f = sp.Matrix([m*g*e3-F*sp.Matrix(R)*e3, tau - om.cross(J @ om), mL*g*e3])
    p = rL - r

    A = sp.zeros(2, 9)
    A[0, 0:3] = -p.T
    A[0, 6:9] = p.T
    A[1, 0:3] = -sp.transpose(sp.Matrix(R) @ e2)
    A[1, 3:6] = -p.T @ sp.Matrix(R) @ hat(e2)
    A[1, 6:9] = sp.transpose(sp.Matrix(R) @ e2)

    b = sp.zeros(2, 1)
    b[0] = -sp.diff(p, t).T @ sp.diff(p, t)
    b[1] = 2*sp.diff(p, t).T @ sp.Matrix(R) @ hat(e2) @ om - p.T @ sp.Matrix(R) @ hat(om) @ hat(om) @ e2

    mat = A @ sp.sqrt(M).inv()
    # pinv = sp.simplify(mat.T @ (sp.simplify)**-1)
    Fc = sp.sqrt(M) @ mat.T @ (mat @ mat.T).LUsolve(b - A @ M.inv() @ f)
    print(Fc[2])


# payload_dyn(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))


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