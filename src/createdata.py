import numpy as np


def Cond_matrix(Z):
    if type(Z) != dict:
        Z = {'Z': Z}

    N = Z['Z'].shape[1]
    Y = np.zeros(Z['Z'].shape, dtype=np.complex)
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                if Z['Z'][i, j] != 0:
                    Y[i, j] = -1 / Z['Z'][i, j]
                if 'T' in Z and Z['T'] is not None and (Z['T'][j, i] != 0):
                    a = max(Z['T'][i, j], Z['T'][j, i])
                    Y[i, j] = Y[i, j] / a
    for i in range(0, N):
        for j in range(0, N):
            if i != j:
                Y[i, i] = Y[i, i] - Y[i, j]
                if 'bg' in Z and Z['bg'] is not None and (Z['bg'][i, j] != 0):
                    Y[i, i] = Y[i, i] + Z['bg'][i, j] / 2
                    if 'T' in Z and Z['T'] is not None and Z['T'][i, j] != 0:
                        Y[i, i] = Y[i, i] + Y[i, j] - Y[i, j] / Z['T'][i, j]
        # if Z.get('vzd'):
        #     Y[i, i] = Y[i, i] + Z['vzd'][i]
    # print(Y)
    return Y


def generate_cubic_system(S, Z, num_node, ballance_node, U_0):
    c_matr = Cond_matrix(Z)
    G = c_matr.real
    B = c_matr.imag
    G_sys = G
    B_sys = B
    g_0 = (G[:, ballance_node] + B[:, ballance_node] * 1j).real * U_0
    g_0 = np.delete(g_0, ballance_node)
    b_0 = (G[:, ballance_node] + B[:, ballance_node] * 1j).imag * U_0
    b_0 = np.delete(b_0, ballance_node)
    G_sys = np.delete(G_sys, ballance_node, axis=1)
    G_sys = np.delete(G_sys, ballance_node, axis=0)
    B_sys = np.delete(B_sys, ballance_node, axis=1)
    B_sys = np.delete(B_sys, ballance_node, axis=0)
    Y_sys = np.bmat([[G_sys, -B_sys], [B_sys, G_sys]]).A

    def u(x): return x[:x.size // 2]

    def v(x): return x[x.size // 2:]

    def D(x): return np.bmat([[np.diag(u(x)), np.diag(v(x))], [np.diag(v(x)), np.diag(-u(x))]]).A

    def F(x): return np.concatenate((S.real, S.imag)) + D(x)@(Y_sys @ x + np.concatenate((g_0, b_0)))

    def D_F(x): return np.bmat([[np.diag(F(x)[:x.size // 2]), -np.diag(F(x)[x.size // 2:])],
                                [np.diag(F(x)[x.size // 2:]), np.diag(F(x)[:x.size // 2])]
                                ]).A

    def J(x): return np.bmat([[np.diag((G_sys@u(x) - B_sys@v(x))), np.diag((B_sys.dot(u(x)) + G_sys.dot(v(x))))],
                              [-np.diag((B_sys@u(x)) + G_sys@v(x)), np.diag((G_sys@u(x)) - B_sys@v(x))]
                              ]).A + np.bmat([[np.diag(g_0), np.diag(b_0)], [-np.diag(b_0), np.diag(g_0)]]).A + D(x) @ Y_sys

    def cs_g(x): return F(x) @ J(x)

    def cs_H(x): return J(x).T @ J(x) + D_F(x) @ Y_sys + Y_sys.T @ D_F(x).T

    def F_min(x): return 0.5 * np.linalg.norm(F(x))**2

    # test = np.array([100, 100, 1, 1])
    # print("g: ", cs_g(test))
    # print("H: ", cs_H(test))

    return F_min, cs_g, cs_H


def generate_polar_system(F_min, cs_g, cs_H):

    def v(x): return x[:x.size // 2]

    def th(x): return x[x.size // 2:]

    def convert_p_to_d(x): return np.append(v(x) * np.cos(th(x)), v(x) * np.sin(th(x)))

    def T(x): return np.bmat([[np.diag(np.cos(th(x))), np.diag(-v(x) * np.sin(th(x)))],
                              [np.diag(np.sin(th(x))), np.diag(v(x) * np.cos(th(x)))]]).A

    def polar_F_min(x): return F_min(convert_p_to_d(x))

    def polar_g(x): return cs_g(convert_p_to_d(x))@T(x)

    def polar_H(x):
        H = cs_H(convert_p_to_d(x))
        g = cs_g(convert_p_to_d(x))
        c = np.cos(th(x))
        s = np.sin(th(x))
        V = v(x)
        l = x.size // 2
        A = np.array([[
            H[i, j] * c[i] * c[j] +
            H[i, j + l] * c[i] * s[j] +
            H[i + l, j] * s[i] * c[j] +
            H[i + l, j + l] * s[i] * s[j]
            for i in range(0, l)]
            for j in range(0, l)
        ])
        B = np.array([[
            -H[i, j] * c[i] * V[j] * s[j] +
            H[i, j + l] * c[i] * V[j] * c[j] -
            H[i + l, j] * s[i] * V[j] * s[j] +
            H[i + l, j + l] * s[i] * V[j] * c[j] +
            (-g[i] * s[i] + g[i + l] * c[i] if i == j else 0)
            for i in range(0, l)]
            for j in range(0, l)
        ])
        C = np.array([[
            -H[i, j] * V[i] * s[i] * c[j] -
            H[i, j + l] * V[i] * s[i] * s[j] +
            H[i + l, j] * V[i] * c[i] * c[j] +
            H[i + l, j + l] * V[i] * c[i] * s[j] +
            (-g[i] * s[i] + g[i + l] * c[i] if i == j else 0)
            for i in range(0, l)]
            for j in range(0, l)
        ])
        D = np.array([[
            H[i, j] * V[i] * s[i] * V[j] * s[j]
            - H[i, j + l] * V[i] * s[i] * V[j] * c[j]
            - H[i + l, j] * V[i] * c[i] * V[j] * s[j]
            + H[i + l, j + l] * V[i] * c[i] * V[j] * c[j] +
            (-g[i] * V[i] * c[i] - g[i + l] * V[i] * s[i] if i == j else 0)
            for i in range(0, l)]
            for j in range(0, l)
        ])
        return np.bmat([[A, B], [C, D]]).A

    # orig_test = np.array([100, 100, 1, 1])
    # test = np.append(abs(v(orig_test) + th(orig_test) * 1.j), np.angle(v(orig_test) + th(orig_test) * 1.j))
    return polar_F_min, polar_g, polar_H


def test_U(U, Z, ballance_node, U_0):
    Y = Cond_matrix(Z)
    Y = np.delete(Y, ballance_node, axis=0)
    Y_ballance = Y[:, ballance_node].reshape(Y[:, ballance_node].size, 1)
    Y = np.delete(Y, ballance_node, axis=1)
    U_restore = np.zeros(int(U.size / 2), dtype='complex').reshape(int(U.size / 2), 1)
    for i in range(0, int(U.size / 2)):
        U_restore[i] = U[i] + U[int(U.size / 2) + i] * 1.j
    S = (np.diag(U_restore.ravel())).conjugate().dot(Y.dot(U_restore) + Y_ballance.dot(U_0))
    S = -S.conjugate()
    return S.ravel()
