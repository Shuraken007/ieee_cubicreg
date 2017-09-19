import numpy as np


def Cond_matrix(Z):
    if type(Z) != dict:
        res = {'Z': Z}

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

    P = S.real.copy()
    P.resize(P.size, 1)
    Q = S.imag.copy()
    Q.resize(Q.size, 1)
    c_matr = Cond_matrix(Z)
    G = c_matr.real
    B = c_matr.imag
    G_sys = G
    B_sys = B
    g_0 = (G[:, ballance_node] + B[:, ballance_node] * 1j).real * U_0
    g_0.resize(g_0.size, 1)
    g_0 = np.delete(g_0, ballance_node, axis=0)
    b_0 = (G[:, ballance_node] + B[:, ballance_node] * 1j).imag * U_0
    b_0.resize(b_0.size, 1)
    b_0 = np.delete(b_0, ballance_node, axis=0)
    G_sys = np.delete(G_sys, ballance_node, axis=1)
    G_sys = np.delete(G_sys, ballance_node, axis=0)
    B_sys = np.delete(B_sys, ballance_node, axis=1)
    B_sys = np.delete(B_sys, ballance_node, axis=0)
    Y_sys = np.bmat([[G_sys, -B_sys], [B_sys, G_sys]])
    a = np.ones((1, 30)).ravel()

    def u(x): print(x); return x.reshape(x.size, 1)[0: int(x.size / 2)]

    def v(x): return x.reshape(x.size, 1)[int(x.size / 2): x.size]

    def D(x): return np.bmat([[np.diag(u(x).ravel()), np.diag(v(x).ravel())], [np.diag(v(x).ravel()), np.diag(-u(x).ravel())]])

    def F(x): return np.asarray(np.concatenate((P, Q)) + D(x) * (Y_sys * np.asmatrix(x.reshape(x.size, 1)) + np.asmatrix(np.concatenate((g_0, b_0)))))

    def D_F(x): return np.bmat([[np.diag(F(x)[0:num_node - 1].ravel()), -np.diag(F(x)[num_node - 1:2 * (num_node - 1)].ravel())],
                                [np.diag(F(x)[num_node - 1:2 * (num_node - 1)].ravel()), np.diag(F(x)[0:num_node - 1].ravel())]
                                ])

    def J(x): return np.bmat([[np.diag((G_sys.dot(u(x)) - B_sys.dot(v(x))).ravel()), np.diag((B_sys.dot(u(x)) + G_sys.dot(v(x))).ravel())],
                              [-np.diag((B_sys.dot(u(x)) + G_sys.dot(v(x))).ravel()), np.diag((G_sys.dot(u(x)) - B_sys.dot(v(x))).ravel())]
                              ]) + np.bmat([[np.diag(g_0.ravel()), np.diag(b_0.ravel())], [-np.diag(b_0.ravel()), np.diag(g_0.ravel())]]) + D(x) * Y_sys

    def cs_g(x): return np.asarray(F(x).ravel() * J(x)).ravel()

    def cs_H(x): return np.asarray(J(x).T * J(x) + D_F(x) * Y_sys + Y_sys.T * D_F(x).T)

    def F_min(x): return 0.5 * np.linalg.norm(F(x))**2
    # a = np.ones((30, 1))
    return F_min, cs_g, cs_H


def generate_cubic_system_polar(S, Z, num_node, ballance_node, U_0):
    P = S.real.copy()
    P.resize(P.size, 1)
    Q = S.imag.copy()
    Q.resize(Q.size, 1)
    c_matr = Cond_matrix(Z)
    G = c_matr.real
    B = c_matr.imag
    G_sys = G
    B_sys = B
    G_sys = np.delete(G_sys, ballance_node, axis=1)
    G_sys = np.delete(G_sys, ballance_node, axis=0)
    B_sys = np.delete(B_sys, ballance_node, axis=1)
    B_sys = np.delete(B_sys, ballance_node, axis=0)

    def d(x): return x.reshape(x.size, 1)[0:int(x.size / 2)]

    def v(x): return x.reshape(x.size, 1)[int(x.size / 2):x.size]

    def D(x): return np.bmat([[np.diag(u(x).ravel()), np.diag(v(x).ravel())], [np.diag(v(x).ravel()), np.diag(-u(x).ravel())]])

    def F(x): return np.asarray(np.concatenate((P, Q)) + D(x) * (Y_sys * np.asmatrix(x.reshape(x.size, 1)) + np.asmatrix(np.concatenate((g_0, b_0)))))

    def D_F(x): return np.bmat([[np.diag(F(x)[0:num_node - 1].ravel()), -np.diag(F(x)[num_node - 1:2 * (num_node - 1)].ravel())],
                                [np.diag(F(x)[num_node - 1:2 * (num_node - 1)].ravel()), np.diag(F(x)[0:num_node - 1].ravel())]
                                ])

    def J(x): return np.bmat([[H(d), N(d)],
                              [-np.diag((B_sys.dot(u(x)) + G_sys.dot(v(x))).ravel()), np.diag((G_sys.dot(u(x)) - B_sys.dot(v(x))).ravel())]
                              ]) + np.bmat([[np.diag(g_0.ravel()), np.diag(b_0.ravel())], [-np.diag(b_0.ravel()), np.diag(g_0.ravel())]]) + D(x) * Y_sys

    def cs_g(x): return np.asarray(F(x).ravel() * J(x)).ravel()

    def cs_H(x): return np.asarray(J(x).T * J(x) + D_F(x) * Y_sys + Y_sys.T * D_F(x).T)

    def F_min(x): return 0.5 * np.linalg.norm(F(x))**2
    # a = np.ones((30, 1))
    return F_min, cs_g, cs_H


def test_U(U, Z, ballance_node, U_0):
    Y = Cond_matrix(Z)
    Y = np.delete(Y, ballance_node, axis=0)
    Y_ballance = Y[:, ballance_node].reshape(Y[:, ballance_node].size, 1)
    Y = np.delete(Y, ballance_node, axis=1)
    U_restore = np.zeros(int(U.size / 2), dtype='complex').reshape(int(U.size / 2), 1)
    for i in range(0, int(U.size / 2)):
        U_restore[i] = U[i] + U[U.size / 2 + i] * 1.j
    S = (np.diag(U_restore.ravel())).conjugate().dot(Y.dot(U_restore) + Y_ballance.dot(U_0))
    S = -S.conjugate()
    return S.ravel()
