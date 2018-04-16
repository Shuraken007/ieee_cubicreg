import numpy as np
import scipy.linalg
from scipy.optimize import newton

class _AuxiliaryProblem:
    """
    Solve the cubic subproblem as described in Conn et. al (2000) (see reference at top of file)
    The notation in this function follows that of the above reference.
    """

    def __init__(self, x, gradient, hessian, M, lambda_nplus, kappa_easy, submaxiter, stepmin, epsilon=np.sqrt(np.finfo(float).eps)):
        """
        :param x: Current location of cubic regularization algorithm
        :param gradient: Gradient at current point
        :param hessian: Hessian at current point
        :param M: Current value used for M in cubic upper approximation to f at x_new
        :param lambda_nplus: max(-1*smallest eigenvalue of hessian of f at x, 0)
        :param kappa_easy: Convergence tolerance
        """
        self.x = x
        self.grad_x = gradient
        self.hess_x = hessian
        self.M = M
        self.lambda_nplus = lambda_nplus
        self.kappa_easy = kappa_easy
        self.maxiter = submaxiter
        # Function to compute H(x)+lambda*I as function of lambda
        self.H_lambda = lambda lambduh: self.hess_x + \
            lambduh * np.identity(np.size(self.hess_x, 0))
        # Constant to add to lambda_nplus so that you're not at the zero where the eigenvalue is
        self.epsilon = epsilon
        self.lambda_const = (1 + self.lambda_nplus) * self.epsilon
        self.stepmin = stepmin
        self.im_id = 0

    def _compute_s(self, lambduh):
        """
        Compute L in H_lambda = LL^T and then solve LL^Ts = -g
        :param lambduh: value for lambda in H_lambda
        :return: s, L
        """
        try:
            # Numpy's Cholesky seems more numerically stable than scipy's Cholesky
            L = np.linalg.cholesky(self.H_lambda(lambduh)).T
        except:
            # See p. 516 of Gould et al. (1999) (see reference at top of file)
            # self.lambda_const *= 2
            k = 0.98
            corr = np.min(np.linalg.eigvals(self.hess_x))*k
            if corr < 0:
                self.lambda_nplus = - corr
            try:
                s, L = self._compute_s(self.lambda_nplus)
            except:
                # with open('cr_16_node_mode_' + self.stepmin + '.txt', 'a') as fi:
                    # print(self.H_lambda(lambduh), file=fi)
                    # print('{:<28}'.format('eigen vals, hessian(x_opt):'), np.linalg.eig(self.H_lambda(lambduh))[0], file=fi)
                    # print('{:<28}'.format('e_min:'), np.linalg.eig(self.H_lambda(lambduh))[0].min(), file=fi)
                print('Cholesky problems')
                return np.zeros_like(self.grad_x), [], 1
        s = scipy.linalg.cho_solve((L, False), -self.grad_x)
        return s, L, 0

    def _update_lambda(self, lambduh, s, L):
        """
        Update lambda by taking a Newton step
        :param lambduh: Current value of lambda
        :param s: Current value of -(H+lambda I)^(-1)g
        :param L: Matrix L from Cholesky factorization of H_lambda
        :return: lambduh - phi/phi_prime: Next value of lambda
        """
        w = scipy.linalg.solve_triangular(L.T, s, lower=True)
        norm_s = np.linalg.norm(s)
        phi = 1 / norm_s - self.M / (2 * lambduh)
        phi_prime = np.linalg.norm(
            w)**2 / (norm_s**3) + self.M / (2 * lambduh**2)
        return lambduh - phi / phi_prime

    def _converged(self, s, lambduh):
        """
        Check whether the algorithm from the subproblem has converged
        :param s: Current estimate of -(H+ lambda I)^(-1)g
        :param lambduh: Current estimate of lambda := Mr/2
        :return: True/False based on whether the convergence criterion has been met
        """
        r = 2 * lambduh / self.M
        if abs(np.linalg.norm(s) - r) <= self.kappa_easy:
            return True
        else:
            return False

    def find_min_r(self):
        k = 0.98
        corr = np.min(np.linalg.eigvals(self.hess_x))*k
        r0 = -2*corr/self.M
        r0 = max(0, r0)
        return r0

    def solve(self):
        """
        Solve the cubic regularization subproblem. See algorithm 7.3.6 in Conn et al. (2000).
        :return: s: Step for the cubic regularization algorithm
        """
        if self.lambda_nplus == 0:
            lambduh = 0
        else:
            lambduh = self.lambda_nplus + self.lambda_const
        s, L, flag = self._compute_s(lambduh)
        if flag != 0:
            return s, flag
        r = 2 * lambduh / self.M
        if np.linalg.norm(s) <= r:
            if lambduh == 0 or np.linalg.norm(s) == r:
                return s, 0
            else:
                Lambda, U = np.linalg.eigh(self.H_lambda(self.lambda_nplus))
                s_cri = -U.T.dot(np.linalg.pinv(np.diag(Lambda))
                                 ).dot(U).dot(self.grad_x)
                alpha = max(np.roots([np.dot(U[:, 0], U[:, 0]), 2 * np.dot(
                    U[:, 0], s_cri), np.dot(s_cri, s_cri) - 4 * self.lambda_nplus**2 / self.M**2]))
                s = s_cri + alpha * U[:, 0]
                return s, 0
        if lambduh == 0:
            lambduh += self.lambda_const
        iter = 0
        while not self._converged(s, lambduh) and iter < self.maxiter:
            iter += 1
            lambduh = self._update_lambda(lambduh, s, L)
            s, L, flag = self._compute_s(lambduh)
            if flag != 0:
                return s, flag
            if iter == self.maxiter:
                print(RuntimeWarning(
                    'Warning: Could not compute s: maximum number of iterations reached'))
        return s, 0

    def solve_diff(self, r0):
        I = np.identity(np.size(self.hess_x, 0))
        H = self.hess_x
        g = self.grad_x
        M = self.M

        def x(r): return np.linalg.inv(H + M * r * I / 2).dot(g)

        def F_n(r): return r * r - np.linalg.norm(x(r)) ** 2

        def diff_x(r): return -M / 2 * \
            (np.linalg.inv(H + M * r * I / 2)).dot(x(r))

        def diff_F(r): return 2 * r - 2 * np.sum(x(r).dot(diff_x(r)))

        _r0 = self.find_min_r()
        r0 = max(r0, _r0)
        # print(r0)
        r = newton(F_n, r0, fprime=diff_F, args=(), tol=1.48e-06, maxiter=1000)

        # _, _, r1 = self.solve_bifind()
        # def h(r): return np.linalg.norm(-np.linalg.inv(H + M * r * I / 2)@g)**2
        # def tau(r): return r**2

        # # a = r0 - (max(r1,r)-r0)*2
        # # b = r0 + (max(r1,r)-r0)*2
        # a = r0 - 1
        # b = r0 + 1
        # X = np.arange(a, b, (b-a)/1000)
        # _h = [h(r) for r in X]
        # _tau = [tau(r) for r in X]
        # plt.figure()
        # plt.plot(X, _h, color = 'blue')
        # plt.plot(X, _tau, color = 'red')
        # global im_id
        # im_id += 1
        # plt.draw()
        # plt.savefig('graph\\graph' + str(im_id) + '.jpg')
        # plt.close()

        # print(r**2, r0)
        # print(r)
        return -x(r), 0, r

    def solve_poly(self):
        I = np.identity(np.size(self.hess_x, 0))
        H = self.hess_x
        g = self.grad_x
        M = self.M

        e, v = np.linalg.eig(H)
        e_min = max([0, -np.linalg.eig(H)[0].min()])
        g_new = np.linalg.inv(v).dot(g)
        for x in np.nditer(g_new, op_flags=['readwrite']):
            if x == 0:
                x[...] = x + self.g_add

        r = symbols('r')
        F = r * r
        for i in range(g_new.size):
            F = F * (e[i] + r * M / 2) ** 2

        for i in range(g_new.size):
            temp = g_new[i] ** 2
            for j in range(g_new.size):
                if j != i:
                    temp = temp * (e[j] + r * M / 2) ** 2
            F = F - temp

        all_roots = np.roots(Poly(F).all_coeffs())
        all_roots = all_roots[all_roots >= e_min]

        best_roots = []
        best_roots_val = []
        best_roots_id = []
        index = -1

        def point(r): return -np.linalg.inv(H + M * r * I / 2).dot(g)

        def Func_T_M(x): return g.dot(x) + (H.dot(x).dot(x)) / \
            2 + M / 6 * np.linalg.norm(x) ** 3

        for i in range(0, all_roots.size):
            if all_roots[i].imag == 0:
                if index == -1:
                    best_roots.append(point(all_roots[i].real))
                    best_roots_val.append(Func_T_M(best_roots[0]))
                    best_roots_id.append(i)
                    index = 0
                else:
                    if np.linalg.norm(best_roots[index] - (point(all_roots[i].real))) > 0.01:
                        index = index + 1
                        best_roots.append(point(all_roots[i].real))
                        best_roots_val.append(Func_T_M(best_roots[index]))
                        best_roots_id.append(i)
        best_roots_val = np.array(best_roots_val)

        return best_roots[best_roots_val.argmin()], 0

    def solve_bifind(self, r0):
        I = np.identity(np.size(self.hess_x, 0))
        H = self.hess_x
        g = self.grad_x
        M = self.M

        def h(r): return np.linalg.norm(-np.linalg.inv(H + M * r * I / 2)@g)**2

        def tau(r): return r**2

        def x(r): return np.linalg.inv(H + M * r * I / 2)@g

        # def D(r): return g@h(r) + 1 / 2 * (H@h(r))@h(r) + M / 6 * tau(r)**(3 / 2) + M / 4 * r * (np.linalg.norm(h(r))**2 - tau(r))
        # def D(r): return h(r) - tau(r)
        r0 = self.find_min_r()

        r, flag = self.bin_search(h, tau, r0)
        # print(r, r0)
        return -x(r), flag, r

    def bin_search(self, h, tau, r0, eps=0.1, step=2, maxiter=1000):
        if tau(r0) > h(r0):
            return r0, 0
        a = r0
        b = a + step
        iter = 0
        while tau(b) < h(a) and iter < maxiter:
            b *= step
            iter += 1
        r = (b+a)/2
        if iter == maxiter:
            print('wrong tau(b) < h(a)')
            return r0, 1
        iter = 0
        while abs(tau(b) - h(a)) > eps and iter < maxiter:
            # print(b, a, tau(b) - h(a))
            if tau(r) > h(r):
                b = r
            else:
                a = r
            r = (b+a)/2
            iter += 1
        if iter == maxiter:
            print('abs(tau(b) - h(a)) > eps')
            return r0, 1
        return r, 0
