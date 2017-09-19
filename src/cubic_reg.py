"""
This module implements cubic regularization of Newton's method, as described in Nesterov and Polyak (2006) and also
the adaptive cubic regularization algorithm described in Cartis et al. (2011). This code solves the cubic subproblem
according to slight modifications of Algorithm 7.3.6 of Conn et. al (2000). Cubic regularization solves unconstrained
minimization problems by minimizing a cubic upper bound to the function at each iteration.

Implementation by Corinne Jones
cjones6@uw.edu
June 2016

References:
- Nesterov, Y., & Polyak, B. T. (2006). Cubic regularization of Newton method and its global performance.
  Mathematical Programming, 108(1), 177-205.
- Cartis, C., Gould, N. I., & Toint, P. L. (2011). Adaptive cubic regularisation methods for unconstrained optimization.
  Part I: motivation, convergence and numerical results. Mathematical Programming, 127(2), 245-295.
- Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods (Vol. 1). Siam.
- Gould, N. I., Lucidi, S., Roma, M., & Toint, P. L. (1999). Solving the trust-region subproblem using the Lanczos
  method. SIAM Journal on Optimization, 9(2), 504-525.
"""

from sympy import *
import numpy as np
import scipy.linalg
import time
from scipy.optimize import newton


class Algorithm:
    def __init__(self, x0, f=None, gradient=None, hessian=None, L=None, L0=None, kappa_easy=0.0001, maxiter=10000,
                 submaxiter=100000, conv_tol=1e-5, conv_criterion='gradient', epsilon=2 * np.sqrt(np.finfo(float).eps),
                 subepsilon=2 * np.sqrt(np.finfo(float).eps), print_data=None, stepmin='base', g_add=1e-3):
        """
        Collect all the inputs to the cubic regularization algorithm.
        Required inputs: function or all of gradient and Hessian and L. If you choose conv_criterion='Nesterov', you must also supply L.
        :param x0: Starting point for cubic regularization algorithm
        :param f: Function to be minimized
        :param gradient: Gradient of f (input as a function that returns a numpy array)
        :param hessian: Hessian of f (input as a function that returns a numpy array)
        :param L: Lipschitz constant on the Hessian
        :param L0: Starting point for line search for M
        :param kappa_easy: Convergence tolerance for the cubic subproblem
        :param maxiter: Maximum number of cubic regularization iterations
        :param submaxiter: Maximum number of iterations for the cubic subproblem
        :param conv_tol: Convergence tolerance
        :param conv_criterion: Criterion for convergence: 'gradient' or 'nesterov'. Gradient uses norm of gradient.
                                Nesterov's uses max(sqrt(2/(L+M)norm(f'(x)), -2/(2L+M)lambda_min(f''(x))).
        :param epsilon: Value added/subtracted from x when approximating gradients and Hessians
        """
        self.f = f
        self.gradient = gradient
        self.hessian = hessian
        self.x0 = np.array(x0) * 1.0
        self.maxiter = maxiter
        self.submaxiter = submaxiter
        self.conv_tol = conv_tol
        self.conv_criterion = conv_criterion.lower()
        self.epsilon = epsilon
        self.subepsilon = subepsilon
        self.L = L
        self.L0 = L0
        self.kappa_easy = kappa_easy
        self.n = len(x0)
        self.print_data = print_data
        self.stepmin = stepmin
        self.g_add = g_add
        self._check_inputs()
        # Estimate the gradient, hessian, and find a lower bound L0 for L if necessary
        if gradient is None:
            self.gradient = self.approx_grad
        if hessian is None:
            self.hessian = self.approx_hess
        if L0 is None and L is None:
            self.L0 = np.linalg.norm(self.hessian(self.x0) - self.hessian(self.x0 + np.ones_like(self.x0)), ord=2) / np.linalg.norm(np.ones_like(self.x0)) + self.epsilon

        self.grad_x = self.gradient(self.x0)
        self.hess_x = self.hessian(self.x0)
        self.lambda_nplus = self._compute_lambda_nplus()[0]
        self.print_data = print_data

    def _check_inputs(self):
        """
        Ensure that the inputs are of the right form and all necessary inputs have been supplied
        """
        if not isinstance(self.x0, (tuple, list, np.ndarray)):
            raise TypeError('Invalid input type for x0')
        if len(self.x0) < 1:
            raise ValueError('x0 must have length > 0')
        if not (self.f is not None or (self.gradient is not None and self.hessian is not None and self.L is not None)):
            raise AttributeError('You must specify f and/or each of the following: gradient, hessian, and L')
        if not((not self.L or self.L > 0)and (not self.L0 or self.L0 > 0) and self.kappa_easy > 0 and self.maxiter > 0 and self.conv_tol > 0 and self.epsilon > 0):
            raise ValueError('All inputs that are constants must be larger than 0')
        if self.f is not None:
            try:
                self.f(self.x0)
            except TypeError:
                raise TypeError('x0 is not a valid input to function f')
        if self.gradient is not None:
            try:
                self.gradient(self.x0)
            except TypeError:
                raise TypeError('x0 is not a valid input to the gradient. Is the gradient a function with input dimension length(x0)?')
        if self.hessian is not None:
            try:
                self.hessian(self.x0)
            except TypeError:
                raise TypeError('x0 is not a valid input to the hessian. Is the hessian a function with input dimension length(x0)?')
        if not (self.conv_criterion == 'gradient' or self.conv_criterion == 'nesterov'):
            raise ValueError('Invalid input for convergence criterion')
        if self.conv_criterion == 'nesterov' and self.L is None:
            raise ValueError("With Nesterov's convergence criterion you must specify L")

    @staticmethod
    def _std_basis(size, idx):
        """
        Compute the idx'th standard basis vector
        :param size: Length of the vector
        :param idx: Index of value 1 in the vector
        :return: ei: Standard basis vector with 1 in the idx'th position
        """
        ei = np.zeros(size)
        ei[idx] = 1
        return ei

    def approx_grad(self, x):
        """
        Approximate the gradient of the function self.f at x
        :param x: Point at which the gradient will be approximated
        :return: Estimated gradient at x
        """
        return np.asarray([(self.f(x + self.epsilon * self._std_basis(self.n, i)) - self.f(x - self.epsilon * self._std_basis(self.n, i))) / (2 * self.epsilon) for i in range(0, self.n)])

    def approx_hess(self, x):
        """
        Approximate the hessian of the function self.x at x
        :param x: Point at which the Hessian will be approximated
        :return: Estimated Hessian at x
        """
        grad_x0 = self.gradient(x)
        hessian = np.zeros((self.n, self.n))
        for j in range(0, self.n):
            grad_x_plus_eps = self.gradient(x + self.epsilon * self._std_basis(self.n, j))
            for i in range(0, self.n):
                hessian[i, j] = (grad_x_plus_eps[i] - grad_x0[i]) / self.epsilon
        return hessian

    def _compute_lambda_nplus(self):
        """
        Compute max(-1*smallest eigenvalue of hessian of f at x, 0)
        :return: max(-1*smallest eigenvalue of hessian of f at x, 0)
        :return: lambda_n: Smallest eigenvaleu of hessian of f at x
        """
        lambda_n = scipy.linalg.eigh(self.hess_x, eigvals_only=True, eigvals=(0, 0))
        return max(-lambda_n[0], 0), lambda_n

    def _check_convergence(self, lambda_min, M):
        """
        Check whether the cubic regularization algorithm has converged
        :param lambda_min: Minimum eigenvalue at current point
        :param M: Current value used for M in cubic upper approximation to f at x_new
        :return: True/False depending on whether the convergence criterion has been satisfied
        """
        if self.conv_criterion == 'gradient':
            if np.linalg.norm(self.grad_x) <= self.conv_tol:
                return True
            else:
                return False
        elif self.conv_criterion == 'nesterov':
            if max(np.sqrt(2 / (self.L + M) * np.linalg.norm(self.grad_x)), -2 / (2 * L + M) * lambda_min) <= self.conv_tol:
                return True
            else:
                return False


class CubicRegularization(Algorithm):
    def __init__(self, x0, f=None, gradient=None, hessian=None, L=None, L0=None, kappa_easy=0.0001, maxiter=10000, submaxiter=10000,
                 conv_tol=1e-5, conv_criterion='gradient', epsilon=2 * np.sqrt(np.finfo(float).eps),
                 subepsilon=2 * np.sqrt(np.finfo(float).eps), print_data=None, stepmin='base', g_add=1e-3):
        Algorithm.__init__(self, x0, f=f, gradient=gradient, hessian=hessian, L=L, L0=L0, kappa_easy=kappa_easy,
                           maxiter=maxiter, submaxiter=submaxiter, conv_tol=conv_tol, conv_criterion=conv_criterion,
                           epsilon=epsilon, subepsilon=subepsilon, print_data=print_data, stepmin=stepmin, g_add=g_add)

    def cubic_reg(self):
        """
        Run the cubic regularization algorithm
        :return: x_new: Final point
        :return: intermediate_points: All points visited by the cubic regularization algorithm on the way to x_new
        :return: iter: Number of iterations of cubic regularization
        """
        self.iter = flag = 0
        converged = False
        x_new = self.x0
        mk = self.L0
        intermediate_points = [x_new]
        self.print_data.print_phase('main_iter', {'iter': self.iter,
                                                  'grad': np.linalg.norm(self.grad_x),
                                                  'value': self.f(x_new),
                                                  'vector': x_new.ravel(),
                                                  'e_min': np.linalg.eig(self.hess_x)[0].min(),
                                                  'time': time.time() - time.time(),
                                                  'M': mk
                                                  })
        self.prevtime = time.time()
        while self.iter < self.maxiter and converged is False:
            x_old = x_new
            x_new, mk, flag = self._find_x_new(x_old, mk)
            self.grad_x = self.gradient(x_new)
            self.hess_x = self.hessian(x_new)
            self.lambda_nplus, lambda_min = self._compute_lambda_nplus()
            converged = self._check_convergence(lambda_min, mk)
            self.print_data.print_phase('main_iter', {'iter': self.iter + 1,
                                                      'grad': np.linalg.norm(self.grad_x),
                                                      'value': self.f(x_new),
                                                      'vector': x_new.ravel(),
                                                      'e_min': np.linalg.eig(self.hess_x)[0].min(),
                                                      'time': time.time() - self.prevtime,
                                                      'M': mk
                                                      })
            self.prevtime = time.time()
            if flag != 0:
                print(RuntimeWarning('Convergence criteria not met, likely due to round-off error or ill-conditioned Hessian.'))
                return x_new, intermediate_points, self.iter, flag
            intermediate_points.append(x_new)
            self.iter += 1
        return x_new, intermediate_points, self.iter, flag

    def _find_x_new(self, x_old, mk):
        """
        Determine what M_k should be and compute the next point for the cubic regularization algorithm
        :param x_old: Previous point
        :param mk: Previous value of M_k (will start with this if L isn't specified)
        :return: x_new: New point
        :return: mk: New value of M_k
        """
        if self.L is not None:
            aux_problem = _AuxiliaryProblem(x_old, self.grad_x, self.hess_x, self.L, self.lambda_nplus, self.kappa_easy, self.submaxiter, self.stepmin, self.subepsilon)
            s, flag = aux_problem.solve()
            x_new = s + x_old
            return x_new, self.L, flag
        else:
            decreased = False
            iter = 0
            f_xold = self.f(x_old)
            while not decreased and iter < self.submaxiter:
                mk *= 2
                if mk > 20000:
                    self.L0 = 0.3 * self.L0
                    mk = self.L0
                aux_problem = _AuxiliaryProblem(x_old, self.grad_x, self.hess_x, mk, self.lambda_nplus, self.kappa_easy, self.submaxiter, self.stepmin, self.subepsilon)
                if self.stepmin == 'base':
                    s, flag = aux_problem.solve()
                elif self.stepmin == 'diff':
                    if self.iter >= 1:
                        s, flag, self.stepmin_diff_x = aux_problem.solve_diff(self.stepmin_diff_x)
                    else:
                        s, flag = aux_problem.solve()
                        I = np.identity(np.size(self.hess_x, 0))
                        H = self.hess_x
                        g = self.grad_x

                        def x(r): return np.linalg.inv(H + mk * r * I / 2).dot(g)
                        x0 = (2 * (g - H.dot(x(-s))) / (mk * I.dot(x(-s))))[1]
                        self.stepmin_diff_x = x0
                elif self.stepmin == 'poly':
                    s, flag = aux_problem.solve_poly()

                x_new = s + x_old
                decreased = (self.f(x_new) - f_xold <= 0)
                iter += 1
                if not decreased and iter < self.submaxiter:
                    self.print_data.print_phase('sub_M', {
                        'M': mk
                    })
                if iter == self.submaxiter:
                    raise RuntimeError('Could not find cubic upper approximation')
            mk = max(0.5 * mk, self.L0)
            return x_new, mk, flag


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
        self.H_lambda = lambda lambduh: self.hess_x + lambduh * np.identity(np.size(self.hess_x, 0))
        # Constant to add to lambda_nplus so that you're not at the zero where the eigenvalue is
        self.epsilon = epsilon
        self.lambda_const = (1 + self.lambda_nplus) * self.epsilon
        self.stepmin = stepmin

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
            self.lambda_const *= 2
            try:
                s, L = self._compute_s(self.lambda_nplus + self.lambda_const)
            except:
                # with open('cr_16_node_mode_' + self.stepmin + '.txt', 'a') as fi:
                    # print(self.H_lambda(lambduh), file=fi)
                    # print('{:<28}'.format('eigen vals, hessian(x_opt):'), np.linalg.eig(self.H_lambda(lambduh))[0], file=fi)
                    # print('{:<28}'.format('e_min:'), np.linalg.eig(self.H_lambda(lambduh))[0].min(), file=fi)

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
        phi_prime = np.linalg.norm(w)**2 / (norm_s**3) + self.M / (2 * lambduh**2)
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
                s_cri = -U.T.dot(np.linalg.pinv(np.diag(Lambda))).dot(U).dot(self.grad_x)
                alpha = max(np.roots([np.dot(U[:, 0], U[:, 0]), 2 * np.dot(U[:, 0], s_cri), np.dot(s_cri, s_cri) - 4 * self.lambda_nplus**2 / self.M**2]))
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
                print(RuntimeWarning('Warning: Could not compute s: maximum number of iterations reached'))
        return s, 0

    def solve_diff(self, x0):
        I = np.identity(np.size(self.hess_x, 0))
        H = self.hess_x
        g = self.grad_x
        M = self.M

        def x(r): return np.linalg.inv(H + M * r * I / 2).dot(g)

        def F_n(r): return r * r - np.linalg.norm(x(r)) ** 2

        def diff_x(r): return -M / 2 * (np.linalg.inv(H + M * r * I / 2)).dot(x(r))

        def diff_F(r): return 2 * r - 2 * np.sum(x(r).dot(diff_x(r)))
        result = newton(F_n, x0, fprime=diff_F, args=(), tol=1.48e-08, maxiter=1000)
        return -x(result), 0, result

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

        def Func_T_M(x): return g.dot(x) + (H.dot(x).dot(x)) / 2 + M / 6 * np.linalg.norm(x) ** 3

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
