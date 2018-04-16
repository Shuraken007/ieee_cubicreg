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

import numpy as np
from sympy import *
import scipy.linalg
import time
import matplotlib.pyplot as plt
from .AuxiliaryProblem import _AuxiliaryProblem

im_id = 0

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
                 subepsilon=2 * np.sqrt(np.finfo(float).eps), print_data=None, stepmin='base', g_add=1e-3, S = None):
        Algorithm.__init__(self, x0, f=f, gradient=gradient, hessian=hessian, L=L, L0=L0, kappa_easy=kappa_easy,
                           maxiter=maxiter, submaxiter=submaxiter, conv_tol=conv_tol, conv_criterion=conv_criterion,
                           epsilon=epsilon, subepsilon=subepsilon, print_data=print_data, stepmin=stepmin, g_add=g_add)
        self.S = S

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
        if self.S is not None:
            s_n = np.linalg.norm(self.S)
        else:
            s_n = -1
        self.print_data.print_phase('last_phase', {'iter': self.iter + 1,
                                                  'grad': np.linalg.norm(self.grad_x),
                                                  'value': self.f(x_new),
                                                  'e_min': np.linalg.eig(self.hess_x)[0].min(),
                                                  'S': s_n
                                                  })
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
                elif self.stepmin == 'diff' or self.stepmin == 'bifind':
                    if self.stepmin == "diff":
                        meth = aux_problem.solve_diff
                    else:
                        meth = aux_problem.solve_bifind

                    if self.iter >= 1:
                        s, flag, self.stepmin_diff_x = meth(self.stepmin_diff_x)
                    else:
                        s, flag = aux_problem.solve()
                        I = np.identity(np.size(self.hess_x, 0))
                        H = self.hess_x
                        g = self.grad_x

                        def x(r): return np.linalg.inv(H + mk * r @ I / 2)@g
                        x0 = (2 * (g - H@x(-s)) / (mk * I@x(-s)))[1]
                        self.stepmin_diff_x = x0
                elif self.stepmin == "base+diff" or self.stepmin == "base+bifind":
                    if self.stepmin == "base+diff":
                        meth = aux_problem.solve_diff
                    else:
                        meth = aux_problem.solve_bifind

                    self.itertime_s = time.time()
                    if self.iter >= 1 and self.bd_iter_timer > 0 and self.bd_iter_timer_value > 1:
                        s, flag, self.stepmin_diff_x = aux_problem.solve_diff(self.stepmin_diff_x)
                        self.bd_iter_timer -= 1
                    else:
                        if self.iter >= 1 and self.bd_iter_timer == 0:
                            self.itertimep_test_s = time.time()
                        s, flag = aux_problem.solve()
                        I = np.identity(np.size(self.hess_x, 0))
                        H = self.hess_x
                        g = self.grad_x

                        def x(r): return np.linalg.inv(H + mk * r @ I / 2)@g
                        x0 = (2 * (g - H@x(-s)) / (mk * I@x(-s)))[1]
                        self.stepmin_diff_x = x0

                        if self.iter == 1 and self.bd_iter_timer == 0:
                            self.itertimep_test_f = time.time()
                            degree = (self.itertimep_test_f - self.itertimep_test_s) - (self.itertime_s - self.itertime_f)
                            if degree > 0:
                                self.bd_iter_timer_value *= 2
                            else:
                                self.bd_iter_timer_value /= 2
                            self.bd_iter_timer = self.bd_iter_timer_value
                        else:
                            self.bd_iter_timer_value = 10
                            self.bd_iter_timer = self.bd_iter_timer_value
                    self.itertime_f = time.time()
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
