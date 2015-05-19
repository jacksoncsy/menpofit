from __future__ import division
import numpy as np


class mlr(object):
    r"""
    Multivariate Linear Regression

    Parameters
    ----------
    X: numpy.array
        The regression features used to create the coefficient matrix.
    T: numpy.array
        The shapes differential that denote the dependent variable.
    """
    def __init__(self, X, T):
        XX = np.dot(X.T, X)
        XX = (XX + XX.T) / 2
        XT = np.dot(X.T, T)
        self.R = np.linalg.solve(XX, XT)

    def __call__(self, x):
        return np.dot(x, self.R)

class rls(object):
    r"""
    Multivariate Recursive Least Linear Regression

    Parameters
    ----------

    """
    def __init__(self, W, V, multi_step=False):
        self.multi_step = multi_step
        self.R = W
        self.V = V

    def __call__(self, x):
        return np.dot(x, self.R)

    def update_model(self, X, T):
        """
            function to update regression models.
        """
        if self.multi_step:
            n = X.shape[0]
            a = self.V.dot(X.T)
            b = np.linalg.solve(np.eye(n) + X.dot(self.V).dot(X.T), X)
            # update with many examples
            delta_V = -np.dot(a, b.dot(self.V))
            self.V += delta_V
            delta_R = np.dot(self.V.dot(X.T), T) - np.dot(a, b.dot(self.R))
            self.R += delta_R
        else:
            for i, (x, y) in enumerate(zip(X, T)):
                x = x[None, :]
                y = y[None, :]
                delta_V = - np.dot(self.V.dot(x.T), x.dot(self.V)) / (1.0 + x.dot(self.V).dot(x.T))
                self.V += delta_V
                term = self.V.dot(x.T)
                delta_R = np.dot(term, y - x.dot(self.R))
                self.R += delta_R

class mlr_svd(object):
    r"""
    Multivariate Linear Regression using SVD decomposition

    Parameters
    ----------
    X: numpy.array
        The regression features used to create the coefficient matrix.
    T: numpy.array
        The shapes differential that denote the dependent variable.
    variance: float or None, Optional
        The SVD variance.

        Default: None

    Raises
    ------
    ValueError
        variance must be set to a number between 0 and 1
    """
    def __init__(self, X, T, variance=None):
        self.R, _, _, _ = _svd_regression(X, T, variance=variance)

    def __call__(self, x):
        return np.dot(x, self.R)


class mlr_pca(object):
    r"""
    Multivariate Linear Regression using PCA reconstructions

    Parameters
    ----------
    X: numpy.array
        The regression features used to create the coefficient matrix.
    T: numpy.array
        The shapes differential that denote the dependent variable.
    variance: float or None, Optional
        The SVD variance.

        Default: None

    Raises
    ------
    ValueError
        variance must be set to a number between 0 and 1
    """
    def __init__(self, X, T, variance=None):
        self.R, _, _, self.V = _svd_regression(X, T, variance=variance)

    def _call__(self, x):
        x = np.dot(np.dot(x, self.V.T), self.V)
        return np.dot(x, self.R)


class mlr_pca_weights(object):
    r"""
    Multivariate Linear Regression using PCA weights

    Parameters
    ----------
    X: numpy.array
        The regression features used to create the coefficient matrix.
    T: numpy.array
        The shapes differential that denote the dependent variable.
    variance: float or None, Optional
        The SVD variance.

        Default: None

    Raises
    ------
    ValueError
        variance must be set to a number between 0 and 1
    """
    def __init__(self, X, T, variance=None):
        _, _, _, self.V = _svd_regression(X, T, variance=variance)
        W = np.dot(X, self.V.T)
        self.R, _, _, _ = _svd_regression(W, T)

    def __call__(self, x):
        w = np.dot(x, self.V.T)
        return np.dot(w, self.R)


def _svd_regression(X, T, variance=None):
    r"""
    SVD decomposition for regression.

    Parameters
    ----------
    X: numpy.array
        The regression features used to create the coefficient matrix.
    T: numpy.array
        The shapes differential that denote the dependent variable.
    variance: float or None, Optional
        The SVD variance.

        Default: None

    Raises
    ------
    ValueError
        variance must be set to a number between 0 and 1
    """
    if variance is not None and not (0 < variance <= 1):
        raise ValueError("variance must be set to a number between 0 and 1.")

    U, s, V = np.linalg.svd(X)
    if variance:
        total = sum(s)
        acc = 0
        for j, y in enumerate(s):
            acc += y
            if acc / total >= variance:
                r = j+1
                break
    else:
        tol = np.max(X.shape) * np.spacing(np.max(s))
        r = np.sum(s > tol)
    U = U[:, :r]
    s = 1 / s[:r]
    V = V[:r, :]
    R = np.dot(np.dot(V.T * s, U.T), T)

    return R, U, s, V
