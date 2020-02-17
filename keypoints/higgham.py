import torch
import numpy as np
from numpy import linalg as la


def np_nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not np_isPD(A3):
        evals = la.eigvals(A3)
        real_evals = np.real(evals)
        mineig = np.min(real_evals)
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def np_isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def isPD(A):
    try:
        torch.cholesky(A)
        return True
    except Exception:
        return False


def nearestPD(A):
    """ dont use this it doesn't work"""
    B = A + A.T / 2
    u, s, v = torch.svd(B)
    H = torch.matmul(v.T, torch.matmul(s.diag_embed(), v))
    A2 = (B + H) / 2
    A3 = (A2 + A2.t()) / 2
    if isPD(A3):
        return A3

    spacing = torch.finfo(A3.dtype).eps
    I = torch.eye(A.size(0), device=A.device)
    k = 1
    while not isPD(A3):
        val, vecs = A3.eig()
        mineig = val[0].min()
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def nearestPDHack(c):
    """ this probably kinda works but is really slow"""
    eps = torch.finfo(c.dtype).eps
    k = 1
    while not isPD(c):
        # fix it so we have positive definite matrix
        # could also use the Higham algorithm for more accuracy
        #  N.J. Higham, "Computing a nearest symmetric positive semidefinite
        # https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
        print('covariance matrix not positive definite, attempting recovery')
        e, v = torch.symeig(c, eigenvectors=True)
        bump = eps * k ** 2
        e[e < bump] += bump
        c = torch.matmul(v, torch.matmul(e.diag_embed(), v.t()))
        k += 1

    return c