import numpy as np
from typing import Tuple

from .smoothing import eval_H_smooth
from .hamiltonian import compute_H

def pack_unknowns(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Flatten state and costate trajectories into a single vector.

    X has shape (N+1, n) and contains x_0,...,x_N.
    P has shape (N+1, n) and contains p_0,...,p_N.
    x_0 is fixed by the problem and should not be included in the unknown vector.

    The returned vector z concatenates x_1,...,x_N followed by p_0,...,p_N.
    """
    N_plus_1, n = X.shape
    N = N_plus_1 - 1
    z = np.zeros((N * n + (N + 1) * n,))
    # pack x_1,...,x_N
    z[0:N * n] = X[1:, :].reshape(N * n)
    # pack p_0,...,p_N
    z[N * n:] = P.reshape((N + 1) * n)
    return z

def unpack_unknowns(z: np.ndarray, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of pack_unknowns.

    Parameters
    ----------
    z : np.ndarray
        Flattened vector of unknowns with length N*n + (N+1)*n.
    x0 : np.ndarray
        The initial state (x_0), which is fixed and not included in z.

    Returns
    -------
    X : np.ndarray
        State trajectory array of shape (N+1, n).
    P : np.ndarray
        Costate trajectory array of shape (N+1, n).
    """
    n = x0.size
    # deduce N such that z.size = (2*N + 1) * n
    total = z.size // n
    # total = 2*N + 1 -> N = (total - 1) // 2
    N = (total - 1) // 2
    N_plus_1 = N + 1
    X = np.zeros((N_plus_1, n))
    P = np.zeros((N_plus_1, n))
    # assign x_0
    X[0, :] = x0
    # x_1,...,x_N
    X[1:, :] = z[0:N * n].reshape((N, n))
    # p_0,...,p_N
    P[:, :] = z[N * n:].reshape((N_plus_1, n))
    return X, P

def assemble_residual(problem, t_nodes: np.ndarray, X: np.ndarray, P: np.ndarray, bundle, delta: float) -> np.ndarray:
    """
    Assemble the residual vector for the symplectic Euler discretisation.

    Parameters
    ----------
    problem : OCPProblem
        Optimal control problem instance.
    t_nodes : np.ndarray
        Array of time nodes of length N+1.
    X : np.ndarray
        Array of shape (N+1, n) containing state trajectory (including x_0).
    P : np.ndarray
        Array of shape (N+1, n) containing costate trajectory.
    bundle : PABundle
        Bundle used for smoothing in the Hamiltonian.
    delta : float
        Smoothing parameter.

    Returns
    -------
    np.ndarray
        Residual vector of length 2*N*n + n.
    """
    N_plus_1 = t_nodes.size
    N = N_plus_1 - 1
    n = X.shape[1]
    residual = np.zeros((2 * N * n + n,))
    offset = 0
    for i in range(N):
        dt = t_nodes[i + 1] - t_nodes[i]
        x_i = X[i]
        x_ip1 = X[i + 1]
        p_i = P[i]
        p_ip1 = P[i + 1]
        # gradient at start
        _, grad_p_i, _ = eval_H_smooth(problem, bundle, p_i, x_i, t_nodes[i], delta)
        # gradient at end (for costate update)
        _, _, grad_x_ip1 = eval_H_smooth(problem, bundle, p_ip1, x_ip1, t_nodes[i + 1], delta)
        # state residual r_x = x_i + dt * grad_p - x_{i+1}
        r_x = x_i + dt * grad_p_i - x_ip1
        residual[offset:offset + n] = r_x
        offset += n
        # costate residual r_p = p_{i+1} + dt * grad_x_ip1 - p_i
        r_p = p_ip1 + dt * grad_x_ip1 - p_i
        residual[offset:offset + n] = r_p
        offset += n
    # terminal boundary condition: p_N + ∇g(x_N) = 0
    x_N = X[-1]
    p_N = P[-1]
    # gradient of g by finite difference
    g_grad = np.zeros_like(p_N)
    eps = 1e-6
    for j in range(n):
        x_plus = x_N.copy()
        x_minus = x_N.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        g_plus = problem.g(x_plus)
        g_minus = problem.g(x_minus)
        g_grad[j] = (g_plus - g_minus) / (2 * eps)
    r_bc = p_N + g_grad
    residual[offset:] = r_bc
    return residual

def assemble_jacobian(problem, t_nodes: np.ndarray, X: np.ndarray, P: np.ndarray, bundle, delta: float) -> np.ndarray:
    """
    Assemble the Jacobian matrix of the residual with respect to the unknowns.

    The unknown vector consists of x_1,...,x_N, p_0,...,p_N.  The Jacobian
    therefore has shape ((2*N*n + n), (2*N*n + n)).  We compute it by
    finite differences on the residual function.  This is expensive for large
    problems but suffices for moderate N and n.

    Returns
    -------
    np.ndarray
        Full Jacobian matrix.
    """
    # pack current unknown vector
    z = pack_unknowns(X, P)
    # function to compute residual
    def res_fun(z_vec: np.ndarray) -> np.ndarray:
        X_new, P_new = unpack_unknowns(z_vec, X[0])
        return assemble_residual(problem, t_nodes, X_new, P_new, bundle, delta)
    F0 = res_fun(z)
    m = z.size
    k = F0.size
    J = np.zeros((k, m))
    eps = 1e-6
    # finite differences
    for j in range(m):
        z_plus = z.copy()
        z_minus = z.copy()
        z_plus[j] += eps
        z_minus[j] -= eps
        F_plus = res_fun(z_plus)
        F_minus = res_fun(z_minus)
        J[:, j] = (F_plus - F_minus) / (2 * eps)
    return J