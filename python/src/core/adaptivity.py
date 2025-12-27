import numpy as np

from .pa_bundle import PABundle
from .smoothing import eval_H_smooth
from .hamiltonian import compute_H
from .newton import solve_tpbvp

def solve_optimal_control(
    problem,
    initial_mesh: np.ndarray,
    tol_time: float = 1e-3,
    tol_PA: float = 1e-3,
    tol_delta: float = 1e-3,
    max_iters: int = 10,
    delta0: float = 0.1,
):
    """
    Solve an optimal control problem adaptively by refining time mesh,
    adding new control planes, and reducing the smoothing parameter.

    Parameters
    ----------
    problem : OCPProblem
        Problem definition.
    initial_mesh : np.ndarray
        Initial time grid (including 0 and T).  Should be sorted.
    tol_time : float
        Tolerance for the time discretisation error indicator.
    tol_PA : float
        Tolerance for the PA surrogate error indicator.
    tol_delta : float
        Tolerance for the smoothing error indicator.
    max_iters : int
        Maximum number of outer adaptivity iterations.
    delta0 : float
        Initial smoothing parameter.

    Returns
    -------
    dict
        Dictionary with solution, mesh, bundle, delta, and log information.
    """
    # copy mesh
    t_nodes = np.asarray(initial_mesh, dtype=float).copy()
    # initialize PA bundle with zero control if dimension known, otherwise empty
    bundle = PABundle()
    # try to add zero control (or mean of bounds) if possible
    bounds = problem.control_bounds_tuple()
    m = problem.m
    if m is None and bounds is not None:
        m = bounds[0].size
    if m is not None:
        if bounds is not None:
            u0 = 0.5 * (bounds[0] + bounds[1])
        else:
            u0 = np.zeros(m)
        bundle.add_control(u0)
    delta = delta0
    log = []
    # initial guesses for X and P: None (will be set in Newton)
    X_guess = None
    P_guess = None
    for k in range(max_iters):
        # solve TPBVP on current mesh with current bundle and delta
        X, P, info = solve_tpbvp(problem, t_nodes, bundle, delta, X_guess, P_guess)
        # compute error indicators
        # time discretisation error
        N = len(t_nodes) - 1
        n = problem.n
        eta_time_local = np.zeros(N)
        grad_p_list = []
        grad_x_list = []
        # compute gradients at nodes for time error and smoothing error
        for i in range(N + 1):
            # grad_p, grad_x at each node
            # use smoothing evaluation
            _, grad_p_i, grad_x_i = eval_H_smooth(problem, bundle, P[i], X[i], t_nodes[i], delta)
            grad_p_list.append(grad_p_i)
            grad_x_list.append(grad_x_i)
        # compute local error indicator as difference of gradient across interval
        for i in range(N):
            dt = t_nodes[i + 1] - t_nodes[i]
            gp0 = grad_p_list[i]
            gp1 = grad_p_list[i + 1]
            gx0 = grad_x_list[i]
            gx1 = grad_x_list[i + 1]
            # measure change
            diff_gp = gp1 - gp0
            diff_gx = gx1 - gx0
            eta_time_local[i] = dt * (np.linalg.norm(diff_gp) + np.linalg.norm(diff_gx))
        eta_time = np.max(eta_time_local) if N > 0 else 0.0
        # PA error: integrate (Hbar - H)
        eta_PA = 0.0
        for i in range(N):
            # at node i and i+1, compute gap
            Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
            Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])
            # compute true H (restricted) at i and i+1
            H_i, _ = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True)
            H_ip1, _ = compute_H(problem, P[i + 1], X[i + 1], t_nodes[i + 1], bundle.controls, restricted=True)
            gap_i = Hbar_i - H_i
            gap_ip1 = Hbar_ip1 - H_ip1
            dt = t_nodes[i + 1] - t_nodes[i]
            eta_PA += 0.5 * (gap_i + gap_ip1) * dt
        # smoothing error: integrate (H_delta - Hbar)
        eta_delta = 0.0
        for i in range(N):
            Hdelta_i, _, _ = eval_H_smooth(problem, bundle, P[i], X[i], t_nodes[i], delta)
            Hdelta_ip1, _, _ = eval_H_smooth(problem, bundle, P[i + 1], X[i + 1], t_nodes[i + 1], delta)
            Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
            Hbar_ip1, _ = bundle.evaluate(problem, P[i + 1], X[i + 1], t_nodes[i + 1])
            diff_i = Hdelta_i - Hbar_i
            diff_ip1 = Hdelta_ip1 - Hbar_ip1
            dt = t_nodes[i + 1] - t_nodes[i]
            eta_delta += 0.5 * (diff_i + diff_ip1) * dt
        log.append({
            'iteration': k,
            'N': N,
            'M': bundle.num_planes(),
            'delta': delta,
            'eta_time': eta_time,
            'eta_PA': eta_PA,
            'eta_delta': eta_delta,
            'newton_iter': info['iterations'],
            'newton_residual': info['residual_norm'],
        })
        # check convergence
        if (eta_time <= tol_time) and (eta_PA <= tol_PA) and (eta_delta <= tol_delta):
            break
        # priority: refine time first, then PA planes, then reduce delta
        if eta_time > tol_time:
            # refine time mesh: subdivide intervals with high local error
            new_nodes = [t_nodes[0]]
            X_new = [X[0]]
            P_new = [P[0]]
            for i in range(N):
                dt = t_nodes[i + 1] - t_nodes[i]
                # compute midpoint and error indicator
                err = eta_time_local[i]
                if err > tol_time:
                    # insert midpoint
                    t_mid = 0.5 * (t_nodes[i] + t_nodes[i + 1])
                    # linear interpolate X and P
                    alpha = (t_mid - t_nodes[i]) / dt
                    x_mid = (1 - alpha) * X[i] + alpha * X[i + 1]
                    p_mid = (1 - alpha) * P[i] + alpha * P[i + 1]
                    new_nodes.extend([t_mid])
                    X_new.extend([x_mid])
                    P_new.extend([p_mid])
                new_nodes.append(t_nodes[i + 1])
                X_new.append(X[i + 1])
                P_new.append(P[i + 1])
            t_nodes = np.array(new_nodes, dtype=float)
            X_guess = np.array(X_new)
            P_guess = np.array(P_new)
            continue
        if eta_PA > tol_PA:
            # add new plane: find worst gap index
            max_gap = -np.inf
            max_idx = 0
            for i in range(N + 1):
                Hbar_i, _ = bundle.evaluate(problem, P[i], X[i], t_nodes[i])
                H_i, u_star = compute_H(problem, P[i], X[i], t_nodes[i], bundle.controls, restricted=True)
                gap = Hbar_i - H_i
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
                    best_u = u_star
            # add best_u to bundle
            if best_u is not None:
                bundle.add_control(best_u)
            X_guess = X
            P_guess = P
            continue
        # else reduce delta
        if eta_delta > tol_delta:
            delta = delta * 0.5
            # do not change mesh or bundle
            X_guess = X
            P_guess = P
            continue
    # return final solution and log
    return {
        't_nodes': t_nodes,
        'X': X,
        'P': P,
        'bundle': bundle,
        'delta': delta,
        'log': log
    }