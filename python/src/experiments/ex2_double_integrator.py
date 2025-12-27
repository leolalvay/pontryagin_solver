"""
Example 2: Minimum-time double integrator with box state constraint.

The problem seeks to transfer the double integrator from x(0) = [-1,0] to
x(T) = [0,0] in minimum time subject to bounded control |u| ≤ 1 and state
constraint x1 ≤ 0.  A soft terminal penalty encourages reaching the target.
"""
import numpy as np

from core.problem import OCPProblem
from core.adaptivity import solve_optimal_control
from core.hamiltonian import compute_H


def run_example():
    # dynamics: double integrator
    def dynamics(x, u, t):
        # x = [x1, x2]
        return np.array([x[1], u[0]])
    # running cost: approximate minimum time with small quadratic control cost
    def stage_cost(x, u, t):
        return 1.0 + 0.01 * (u[0] ** 2)
    # target and penalty
    target = np.array([0.0, 0.0])
    penalty_weight = 100.0
    def terminal_cost(x):
        diff = x - target
        return penalty_weight * diff.dot(diff)
    # initial state and horizon guess
    x0 = np.array([-1.0, 0.0])
    T = 2.0
    # bounds on control [-1,1]
    u_min = np.array([-1.0])
    u_max = np.array([1.0])
    # state constraint: x1 <= 0; encode as bounds [-∞, 0] for x1, none for x2
    x_min = np.array([-np.inf, -np.inf])
    x_max = np.array([0.0, np.inf])
    # create problem
    prob = OCPProblem(dynamics, stage_cost, terminal_cost, x0, T,
                      control_bounds=(u_min, u_max), state_bounds=(x_min, x_max))
    # initial mesh: uniform 20 segments
    t_nodes = np.linspace(0.0, T, 21)
    # solve adaptively
    result = solve_optimal_control(prob, t_nodes, tol_time=1e-3, tol_PA=1e-3, tol_delta=1e-3, max_iters=10, delta0=0.2)
    # extract solution
    X = result['X']
    P = result['P']
    mesh = result['t_nodes']
    bundle = result['bundle']
    # approximate controls
    controls = []
    for i in range(len(mesh)):
        _, u_star = compute_H(prob, P[i], X[i], mesh[i], bundle.controls, restricted=True)
        controls.append(u_star)
    controls = np.asarray(controls)
    # approximate final time (time when x1 crosses zero)
    final_time = mesh[-1]
    for i in range(len(mesh) - 1):
        if X[i, 0] <= 0 <= X[i + 1, 0]:
            # linear interpolation
            alpha = (0.0 - X[i, 0]) / (X[i + 1, 0] - X[i, 0] + 1e-12)
            final_time = mesh[i] + alpha * (mesh[i + 1] - mesh[i])
            break
    print("Double Integrator Example")
    print(f"Mesh points: {len(mesh)}")
    print(f"Planes: {bundle.num_planes()}")
    print(f"Estimated final time: {final_time}")
    print("Indicator history:")
    for entry in result['log']:
        print(entry)
    return result


if __name__ == '__main__':
    run_example()