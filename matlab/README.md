# Smoothed PMP Solver (MATLAB)

This repository contains a MATLAB implementation of the adaptive optimal control solver described in the DeepResearch blueprint.  The solver follows Pontryagin's Maximum Principle with a universally smoothed Hamiltonian, an adaptive time mesh, and a piecewise‐affine (PA) bundle surrogate to handle nonsmooth Hamiltonians.  It provides a flexible framework to solve a wide range of Bolza optimal control problems with optional control and state constraints.

## Contents

- `+core/` – Core solver modules:
  - `problem.m` – Defines the optimal control problem structure (dynamics, costs, bounds, etc.).
  - `hamiltonian.m` – Implements true and restricted Hamiltonian evaluation.
  - `pa_bundle.m` – Manages the piecewise‐affine surrogate bundle.
  - `smoothing.m` – Implements the log–sum–exp smoothing of the Hamiltonian.
  - `integrators.m` – Provides symplectic Euler discretisation, residuals and Jacobian assembly.
  - `shooting.m` – Assembles the full multiple–shooting residual and Jacobian.
  - `newton.m` – Damped Newton solver with Armijo line search for the TPBVP.
  - `constraints.m` – Utilities for enforcing state constraints and viability.
  - `adaptivity.m` – Orchestrates the outer adaptive loop (Algorithm 3.1): refines time mesh, adds planes and reduces smoothing according to a posteriori error indicators.
  - `utils.m` – Miscellaneous helper functions (currently unused placeholder).
- `+experiments/` – Scripts demonstrating the solver on three examples:
  - `ex1_lqr.m` – Linear quadratic regulator (LQR) benchmark; smooth problem with analytic solution.
  - `ex2_double_integrator.m` – Minimum‐time double integrator with box constraint; exhibits bang–bang control.
  - `ex3_dubins.m` – Dubins car with bounded turn rate; a nonlinear problem with one control switch.
- `figs/` – Recommended location to save plots produced by the example scripts.

## Usage

1. **Prepare MATLAB environment:** Ensure you are using MATLAB R2023a or later.  Clone or download this repository and add the `matlab` directory to your MATLAB path.

2. **Run an example:** From the MATLAB command window, navigate to the `matlab` folder and execute one of the experiment scripts.  For instance:

```matlab
% Run Example 1 (LQR)
result = experiments.ex1_lqr();

% Run Example 2 (Minimum‐time double integrator)
result = experiments.ex2_double_integrator();

% Run Example 3 (Dubins car)
result = experiments.ex3_dubins();
```

Each script constructs the problem, invokes the adaptive solver, prints a summary of the results and generates plots of the state and control trajectories.  The returned `result` struct contains fields:

- `t` – time nodes of the final adaptive mesh;
- `X` – state trajectory (n×(N+1));
- `P` – costate trajectory (n×(N+1));
- `u` – control sequence (m×(N+1));
- `bundle` – final PA bundle (with method `num_planes()`);
- `delta` – final smoothing parameter;
- `log` – history of iteration diagnostics (mesh size, plane count, errors);
- `iterations` – number of outer adaptivity iterations performed.

3. **Interpreting results:**
   - *Example 1* should reproduce the known LQR optimal cost (around 1.074) and require minimal adaptivity (one plane and modest number of time steps).
   - *Example 2* demonstrates a bang–bang control with one switch near `t = 1.0`.  The approximate final time reported by the script should be close to 2.0 seconds.  The adaptive solver will refine the mesh around the switching point and add two control planes (±1).
   - *Example 3* solves a nonlinear Dubins car problem.  The trajectory comprises two curved arcs with a single change in steering direction.  The final state should match the target position `(1,1)` and heading `π/2` within the solver tolerance.

## Notes

- The solver uses a smoothed Hamiltonian (log–sum–exp) to enable gradient‐based Newton steps.  The smoothing parameter `delta` is progressively halved once time and PA surrogate errors are sufficiently small.
- Error tolerances for time discretisation, PA model and smoothing bias are set to `1e-3` by default but can be adjusted via parameters to `solve_optimal_control`.
- The implementation uses simple finite differences for some derivatives.  For production use one may provide analytic derivatives for improved accuracy and efficiency.
- Although the example problems use quadratic terminal penalties for convenience, the solver supports any smooth terminal cost function via the `terminal_cost_func` of `OCPProblem`.

For further details on the algorithmic design, refer to the DeepResearch blueprint (Sections A–H) included with this repository.