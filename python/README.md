# Adaptive Pontryagin Solver (Python)

This repository implements a generic solver for optimal control problems based on
Pontryagin's Maximum Principle (PMP) with an adaptive discretisation
and smoothing strategy.  The solver is designed to handle problems with
nonsmooth Hamiltonians through a piecewise‑affine (PA) bundle surrogate and
a log‑sum‑exp smoothing function.  An outer adaptivity loop refines the time
mesh, adds new controls to the bundle, and reduces the smoothing parameter
until a posteriori error indicators fall below user‑specified tolerances.

## Repository Structure

```
python/
  README.md            # this file
  src/
    core/
      problem.py       # problem definition (dynamics, cost, constraints)
      pa_bundle.py     # PA bundle surrogate (control candidates)
      hamiltonian.py   # Hamiltonian evaluation routines
      smoothing.py     # smoothed Hamiltonian and gradients via log-sum-exp
      integrators.py   # symplectic Euler residual/Jacobian assembly
      shooting.py      # wrapper for residual/Jacobian using packed vectors
      newton.py        # damped Newton solver with Armijo line search
      adaptivity.py    # adaptive outer loop implementing Algorithm 3.1
      constraints.py   # clipping utilities for states and controls
      utils.py         # miscellaneous helper functions
    experiments/
      ex1_lqr.py       # Example 1: LQR problem
      ex2_double_integrator.py  # Example 2: minimum‑time double integrator
      ex3_dubins.py    # Example 3: Dubins car
```

## Usage

All dependencies are part of the Python standard library except for
NumPy, which is required.  To run an example from the command line:

```bash
cd python/src
python -m experiments.ex1_lqr
python -m experiments.ex2_double_integrator
python -m experiments.ex3_dubins
```

Each example constructs an `OCPProblem`, sets up an initial time mesh and
tolerances, calls the adaptive solver, and prints a summary of the solution
including the number of time nodes, number of control planes in the bundle,
and the history of error indicators per outer iteration.

## Notes

- This implementation follows the design laid out in the DeepResearch
  blueprint.  In particular, the solver uses a symplectic Euler
  discretisation of the canonical system, a damped Newton method to solve
  the resulting TPBVP, and a posteriori indicators to guide adaptivity.
- While the code aims to be robust, it is intended primarily as an
  educational reference and may require tuning of tolerances and initial
  guesses for challenging problems.