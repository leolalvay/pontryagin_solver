# `core/integrators.py` — Symplectic Euler residual + finite-difference Jacobian

This module turns the **δ-smoothed Pontryagin system** into a **nonlinear algebraic system**
$$
F(z)=0
$$
that can be solved by Newton (via the shooting layer). The key is that we do **not** “integrate forward” as a standalone simulator; instead, we **assemble residual blocks** that enforce the symplectic Euler updates and the terminal boundary condition.

---

## 1) Continuous system being discretized (what we want to enforce)

Using the smoothed Hamiltonian $H_\delta(p,x,t)$, the canonical PMP dynamics are
$$
\dot x(t)=\nabla_p H_\delta(p(t),x(t),t),
\qquad
-\dot p(t)=\nabla_x H_\delta(p(t),x(t),t),
$$
with boundary conditions
$$
x(0)=x_0,
\qquad
p(T)+\nabla g(x(T))=0.
$$

In the implementation, $\nabla_p H_\delta$ and $\nabla_x H_\delta$ are obtained from
`eval_H_smooth(...)`, and $\nabla g$ is approximated by finite differences.

---

## 2) Unknown vector $z$ (how the solver stores $(X,P)$)

On a mesh $0=t_0<\dots<t_N=T$, we store samples
$$
X_i\approx x(t_i),\qquad P_i\approx p(t_i).
$$

The initial state $X_0=x_0$ is **fixed**, so the unknown vector concatenates
$$
z = (X_1,\dots,X_N,\;P_0,\dots,P_N).
$$

This is exactly what `pack_unknowns` does:

```python
def pack_unknowns(X, P):
    N_plus_1, n = X.shape
    N = N_plus_1 - 1
    z = np.zeros((N * n + (N + 1) * n,)) #z=(X_1,..,P_N)
    z[0:N * n] = X[1:, :].reshape(N * n)      # x_1,...,x_N
    z[N * n:]  = P.reshape((N + 1) * n)       # p_0,...,p_N
    return z
```

And `unpack_unknowns` reconstructs $(X,P)$ from $z$ while reinserting $X_0=x_0$:

```python
def unpack_unknowns(z, x0):
    n = x0.size
    total = z.size // n
    N = (total - 1) // 2
    X = np.zeros((N + 1, n))
    P = np.zeros((N + 1, n))
    X[0, :]   = x0
    X[1:, :]  = z[0:N * n].reshape((N, n))
    P[:, :]   = z[N * n:].reshape((N + 1, n))
    return X, P
```

---

## 3) Symplectic Euler discretization (residual blocks)

For each step $i=0,\dots,N-1$ with $\Delta t_i=t_{i+1}-t_i$, symplectic Euler is enforced as:

**State update (gradient at the start):**
$$
X_{i+1} = X_i + \Delta t_i\,\nabla_p H_\delta(P_i,X_i,t_i).
$$

**Costate update (gradient at the end):**
$$
P_i = P_{i+1} + \Delta t_i\,\nabla_x H_\delta(P_{i+1},X_{i+1},t_{i+1}).
$$

The code enforces these by assembling residuals
$$
r_x^{(i)} = X_i + \Delta t_i\,\nabla_p H_\delta(P_i,X_i,t_i) - X_{i+1},
$$
$$
r_p^{(i)} = P_{i+1} + \Delta t_i\,\nabla_x H_\delta(P_{i+1},X_{i+1},t_{i+1}) - P_i.
$$

Here is the exact implementation pattern inside `assemble_residual`:

```python
for i in range(N):
    dt = t_nodes[i + 1] - t_nodes[i]
    x_i, x_ip1 = X[i], X[i + 1]
    p_i, p_ip1 = P[i], P[i + 1]

    # gradient at start (for state update)
    _, grad_p_i, _ = eval_H_smooth(problem, bundle, p_i,   x_i,   t_nodes[i],     delta)

    # gradient at end (for costate update)
    _, _, grad_x_ip1 = eval_H_smooth(problem, bundle, p_ip1, x_ip1, t_nodes[i + 1], delta)

    r_x = x_i   + dt * grad_p_i   - x_ip1
    r_p = p_ip1 + dt * grad_x_ip1 - p_i
```

So the “math-to-code” correspondence is direct: each residual block is literally the discretized equation moved to the left-hand side.

---

## 4) Terminal boundary condition $p(T)+\nabla g(x(T))=0$

At the final node $(X_N,P_N)$, the code appends the boundary residual
$$
r_{\mathrm{bc}} = P_N + \nabla g(X_N).
$$

Since the problem object exposes $g(x)$, the gradient is computed by central finite differences:

```python
x_N = X[-1]
p_N = P[-1]
g_grad = np.zeros_like(p_N)
eps = 1e-6
for j in range(n):
    x_plus  = x_N.copy(); x_plus[j]  += eps
    x_minus = x_N.copy(); x_minus[j] -= eps
    g_plus  = problem.g(x_plus)
    g_minus = problem.g(x_minus)
    g_grad[j] = (g_plus - g_minus) / (2 * eps)

r_bc = p_N + g_grad
```

---

## 5) Jacobian assembly (finite differences on the full residual map)

Newton needs a Jacobian $J\approx \partial F/\partial z$. Here it is computed by finite differences on $F(z)$ itself:

$$
J_{:,j} \approx \frac{F(z+\varepsilon e_j)-F(z-\varepsilon e_j)}{2\varepsilon}.
$$

Implementation outline in `assemble_jacobian`:

```python
z = pack_unknowns(X, P)

def res_fun(z_vec):
    X_new, P_new = unpack_unknowns(z_vec, X[0])
    return assemble_residual(problem, t_nodes, X_new, P_new, bundle, delta)

F0 = res_fun(z)
J = np.zeros((F0.size, z.size))
eps = 1e-6

for j in range(z.size):
    z_plus  = z.copy(); z_plus[j]  += eps
    z_minus = z.copy(); z_minus[j] -= eps
    J[:, j] = (res_fun(z_plus) - res_fun(z_minus)) / (2 * eps)
```

This is expensive (it requires two residual evaluations per unknown), but it is simple and robust for a first version and small-to-moderate problem sizes.

---
