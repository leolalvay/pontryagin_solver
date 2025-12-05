function M = integrators()
%INTEGRATORS Module factory (Python-like).
M.pack_unknowns     = @pack_unknowns;
M.unpack_unknowns   = @unpack_unknowns;
M.assemble_residual = @assemble_residual;
M.assemble_jacobian = @assemble_jacobian;
end

function R = assemble_residual(problem, bundle, delta, t_nodes, X, P)
%ASSEMBLE_RESIDUAL Assemble the residual vector for the canonical system.
%   R = assemble_residual(problem, bundle, delta, t_nodes, X, P) returns
%   the residual vector of length 2*n*N + n for the symplectic Euler
%   discretization.  X and P are n×(N+1) matrices of state and costate
%   nodes, respectively.  The time mesh is given by t_nodes (1×(N+1)).
%   See Section 6.5 of the DeepResearch plan for details.

n = size(X,1);
N = length(t_nodes) - 1;
R = zeros(2*n*N + n, 1);
idx = 1;
for i = 1:N
    dt = t_nodes(i+1) - t_nodes(i);
    xi   = X(:,i);
    pi   = P(:,i);
    xip1 = X(:,i+1);
    pip1 = P(:,i+1);
    ti   = t_nodes(i);
    tip1 = t_nodes(i+1);
    % Evaluate gradients of smooth Hamiltonian at i and i+1
    [~, gradp_i, ~] = core.smoothing().eval_H_smooth(bundle, problem, pi, xi, ti, delta);
    [~, ~, gradx_ip1] = core.smoothing().eval_H_smooth(bundle, problem, pip1, xip1, tip1, delta);
    % Residuals: r_x = xi + dt*gradp_i - x_{i+1}; r_p = p_{i+1} + dt*gradx_{i+1} - p_i
    r_x = xi + dt * gradp_i - xip1;
    r_p = pip1 + dt * gradx_ip1 - pi;
    R(idx:idx+n-1)     = r_x;
    R(idx+n:idx+2*n-1) = r_p;
    idx = idx + 2*n;
end
% Terminal boundary condition: p(T) + ∂g/∂x(x(T)) = 0
xT = X(:,N+1);
pT = P(:,N+1);
grad_g = gradient_terminal_cost(problem, xT, t_nodes(end));
R(end-n+1:end) = pT + grad_g;
end

function grad_g = gradient_terminal_cost(problem, xT, T)
%GRADIENT_TERMINAL_COST Approximate gradient of terminal cost g w.r.t x.
%   Finite difference approximation of ∂g/∂x at xT.
n = length(xT);
grad_g = zeros(n,1);
eps_fd = 1e-6;
base = problem.terminal_cost(xT, T);
for j = 1:n
    e = zeros(n,1);
    e(j) = 1;
    fwd = problem.terminal_cost(xT + eps_fd * e, T);
    grad_g(j) = (fwd - base) / eps_fd;
end
end

function J = assemble_jacobian(problem, bundle, delta, t_nodes, X, P)
%ASSEMBLE_JACOBIAN Assemble Jacobian matrix of the residual via finite difference.
%   J = assemble_jacobian(problem, bundle, delta, t_nodes, X, P) returns
%   the Jacobian of assemble_residual with respect to unknown variables.
%   Unknowns consist of X(:,2:N+1) and P(:,1:N+1).  x0 is fixed.  The
%   total number of unknowns is n*N + n*(N+1) = n*(2*N+1).  The Jacobian
%   is returned as a dense matrix of size (2*n*N+n) × (n*(2*N+1)).

n = size(X,1);
N = length(t_nodes) - 1;
% Pack unknown vector
z = pack_unknowns(X, P);
R0 = assemble_residual(problem, bundle, delta, t_nodes, X, P);
m = length(z);
k = length(R0);
J = zeros(k, m);
eps_fd = 1e-6;
for j = 1:m
    z_pert = z;
    z_pert(j) = z_pert(j) + eps_fd;
    [Xpert, Ppert] = unpack_unknowns(z_pert, n, N);
    Rj = assemble_residual(problem, bundle, delta, t_nodes, Xpert, Ppert);
    J(:,j) = (Rj - R0) / eps_fd;
end
end

function z = pack_unknowns(X, P)
%PACK_UNKNOWNS Flatten the unknown state and costate variables into a vector.
%   Unknowns include x1..xN and p0..pN.  x0 is excluded.
n = size(X,1);
N = size(X,2) - 1;
z = zeros(n*(2*N+1),1);
pos = 1;
% x_1 .. x_N
for i = 2:N+1
    z(pos:pos+n-1) = X(:,i);
    pos = pos + n;
end
% p_0 .. p_N
for i = 1:N+1
    z(pos:pos+n-1) = P(:,i);
    pos = pos + n;
end
end

function [X, P] = unpack_unknowns(z, n, N)
%UNPACK_UNKNOWNS Reconstruct state and costate matrices from vector z.
X = zeros(n, N+1);
P = zeros(n, N+1);
pos = 1;
% x_0 will be set by caller; here we fill x1..xN leaving x0 untouched
for i = 2:N+1
    X(:,i) = z(pos:pos+n-1);
    pos = pos + n;
end
% p_0 .. p_N
for i = 1:N+1
    P(:,i) = z(pos:pos+n-1);
    pos = pos + n;
end
end