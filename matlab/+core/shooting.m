function R = shooting_residual(problem, bundle, delta, t_nodes, z)
%SHOOTING_RESIDUAL Residual of the TPBVP for a packed unknown vector.
%   R = shooting_residual(problem, bundle, delta, t_nodes, z) constructs
%   state and costate trajectories from the unknown vector z and
%   computes the residual vector using assemble_residual.  The initial
%   state x0 is taken from problem.x0 and is not part of z.

n = length(problem.x0);
N = length(t_nodes) - 1;
[X, P] = core.integrators.unpack_unknowns(z, n, N);
% Insert fixed initial state
X(:,1) = problem.x0(:);
R = core.integrators.assemble_residual(problem, bundle, delta, t_nodes, X, P);
end

function J = shooting_jacobian(problem, bundle, delta, t_nodes, z)
%SHOOTING_JACOBIAN Jacobian of the residual with respect to unknown vector.
%   Uses finite difference assembly from integrators.  The initial
%   state x0 is fixed and excluded from z.

n = length(problem.x0);
N = length(t_nodes) - 1;
[X, P] = core.integrators.unpack_unknowns(z, n, N);
X(:,1) = problem.x0(:);
% Compute Jacobian via integrators.  assemble_jacobian assumes x0 is in
% X(:,1), so we pass X with x0 and P.  It will pack unknowns in the same
% order as z.
J = core.integrators.assemble_jacobian(problem, bundle, delta, t_nodes, X, P);
end