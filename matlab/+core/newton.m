function [X, P, info] = solve_tpbvp(problem, bundle, delta, t_nodes, Xinit, Pinit, options)
%SOLVE_TPBVP Solve the canonical TPBVP using damped Newton method.
%   [X,P,info] = solve_tpbvp(problem, bundle, delta, t_nodes, Xinit, Pinit, options)
%   solves the two‑point boundary value problem defined by the
%   Pontryagin minimum principle discretised via symplectic Euler and
%   smoothed Hamiltonian.  The state X and costate P are returned as
%   matrices of size n×(N+1).  options fields:
%       tol      – stopping tolerance for residual norm (default 1e-8)
%       maxIter  – maximum Newton iterations (default 50)
%       alpha    – Armijo slope reduction parameter (default 1e-4)
%       beta     – step reduction factor in line search (default 0.5)
%       lam_min  – minimum step length for line search (default 1e-8)
%   info contains diagnostic fields (iterations, final residual etc.).

n  = length(problem.x0);
N  = length(t_nodes) - 1;
% Flatten initial guess into z.  Ensure Xinit(:,1) equals x0
if ~isequal(Xinit(:,1), problem.x0(:))
    Xinit(:,1) = problem.x0(:);
end
z = core.integrators.pack_unknowns(Xinit, Pinit);
% Parse options
if nargin < 7
    options = struct();
end
tol = get_option(options, 'tol', 1e-8);
maxIter = get_option(options, 'maxIter', 50);
alpha = get_option(options, 'alpha', 1e-4);
beta = get_option(options, 'beta', 0.5);
lam_min = get_option(options, 'lam_min', 1e-8);

info.iterations = 0;
info.residual_norms = [];
for k = 1:maxIter
    info.iterations = k;
    R = core.shooting.shooting_residual(problem, bundle, delta, t_nodes, z);
    res_norm = norm(R, Inf);
    info.residual_norms(end+1) = res_norm;
    if res_norm < tol
        break;
    end
    % Assemble Jacobian
    J = core.shooting.shooting_jacobian(problem, bundle, delta, t_nodes, z);
    % Solve linear system J * dz = -R
    % Use MATLAB backslash for efficiency; may use sparse J if large
    dz = -J \ R;
    % Line search on Newton step
    lam = 1.0;
    f0 = res_norm;
    z_new = z + lam * dz;
    % Evaluate new residual
    R_new = core.shooting.shooting_residual(problem, bundle, delta, t_nodes, z_new);
    f_new = norm(R_new, Inf);
    % Armijo condition: f_new <= (1 - alpha*lam) * f0
    while f_new > (1 - alpha * lam) * f0
        lam = beta * lam;
        if lam < lam_min
            break;
        end
        z_new = z + lam * dz;
        R_new = core.shooting.shooting_residual(problem, bundle, delta, t_nodes, z_new);
        f_new = norm(R_new, Inf);
    end
    z = z_new;
end
[X, P] = core.integrators.unpack_unknowns(z, n, N);
X(:,1) = problem.x0(:);
info.final_residual = info.residual_norms(end);
end

function val = get_option(opts, name, default)
if isfield(opts, name)
    val = opts.(name);
else
    val = default;
end
end