function result = adaptivity(problem, initial_mesh, tol_time, tol_PA, tol_delta, max_iters)
%SOLVE_OPTIMAL_CONTROL Adaptive solver for optimal control problems.
%   result = solve_optimal_control(problem, initial_mesh, tol_time, tol_PA, tol_delta, max_iters)
%   runs the outer adaptivity loop described in Algorithm 3.1 of the
%   DeepResearch plan.  It iteratively solves the TPBVP, computes
%   a posteriori error indicators for time discretisation, PA surrogate
%   error and smoothing error, and refines the discretisation until
%   tolerances are met.  The returned struct 'result' contains the
%   final trajectories X,P, control sequence u, time mesh and logs.

% Set default tolerances if missing
if nargin < 3 || isempty(tol_time);  tol_time = 1e-3; end
if nargin < 4 || isempty(tol_PA);    tol_PA   = 1e-3; end
if nargin < 5 || isempty(tol_delta); tol_delta= 1e-3; end
if nargin < 6 || isempty(max_iters); max_iters = 20; end

% Initialise mesh, bundle and smoothing parameter
t_nodes = initial_mesh(:)';
if t_nodes(1) ~= 0
    error('Initial mesh must start at t=0');
end
% If final time unspecified in problem, use last node of initial mesh
if isempty(problem.T)
    problem.T = t_nodes(end);
elseif abs(problem.T - t_nodes(end)) > 1e-12
    % adjust mesh to end at specified T
    t_nodes(end) = problem.T;
end
% Initialise PA bundle with one control candidate: zero or lower bound
m = length(problem.getControlBounds());
[amin, amax] = problem.getControlBounds();
if isempty(amin)
    init_ctrl = zeros(0,1);
else
    init_ctrl = (amin + amax)/2;
end
bundle = core.PABundle(init_ctrl);
% Initial smoothing parameter delta
delta = 0.1;
% Initial guess for state and costate trajectories
n = length(problem.x0);
N = length(t_nodes) - 1;
Xinit = repmat(problem.x0(:), 1, N+1);
Pinit = zeros(n, N+1);

log = struct();
log.iter = [];
log.N  = [];
log.M  = [];
log.delta = [];
log.eta_time = [];
log.eta_PA   = [];
log.eta_delta= [];

for k = 1:max_iters
    % Solve TPBVP on current discretisation
    opts = struct('tol', 1e-8, 'maxIter', 50, 'alpha', 1e-4, 'beta', 0.5, 'lam_min', 1e-8);
    [X, P, info] = core.newton().solve_tpbvp(problem, bundle, delta, t_nodes, Xinit, Pinit, opts);
    % Compute a posteriori error indicators
    [eta_time, local_time_err] = estimate_time_error(problem, bundle, delta, t_nodes, X, P);
    eta_PA   = estimate_PA_error(problem, bundle, t_nodes, X, P);
    eta_delta= estimate_smoothing_error(problem, bundle, delta, t_nodes, X, P);
    % Log iteration info
    log.iter(end+1)     = k;
    log.N(end+1)        = length(t_nodes)-1;
    log.M(end+1)        = bundle.num_planes();
    log.delta(end+1)    = delta;
    log.eta_time(end+1) = eta_time;
    log.eta_PA(end+1)   = eta_PA;
    log.eta_delta(end+1)= eta_delta;
    % Check stopping criteria
    if (eta_time <= tol_time) && (eta_PA <= tol_PA) && (eta_delta <= tol_delta)
        break;
    end
    % Determine refinement: priority: time, PA, delta
    if eta_time > tol_time
        % Refine mesh: split intervals where local error > tol_time
        old_nodes = t_nodes;
        t_nodes = refine_time_mesh(t_nodes, local_time_err, tol_time);
        % Interpolate X and P from old mesh to new mesh by simple linear interpolation
        [Xinit, Pinit] = interpolate_to_new_mesh(t_nodes, X, P, old_nodes);
    elseif eta_PA > tol_PA
        % Add plane: find worst gap and add control
        [t_idx, a_new] = find_worst_plane(problem, bundle, t_nodes, X, P);
        bundle = bundle.add_control(a_new);
        % Reuse current X,P as initial guess
        Xinit = X;
        Pinit = P;
    elseif eta_delta > tol_delta
        % Reduce smoothing
        delta = delta / 2;
        % Reuse current X,P
        Xinit = X;
        Pinit = P;
    else
        % Should not reach; but ensure progress
        delta = delta / 2;
        Xinit = X;
        Pinit = P;
    end
end

% Extract control trajectory
N = length(t_nodes)-1;
m = size(bundle.controls,1);
u = zeros(m, N+1);
for i = 1:N+1
    p_i = P(:,i);
    x_i = X(:,i);
    t_i = t_nodes(i);
    [~, a_star] = core.hamiltonian.compute_H(problem, p_i, x_i, t_i, bundle, false);
    u(:,i) = a_star;
end

% Package result
result = struct();
result.t   = t_nodes;
result.X   = X;
result.P   = P;
result.u   = u;
result.bundle = bundle;
result.delta  = delta;
result.log    = log;
result.iterations = k;
end

function [eta_time, local_err] = estimate_time_error(problem, bundle, delta, t_nodes, X, P)
%ESTIMATE_TIME_ERROR Estimate the time discretisation error.
%   Returns a global indicator eta_time and per‑interval error local_err.
N = length(t_nodes) - 1;
n = size(X,1);
local_err = zeros(1, N);
% Use difference of gradient of H_delta across interval as proxy for
% curvature; larger changes suggest insufficient time resolution.
for i = 1:N
    ti   = t_nodes(i);
    tip1 = t_nodes(i+1);
    xi   = X(:,i);
    xip1 = X(:,i+1);
    pi   = P(:,i);
    pip1 = P(:,i+1);
    [~, gradp_i, ~] = core.smoothing.eval_H_smooth(bundle, problem, pi, xi, ti, delta);
    [~, gradp_ip1, ~] = core.smoothing.eval_H_smooth(bundle, problem, pip1, xip1, tip1, delta);
    dt = tip1 - ti;
    % Error estimate: change in gradient multiplied by dt
    local_err(i) = norm(gradp_ip1 - gradp_i, 2) * dt;
end
% Global error indicator: maximum local error
eta_time = max(local_err);
end

function eta_PA = estimate_PA_error(problem, bundle, t_nodes, X, P)
%ESTIMATE_PA_ERROR Estimate the model (bundle) error.
%   eta_PA = ∫ (\bar{H} - H) dt approximated by trapezoid rule.
N = length(t_nodes) - 1;
vals = zeros(1, N+1);
for i = 1:N+1
    p_i = P(:,i);
    x_i = X(:,i);
    t_i = t_nodes(i);
    % Surrogate Hbar
    [Hbar, ~] = bundle.eval(problem, p_i, x_i, t_i);
    % True H_K (state constraints) – restrict to tangent cone
    [Htrue, ~] = core.hamiltonian.compute_H(problem, p_i, x_i, t_i, bundle, true);
    vals(i) = Hbar - Htrue;
end
% Integrate by trapezoidal rule
eta_PA = sum(0.5 * (vals(1:end-1) + vals(2:end)) .* diff(t_nodes));
end

function eta_delta = estimate_smoothing_error(problem, bundle, delta, t_nodes, X, P)
%ESTIMATE_SMOOTHING_ERROR Estimate the smoothing bias error.
%   eta_delta = ∫ (H_delta - \bar{H}) dt approximated by trapezoid rule.
N = length(t_nodes) - 1;
vals = zeros(1, N+1);
for i = 1:N+1
    p_i = P(:,i);
    x_i = X(:,i);
    t_i = t_nodes(i);
    [Hdelta, ~, ~] = core.smoothing.eval_H_smooth(bundle, problem, p_i, x_i, t_i, delta);
    [Hbar, ~] = bundle.eval(problem, p_i, x_i, t_i);
    vals(i) = Hdelta - Hbar;
end
eta_delta = sum(0.5 * (vals(1:end-1) + vals(2:end)) .* diff(t_nodes));
end

function new_mesh = refine_time_mesh(t_nodes, local_err, tol)
%REFINE_TIME_MESH Refine time mesh based on local error indicators.
%   Splits intervals where the local error exceeds tol by inserting the
%   midpoint.  Returns the new sorted mesh.
N = length(t_nodes) - 1;
new_mesh = t_nodes;
for i = 1:N
    if local_err(i) > tol
        mid = (t_nodes(i) + t_nodes(i+1)) / 2;
        new_mesh = [new_mesh, mid];
    end
end
new_mesh = sort(new_mesh);
% Remove duplicates (within numerical tolerance)
tol_time = 1e-12;
new_mesh = unique(round(new_mesh/ tol_time)*tol_time);
end

function [Xnew, Pnew] = interpolate_to_new_mesh(t_new, X, P, t_old)
%INTERPOLATE_TO_NEW_MESH Interpolate X and P to a refined mesh.
%   [Xnew, Pnew] = interpolate_to_new_mesh(t_new, X, P, t_old) returns
%   the state and costate trajectories Xnew and Pnew on the new mesh
%   `t_new`, given the trajectories X and P on the old mesh `t_old`.
%   Linear interpolation is used for each state and costate component.

    n = size(X,1);
    Xnew = zeros(n, length(t_new));
    Pnew = zeros(n, length(t_new));
    for j = 1:n
        Xnew(j,:) = interp1(t_old, X(j,:), t_new, 'linear');
        Pnew(j,:) = interp1(t_old, P(j,:), t_new, 'linear');
    end
    % Ensure exact x0 at t=0 if present in old mesh
    if abs(t_new(1) - t_old(1)) < 1e-12
        Xnew(:,1) = X(:,1);
    end
end

function [t_idx, a_new] = find_worst_plane(problem, bundle, t_nodes, X, P)
%FIND_WORST_PLANE Identify time index and control that maximally reduces gap.
%   Evaluates Hbar - Htrue at each node and returns the index of the
%   largest gap along with the true minimizer a_new to add as plane.
N = length(t_nodes) - 1;
gaps = zeros(1, N+1);
a_cands = zeros(length(bundle.controls(:,1)), N+1);
for i = 1:N+1
    p_i = P(:,i);
    x_i = X(:,i);
    t_i = t_nodes(i);
    [Hbar, ~] = bundle.eval(problem, p_i, x_i, t_i);
    [Htrue, a_star] = core.hamiltonian.compute_H(problem, p_i, x_i, t_i, bundle, true);
    gaps(i) = Hbar - Htrue;
    a_cands(:,i) = a_star;
end
[~, t_idx] = max(gaps);
a_new = a_cands(:,t_idx);
end