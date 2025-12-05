function result = ex2_double_integrator()
%EX2_DOUBLE_INTEGRATOR Run Example 2: Minimum‑time double integrator with box constraint.
%   This script sets up the minimum‑time problem for a double integrator
%   with control bound u ∈ [−1,1] and state constraint x₁ ≤ 0.  The
%   objective is to drive the state from x(0) = [−1;0] to x(T) = [0;0]
%   in minimum time.  We approximate the time‑optimal problem by
%   penalising stage cost ℓ(x,a,t) = 1 + 0.01 a^2 and a large
%   quadratic terminal penalty on the final state.  We fix the final
%   horizon T=2 as the analytically optimal time and allow the adaptive
%   solver to refine the control and mesh.  Diagnostics and plots are
%   produced at the end.

% Problem data
A = [0, 1; 0, 0];
B = [0; 1];
x0 = [-1; 0];
T  = 2.0;
target = [0; 0];

% Control bounds
ctrl_lower = -1;
ctrl_upper =  1;

% State box constraint: x1 ∈ [−Inf, 0], x2 unconstrained
state_lower = [-Inf; -Inf];
state_upper = [0; Inf];

% Running cost: constant plus small control effort term
stage = @(x,a,t) (1 + 0.01*a.^2);

% Terminal cost: large penalty on deviation from target
alpha = 100;
term  = @(xT,T) alpha * sum((xT - target).^2);

% Dynamics
dyn = @(x,a,t) A*x + B*a;

% Construct OCP problem
prob = core.OCPProblem(dyn, stage, term, x0, T, 'ControlLower', ctrl_lower, 'ControlUpper', ctrl_upper, 'StateLower', state_lower, 'StateUpper', state_upper, 'Target', target);

% Initial mesh guess
N0 = 20;
t_nodes = linspace(0, T, N0+1);

% Solve using adaptive solver
res = core.adaptivity.solve_optimal_control(prob, t_nodes, 1e-3, 1e-3, 1e-3, 20);

% Compute approximate final time when x1 reaches zero
t = res.t;
X = res.X;
u = res.u;
% Find index where x1 crosses zero
idx = find(X(1,:) >= 0, 1, 'first');
if isempty(idx)
    T_final = t(end);
else
    T_final = t(idx);
end

% Print diagnostics
fprintf('Example 2 (Min‑Time Double Integrator) results:\n');
fprintf('  Approximate final time T = %.6f\n', T_final);
fprintf('  Number of mesh intervals N = %d\n', length(t)-1);
fprintf('  Number of PA planes M = %d\n', res.bundle.num_planes());
fprintf('  Final smoothing delta = %.3e\n', res.delta);
fprintf('  Outer iterations = %d\n', res.iterations);

% Plot state and control trajectories
figure;
subplot(3,1,1);
plot(t, X(1,:), '-o'); grid on;
xlabel('t'); ylabel('x_1'); title('Position x_1');

subplot(3,1,2);
plot(t, X(2,:), '-o'); grid on;
xlabel('t'); ylabel('x_2'); title('Velocity x_2');

subplot(3,1,3);
plot(t, u(1,:), '-o'); grid on;
xlabel('t'); ylabel('u'); title('Control u');

result = res;
end