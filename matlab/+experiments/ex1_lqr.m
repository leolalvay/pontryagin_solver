function result = ex1_lqr()
%EX1_LQR Run Example 1: Linear Quadratic Regulator.
%   This script constructs the LQR optimal control problem described in
%   Section D of the DeepResearch blueprint and solves it using the
%   adaptive PMP solver.  The system is a double integrator with
%   dynamics \dot{x} = A x + B u, quadratic running cost and terminal cost.
%   The optimal control is linear feedback and can be compared against
%   the analytic Riccati solution.  The script prints diagnostic
%   quantities and plots the resulting trajectories.

% Define problem data
A = [0, 1; 0, 0];
B = [0; 1];
Q = eye(2);
R = 1e-2;
Qf = Q;
x0 = [1; 0];
T  = 1.0;

% Dynamics and costs
dyn  = @(x,a,t) A*x + B*a;
stage= @(x,a,t) (x.'*Q*x + a.'*R*a);
term = @(xT,T) xT.'*Qf*xT;

% Construct problem (no control or state constraints)
prob = core.OCPProblem(dyn, stage, term, x0, T);

% Initial mesh (uniform)
N0 = 20;
t_nodes = linspace(0, T, N0+1);

% Solve using adaptive solver
res = core.adaptivity(prob, t_nodes, 1e-3, 1e-3, 1e-3, 10);

% Compute objective value J = x(T)^T Qf x(T) + ∫0^T (x^T Q x + u^T R u) dt
t = res.t;
X = res.X;
u = res.u;
J = X(:,end).'*Qf*X(:,end);
for i = 1:length(t)-1
    dt = t(i+1) - t(i);
    xi = X(:,i);
    ui = u(:,i);
    J = J + (xi.'*Q*xi + ui.'*R*ui)*dt;
end

% Print diagnostics
fprintf('Example 1 (LQR) results:\n');
fprintf('  Final objective J = %.6f\n', J);
fprintf('  Number of mesh intervals N = %d\n', length(t)-1);
fprintf('  Number of PA planes M = %d\n', res.bundle.num_planes());
fprintf('  Final smoothing delta = %.3e\n', res.delta);
fprintf('  Outer iterations = %d\n', res.iterations);

% Plot state and control trajectories
figure;
subplot(3,1,1);
plot(t, X(1,:), '-o'); grid on;
xlabel('t'); ylabel('x_1'); title('State x_1');

subplot(3,1,2);
plot(t, X(2,:), '-o'); grid on;
xlabel('t'); ylabel('x_2'); title('State x_2');

subplot(3,1,3);
plot(t, u(1,:), '-o'); grid on;
xlabel('t'); ylabel('u'); title('Control u');

result = res;
end