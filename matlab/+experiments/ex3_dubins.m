function result = ex3_dubins()
%EX3_DUBINS Run Example 3: Dubins car with bounded turn rate.
%   This script sets up the Dubins car optimal control problem from
%   Section D of the DeepResearch blueprint.  The state is (x,y,θ) with
%   dynamics \dot{x} = cos(θ), \dot{y} = sin(θ), \dot{θ} = u.  The
%   control u is bounded in [−1,1] and the cost consists of a small
%   running penalty ℓ(x,u) = 0.1 + u^2 and a large terminal penalty for
%   mismatch from the target state (1,1,π/2).  The horizon is fixed at
%   T=3.  The solver is applied with adaptive refinement.  Diagnostics
%   and trajectory plots are produced at the end.

% Target state
target = [1; 1; pi/2];

% Control bounds
ctrl_lower = -1;
ctrl_upper =  1;

% No state box constraints
state_lower = [];
state_upper = [];

% Stage cost: small constant plus quadratic in u
stage = @(x,a,t) (0.1 + a.^2);

% Terminal cost: large quadratic penalty on deviation from target
alpha_pos = 100;
alpha_theta = 10;
term = @(xT,T) alpha_pos*((xT(1)-target(1))^2 + (xT(2)-target(2))^2) + alpha_theta*((xT(3)-target(3))^2);

% Dynamics
dyn = @(x,a,t) [cos(x(3)); sin(x(3)); a];

% Initial state
x0 = [0; 0; 0];
T  = 3.0;

% Construct OCP problem
prob = core.OCPProblem(dyn, stage, term, x0, T, 'ControlLower', ctrl_lower, 'ControlUpper', ctrl_upper, 'StateLower', state_lower, 'StateUpper', state_upper, 'Target', target);

% Initial mesh guess
N0 = 30;
t_nodes = linspace(0, T, N0+1);

% Solve using adaptive solver
res = core.adaptivity.solve_optimal_control(prob, t_nodes, 1e-3, 1e-3, 1e-3, 20);

% Print diagnostics
fprintf('Example 3 (Dubins Car) results:\n');
fprintf('  Final time horizon T = %.2f\n', T);
fprintf('  Number of mesh intervals N = %d\n', length(res.t)-1);
fprintf('  Number of PA planes M = %d\n', res.bundle.num_planes());
fprintf('  Final smoothing delta = %.3e\n', res.delta);
fprintf('  Outer iterations = %d\n', res.iterations);

% Plot state and control trajectories
t = res.t;
X = res.X;
u = res.u;

figure;
subplot(4,1,1);
plot(t, X(1,:), '-o'); grid on;
xlabel('t'); ylabel('x'); title('Position x');

subplot(4,1,2);
plot(t, X(2,:), '-o'); grid on;
xlabel('t'); ylabel('y'); title('Position y');

subplot(4,1,3);
plot(t, X(3,:), '-o'); grid on;
xlabel('t'); ylabel('\theta'); title('Heading θ');

subplot(4,1,4);
plot(t, u(1,:), '-o'); grid on;
xlabel('t'); ylabel('u'); title('Control u');

result = res;
end