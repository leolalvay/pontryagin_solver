function M = smoothing()
%SMOOTHING Module factory (Python-like).
M.eval_H_smooth = @eval_H_smooth;
M.eval_f_smooth = @eval_f_smooth;
M.eval_H        = @eval_H;
M.eval_f        = @eval_f;
end

function [Hdelta, gradp, gradx] = eval_H_smooth(bundle, problem, p, x, t, delta)
%EVAL_H_SMOOTH Smooth approximation of the PA Hamiltonian.
%   Computes the smoothed Hamiltonian H_delta(p,x,t) for the given
%   bundle of controls, problem definition and smoothing parameter delta.
%   Uses the log‑sum‑exp formulation:
%     H_delta = -delta * log(sum_i exp(-L_i/delta)),
%   where L_i = p·f(x,a_i,t) + ℓ(x,a_i,t).  The gradients with
%   respect to p and x are also returned.  The gradient with respect
%   x is approximated by finite differences if necessary.  See
%   Section 6.4 of the DeepResearch plan for definitions.

% Ensure the bundle has at least one plane
if bundle.num_planes() == 0
    % If no planes, return Inf to signal missing data
    Hdelta = Inf;
    gradp  = zeros(size(p));
    gradx  = zeros(size(x));
    return;
end

M = bundle.num_planes();
L = zeros(1,M);
fvals = zeros(length(x), M);
% Compute L_i and f(x,a_i,t) for each plane
for i = 1:M
    a_i = bundle.controls(:,i);
    f_i = problem.dynamics(x, a_i, t);
    ell_i = problem.stage_cost(x, a_i, t);
    fvals(:,i) = f_i;
    L(i) = p(:).' * f_i(:) + ell_i;
end
% Use log‑sum‑exp to compute the smooth minimum robustly.  Shift by
% minimum L to avoid overflow.
Lmin = min(L);
weights = exp(-(L - Lmin) / delta);
Z = sum(weights);
Hdelta = -delta * (log(Z) + Lmin/delta);
% Compute weights normalized
w = weights / Z;
% Gradient w.r.t p: weighted average of f_i
gradp = fvals * w(:);
% Gradient w.r.t x: derivative of L_i w.r.t x weighted.  Use finite
% difference for p·f + ℓ.  Compute ∂/∂x (p·f(x,a_i) + ℓ(x,a_i)) as
% (L_i(x+eps) - L_i(x-eps)) /(2 eps).  Combine across planes.
n = length(x);
gradx = zeros(n,1);
eps_fd = 1e-6;
for j = 1:n
    e = zeros(n,1);
    e(j) = 1;
    x_plus  = x + eps_fd * e;
    x_minus = x - eps_fd * e;
    L_plus  = zeros(1,M);
    L_minus = zeros(1,M);
    for i = 1:M
        a_i = bundle.controls(:,i);
        f_plus  = problem.dynamics(x_plus, a_i, t);
        ell_plus= problem.stage_cost(x_plus, a_i, t);
        L_plus(i)  = p(:).' * f_plus(:) + ell_plus;
        f_minus = problem.dynamics(x_minus, a_i, t);
        ell_minus= problem.stage_cost(x_minus, a_i, t);
        L_minus(i) = p(:).' * f_minus(:) + ell_minus;
    end
    % Compute smoothed derivative by weighting L derivatives
    dLdx = (L_plus - L_minus) / (2*eps_fd);
    gradx(j) = w * dLdx.';
end
end