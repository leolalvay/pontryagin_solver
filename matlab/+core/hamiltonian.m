function [Hval, a_star] = compute_H(problem, p, x, t, bundle, restricted)
%COMPUTE_H Evaluate the true Hamiltonian H or restricted H_K.
%   [Hval, a_star] = compute_H(problem, p, x, t, bundle, restricted)
%   computes the value of the Hamiltonian H(p,x,t) by minimizing over
%   admissible controls a:
%       H(p,x,t) = min_a p·f(x,a,t) + ℓ(x,a,t).
%   If restricted = true, only controls leading to velocities in the
%   tangent cone at x are considered (H_K).  The candidate control set
%   includes controls already present in the bundle and the extreme
%   values of the control bounds.  The minimizing control is returned in
%   a_star.

if nargin < 6
    restricted = false;
end

% Determine candidate controls
[amin, amax] = problem.getControlBounds();
control_list = [];
m = length(amin);

% Add existing controls from the bundle
if ~isempty(bundle) && bundle.num_planes() > 0
    control_list = [control_list, bundle.controls];
end
% Add extremes of control bounds (corners of A)
if ~isempty(amin) && ~isempty(amax)
    % All combinations of minima and maxima for each dimension
    combos = allBinaryCombinations(m);
    for j = 1:size(combos,1)
        ctrl = zeros(m,1);
        for d = 1:m
            if combos(j,d) == 0
                ctrl(d) = amin(d);
            else
                ctrl(d) = amax(d);
            end
        end
        control_list = [control_list, ctrl];
    end
else
    % If no bounds, we include zero control and maybe an analytic optimum
    control_list = [control_list, zeros(m,1)];
end
% Remove duplicate columns
if ~isempty(control_list)
    % unique columns using tolerance
    tmp = unique(round(control_list'*1e12)/1e12, 'rows');
    control_list = tmp';
end

% Evaluate Hamiltonian for each candidate
num_c = size(control_list,2);
vals = inf(1,num_c);
for k = 1:num_c
    a = control_list(:,k);
    % Check admissibility
    if ~problem.admissible_control(a, x, t)
        continue;
    end
    % Compute velocity
    fval = problem.dynamics(x, a, t);
    % For restricted case, ensure viability
    if restricted
        mask = problem.tangent_cone_filter(x, fval);
        if ~all(mask)
            continue;
        end
    end
    % Stage cost
    ell  = problem.stage_cost(x, a, t);
    vals(k) = p(:).' * fval(:) + ell;
end
% Return minimum value and corresponding control
[Hval, idx] = min(vals);
if isempty(idx) || Hval == Inf
    % If no valid control found, return Inf
    a_star = NaN(length(amin),1);
else
    a_star = control_list(:,idx);
end
end

function combos = allBinaryCombinations(m)
%ALLBINARYCOMBINATIONS Generate all binary vectors of length m.
%   Returns a 2^m × m matrix where each row is a binary vector.
if m == 0
    combos = zeros(0,0);
    return;
end
num = 2^m;
combos = zeros(num, m);
for i = 0:num-1
    b = dec2bin(i, m) - '0';
    combos(i+1,:) = b;
end
end