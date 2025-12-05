function x_clipped = enforce_state_bounds(problem, x)
%ENFORCE_STATE_BOUNDS Clip state x into the admissible state box.
%   x_clipped = enforce_state_bounds(problem, x) returns x clipped
%   elementwise to lie within [state_lower, state_upper].  If no
%   state bounds are defined the original x is returned.

if isempty(problem.state_lower) && isempty(problem.state_upper)
    x_clipped = x;
    return;
end
x_clipped = x;
if ~isempty(problem.state_lower)
    x_clipped = max(x_clipped, problem.state_lower);
end
if ~isempty(problem.state_upper)
    x_clipped = min(x_clipped, problem.state_upper);
end
end

function a_proj = project_control(problem, a)
%PROJECT_CONTROL Project control a into the control bounds.
%   a_proj = project_control(problem,a) clips each component of a to
%   [control_lower, control_upper] if defined.

if isempty(problem.control_lower) && isempty(problem.control_upper)
    a_proj = a;
    return;
end
a_proj = a;
if ~isempty(problem.control_lower)
    a_proj = max(a_proj, problem.control_lower);
end
if ~isempty(problem.control_upper)
    a_proj = min(a_proj, problem.control_upper);
end
end