classdef OCPProblem
    %OCPProblem General optimal control problem definition.
    %   This class encapsulates the ingredients of a Bolza optimal control
    %   problem with control bounds and optional state box constraints.  It
    %   stores user‑provided dynamics, running cost and terminal cost
    %   functions and provides helper methods for computing admissibility
    %   and tangent‑cone viability checks.  See Section B of the
    %   DeepResearch report for the formal specification of these
    %   interfaces.

    properties
        % Function handle for the state dynamics f(x,a,t) → ℝ^n.
        dynamics_func
        % Function handle for the running cost ℓ(x,a,t) → ℝ.
        stage_cost_func
        % Function handle for the terminal cost g(x,T) → ℝ.
        terminal_cost_func
        % Initial state vector x₀ (n×1).
        x0
        % Desired final state vector (n×1) for reference.  If empty the
        % problem does not have a hard terminal target.
        target
        % Fixed final time horizon T (scalar).  Use [] if free terminal time.
        T
        % Lower and upper bounds on control, stored as column vectors (m×1).
        control_lower
        control_upper
        % Lower and upper bounds on state (box constraints).  If empty the
        % state is unconstrained.  Each is an n×1 column vector.
        state_lower
        state_upper
    end

    methods
        function obj = OCPProblem(dynamics_func, stage_cost_func, terminal_cost_func, x0, T, varargin)
            %OCPProblem Construct an instance of this class
            %   dynamics_func: function handle f(x,a,t)
            %   stage_cost_func: function handle ℓ(x,a,t)
            %   terminal_cost_func: function handle g(x,T)
            %   x0: initial state (n×1)
            %   T: fixed final time (scalar) or [] if free
            %   Optional name/value arguments:
            %     'ControlLower', lower bound on control (m×1)
            %     'ControlUpper', upper bound on control (m×1)
            %     'StateLower', lower bound on state (n×1)
            %     'StateUpper', upper bound on state (n×1)
            %     'Target', desired final state (n×1)
            obj.dynamics_func     = dynamics_func;
            obj.stage_cost_func   = stage_cost_func;
            obj.terminal_cost_func= terminal_cost_func;
            obj.x0                = x0;
            obj.T                 = T;
            obj.control_lower     = [];
            obj.control_upper     = [];
            obj.state_lower       = [];
            obj.state_upper       = [];
            obj.target            = [];
            % Parse optional arguments
            for i = 1:2:length(varargin)
                switch lower(varargin{i})
                    case 'controllower'
                        obj.control_lower = varargin{i+1};
                    case 'controlupper'
                        obj.control_upper = varargin{i+1};
                    case 'statelower'
                        obj.state_lower = varargin{i+1};
                    case 'stateupper'
                        obj.state_upper = varargin{i+1};
                    case 'target'
                        obj.target = varargin{i+1};
                    otherwise
                        error('Unknown option %s', varargin{i});
                end
            end
            % Force column vectors for bounds
            if ~isempty(obj.control_lower)
                obj.control_lower = obj.control_lower(:);
            end
            if ~isempty(obj.control_upper)
                obj.control_upper = obj.control_upper(:);
            end
            if ~isempty(obj.state_lower)
                obj.state_lower = obj.state_lower(:);
            end
            if ~isempty(obj.state_upper)
                obj.state_upper = obj.state_upper(:);
            end
        end

        function f = dynamics(obj, x, a, t)
            %DYNAMICS Evaluate the state dynamics f(x,a,t).
            %   x is an n×1 state vector, a is an m×1 control vector, t is scalar.
            f = obj.dynamics_func(x, a, t);
        end

        function ell = stage_cost(obj, x, a, t)
            %STAGE_COST Evaluate the running cost ℓ(x,a,t).
            ell = obj.stage_cost_func(x, a, t);
        end

        function phi = terminal_cost(obj, xT, T)
            %TERMINAL_COST Evaluate the terminal cost g(x,T).
            phi = obj.terminal_cost_func(xT, T);
        end

        function [amin, amax] = getControlBounds(obj)
            %GETCONTROLBOUNDS Return control bounds as column vectors (m×1 each).
            amin = obj.control_lower;
            amax = obj.control_upper;
        end

        function tf = admissible_control(obj, a, ~, ~)
            %ADMISSIBLE_CONTROL Check if control lies within A.
            %   Returns true if a ∈ [control_lower, control_upper] elementwise.
            if isempty(obj.control_lower) && isempty(obj.control_upper)
                tf = true;
                return;
            end
            tf = true;
            if ~isempty(obj.control_lower)
                tf = tf && all(a >= obj.control_lower - 1e-12);
            end
            if ~isempty(obj.control_upper)
                tf = tf && all(a <= obj.control_upper + 1e-12);
            end
        end

        function [xmin, xmax] = getStateBounds(obj)
            %GETSTATEBOUNDS Return state bounds as column vectors (n×1 each).
            xmin = obj.state_lower;
            xmax = obj.state_upper;
        end

        function tf = admissible_state(obj, x)
            %ADMISSIBLE_STATE Check if the state lies inside K (box constraints).
            if isempty(obj.state_lower) && isempty(obj.state_upper)
                tf = true;
                return;
            end
            tf = true;
            if ~isempty(obj.state_lower)
                tf = tf && all(x >= obj.state_lower - 1e-12);
            end
            if ~isempty(obj.state_upper)
                tf = tf && all(x <= obj.state_upper + 1e-12);
            end
        end

        function mask = tangent_cone_filter(obj, x, f_cands)
            %TANGENT_CONE_FILTER Filter candidate velocities via tangent cone.
            %   Given a state x and a set of candidate state velocities
            %   f_cands (n×M) columns, returns a logical mask (1×M) of
            %   admissible velocities such that, for each state component
            %   approaching a boundary, the velocity points inward or is zero.
            %   See Section 6 of the DeepResearch plan and Eq.(2.4) for
            %   tangential viability.
            [n, M] = size(f_cands);
            mask = true(1, M);
            % No constraints
            if isempty(obj.state_lower) && isempty(obj.state_upper)
                return;
            end
            for i = 1:n
                % Check lower bound
                if ~isempty(obj.state_lower)
                    if abs(x(i) - obj.state_lower(i)) < 1e-10
                        % On lower boundary; require f(i,:) >= 0
                        mask = mask & (f_cands(i,:) >= -1e-12);
                    end
                end
                % Check upper bound
                if ~isempty(obj.state_upper)
                    if abs(x(i) - obj.state_upper(i)) < 1e-10
                        % On upper boundary; require f(i,:) <= 0
                        mask = mask & (f_cands(i,:) <= 1e-12);
                    end
                end
            end
        end
    end
end