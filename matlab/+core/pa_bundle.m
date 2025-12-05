classdef PABundle
    %PABundle Piecewise‑affine Hamiltonian surrogate bundle.
    %   Stores a finite set of control candidates defining supporting
    %   hyperplanes for the Hamiltonian.  Each column of the controls
    %   property is an m×1 control vector.  The bundle is used to
    %   evaluate the surrogate Hamiltonian \bar{H}(p,x,t) = min_i
    %   p·f(x,a_i,t) + ℓ(x,a_i,t).  New planes can be added via
    %   add_control() to improve the approximation.  See Section 6.3 and
    %   Section 7 of the DeepResearch plan.

    properties
        % Matrix whose columns are control vectors a_i (m×M).
        controls
    end

    methods
        function obj = PABundle(initial_controls)
            %PABundle Construct a new bundle
            %   initial_controls: m×M matrix of control candidates.  Each
            %   column will define a supporting plane in the surrogate.
            if nargin < 1 || isempty(initial_controls)
                obj.controls = [];
            else
                obj.controls = initial_controls;
            end
        end

        function M = num_planes(obj)
            %NUM_PLANES Return the number of planes (columns of controls).
            if isempty(obj.controls)
                M = 0;
            else
                M = size(obj.controls,2);
            end
        end

        function [Hbar, idx] = eval(obj, problem, p, x, t)
            %EVAL Evaluate the PA surrogate \bar{H}(p,x,t).
            %   Returns the minimum p·f(x,a_i,t) + ℓ(x,a_i,t) over all
            %   stored control candidates.  idx is the index of the active
            %   plane achieving the minimum.  If the bundle has no
            %   planes, Hbar is Inf and idx = 0.
            if obj.num_planes() == 0
                Hbar = Inf;
                idx  = 0;
                return;
            end
            M = obj.num_planes();
            vals = zeros(1, M);
            for k = 1:M
                a = obj.controls(:,k);
                fval = problem.dynamics(x, a, t);
                ell  = problem.stage_cost(x, a, t);
                vals(k) = p(:).'*fval(:) + ell;
            end
            [Hbar, idx] = min(vals);
        end

        function obj = add_control(obj, a, tol)
            %ADD_CONTROL Add a new control vector to the bundle if unique.
            %   a: m×1 control vector to add.  If it is already present
            %   (within tolerance tol), the bundle is unchanged.  tol
            %   defaults to 1e-12.
            if nargin < 3
                tol = 1e-12;
            end
            if isempty(obj.controls)
                obj.controls = a(:);
                return;
            end
            % Check if a is already in the set
            diffs = obj.controls - a(:);
            norms = sqrt(sum(diffs.^2,1));
            if any(norms < tol)
                return;
            end
            obj.controls(:,end+1) = a(:);
        end
    end
end