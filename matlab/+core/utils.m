function grad = fd_gradient(fun, x, h)
%FD_GRADIENT Finite difference approximation to gradient.
%   grad = fd_gradient(fun, x, h) computes a central finite difference
%   approximation of the gradient of a scalar function handle FUN at
%   point x.  The function FUN should accept a vector input and return
%   a scalar.  h specifies the perturbation step (default 1e-6).
%   This utility can be used in unit tests to verify analytic
%   gradients.  The output grad has the same size as x.

if nargin < 3 || isempty(h)
    h = 1e-6;
end
n = numel(x);
grad = zeros(size(x));
for i = 1:n
    e = zeros(size(x));
    e(i) = 1;
    f_plus  = fun(x + h * e);
    f_minus = fun(x - h * e);
    grad(i) = (f_plus - f_minus) / (2*h);
end
grad = reshape(grad, size(x));
end