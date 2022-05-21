function [obj_val, grad] = fd(x, func, epsilon, varargin)
% Finite-difference
%
% Arguments:
%   x:      point
%   func:   Function handle which returns [obj_val]
%   varargin:   Addiitonal args to func

grad = zeros(size(x));

obj_val = func(x, varargin{:});

for i = 1:length(x)
    x2 = x;
    x2(i) = x2(i) + epsilon;
    
    obj_new = func(x2, varargin{:});
    
    grad(i) = (obj_new - obj_val) / epsilon;
end

end

