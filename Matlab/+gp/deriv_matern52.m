function val = deriv_matern52(x, y, l, sigma)
% Derivative of Matérn v=5/2 w.r.t. x
% k(x, y) = ...
%
% Arguments:
%   x, y: vector or scalar input
%   sigma, l: hyperparameters
%

r = sqrt(sum((x - y).^2));
val = 5 * sigma^2 / (3*l^3) * (l + sqrt(5) * r) * r * exp(-sqrt(5)*r/l);

end

