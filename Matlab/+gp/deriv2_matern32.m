function val = deriv2_matern32(x, y, l, sigma)
% Derivative of Matérn v=3/2 w.r.t. x
% d^2K/dxdy = d^K/dr^2 * dr/dx * dr/dy
%           = sig^2 * 3 * (l-sqrt(3)*r)/l^3 * e^(...)
%
% Arguments:
%   x, y: vector or scalar input
%   sigma, l: hyperparameters
%

r = sqrt(sum((x - y).^2));

val = sigma^2 * 3 / l^3 * (l - sqrt(3) * r) * exp(-sqrt(3)/l * r);

end

