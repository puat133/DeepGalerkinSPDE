function val = deriv2_matern52(x, y, l, sigma)
% Derivative of Matérn v=3/2 w.r.t. x
% d^2K/dxdy = d^K/dr^2 * dr/dx * dr/dy
%           = sig^2 * 3 * (l-sqrt(3)*r)/l^3 * e^(...)
%
% Arguments:
%   x, y: vector or scalar input
%   sigma, l: hyperparameters
%

r = sqrt(sum((x - y).^2));

val = sigma^2 * 5 * (l^2+sqrt(5)*l*r-5*r^2) / (3*l^4) * exp(-sqrt(5)*r/l);

end

