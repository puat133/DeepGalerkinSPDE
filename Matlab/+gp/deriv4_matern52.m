function val = deriv4_matern52(x, y, l, sigma)
% Derivative of Matérn v=5/2 w.r.t. x
% d^4K/dx^2dy^2 
%
% Arguments:
%   x, y: vector or scalar input
%   sigma, l: hyperparameters
%

r = sqrt(sum((x - y).^2));

val = sigma^2 * 25 / (3*l^6) * (3*l^2-5*sqrt(5)*l*r+5*r^2) * exp(-sqrt(5)*r/l);

end

