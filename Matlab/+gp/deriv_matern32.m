function val = deriv_matern32(x, y, l, sigma)
% Derivative of Matérn v=3/2 w.r.t. x
% dK/dx = ... 
% Remember to take negative if you want to get dK/dy
%
% Arguments:
%   x, y: vector or scalar input
%   sigma, l: hyperparameters
%

r = sqrt(sum((x - y).^2));
val = sigma^2 * 3 / l^2 * r * exp(-sqrt(3)/l * r);

end

