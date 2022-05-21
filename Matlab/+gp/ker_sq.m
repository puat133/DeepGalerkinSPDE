function val = ker_sq(x, y, l, sigma)
% Squared exponential GP kernel
% k(x, y) = sigma^2 * exp (-|x-y|^2/l)
%
% Arguments:
%   x, y: vector or scalar input
%   sigma, l: hyperparameters

val = sigma^2 * exp(-sum((x - y).^2) / (2*l^2));

end

