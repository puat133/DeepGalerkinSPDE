function val = ker_periodic(x, y, w0, l, sigma)
% Periodic GP kernel
% k(x, y) = sigma^2 * exp (-|x-y|^2/l)
%
% Arguments:
%   x, y: vector or scalar input
%   w0, sigma, l: hyperparameters
%
% Copyrighters are anonymized for doub-blind review  (c) 2019
% 
%

r = pdist2(x, y);

val = sigma^2 * exp(-(2 * sin(w0 * r / 2)^2) / l^2);

end

