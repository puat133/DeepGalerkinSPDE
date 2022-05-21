%
% Sigmas and weights for spherical cubature
%

% m = m0;
[WM,W,c] = ut_mweights(size(m,1),1,0,0);
XI = sqrt(c)*[zeros(size(m)) eye(size(m,1)) -eye(size(m,1))];

% method = 'CT';

