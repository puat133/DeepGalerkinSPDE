function [F, L, q, H, P_inf, lam, dAdl, dAds, dQdl, dQds] = gp_to_state(kernel, l, sigma, disc, dtau)
% Converting GP to state-space model according to kernel
%
% Argument:
%   kernel:     the GP kernel function name
%   l, sigma:   GP kernel parameters
%   disc:       'true' will covert the continuous LTI into a discrete model.
%   dt:         if disc is 'true', then dt is the time interval or symbolic
%                   thing
%
% Return:
%   [F, L, q, H] or [A, 0, Q, H]. 
%   P_inf:  stationary covariance (or P0).
%
% Copyrighters are anonymized for doub-blind review  @ 2018
%

switch kernel
    case 'gp.ker_sq'
        error('not implemented for sq kernel yet. 0 0 ')
        
    case 'gp.ker_ou'
        error('please use matern12 instead. ')
        
    case 'gp.ker_matern12'
        lam = 1 / l;
        F = -1 / l;
        L = 1;
        q = 2 * sigma^2 / l;
        H = 1;
        P_inf = sigma^2;
        D = 1;
        
    case 'gp.ker_matern32'
        lam = sqrt(3) / l;
        F = [0 1; -lam^2 -2*lam];
        L = [0; 1];
        q = 4 * lam^3 * sigma^2;
        H = [1 0];
        P_inf = [sigma^2 0; 0 lam^2*sigma^2];
        D = 2;
        
    case 'gp.ker_matern52'
        lam = sqrt(5) / l;
        F = [0 1 0; 0 0 1; -lam^3 -3*lam^2 -3*lam];
        L = [0; 0; 1];
        q = 16 / 3 * sigma^2 * lam^5;
        H = [1 0 0];
        
        % Check matern52_stationary_cov_derivation.nb for derivation
        P_inf = [sigma^2 0 -1/3*lam^2*sigma^2; 
                  0 1/3*lam^2*sigma^2 0; 
                  -1/3*lam^2*sigma^2 0 lam^4*sigma^2];
        D = 3;
end

q = sigma^2 * (factorial(D-1)^2 / factorial(2*D-2)) * (2*lam)^(2*D-1);

if disc
    if ~isnumeric(dtau)
        [A_sym, Q_sym, dAdl, dAds, dQdl, dQds] = tools.matern_to_state_aug(D);
        F = matlabFunction(A_sym);
        q = matlabFunction(Q_sym);
        dAdl = matlabFunction(dAdl);
        dAds = matlabFunction(dAds);
        dQdl = matlabFunction(dQdl);
        dQds = matlabFunction(dQds);
    else
        [F, q] = tools.lti_disc(F, L, q, dtau);
        L = 0;
    end
    
end

end

