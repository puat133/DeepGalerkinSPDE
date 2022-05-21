function [obj_val, grad, MM, PP] = EKFS_LIK(p, ss)
% Perform MAP hyperpara estimation in EKFS
%
% U_{k+1} = a(U_{k}) + q_k,   q_k ~ N(0, Q(U_k))
% the EKFS approx. as 
% U_{k+1} = A(U_{k}) U_k + q
%
% Arguments:
%   p:          Value of hyperparameter vector
%   ss:         The SSDGP object, with loaded data in DGP.
%
% Returns:
%   obj_val:    objective value
%   grad:       gradient
%
%
% Copyrighters are anonymized for doub-blind review   (c) 2020
% 
%

%% Check if settings are correct
% Check if SS is legit
if ~isa(ss, 'ssdgp.SSDGP')
    error('SS is not a ssdgp.SSDGP class');
end

% Check if everything is okay
if ~ss.dgp.compiled
    error('DGP is not compiled.');
end

if ~ss.dgp.data_loaded
    error('Data is not load. Pls use dgp.load_data method.');
end

%% Initialize
% Here we will try to avoid directly operating on the object properties for
% performance issue, though it might be memory-inefficient

m0 = ss.m0;
P0 = ss.P0;

% Model
a = ss.F;
A = ss.dFdu;
Q = ss.Q;

dAdp = ss.dFdudp;
dQdp = ss.dQdp;

H = zeros(1, length(m0));
H(1) = 1;

%% Data
T = ss.dgp.x;
y = ss.dgp.y;
R = ss.dgp.R(1, 1);
Y = y(:);

dt = diff(T);
dt = [dt(1); dt];

N = length(T);

MM = zeros(length(m0), N);
PP = zeros(length(m0), length(m0), N);

%% Filtering pass
m = m0;
P = P0;

obj_val = 0.5 * ((p - ss.dgp.p_mean)' / ss.dgp.p_cov * (p - ss.dgp.p_mean));
grad = ss.dgp.p_cov * (p - ss.dgp.p_mean);

% obj_val = 0;
% grad = zeros(length(p), 1);

dmdp = zeros(length(m), length(p));
dPdp = zeros(length(m), length(m), length(p));
dvdp = zeros(1, length(p));
dSdp = dvdp;
dKdp = zeros(length(m), length(p));

for k = 1:N

    % Gradient
    for i = 1:length(p)
        dmdp(:, i) = dAdp{i}(dt(k), m, p) * m + A(dt(k), m, p) * dmdp(:, i);
        dPdp(:, :, i) = dAdp{i}(dt(k), m, p) * P * A(dt(k), m, p)' ...
                      + A(dt(k), m, p) * dPdp(:, :, i) * A(dt(k), m, p)' ...
                      + A(dt(k), m, p) * P * dAdp{i}(dt(k), m, p) ...
                      + dQdp{i}(dt(k), m, p);
    end
    
    % Prediction
    P = A(dt(k), m, p) * P * A(dt(k), m, p)' + Q(dt(k), m, p);
    m = a(dt(k), m, p);

    % Update
    S = H * P * H' + R;
    K = P * H' / S;
    v = (Y(k) - H * m);

    % Gradient
    for i = 1:length(p)
        dvdp(:, i) = - H * dmdp(:, i);
        dSdp(:, :, i) = H * dPdp(:, :, i) * H';
        dKdp(:, i) = dPdp(:, :, i) * H' / S - P * H' / S * dSdp(i) / S;
        dmdp(:, i) = dmdp(:, i) + dKdp(:, i) * v + K * dvdp(i);
        dPdp(:, :, i) = dPdp(:, :, i) - dKdp(:, i) * S * K' ...
                      - K * dSdp(i) * K' ...
                      - K * S * dKdp(:, i)';
        
        grad(i) = grad(i) + 0.5 * trace(S \ dSdp(i)) ...
                          + v' / S * dvdp(i) ...
                          - 0.5 * v' / S * dSdp(i) / S * v;
    end
    
    m = m + K * v;
    P = P - K * S * K';
    
    MM(:, k) = m;
    PP(:, :, k) = P;
    
    % Objective value
    obj_val = obj_val + 0.5 * (tools.log_det(2*pi*S) + v' / S * v);

end
    
end