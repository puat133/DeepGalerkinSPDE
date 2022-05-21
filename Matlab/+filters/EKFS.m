function [MM, PP, MS, PS] = EKFS(p, ss)
% EKFS
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

MM_pred = MM;
PP_pred = PP;

%% Filtering pass
m = m0;
P = P0;

int_steps = 2;

for k = 1:N
    
    % Prediction
    for j = 1:int_steps
        P = A(dt(k) / int_steps, m, p) * P * A(dt(k) / int_steps, m, p)' + Q(dt(k) / int_steps, m, p);
        m = a(dt(k) / int_steps, m, p);
    end
    
    MM_pred(:, k) = m;
    PP_pred(:, :, k) = P;

    % Update
    S = H * P * H' + R;
    K = P * H' / S;
    v = (Y(k) - H * m);

    m = m + K * v;
    P = P - K * S * K';
    
    MM(:, k) = m;
    PP(:, :, k) = P;

end

%% Smoothing pass

MS = MM;
PS = PP;

for k = N-1:-1:1
    
    D = PP(:, :, k) * A(dt(k), MM(:, k), p)';
    G = D / PP_pred(:, :, k+1);
    MS(:, k) = MM(:, k) + G * (MS(:, k+1) - MM_pred(:, k+1));
    PS(:, :, k) = PP(:, :, k) + G * (PS(:, :, k+1) - PP_pred(:, :, k+1)) * G';

end
    
end