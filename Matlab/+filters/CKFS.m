function [MM, PP, MS, PS] = CKFS(p, ss)
% Generic cubature filter and smoother on a discretised model
% x_k = F(x_{k-1}) + q_{k-1},
% y_k = H x_k + r_k.
%
% Arguments:
%   SS:         The SSDGP object, with loaded data in DGP.
%   query:      The query/interpolation/integration positions
%   to_ss:      Push results to ss object?
%
% Returns:
%   x_post:     The location where posterior is evaluated
%   post:       The posterior estimates in a cell. {MM, PP, MS, PS}
%   time:       The CPU time {time_f, time_s} for filter and smoother
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
AX = ss.D;
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

%% Initialize cubature sigma points
dim_x = length(m0);

XI = [eye(dim_x) -eye(dim_x)];
XI = sqrt(dim_x) * XI;
WM = ones(1, 2*dim_x) / (2 * dim_x);
WC = WM;

%% Filtering pass
m = m0;
P = P0;

sigp = chol(P)' * XI + repmat(m, 1, 2*dim_x);
ax = sigp;
Bx = zeros(dim_x, dim_x, 2*dim_x);

int_steps = 2;

for k = 1:N
    
    % Prediction step
    for z = 1:int_steps
        sigp = chol(P)' * XI + repmat(m, 1, 2*dim_x);
        for j = 1:2*dim_x
            ax(:, j) = a(dt(k) / int_steps, sigp(:, j), p);
            Bx(:, :, j) = Q(dt(k) / int_steps, sigp(:, j), p) + ax(:, j) * ax(:, j)';
        end
        m = sum(ax, 2) / (2 * dim_x);
        P = sum(Bx, 3) / (2 * dim_x) - m * m';
    end
    
    MM_pred(:, k) = m;
    PP_pred(:, :, k) = P;

    % Update step
    S = H * P * H' + R;
    K = P * H' / S;
    m = m + K * (Y(k) - H * m);
    P = P - K * S * K';
    
    MM(:, k) = m;
    PP(:, :, k) = P;
end

%% Smoothing pass

MS = MM;
PS = PP;

for k = N-1:-1:1

    sigp = chol(PP(:, :, k))' * XI + repmat(MM(:, k), 1, 2*dim_x);   
    % Calculate cross-cov D
    for j=1:2*dim_x
        Bx(:, :, j) = AX(dt(k+1), sigp(:, j), p);
    end
    D = sum(Bx, 3) / (2 * dim_x) - MM(:, k) * MM_pred(:, k+1)';
    % Smoothing Gain
    G = D / PP_pred(:, :, k+1);
    % Smooth
    MS(:, k) = MM(:, k) + G * (MS(:, k+1) - MM_pred(:, k+1));
    PS(:, :, k) = PP(:, :, k) + G * (PS(:, :, k+1) - PP_pred(:, :, k+1)) * G';

end

    
end
