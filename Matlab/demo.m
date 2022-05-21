% This is a demo script of deep state-space Gaussian processes on some test
% signals
%
% Please refer to our paper "Fast optimize-and-sample method for 
% differentiable Galerkin approximations of multi-layered Gaussian process priors" 
% for details.
%
% 
clear
close all
clc
rng('default')
rng(666)

%% Simualte the rectangle signale
dt = 0.005;
T = [dt:dt:1];

xt = toymodels.rect(T, 'rect-2');
R = 0.01 * eye(length(T));
y = xt + chol(R)' * randn(size(xt));

%% Build DGP model
g = "exp";

u22 = dgp.PriorNode('u22', 3, 1);

f = dgp.DGPNode('f', [], g, 'f', @gp.matern32_ns, {1, u22});

u31 = dgp.PriorNode('u31', -3, 1);
u32 = dgp.PriorNode('u32', 1, 1);

u21 = dgp.DGPNode(f, 'l', g, 'u21', @gp.matern32, {u31, u32});

my_dgp = dgp.DGP(f);
my_dgp.compile();

ss = ssdgp.SSDGP(my_dgp, "TME-3");

my_dgp.load_data(T, y, R);

%% Optimize w.r.t. hyperparas using finite difference
p = zeros(3, 1);

solver = 'EKFS';
optimizer = 'BFGS';

% Decide which function handles
if strcmp(solver, 'EKFS')
    fmincon_func = @(x) filters.EKFS_LIK(x, ss);
    gd_func = @filters.EKFS_LIK;
    smoother_func = @filters.EKFS;
    
elseif strcmp(solver, 'CKFS')
    fmincon_func = @(x) filters.CKFS_LIK(x, ss);
    gd_func = @filters.CKFS_LIK;
    smoother_func = @filters.CKFS;
    
end

% Decice which optimizer
if strcmp(optimizer, 'RMSPROP')
    options.max_iter = 1000;
    options.lr = 1e-4;
    options.alp = 1e-2;
    options.beta = 0.9;
    options.beta1 = 0.9;
    options.beta2 = 0.999;
    options.epsilon = 1e-7;
    options.stop_obj = 1e-7;
    options.verbose = 1;

    [p_new] = opts.rmsprop(p, @opts.fd, options, gd_func, 1e-8, ss);
    
elseif strcmp(optimizer, 'ADAM')
    options.max_iter = 1000;
    options.lr = 1e-4;
    options.alp = 1e-2;
    options.beta = 0.9;
    options.beta1 = 0.9;
    options.beta2 = 0.999;
    options.epsilon = 1e-7;
    options.stop_obj = 1e-7;
    options.verbose = 1;

    [p_new] = opts.adam(p, @opts.fd, options, gd_func, 1e-8, ss);
    
elseif strcmp(optimizer, 'BFGS')
    options = optimoptions('fmincon','Algorithm','Interior-Point', ...
                        'HessianApproximation', 'lbfgs', ...
                        'SpecifyObjectiveGradient',false, 'Display', ...
                        'iter-detailed', 'MaxIterations', 1000, ...
                        'CheckGradients', false);
                    
    [p_new, neg_log_lik] = fmincon(fmincon_func, p, [], [], [], [], -8*ones(3,1), 4*ones(3,1), [], options);
    
end

%% Smoothing and calculate those metrics
[MM, PP, MS, PS] = smoother_func(p_new, ss);

rmse = metrics.RMSE(xt, MS(1, :));
l2_err = metrics.L2(xt, MS(1, :));
psnr = metrics.PSNR(xt, MS(1, :));

%% Plot f
figure()
plot(T, xt, 'DisplayName', 'True signal');
hold on
scatter(T, y, 'DisplayName', 'Measurements');
prop_line = {'Color', 'k', 'LineWidth', 2.5, 'DisplayName', 'Estimates'};
prop_patch = {'FaceColor', 'k', 'EdgeColor', 'none', 'FaceAlpha', 0.2, ...
              'HandleVisibility', 'off'};
tools.errBar(T, MS(1, :), 1.96 * sqrt(squeeze(PS(1, 1, :))), prop_line, prop_patch);
title('Estiamte of f(t)')

%% Plot ell
figure()
tools.errBar(T, MS(3, :), 1.96 * sqrt(squeeze(PS(3, 3, :))), prop_line, prop_patch);
title('Estiamte of ell(t)')

%% Plot sigma
figure()
tools.errBar(T, MS(4, :), 1.96 * sqrt(squeeze(PS(4, 4, :))), prop_line, prop_patch);
title('Estiamte of sigma(t)')