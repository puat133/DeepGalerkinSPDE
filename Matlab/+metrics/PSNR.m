function val = PSNR(x1, x2)

% x1: Ground truth
% x2: Estimate

mse = metrics.MSE(x1, x2);
val = 10 * log10(max(x2).^2 / mse);

end