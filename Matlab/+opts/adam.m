function [x, obj_val] = adam(x0, f, options, varargin)
%Adam optimization
%
% arg min_x f(x)
%
% https://arxiv.org/pdf/1412.6980.pdf
%
% Arguments:
%   x0:         Initial value
%   f:          Function handle which returns [obj_val, grad]
%   options:    Option struct
%                   .max_iter:      Maximum iteration
%                   .lr:            Learning rate
%                   .stop_obj:      Stopping obj difference
%                   .stop_grad:     Stopping grad size
%                   .stop_x:        Stopping x difference
%                   .verbose:       Verbose
%   varargin:   Arguments to function handle f
%
% Returns:
%   x:          Solution
%   obj_val:    obj val
%
% Copyrighters are anonymized for doub-blind review 
% 
%

% Parse options
alp = options.alp;
beta1 = options.beta1;
beta2 = options.beta2;
epsilon = options.epsilon;
max_iter = options.max_iter;
stop_obj = options.stop_obj;
verbose = options.verbose;

m0 = zeros(size(x0));
v0 = zeros(size(x0));

iter = 0;

x = x0;
m = m0;
v = v0;

history_obj = zeros(max_iter + 1, 1);
diff_obj = 100;

if verbose > 0
	fprintf('Iter      obj_val \n');
    fprintf('------------------\n');
end

if verbose > 1
    figure()
    hold on
end

tic
while (iter <= max_iter) && (diff_obj >= stop_obj)
    
    iter = iter + 1;
    
    [obj_val, grad] = f(x, varargin{:});
    history_obj(iter + 1) = obj_val;
    
    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * grad .^ 2;
    m_hat = m / (1 - beta1^iter);
    v_hat = v / (1 - beta2^iter);
    
    x = x - alp .* m_hat ./ (sqrt(v_hat) + epsilon);
    
    diff_obj = abs(obj_val - history_obj(iter));
    
    if verbose > 0
        fprintf('%d       %.4e\n', iter, obj_val);
    end
    
    if verbose > 1
        drawnow
        scatter(iter, obj_val, 'black', 'filled');
    end
    
end
time = toc;

end

