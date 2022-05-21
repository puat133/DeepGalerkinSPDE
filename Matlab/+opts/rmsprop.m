function [x, obj_val] = rmsprop(x0, f, options, varargin)
%RMSprop gradient descent optimization
%
% arg min_x f(x)
%
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
epsilon = options.epsilon;
beta = options.beta;
max_iter = options.max_iter;
stop_obj = options.stop_obj;
verbose = options.verbose;

v0 = zeros(size(x0));

iter = 0;

x = x0;
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
    
    v = beta .* v + (1 - beta) .* grad .^ 2;
    x = x - alp .* grad ./ (epsilon + sqrt(v));
    
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
