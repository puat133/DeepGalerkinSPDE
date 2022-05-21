function [x, obj_val, history_obj] = gd(x0, f, options, varargin)
%Simple gradient descent implementation
% arg min_x f(x)
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
%   history_obj:All history obj vals
%
% Copyrighters are anonymized for doub-blind review 
% 
%

% Parse options
max_iter = options.max_iter;
lr = options.lr;
stop_obj = options.stop_obj;
% stop_grad = options.stop_grad;
% stop_x = options.stop_x;
verbose = options.verbose;

% Init
x = x0;
iter = 0;

history_obj = zeros(max_iter + 1, 1);
diff_obj = 100;

if verbose > 0
	fprintf('Iter      obj_val\n');
    fprintf('-----------------\n');
end

if verbose > 1
    figure()
    hold on
end

while (iter <= max_iter) && (diff_obj >= stop_obj)
    
    iter = iter + 1;
    
    [obj_val, grad] = f(x, varargin{:});
    history_obj(iter + 1) = obj_val;
    
    x = x - lr .* grad;
        
    diff_obj = abs(obj_val - history_obj(iter));
    
    if verbose > 0
        fprintf('%d       %.4e\n', iter, obj_val);
    end
    
    if verbose > 1
        drawnow
        scatter(iter, obj_val, 'black', 'filled');
    end
    
end

end
