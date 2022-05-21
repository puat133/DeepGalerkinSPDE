classdef PriorNode < handle
    % A DGP Node but is just a conventional Gaussian priro instead of a GP.
    
    properties
        name
        mean
        variance
        
        sym_p
    end
    
    methods
        function obj = PriorNode(name, mean, variance)
            % Constructor
            obj.name = name;
            obj.mean = mean;
            obj.variance = variance;
            obj.sym_p = sym(name, 'real');
        end
        
    end
end

