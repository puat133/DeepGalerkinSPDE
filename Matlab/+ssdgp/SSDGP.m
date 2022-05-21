% Class for State Space Deep Gaussian Processes
%
% The current implementation use the same GP covariance functions uniformly
% for all GP nodes. though in practice it is not limited to this. 
% 
% du = f(u) dt + L(u) dW
% y_k = H u_k + r_k,
%
% where H extract the first component.
%
% Reference https://arxiv.org/pdf/2008.04733.pdf
%
% Copyrighters are anonymized for doub-blind review   (c) 2020
% 
%


classdef SSDGP < handle
    %This is an abstract class for SS-DGP, initialized with DGP object.
    
    properties
        
        % Basic infos
        dim             % Dimension of state
        num_nodes       % Number of GP nodes
        num_dim_node    % Number of dimension per node. [2, 3, ... etc]
        dgp             % To which DGP is this connected
        
        % Symbolc variables of SDE
        sym_u
        sym_f
        sym_L
        sym_q
        
        sym_p
        
        % SDE function handles
        f
        L
        
        % Symbolic variables of discretised state space
        sym_F
        sym_Q
        sym_D
        sym_dt
        
        % Discretised state space function handles
        F
        Q
        QQ     % Vectorized Q, which could be more efficient to evaluate
        dt     % This should be set different for filter or MAP/HMC
        D      % This is for smoother
        H
        
        % Derivatives for continuous model, if needed
        sym_dfdu
        sym_dLdu
        sym_dfdp
        sym_dLdp
        
        dfdu
        dLdu
        dfdp
        dLdp
        
        % Derivatives for discrete model, if needed
        sym_dFdu
        sym_dQdu
        sym_dFdp
        sym_dQdp
        sym_dFdudp
        
        dFdu
        dQdu
        dFdp
        dQdp
        dFdudp
        
        % TME, if needed
        a
        Sigma
        as
        
        % Filtering and smoothing
        m0              % Initial condition is a 0-mean Gaussian
        P0
        
        MM
        PP
        MS
        PS
        x_post
        
        % MAP and HMC
        U               % Similar to dgp.U, this is a matrix containing 
                        % the estimate of MAP. Structure should look
                        % like:
                        % U = [f|0      f|1     f|2     f|3    ...   f|N
                        %     [Df|0     Df|1    Df|2    Df|3   ...   Df|N
                        %     ...
                        %     [u21|0    u21|1   u21|2   u21|3  ...   u21|N
                        %     [u22|0    u22|1   u22|2   u22|3  ...   u22|N
                        %     ...                              ...       ]
                        %     
                        % Notice the direction is a bit different from
                        % dgp.U, where rows are state components. This also
                        % requires more memory, as we estiamte the
                        % derivateis, e.g., Df, Du21 etc.
                        %
                        % For HMC, the structure will look like
                        % U = [f|0  Df|0 ... u21|0 ... f|1  Df|1 ...]'
                        % Recall that MATLAB is a column-priority language
        
    end
    
    methods
        function obj = SSDGP(dgp, disc_method, keep_st)
            %SSDGP Constructor
            
            if nargin < 3
                keep_st = false;
            end
            
            % Check if dgp is compiled
            if ~dgp.compiled
                error('DGP is not compiled');
            end
            
            % Tell DGP this is linked.
            obj.dgp = dgp;
            dgp.ss = obj;
            
            % Figure out number of dimensions etc
            obj.num_nodes = sum(dgp.Li);
            obj.dim = 0;
            obj.num_dim_node = [];
            for i = 1:dgp.L
                for j = 1:dgp.Li(i)
                    ker_info = functions(dgp.nodes{i, j}.gp_ker);
                    if contains(ker_info.function, 'matern12')
                        ddim = 1;
                    elseif contains(ker_info.function, 'matern32')
                        ddim = 2;
                    elseif contains(ker_info.function, 'matern52')
                        ddim = 3;
                    end
                    obj.dim = obj.dim + ddim;
                    obj.num_dim_node = [obj.num_dim_node ddim];
                end
            end
            
            % Convert DGP to SDEs (symbolic and function handle)
            [obj.sym_u, obj.sym_f, obj.sym_L, obj.P0, obj.sym_p] = dgp_to_ss(obj, dgp, "simplify");
            obj.sym_q = sym(eye(obj.num_nodes));
            obj.m0 = zeros(length(obj.sym_u), 1);
            obj.f = matlabFunction(obj.sym_f, 'Vars', {obj.sym_u, obj.sym_p});
            obj.L = matlabFunction(obj.sym_L, 'Vars', {obj.sym_u, obj.sym_p});
            % Use the code below to see the expanded result and check if
            % they are correct.
%             obj.f = matlabFunction(obj.sym_f, 'Vars', obj.sym_u);
%             obj.L = matlabFunction(obj.sym_L, 'Vars', obj.sym_u);
            
            % Discretize using Euler-Maruyama (symbolic and function handle)
            if strcmp(disc_method, "local")
                locally_disc(obj, "simplify");
                fprintf('Discretised using locally linearization. \n')
            else
                disc(obj, disc_method, "simplify", keep_st);
            end
            
            obj.F = matlabFunction(obj.sym_F, 'Vars', {obj.sym_dt, obj.sym_u, obj.sym_p});
            obj.Q = matlabFunction(obj.sym_Q, 'Vars', {obj.sym_dt, obj.sym_u, obj.sym_p});
            obj.D = matlabFunction(obj.sym_D, 'Vars', {obj.sym_dt, obj.sym_u, obj.sym_p});
            obj.H = zeros(1, obj.dim);
            obj.H(1) = 1;
            
            % Bonus. This could 
            obj.QQ = matlabFunction(obj.sym_Q(:), 'Vars', {obj.sym_dt, obj.sym_u, obj.sym_p});

            % Give derivatives if needed
            obj.sym_dfdu = simplify(jacobian(obj.sym_f, obj.sym_u));
            obj.dfdu = matlabFunction(obj.sym_dfdu, 'Vars', {obj.sym_u, obj.sym_p});
            
            obj.sym_dfdp = simplify(jacobian(obj.sym_f, obj.sym_p));
            obj.dfdp = matlabFunction(obj.sym_dfdp, 'Vars', {obj.sym_u, obj.sym_p});
            
            obj.sym_dFdu = simplify(jacobian(obj.sym_F, obj.sym_u));
            obj.sym_dFdp = simplify(jacobian(obj.sym_F, obj.sym_p));
            obj.sym_dFdudp = {};
            obj.sym_dQdu = {};
            obj.sym_dQdp = {};
            for i = 1:obj.dim
                obj.sym_dQdu{i} = simplify(diff(obj.sym_Q, obj.sym_u(i)));
            end
            for i = 1:length(obj.sym_p)
                obj.sym_dQdp{i} = simplify(diff(obj.sym_Q, obj.sym_p(i)));
                obj.sym_dFdudp{i} = simplify(diff(obj.sym_dFdu, obj.sym_p(i)));
            end
            
            obj.dFdu = matlabFunction(obj.sym_dFdu, 'Vars', {obj.sym_dt, obj.sym_u, obj.sym_p});
            obj.dFdp = matlabFunction(obj.sym_dFdp, 'Vars', {obj.sym_dt, obj.sym_u, obj.sym_p});
            obj.dFdudp = {};
            obj.dQdu = {};
            obj.dQdp = {};
            for i = 1:obj.dim
                obj.dQdu{i} = matlabFunction(obj.sym_dQdu{i}, 'Vars', {obj.sym_dt obj.sym_u, obj.sym_p});
            end
            for i = 1:length(obj.sym_p)
                obj.dQdp{i} = matlabFunction(obj.sym_dQdp{i}, 'Vars', {obj.sym_dt obj.sym_u, obj.sym_p});
                obj.dFdudp{i} = matlabFunction(obj.sym_dFdudp{i}, 'Vars', {obj.sym_dt obj.sym_u, obj.sym_p});
            end
            
        end
        
        function [U, f, L, P0, p] = dgp_to_ss(obj, dgp, simp)
            % Transforming the DGP into a continuous state space
            % dU = f(U) dt + L(U)dW,  
            % W is a STANDARD Wiener process, we put everything into L(U)
            % This function can be used as a public method.
            %
            % Argument:
            %   dgp:	The DGP object
            %   simp:   Simplify results?
            %
            % Return:
            %   U:      The state vector
            %   f:      The drift term
            %   L:      The dispersion term
            %   P0:     The initial covariance P(t0) (numeric)
            %   p:      The parameter vector
            %
            
            % Check if DGP is compiled
            if ~dgp.compiled
                error('DGP is not compiled');
            end
            
            % Init SDE
            U = [];
            f = [];
            L = [];
            p = [];
            
            % First loop gives the state vektor
            for i = 1:dgp.L
                for j = 1:dgp.Li(i)
                    ker_info = functions(dgp.nodes{i, j}.gp_ker);
                    if contains(ker_info.function, 'matern12')
                        name1 = dgp.nodes{i, j}.name;
                        u = sym(name1, 'real');                        
                        U_temp = u;
                        
                    elseif contains(ker_info.function, 'matern32')
                        name1 = dgp.nodes{i, j}.name;
                        name2 = ['D' dgp.nodes{i, j}.name];
                        u = sym(name1, 'real');
                        Du = sym(name2, 'real');
                        U_temp = [u; Du];
                        
                    elseif contains(ker_info.function, 'matern52')
                        name1 = dgp.nodes{i, j}.name;
                        name2 = ['D' dgp.nodes{i, j}.name];
                        name3 = ['D2' dgp.nodes{i, j}.name];
                        u = sym(name1, 'real');
                        Du = sym(name2, 'real');
                        D2u = sym(name3, 'real');
                        U_temp = [u; Du; D2u];
                    else
                        error('Unsupported kernel')
                    end
                    
                    dgp.nodes{i, j}.sym_u = U_temp;
                    dgp.nodes{i, j}.SS_idx = length(U) + 1;
                    U = [U; U_temp];
                end
            end
            
            % Init P0
            P0 = zeros(length(U), length(U));
            
            % Second loop gives SDE coefficients.
            for i = 1:dgp.L
                for j = 1:dgp.Li(i)
                    ker_info = functions(dgp.nodes{i, j}.gp_ker);
                    g = dgp.nodes{i, j}.g;
                    
                    % Give g(\ell) and g(\sigma)
                    if strcmp(class(dgp.nodes{i, j}.descendants.l), 'dgp.DGPNode')
                        g_l = g(dgp.nodes{i, j}.descendants.l.sym_u(1));
                        P0_l = g(0);
                    elseif strcmp(class(dgp.nodes{i, j}.descendants.l), 'dgp.PriorNode')
                        p = [p; dgp.nodes{i, j}.descendants.l.sym_p];
                        g_l = g(dgp.nodes{i, j}.descendants.l.sym_p);
                        P0_l = g(0);
                    else
                        g_l = sym(dgp.nodes{i, j}.descendants.l);
                        P0_l = g_l;
                    end
                    
                    if strcmp(class(dgp.nodes{i, j}.descendants.sigma), 'dgp.DGPNode')
                        g_sig = g(dgp.nodes{i, j}.descendants.sigma.sym_u(1));
                        P0_sig = g(0);
                    elseif strcmp(class(dgp.nodes{i, j}.descendants.sigma), 'dgp.PriorNode')
                        p = [p; dgp.nodes{i, j}.descendants.sigma.sym_p];
                        g_sig = g(dgp.nodes{i, j}.descendants.sigma.sym_p);
                        P0_sig = g(0);
                    else
                        g_sig = sym(dgp.nodes{i, j}.descendants.sigma);
                        P0_sig = g_sig;
                    end
                    
                    P0_idx = dgp.nodes{i, j}.SS_idx;
                    
                    if contains(ker_info.function, 'matern12')
                        f_temp = -1 / g_l * dgp.nodes{i, j}.sym_u;
                        L_temp = g_sig * sqrt(sym(2))*g_l^sym(-0.5);
                        P0(P0_idx, P0_idx) = P0_sig^2;
                        
                    elseif contains(ker_info.function, 'matern32')
                        f_temp = [0         1; 
                                  sym(-3)/g_l^sym(2) -2*sym(sqrt(3))/g_l] * ...
                                    dgp.nodes{i, j}.sym_u;
                        L_temp = [0; g_sig * (sym(12)*sqrt(sym(3))/g_l^sym(3))^sym(0.5)];
                        P0(P0_idx:P0_idx+1, P0_idx:P0_idx+1) = [P0_sig^2 0; 
                                                                0 3/P0_l^2*P0_sig^2];
                        
                    elseif contains(ker_info.function, 'matern52')
                        f_temp = [0         1         0;
                                  0         0         1;
                                  -5*sqrt(5)/g_l^3 -15/g_l^2 -3*sqrt(5)/g_l] * ...
                                    dgp.nodes{i, j}.sym_u;
                        L_temp = [0; 0; g_sig * (sym(2)*sqrt(sym(5)))^(sym(5)/sym(2))/(sym(6)*g_l^sym(5))^sym(0.5)];
                        P0(P0_idx:P0_idx+2, P0_idx:P0_idx+2) = [P0_sig^2 0 -1/3*5/P0_l^2*P0_sig^2; 
                                                                0 1/3*5/P0_l^2*P0_sig^2 0; 
                                                                -1/3*5/P0_l^2*P0_sig^2 0 25/P0_l^4*P0_sig^2];
                    else
                        error('Unsupported kernel')
                    end
                    
                    f = [f; f_temp];
                    L = blkdiag(L, L_temp);
                end
            end
            
            if strcmp(simp, "simplify")
                U = simplify(U);
                f = simplify(f);
                L = simplify(L);
                if ~isempty(p)
                    p = simplify(p);
                end
            end
        end
        
        function disc(obj, method, simp, keep_st)
            % Discretize SDE.
            %
            % Arguments:
            %   method:     "EM" Euler-Maruyama; 
            %               "TME" Taylor moment expansion
            %   simp:       "simplify" will simplify
            %   keep_st:    Keep the explicit disc for the last stationary
            %               layer? Default is false.
            %
            % Return:
            %   This is a private method, all returns are 
            %   obj.sym_F
            %   obj.sym_Q
            %   obj.sym_dt
            %
            dt = sym('dt', 'real');
            
            if obj.num_nodes == 1
                % A special case when there is only f in the DGP, which is
                % just a GP. Thus the model would be explicit
                g = obj.dgp.nodes{1,1}.g;
                g_l = g(sym(obj.dgp.nodes{1,1}.descendants.l));
                g_s = g(sym(obj.dgp.nodes{1,1}.descendants.sigma));
                
                [F, Q] = tools.matern_to_state(obj.dim, g_l, g_s, dt);
                
                obj.sym_dt = dt;
                obj.sym_F = F * obj.sym_u;
                obj.sym_Q = Q;
                obj.sym_D = obj.sym_u * obj.sym_F';
                
                return
            end
            
            if strcmp(method, "EM")
                F = obj.sym_u + obj.sym_f * dt;
                Q = obj.sym_L * eye(obj.num_nodes) * obj.sym_L' * dt;
                fprintf('Discretised using Euler Maruyama.\n');
            elseif contains(method, "TME")
                TME_name = char(method);
                order = str2double(TME_name(end));
                [F, Q] = tools.TME(obj.sym_u, obj.sym_f, obj.sym_L, ...
                                    eye(obj.num_nodes), dt, order, 'simplify');
                fprintf('Discretised using %d-th order TME.\n', order);
            else
                error('Unsupported discretisation method.')
            end
            
%             % Replace the last LTI SDEs with explicit discretisation
%             % (OPTIONAL)
            if keep_st
                for i = 1:obj.dgp.L
                    for j = 1:obj.dgp.Li(i)
                        if obj.dgp.has_no_son(obj.dgp.nodes{i,j})
                            idx = obj.dgp.nodes{i, j}.SS_idx;
                            ker_info = functions(obj.dgp.nodes{i, j}.gp_ker);
                            if contains(ker_info.function, 'matern12')
                                alp = 1;
                            elseif contains(ker_info.function, 'matern32')
                                alp = 2;
                            elseif contains(ker_info.function, 'matern52')
                                alp = 3;
                            end
                            g = obj.dgp.nodes{i, j}.g; 
                            g_l = sym(obj.dgp.nodes{i,j}.descendants.l);
                            g_s = sym(obj.dgp.nodes{i,j}.descendants.sigma);
                            [FF, QQ] = tools.matern_to_state(alp, g_l, g_s, dt);
                            F(idx:idx+alp-1) = FF * obj.sym_u(idx:idx+alp-1);
                            Q(idx:idx+alp-1, idx:idx+alp-1) = QQ;
                        end
                    end
                end
                fprintf('The last layer stationary GPs are disc explicitly. \n')
            end
            
            % For smoother
            D = obj.sym_u * F';
            
            obj.sym_dt = dt;
            
            % Simplify
            if strcmp(simp, "simplify")
                obj.sym_F = simplify(F);
                obj.sym_Q = simplify(Q);
                obj.sym_D = simplify(D);
            else
                obj.sym_F = F;
                obj.sym_Q = Q;
                obj.sym_D = D;
            end
        end
        
        function locally_disc(obj, simp)
            % Discretize Matern non-liner SDE using locally linearization
            % The idea is to directly give the non-linear discretised state
            % space model by stacking LTI Matern discretised models.
            % This should yield better approximation compared to EM
            % TODO
            %
            % Return:
            %   This is a private method, all returns are 
            %   obj.sym_F
            %   obj.sym_Q
            %   obj.sym_dt
            %
            F = [];
            Q = [];
            D = [];
            dt = sym('dt', 'real');
            
            for i = 1:obj.dgp.L
                for j = 1:obj.dgp.Li(i)
                    ker_info = functions(obj.dgp.nodes{i, j}.gp_ker);
                    g = obj.dgp.nodes{i, j}.g;
                    
                    if strcmp(class(obj.dgp.nodes{i, j}.descendants.l), 'dgp.DGPNode')
                        g_l = g(obj.dgp.nodes{i, j}.descendants.l.sym_u(1));
                    elseif strcmp(class(obj.dgp.nodes{i, j}.descendants.l), 'dgp.PriorNode')
                        g_l = g(obj.dgp.nodes{i, j}.descendants.l.sym_p);
                    else
                        g_l = sym(obj.dgp.nodes{i, j}.descendants.l);
                    end
                    
                    if strcmp(class(obj.dgp.nodes{i, j}.descendants.sigma), 'dgp.DGPNode')
                        g_sig = g(obj.dgp.nodes{i, j}.descendants.sigma.sym_u(1));
                    elseif strcmp(class(obj.dgp.nodes{i, j}.descendants.sigma), 'dgp.PriorNode')
                        g_sig = g(obj.dgp.nodes{i, j}.descendants.sigma.sym_p);
                    else
                        g_sig = sym(obj.dgp.nodes{i, j}.descendants.sigma);
                    end
                    
                    if contains(ker_info.function, 'matern12')
                        alp = 1;
                    elseif contains(ker_info.function, 'matern32')
                        alp = 2;
                    elseif contains(ker_info.function, 'matern52')
                        alp = 3;
                    else
                        error('Omae wa mou shindeiru. Na..Nani???');
                    end
                    [FF, QQ] = tools.matern_to_state(alp, g_l, g_sig, dt);
                    F = [F; FF * obj.dgp.nodes{i, j}.sym_u];
                    Q = blkdiag(Q, QQ);
                end
            end
            
            % For smoother
            D = obj.sym_u * F';
            
            obj.sym_dt = dt;
            
            % Simplify
            if strcmp(simp, "simplify")
                obj.sym_F = simplify(F);
                obj.sym_Q = simplify(Q);
                obj.sym_D = simplify(D);
            else
                obj.sym_F = F;
                obj.sym_Q = Q;
                obj.sym_D = D;
            end
        end
        
        %% Functions for MAP estimates
        function [neg_log_likeli] = MAP(obj, f_bound, options)
            % Maximum a posteriori inference for DGP in state space.
            % The joint model is
            % p(y_1:N | x_1:N) ∝ ∏ p(y_i|x_i) ∏ p(x_i|x_i-1) p(x0)
            % Use the built-in Matlab optimization toolbox.
            %
            % Argument:
            %   f_bound:    The lower and upper bound of f. Default is Inf
            %   options:    options object for Optimization toolbox
            %
            
            % Initialize obj.U
            obj.U = zeros(obj.dim, obj.dgp.N+1);
%             obj.U = rand(obj.dim, obj.dgp.N+1);
            
            % Set dt
            obj.dt = diff(obj.dgp.x);
            obj.dt = [obj.dt(1); obj.dt];

            % Set optimization options
            if nargin < 3
                options = optimoptions('fmincon','Algorithm','Interior-Point', ...
                                    'HessianApproximation', 'lbfgs', ... 
                                    'SpecifyObjectiveGradient',true, 'Display', ...
                                    'iter-detailed', 'MaxIterations', 100, ...
                                    'CheckGradients', true);
            end
            
            % Give lower bound and upper bound of constraint optimization,
            % the same size with obj.U elementwisely
            if nargin < 2
                f_min = -Inf;
                f_max = Inf;
            else
                f_min = f_bound(1);
                f_max = f_bound(2);
            end
            [lower, upper] = obj.opt_boundary(f_min, f_max);
            
            % Optimize. This function can not use the reference of
            % variables, thus has to copy estimate.
            [obj.U, neg_log_likeli] = fmincon(@obj.ss_MAP_handle, obj.U, ...
                            [], [], [], [], lower, upper, [], options);
            
        end
        
        function [lower, upper] = opt_boundary(obj, f_min, f_max)
            % Give the optimization boundary for the estimates
            % This has to be done for the length scale and sigma etc,
            % otherwise the optimizer will diverge.
            % The boundary for f should be individually set
            % Differnet g functions should have different bound.
            % However, it might be tricky to give bounds for the
            % derivatives component, like Df, Du21 etc. We take a faith on
            % no bounds on them.
            %
            
            % Init
            lower = -Inf + obj.U;
            upper = Inf + obj.U;
            
            % Set f
            lower(1, :) = f_min * ones(1, obj.dgp.N+1);
            upper(1, :) = f_max * ones(1, obj.dgp.N+1);
            
            % Iterate the rest of nodes. Note that the g comes from
            % their father
            for i = 2:obj.dgp.L
                for j = 1:obj.dgp.Li(i)
                    idx = obj.dgp.nodes{i, j}.SS_idx;
                    g_info = functions(obj.dgp.nodes{i, j}.father.g);
                    if strcmp(g_info.function, '@(l)exp(l)')
                        lower(idx, :) = -15 * ones(obj.dgp.N+1, 1);
                        upper(idx, :) = 3 * ones(obj.dgp.N+1, 1);
                    elseif strcmp(g_info.function, '@(l)log(exp(l)+1)')
                        lower(idx, :) = -15 * ones(obj.dgp.N+1, 1);
                        upper(idx, :) = 3 * ones(obj.dgp.N+1, 1);
                    elseif strcmp(g_info.function, '@(l)l.^2')
                        lower(idx, :) = -5 * ones(obj.dgp.N+1, 1);
                        upper(idx, :) = 5 * ones(obj.dgp.N+1, 1);
                    elseif strcmp(g_info.function, '@tools.exp_lin')
                        lower(idx, :) = -6 * ones(obj.dgp.N+1, 1);
                        upper(idx, :) = 20 * ones(obj.dgp.N+1, 1);
                    else
                        error('Unsupported function g().')
                    end
                end
            end
            
        end
        
        function [obj_val, grad] = ss_MAP_handle(obj, U)
            % A Matlab handler for giving the objective function value and
            % gradients of the SS-DGP at current estimate U. Similar design
            % to dgp.dgp_opt_handle.
            % The loss function is 
            % L = -[Σ log p(y_i|x_i) + Σ log p(x_i|x_i-1) + log p(x0)]
            %
            % Argument:
            %   U:          The current estimate (value) of the DGP, at
            %               location x0,x1,...xN
            % Return:
            %   obj_val:    NEGATIVE log of the posterior propotion. 
            %               L = -log [p(y(1:N) | U(1:N)) p(U(1:N))]
            %   grad:       -∂L/∂U (A matrix)
            %
                        
            R = obj.dgp.R(1, 1);     % Simply consider R doesn't change
            N = obj.dgp.N;
            dt = obj.dt;
            
            % Init grad and obj
            grad = zeros(size(U));
            
            % Measurement loss p(y|u)
            obj_val = 0.5 * (...
                    (obj.dgp.y'-U(1, 2:end)) * (obj.dgp.y'-U(1, 2:end))' / R ...
                     + log(det(2*pi*R))...
                     );
                 
            % Loss for initial condition
            obj_val = obj_val + 0.5 * (...
                                U(:, 1)' / obj.P0 * U(:, 1) ...
                                + log(det(2*pi*obj.P0))...
                               );
                           
            % Loss for transitions densities
            for k = 2:N+1
                obj_val = obj_val + 0.5 * (...
                    (U(:, k)-obj.F(dt(k-1), U(:, k-1)))' ...
                    / obj.Q(dt(k-1), U(:, k-1)) * ...
                    (U(:, k)-obj.F(dt(k-1), U(:, k-1))) ...
                    + tools.log_det(2*pi*obj.Q(dt(k-1), U(:, k-1)))...
                    );
            end
            
            % Grad for x0 to x_{N-1}
            grad(:, 1) = obj.grad_log_pdf_u(U, 1);
            for k = 2:N
                grad(:, k) = obj.grad_log_pdf_u(U, k) + ...
                        obj.H' / R * (obj.H * U(:, k) - obj.dgp.y(k-1)); 
            end
            
            % Grad for x_N (last one)
            grad(:, N+1) = (obj.Q(obj.dt(N), U(:, N)) ...
                        \ (U(:, N+1) - obj.F(obj.dt(N), U(:, N)))) + ...
                        obj.H' / R * (obj.H * U(:, N+1) - obj.dgp.y(N));

        end

        %% Log pdf and gradient functions
        function grad = grad_log_pdf_u(obj, U, k)
            % Give the gradients of neg log posterior w.r.t. a state at u_k.
            % ∂L/∂u_k = ∂ -[log p(u_k|u_k-1) + log p(u_k+1|u_k)] / ∂ u_k
            %
            % Arguments:
            %   k:      The index in U. k from 1 to N. Corresponding to 
            %           u_0 to u_N-1. You should calcualte the last uN 
            %           seperately outside of this function.
            %
            % Returns:
            %   grad:   Gradient in vector form
            %
            
            % Grad for quardratic term
            if k == 1
                grad = obj.P0 \ U(:,1);
            else
                grad = obj.Q(obj.dt(k-1), U(:, k-1)) ...
                   \ (U(:, k) - obj.F(obj.dt(k-1), U(:, k-1)));
            end
            
            % Grad for tedious non-linear part
            F = obj.F(obj.dt(k), U(:,k));
            Q = obj.Q(obj.dt(k), U(:,k));
            for i = 1:obj.dim
                dFdu = obj.dFdu(obj.dt(k), U(:,k));
                dFdu = dFdu(:, i);
                dQdu = obj.dQdu{i}(obj.dt(k), U(:,k));
                grad(i, :) = grad(i, :) + 0.5 * ( ...
                        -U(:,k+1)' / Q * dQdu / Q * U(:,k+1)...
                        + 2*dFdu' / Q * (F - U(:, k+1)) ...
                        + F' / Q * dQdu / Q * (2*U(:, k+1) - F) ...
                        + trace(Q \ dQdu) ...
                        );
            end
        end
        
        %% Making predictions
        function [x_post, post, time] = predict(obj, node, query, int_method)
            % Make prediction (interpolation) of f at query points, using
            % the results from MAP/HMC.
            % Use Kalman filter and smoother solution to this ns-GP
            % regression problem.
            % 
            % Arguments:
            %   node:           Which GP node. [i j] index
            %   query:          Query points
            %   int_method:     Interpolation method
            %
            % Return:
            %   ...
            %
            if nargin < 4
                int_method = 'linear';
            end
            
            % Ensure query is column
            query = query(:);
            
            idx = obj.dgp.nodes{node(1), node(2)}.SS_idx;
            idx_l = obj.dgp.nodes{node(1), node(2)}.descendants.l.SS_idx;
            idx_s = obj.dgp.nodes{node(1), node(2)}.descendants.sigma.SS_idx;
            
            % Interpolation
            query_l = interp1(obj.dgp.x, obj.U(idx_l, 2:end), query, int_method);
            query_s = interp1(obj.dgp.x, obj.U(idx_s, 2:end), query, int_method);
            
            % Perform Kalman filter and smoother
            % Here are two plans for the posterior. See DGP.predict for
            % details.
            
            [x_post, post, time] = filters.NS_KF_RTS(obj, node, query_l, query_s, query);
            
        end
        
        %% Some tool function
        function idx = node_idx(obj, name, k)
            % A function that returns the state idx of the corresponding GP
            % node.
            %
            % Arguments:
            %   i, j:   The dgp.nodes{i, j} index
            %   k:      To which property of this GP? E.g., k=2 gives first
            %           derivative 
            % Return:
            %   idx:    The idx
            %
            [i, j] = obj.dgp.find_node_by_name(name);
            idx = obj.dgp.nodes{i, j}.SS_idx + (k-1);
            
        end
    end
end

