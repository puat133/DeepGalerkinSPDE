function z = matern32_ns_grad_s(g1,g2,g3,g4,r)
%MATERN32_NS_GRAD_S
%    Z = MATERN32_NS_GRAD_S(G1,G2,G3,G4,R)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    12-Feb-2020 19:27:07
%
% Tedious derivatives of non-stationary Matern 32 w.r.t. l
% Symbolic tool is the only healthy way to give it, thougn not efficient.
%
% Let 
%   g1 = g(σ(x))
%   g2 = g(σ(x'))
%   g3 = g(l(x))
%   g4 = g(l(x'))
%
% Code 
% syms g1 g2 g3 g4 r
% c(g1, g2, g3, g4, r) = g1*g2*g3^0.25*g4^0.25*(2/(g3+g4))^0.5*(1+r*sqrt(6/(g3+g4)))*exp(-r*sqrt(6/(g3+g4)))
% z = simplify(diff(c, g2))
% f = matlabFunction(z, 'File', 'matern32_ns_grad_s')
% 

t2 = sqrt(6.0);
t3 = g3+g4;
t4 = 1.0./t3;
t5 = sqrt(t4);
z = sqrt(2.0).*g1.*g3.^(1.0./4.0).*g4.^(1.0./4.0).*t5.*exp(-r.*t2.*t5).*(r.*t2.*t5+1.0);
