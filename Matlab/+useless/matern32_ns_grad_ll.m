function z = matern32_ns_grad_ll(g1,g3,r)
%MATERN32_NS_GRAD_LL
%    Z = MATERN32_NS_GRAD_LL(G1,G3,R)

%    This function was generated by the Symbolic Math Toolbox version 8.0.
%    12-Feb-2020 18:53:04

z = g1.^2.*1.0./g3.^2.*r.^2.*exp(-sqrt(3.0).*r.*sqrt(1.0./g3)).*(3.0./2.0);