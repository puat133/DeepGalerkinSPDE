function y = rect(x, type)
%Magnitude-varying Rectangle wave in many types
% 
% Arguments:
%   x:      Input
%   type:   What type of rect signal?


if nargin < 2
    type = "rect-2";
end

switch type
    
    case "rect-1"
        
        t1 = 1/6;
        t2 = 2/6;
        t3 = 3/6;
        t4 = 4/6;
        t5 = 5/6;

        y((x>=0)&(x<t1)) = 0;
        y((x>=t1)&(x<t2)) = 1;
        y((x>=t2)&(x<t3)) = 0;
        y((x>=t3)&(x<t4)) = 0.6;
        y((x>=t4)&(x<t5)) = 0;
        y((x>=t5)) = 0.4;
        
    case "rect-2"
        
        t1 = 1/7;
        t2 = 2/7;
        t3 = 3/7;
        t4 = 4/7;
        t5 = 5/7;
        t6 = 6/7;

        y((x>=0)&(x<t1)) = 0;
        y((x>=t1)&(x<t2)) = 1;
        y((x>=t2)&(x<t3)) = 0;
        y((x>=t3)&(x<t4)) = 0.6;
        y((x>=t4)&(x<t5)) = 0;
        y((x>=t5)&(x<t6)) = 0.4;
        y((x>=t6)) = 0;
        
    case "rect-3"
        
        t1 = 1/3;
        t2 = 2/3;
        
        y((x>=0)&(x<t1)) = 0;
        y((x>=t1)&(x<t2)) = 1;
        y((x>=t2)) = 0.5;
        
    case "rect-4"
        
        t1 = 1/2;
        
        y((x>=0)&(x<t1)) = 0;
        y((x>=t1)) = 1;
        
end

y = y(:);

end
