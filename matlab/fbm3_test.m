% fractal brownian motion testing
function fbm3_test
    % plot scatter grid of points
    gv = 0.1:1:20;
    [X,Y,Z] = meshgrid(gv,gv,gv);
    %[X,Y,Z] = sphere(10);
    X = X(:); Y = Y(:); Z = Z(:);
    C = zeros(length(X),1);
    center = [.5,.5,.5];
    for i=1:length(C)
        p = [X(i),Y(i),Z(i)];
        dist2 = sum((p-center).*(p-center));
        C(i) = fbm3(p);
    end
    C = (C+.3)*1.6667;
    %scatter3(X,Y,Z,3,C);
    %title('Spatially Varying Stiffness');
    %axis square;
    min(C)
    max(C)
end

function [out] = fract(x)
    out = x - floor(x);
end

function [out] = hash(n)
    out = fract(sin(n)*43758.5453123);
end

function [out] = mix(x,y,a)
    out = (1-a)*x + a*y;
end

function [out] = noise3(x)
   p = floor(x);
   f = fract(x);
   f = f.*f.*(3-2*f);
   n = x(1) + 157*x(2) + 113*x(3);
   out = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f(1)), ...
                   mix( hash(n+157.0), hash(n+158.0),f(1)),f(2)), ...
               mix(mix( hash(n+113.0), hash(n+114.0),f(1)), ...
                   mix( hash(n+270.0), hash(n+271.0),f(1)),f(2)),f(3));
end

function [out] = fbm3(p)
    f = 0;
    for i=1:6
       x = pow2(i);
       f = f + (noise3(p*x)-.5)/x;
    end
    out = f;
end