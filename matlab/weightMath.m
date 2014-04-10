% for use in CUDA tests
N = @(d) (0<=d & d<1).*(.5*d.^3-d.^2+2/3) + (1<=d & d<2).*(-1/6*d.^3+d.^2-2*d+4/3);
Nd = @(d) (0<=d & d<1).*(3/2*d.^2-2*d) + (1<=d & d<2).*(-.5*d.^2+2*d-2);

h=.013;
gd = [12,10,3]*h;
xp = [4,4,.1];

d = abs(xp-gd);
n = N(d/h);
w = prod(n);
nx = sign(xp).*Nd(d/h);
wg = [ nx(1)*n(2)*n(3), n(1)*nx(2)*n(3), n(1)*n(2)*nx(3) ];
w
wg

% just to make sure we have the right kernel...
%x = -3:.1:3;
%plot(x,sign(x).*Nd(abs(x)));