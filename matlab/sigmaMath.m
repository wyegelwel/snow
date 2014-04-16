% Simple test
mu = 1;
lambda = 1;
xi = 1;

Fe = eye(3);
Fp = eye(3);

Jep = det(Fe);
Jpp = det(Fp);
Jp = Jep * Jpp;

uFp = mu*exp(xi*(1-Jpp));
lFp = lambda*exp(xi*(1-Jpp));

[P, S, Q] = svd(Fe);
Re = P*Q';
sigmaT = (2*uFp/Jp)*(Fe-Re)*Fe'+(lFp/Jp)*(Jep - 1)*Jep*eye(3);
sigma = sigmaT'

% Complex test
mu = 1;
lambda = 1;
xi = 1;

Fe = [1 3 2; -1 0 2; 3 1 -1];
Fp = eye(3);

Jep = det(Fe);
Jpp = det(Fp);
Jp = Jep * Jpp;

uFp = mu*exp(xi*(1-Jpp));
lFp = lambda*exp(xi*(1-Jpp));

[P, S, Q] = svd(Fe);
Re = P*Q';
sigmaT = (2*uFp/Jp)*(Fe-Re)*Fe'+(lFp/Jp)*(Jepp - 1)*Jep*eye(3);
sigma = sigmaT'

% More complex test
mu = 1;
lambda = 1;
xi = 1;

theta = 30*pi/180;
Fe = [cos(theta), sin(theta), 0; -sin(theta), cos(theta), 0; 0,0,1]*.7;
theta2 = 45*pi/180;
Fp = [1,0,0; 0, cos(theta2), sin(theta2); 0, -sin(theta2), cos(theta2)]*.9;

Jep = det(Fe);
Jpp = det(Fp);
Jp = Jep * Jpp;

uFp = mu*exp(xi*(1-Jpp))
lFp = lambda*exp(xi*(1-Jpp))

[P, S, Q] = svd(Fe);
Re = P*Q';
sigmaT = (2*uFp/Jp)*(Fe-Re)*Fe'+(lFp/Jp)*(Jep - 1)*Jep*eye(3);
sigma = sigmaT'