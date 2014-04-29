%% simple computedR

dF = zeros(3);
Se = eye(3);
Re = eye(3);

V = Re'*dF - dF'*Re;
A = [ Se(1,1) + Se(2,2), S(3,2), -S(3,1); 
      S(3,2), Se(1,1) + Se(3,3), S(2,1);
      -S(3,1), S(2,1), Se(2,2) + Se(3,3) ];

b = [ V(1,2); V(1,3); V(2,3)];

x = A\b;

RTdR = [0, x(1), x(2); 
        -x(1), 0, x(3);
        -x(2), -x(3), 0];
    
dR = Re * RTdR;


%% more complex computedR


dF = ones(3);
Se = [1.0, -.1, 3;  -.1, 1.2, 0; 3, 0, .3 ]';
Re = [cos(pi), -sin(pi), 0; sin(pi), cos(pi), 0; 0, 0, 1];

V = Re'*dF - dF'*Re;
A = [ Se(1,1) + Se(2,2), Se(3,2), -Se(3,1); 
      Se(3,2), Se(1,1) + Se(3,3), Se(2,1);
      -Se(3,1), Se(2,1), Se(2,2) + Se(3,3) ];

b = [ V(1,2); V(1,3); V(2,3)];

x = A\b;

RTdR = [0, x(1), x(2); 
        -x(1), 0, x(3);
        -x(2), -x(3), 0];
    
dR = Re * RTdR;