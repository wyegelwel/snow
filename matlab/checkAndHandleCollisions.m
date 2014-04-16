%simple test
n = [0,1,0];
v = [0,-1,-1];
coeffFriction = .5;
vn = v*n';
vt = v - n*vn;
vRel = (1+coeffFriction*vn)*vt


% simple multiple test, colliding with sphere
n = [0, -1, 0];
v = [.5, 1, -1];
coeffFriction = .5;
vn = v*n';
vt = v-n*vn;
vRel = (1+coeffFriction*vn/norm(vt))*vt