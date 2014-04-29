%% Simple
N = @(d) (0<=d & d<1).*(.5*d.^3-d.^2+2/3) + (1<=d & d<2).*(-1/6*d.^3+d.^2-2*d+4/3);
Nd = @(d) (0<=d & d<1).*(3/2*d.^2-2*d) + (1<=d & d<2).*(-.5*d.^2+2*d-2);


particles_pos = [.2,.2,.2;];
particles_masses = [1e-7];
numParticles = length(particles_masses);

grid_dim = [2,2,2]; grid_nodeDim = grid_dim+1;
grid_h = .2; grid_pos = [0,0,0];
numNodes = prod(grid_nodeDim);

nodes_velocities = ones(numNodes, 3);

dus = ones(numNodes,3);

dF = zeros(3,3, numParticles);

for pI = 1:numParticles
   for i = 1:grid_nodeDim(1)
       for j = 1:grid_nodeDim(2)
           for k = 1:grid_nodeDim(3)
                nodePos = ([i,j,k]-1)*grid_h;
                dx = abs(particles_pos(pI,:) - nodePos );
                n = N(dx);
                nx = sign(particles_pos(pI,:) - nodePos).*Nd(dx);
                wg = [ nx(1)*n(2)*n(3), n(1)*nx(2)*n(3), n(1)*n(2)*nx(3) ];
                
                nodeIndex = (i-1)*grid_nodeDim(2)*grid_nodeDim(3) + (j-1)*grid_nodeDim(3) + k;
                
                du = dus(nodeIndex,:);
                dF(:,:, pI) = dF(:,:, pI) + du'*wg;
           end
       end
   end
end

%% complex
N = @(d) (0<=d & d<1).*(.5*d.^3-d.^2+2/3) + (1<=d & d<2).*(-1/6*d.^3+d.^2-2*d+4/3);
Nd = @(d) (0<=d & d<1).*(3/2*d.^2-2*d) + (1<=d & d<2).*(-.5*d.^2+2*d-2);

dim = 15;
grid_dim = [dim,dim,dim]; grid_nodeDim = grid_dim+1;
grid_h = .2; grid_pos = [0,0,0];
numNodes = prod(grid_nodeDim);

numParticles = 256;
particles_pos = ones(numParticles, 3) .* repmat([0:numParticles-1]'/(dim+1.0) * grid_h, 1, 3);
particles_masses = ones(numParticles,1)*1e-7;


nodes_velocities = ones(numNodes, 3);

dus = ones(numNodes,3);

dF = zeros(3,3, numParticles);
vGradient = zeros(3,3, numParticles);

for pI = 1:numParticles
   for i = 1:grid_nodeDim(1)
       for j = 1:grid_nodeDim(2)
           for k = 1:grid_nodeDim(3)
                nodePos = ([i,j,k]-1);
                d = particles_pos(pI,:)/grid_h - nodePos;
                dx = abs(d );
                n = N(dx);
                nx = sign(d).*Nd(dx);
                wg = [ nx(1)*n(2)*n(3), n(1)*nx(2)*n(3), n(1)*n(2)*nx(3) ];
                
                nodeIndex = (i-1)*grid_nodeDim(2)*grid_nodeDim(3) + (j-1)*grid_nodeDim(3) + k;
                
                du = dus(nodeIndex,:);
                dF(:,:, pI) = dF(:,:, pI) + du'*wg;
                vGradient(:,:,pI) = vGradient(:,:,pI) + nodes_velocities(pI,:)' * wg;
           end
       end
   end
   vGradient(:,:,pI) = vGradient(:,:,pI) + eye(3);
end

