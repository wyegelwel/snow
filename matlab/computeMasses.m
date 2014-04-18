%mass weight tests
N = @(d) (0<=d & d<1).*(.5*d.^3-d.^2+2/3) + (1<=d & d<2).*(-1/6*d.^3+d.^2-2*d+4/3);
particles = [ 0,0,0; 1,1,1];
particleMasses = [1, 2];
cellDim = [1, 1, 1];
h = 1;
masses = zeros(cellDim+1);

for pI = 1:size(particles,1)
   for i = 1:cellDim(1)+1
       for j = 1:cellDim(2)+1
           for k = 1:cellDim(3)+1
                nodePos = [i,j,k]*h-1;
                dx = abs(particles(pI,:) - nodePos );
                w = prod(N(dx/h));
                masses(i,j,k) = masses(i,j,k) + particleMasses(pI)*w;
           end
       end
   end
end

masses

%mass weight tests
N = @(d) (0<=d & d<1).*(.5*d.^3-d.^2+2/3) + (1<=d & d<2).*(-1/6*d.^3+d.^2-2*d+4/3);
particles = [ 0,0,0; 1,1,1];
particleMasses = [1, 2];
cellDim = [5, 5, 5];
h = 1;
masses = zeros(cellDim+1);

for pI = 1:size(particles,1)
   for i = 1:cellDim(1)+1
       for j = 1:cellDim(2)+1
           for k = 1:cellDim(3)+1
                nodePos = [i,j,k]*h-1- [2,2,2];
                dx = abs(particles(pI,:) - nodePos );
                w = prod(N(dx/h));
                masses(i,j,k) = masses(i,j,k) + particleMasses(pI)*w;
           end
       end
   end
end

masses