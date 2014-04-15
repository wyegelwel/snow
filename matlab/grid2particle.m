clear;

alpha = 0.95;

dim = 8;
h = 1.0/dim;
pos = [0, 0, 0];

particleCount = dim*dim*dim;
for i = 0:(dim-1)
    for j = 0:(dim-1)
        for k = 0:(dim-1)
            index = i*dim*dim + j*dim + k + 1;
            particles(index).position = pos + h*[i+0.5 j+0.5 k+0.5];
            particles(index).velocity = [0 -0.124 0];
            particles(index).elasticF = diag([1 1 1]);
            particles(index).plasticF = diag([1 1 1]);
        end
    end
end

for i = 0:dim
    for j = 0:dim
        for k = 0:dim
            index = i*(dim+1)*(dim+1) + j*(dim+1) + k + 1;
            nodes(index).velocity = [0 -0.125 0];
            nodes(index).velocityChange = [0 -0.001 0];
        end
    end
end

N = @( d ) ( 0 <= d && d < 1 ) * ( 0.5*d*d*d - d*d + 2/3 ) + ( 1 <= d && d < 2 ) * ( -1/6*d*d*d + d*d - 2*d + 4/3 );
weight = @( dx ) N(abs(dx(1)))*N(abs(dx(2)))*N(abs(dx(3)));

% test = -3:0.01:3;
% Ntest = zeros(size(test));
% for i = 1:length(test)
%    Ntest(i) = N(abs(test(i)));
% end
% subplot( 1, 2, 1 );
% plot( test, Ntest ); shg;

Nd = @( d ) ( 0 <= d && d < 1 ) * ( 1.5*d*d - 2*d ) + ( 1 <= d && d < 2 ) * ( -0.5*d*d + 2*d - 2 );
weightGradient = @( dx ) sign(dx) .* [Nd(abs(dx(1)))*N(abs(dx(2)))*N(abs(dx(3))) N(abs(dx(1)))*Nd(abs(dx(2)))*N(abs(dx(3))) N(abs(dx(1)))*N(abs(dx(2)))*Nd(abs(dx(3)))];

% Ndtest = zeros(size(test));
% for i = 1:length(test)
%    g = weightGradient([test(i) test(i) test(i)]);
%    Ndtest(i) = g(1);
% end
% subplot( 1, 2, 2 );
% plot( test, Ndtest ); shg;

for iteration = 1:5
    disp( ['Iteration ' num2str(iteration)] );
    for p = 1:particleCount
        velocityGradient = zeros(3, 3);
        particlePos = (particles(p).position - pos)./h;
        v_PIC = [0 0 0];
        dv_FLIP = [0 0 0];
        for i = 0:dim
            dx = i - particlePos(1);
            if ( abs(dx) >= 2 ); continue; end
            for j = 0:dim
                dy = j - particlePos(2);
                if ( abs(dy) >= 2 ); continue; end
                for k = 0:dim
                    dz = k - particlePos(3);
                    if ( abs(dz) >= 2 ); continue; end
                    index = i*dim*dim + j*dim + k + 1;
                    DX = [dx dy dz];
                    w = weight( DX );
                    wg = weightGradient( DX );
                    v_PIC = v_PIC + w*nodes(index).velocity;
                    dv_FLIP = dv_FLIP + w*nodes(index).velocityChange;
                    velocityGradient(2,:) = velocityGradient(2,:) + nodes(index).velocity(2) * wg;
                end
            end
        end
        particles(p).velocity = (1-alpha)*v_PIC + alpha*(particles(p).velocity+dv_FLIP);
        particles(p).velocityGradient = velocityGradient;
    end
end

velocity = zeros( 3, particleCount );
for i = 1:particleCount
    velocity(:,i) = particles(i).velocity;
end

gradientRows = zeros( 3, particleCount );
for i = 1:particleCount
    gradientRows(:, i) = particles(i).velocityGradient(2,:);
    disp( particles(i).velocityGradient );
    disp('   ');
end

