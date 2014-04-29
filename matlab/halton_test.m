function halton_test

n = 1000;
X = zeros(n,3);
base = [3 5 7];

for i=1:n
   for j=1:3
      X(i,j) = halton(i+20,base(j)); 
   end 
end

scatter3(X(:,1),X(:,2),X(:,3));

end

function [ result ] = halton(index, base)
    result = 0;
    f = 1/base;
    i = index;
    while (i > 0)
       result = result + f*(mod(i,base));
       i = floor(i/base);
       f = f / base;
    end
end