
%% Generate d_adjugate_F
syms F0 F1 F2 F3 F4 F5 F6 F7 F8;

F = [F0 F3 F6; F1 F4 F7; F2 F5 F8]; % Note, numbers are column major
adjugate_F = [ F4*F8-F5*F7  F5*F6-F3*F8  F3*F7-F4*F6; 
               F2*F7-F1*F8  F0*F8-F2*F6  F1*F6-F0*F7;  
               F1*F5-F2*F4  F2*F3-F0*F5  F0*F4-F1*F3 ];
        
d_adjugate_F = sym(zeros(9,9));           
           
for col = 1:3
    for row = 1:3
        di = F(row, col);
        d_adjugate_F((row-1)*3+1:row*3, (col-1)*3+1:col*3) = diff(adjugate_F, di);
    end
end

displaySymbolicLatex(d_adjugate_F);

%% Generate d_adjugate_F : dF where A = B : C  imples A(row, col) = sum(sum(B((row-1)*3+1:row*3, (col-1)*3+1:col*3) .* C)) 

syms dF0 dF1 dF2 dF3 dF4 dF5 dF6 dF7 dF8;

dF = [dF0 dF3 dF6; dF1 dF4 dF7; dF2 dF5 dF8]; % Note, numbers are column major

dJF_invTrans = sym(zeros(9,1));

for col = 1:3
    for row = 1:3
        dJF_invTrans(row+(col-1)*3) = sum(sum(d_adjugate_F((row-1)*3+1:row*3, (col-1)*3+1:col*3).*dF));
    end
end


%% compute values:

F_values = [7, -2, 0; -2 6, -2; 0, -2, 5];
dF_values = [17, -2.9, 0; -2.9, -1, -1; 0, -1, 7];

dF_invTrans_values = subs(subs(dJF_invTrans, dF, dF_values), F, F_values)
