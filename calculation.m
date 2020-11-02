% e_ij = [1., 0., 0.;
%         0., 1., 0.;
%         0., 0., 1.];
e_ij = [1., 0.;
        0., 1.];
e1 = [1., 0., 0.];
syms C11 C12 C22 C66 E11 E22 E12
C = [C11, C12, 0.; 
    C12, C22, 0.;
    0., 0., C66];
eps = [E11 E22 2*E12];
eps_originial = [E11, E12;
                E12, E22];

Sigma = C * transpose(eps);
disp(Sigma)

dot(C, eps);

% A = C*e_ij;%transpose(e1) ;
% disp(A)

% A = C*e_ij;%transpose(e1) ;
% disp(A)
% d = dot(C,transpose(e1));
% disp(d)
 

%% Kronnecker delta 
syms d1 d2 d3
e_ij = [d1 d2 d3];

% delta_ij = outer

%% 4th order tensor

% Create a three-dimensional MDA
