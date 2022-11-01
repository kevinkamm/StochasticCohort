function L = generatorMatConst(Li)
%%GENERATORMATCONST returns the solution of the SDE in the Lie Algebra for
% given solved solutions in R^((d-1)^2),
% coefficients 
%
%   Input: 
%       Li ((d-1)^2 x N x M): positiv solutions of SDEs in R^((d-1)^2)
%   Ouput:
%       L ( d x d x N x M): solution in Lie Algebra

Li=reshape(Li,size(Li,1),1,1,size(Li,2),size(Li,3));

% 4x4 basis matrices of the lie algebra
G(1,:,:) = [-1     1     0     0
             0     0     0     0
             0     0     0     0
             0     0     0     0];


G(2,:,:) = [-1     0     1     0
             0     0     0     0
             0     0     0     0
             0     0     0     0];


G(3,:,:) = [-1     0     0     1
             0     0     0     0
             0     0     0     0
             0     0     0     0];


G(4,:,:) = [0     0     0     0
            1    -1     0     0
            0     0     0     0
            0     0     0     0];


G(5,:,:) = [0     0     0     0
            0    -1     1     0
            0     0     0     0
            0     0     0     0];


G(6,:,:) = [0     0     0     0
            0    -1     0     1
            0     0     0     0
            0     0     0     0];


G(7,:,:) = [0     0     0     0
            0     0     0     0
            1     0    -1     0
            0     0     0     0];


G(8,:,:) = [0     0     0     0
            0     0     0     0
            0     1    -1     0
            0     0     0     0];


G(9,:,:) = [0     0     0     0
            0     0     0     0
            0     0    -1     1
            0     0     0     0];

L=squeeze(sum(G.*Li,1));
end