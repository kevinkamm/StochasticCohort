function [G,varargout]=lieAlgebraBasis(n)
%%LIEALGEBRABASIS return the basis of the Lie-Algebra of stochastic
% matrices with absorbing default rate, i.e. last line are zeros. There are
% (n-1)^2 basis elements.
%
%   Input:
%       n (int): dimension of the stochastic matrices
%
%   Output:
%       G (n x n x (n-1)^2 array): basis elements of Lie Algebra

G=zeros((n-1)^2,n,n);
R=cell((n-1)^2,1);
k=1;
for i = 1:1:n-1
    for j=1:1:i-1
        G(k,i,i)=-1;
        G(k,i,j)=1;
        R{k}=sprintf('%d-%d',i,j);
        k=k+1;
    end
    for j=i+1:1:n
        G(k,i,i)=-1;
        G(k,i,j)=1;
        R{k}=sprintf('%d-%d',i,j);
        k=k+1;
    end
end
if nargout>1
    varargout{1}=R;
end
end