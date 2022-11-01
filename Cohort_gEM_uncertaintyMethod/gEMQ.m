function [Rlie,varargout] = gEMQ(a,b,sigma,h,t,tInd,dW,comType)
%%GEM geometric Euler-Maruyama
% x - coefficients for linear combination
% output: M - solution of SDE in Lie group
d = 4;
N = length(t);
M = size(dW,3);
K = sqrt(size(dW,1))+1;
dt = t(end)./(N+1);

h=reshape(h,size(h,1),1);
h=[h;1];
switch comType
    case 'Exp'
        H=h./h';
    case 'JLT'
        H=h;
end
% H=h./h';
% H=h;
ind=ones(K,K,'logical');
ind(eye(K,'logical'))=0;
ind(end,:)=0;
A=zeros(size(ind));
B=zeros(size(ind));
Sigma=zeros(size(ind));
A(ind)=a;
B(ind)=b;
Sigma(ind)=sigma;
B=(B+H.*Sigma)';
A=A';
Sigma=Sigma';
a=A(ind');
b=B(ind');
sigma=Sigma(ind');

t=reshape(t,1,[],1);

% Li=zeros(d-1,d-1,N,M);
dW = reshape(dW,(K-1)^2,N-1,M);
W=zeros((K-1)^2,N,M);
W(:,2:end,:)=dW;
W=cumsum(W,2);
%Euler-Maruyama-increments
% tInd2=[1,tInd];
% for ti=1:1:length(tInd2)-1
%     Li(:,:,tInd2(ti):tInd2(ti+1),:)=b(:,:,ti).*t(1,1,tInd2(ti):tInd2(ti+1))+sigma.*W(:,:,tInd2(ti):tInd2(ti+1),:);
% end
Li=b.*t+sigma.*W;
Li=cumsum(abs(Li).^a,2).*dt;
% fix numerical error
% Li(Li<0)=0;
L = generatorMatConst(Li);
dL=diff(L,1,3);
dL=reshape(dL,d,d,(N-1)*M);

dLexp = zeros(size(dL));

parfor twi=1:size(dLexp,3)
    dLexp(:,:,twi)=expm(dL(:,:,twi));
end

dLexp=reshape(dLexp,d,d,N-1,M);
Rlie=zeros(d,d,length(tInd),M);
temp=repmat(eye(d),1,1,1,M);

k=1;
if tInd(k)==1
    Rlie(:,:,k,:)=temp;
    k=k+1;
end
for ti=1:1:N-1
    temp=pagemtimes(squeeze(temp),squeeze(dLexp(:,:,ti,:)));
    if tInd(k)==ti+1
        Rlie(:,:,k,:)=reshape(temp,d,d,1,M);
        k=k+1;
    end
end
if nargout>1
    varargout{1}=reshape(dL,d,d,N-1,M);
end
end