function W=brownianMotion(t,M,n)
N=length(t);
dt = t(end)/(N+1);
W=zeros(N,M,n);
dW=sqrt(dt).*randn(N-1,M,n);
W(2:end,:,:)=dW;
W=cumsum(W,1);
end