function y=objectiveLsqnonlinQ(h,params,t,tInd,dW,PD,comType)
a=params(1:9);
b=params(10:18);
sigma=params(19:27);
RlieQ = gEMQ(a,b,sigma,h,t,tInd,dW,comType);

y=abs(squeeze(mean(RlieQ(:,end,end,:),4))-PD(:,end));
y=y(:);
end