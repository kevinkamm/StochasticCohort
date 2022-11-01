function y=objectiveFminconP(x,t,tInd,dW,MuGAN,SigmaGAN,SkewGAN,KurtGAN)
a=x(1:9);
b=x(10:18);
sigma=x(19:27);
Rlie = gEM(a,b,sigma,t,tInd,dW);


%% Moments
[MuLie,SigmaLie,SkewLie,KurtLie]=ratingMoments(Rlie);

y=sum(abs(MuLie-MuGAN),'all')+...
  10.*sum(abs(SigmaLie-SigmaGAN),'all');
end