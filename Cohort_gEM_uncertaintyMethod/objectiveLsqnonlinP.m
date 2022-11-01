function y=objectiveLsqnonlinP(x,t,tInd,dW,Rrec,Rcohort)
a=x(1:9);
b=x(10:18);
sigma=x(19:27);
Rlie = gEMP(a,b,sigma,t,tInd,dW);

[MuLie,SigmaLie,~,~]=ratingMoments(Rlie);


y1=abs(MuLie-Rrec);
sigmaCohort=abs(Rrec-Rcohort).^2;

y2=abs(SigmaLie-sigmaCohort);
y=[y1(:);y2(:)];

% lvl=1;
% 
% y2=abs(mean(Rlie(1:end-1,:,:,:)>=lower)-lvl);
% y3=abs(mean(Rlie(1:end-1,:,:,:)<=upper)-lvl);
% y=[y1(:);y2(:);y3(:)];

% y3=1.*abs(SkewLie-SkewGAN);
% y4=1.*abs(KurtLie-KurtGAN);
% y=[y1(:);y2(:);y3(:);y4(:)];
% y=[y1(:);y2(:)];

end