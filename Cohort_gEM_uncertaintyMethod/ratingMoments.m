function [Mu,Sigma,Skew,Kurt]=ratingMoments(R)

Mu=mean(R(1:end-1,:,:,:),4);
Sigma=var(R(1:end-1,:,:,:),0,4);
% Skew=skewness(R(1:end-1,:,:,:),1,4);
% Kurt=kurtosis(R(1:end-1,:,:,:),1,4);
Skew=moment(R(1:end-1,:,:,:),3,4);
Kurt=moment(R(1:end-1,:,:,:),4,4);


end