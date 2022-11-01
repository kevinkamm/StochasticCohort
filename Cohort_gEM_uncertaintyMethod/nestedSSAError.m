clear all; close all; fclose('all');gpuDevice(1);rng(0);
pool=gcp('nocreate');
if isempty(pool)
%     pool=parpool('local');
    pool=parpool('threads');
end
%% Load data
% parameters of autoencoder
AE=560;
% parameters of generator
G=854;
% length of time-series
lenSeq=4;
% batch size
batch=128;
% epochs
epochs=40;
% number of time steps for Brownian motion
N=361;
% number of different ratings
K=4;
% months of time series
months=[1,3,6,12];
T = months(end)/12;

% relative path to data
relDir='Data';

ticLoadData=tic;

year=2022;
Cohortdataset=4;
[Rcohort,Rrec]=loadData(year,Cohortdataset);
% K=size(cohort,1);

% Probability of Default, 0=similar, 1=realistic, 2=unrealistically high
PDdataset=1;
PD=defaultProbabilityGenerator(Rrec,PDdataset);

ctimeLoadData=toc(ticLoadData);
fprintf('Elapsed time for loading data %1.3f\n',ctimeLoadData);
% order of Lie Algebra basis
[~,basisOrder]=lieAlgebraBasis(K);
%% Figure settings
backgroundColor='w';
textColor='k';
%% Uncertainty measure
distCohort=abs(Rrec(:,:,end)-Rcohort(:,:,end));
Rcertain=Rcohort(:,:,end)+(distCohort./sum(distCohort,2)).*distCohort;
% (Rrec(:,:,end)-Rcertain)./(sum(Rrec(:,:,end)-Rcertain,2))
% Rcertain=Rcertain+(Rrec(:,:,end)-Rcertain).^2./(sum(Rrec(:,:,end)-Rcertain,2))
%% Calibration
% comType='JLT';
comType='Exp';
optimizer='lsqnonlin';
M=1000;
% M=100;
% N=50*12+1;
N=25*12+1;

fileName = sprintf('GEM_%s_N%d_M%d_Cohort%d',optimizer,N,M,Cohortdataset);
saveCal = true;
loadCal = true;

disp('Start calibration under P')
[params,errCal,dWP,t,tInd,ctimeCal]=calibrateP(months,T,N,M,Rrec,Rcertain,...
                                    'Optimizer',optimizer,...
                                    'saveCal',saveCal,...
                                    'loadCal',loadCal,...
                                    'fileNameCal',fileName);
fprintf('Elapsed time for calibration under P %g s with error %1.3e\n',...
        ctimeCal,errCal);
aCal=params(1:9);
bCal=params(10:18);
sigmaCal=params(19:27);
paramTable = array2table([aCal,bCal,sigmaCal],...
                         'VariableNames',{'a','b','sigma'},...
                         'RowNames',basisOrder);
disp('Parameters after calibration')
disp(paramTable)

fileName = sprintf('GEM_%s_%s_N%d_M%d_%d',optimizer,comType,N,M,PDdataset);
disp('Start calibration under Q')
[h,errCalQ,ctimeCalQ,dWQ]=calibrateQ(months,T,t,tInd,params,dWP,PD,comType,...
                                'Optimizer',optimizer,...
                                'saveCal',saveCal,...
                                'loadCal',loadCal,...
                                'fileNameCal',fileName);
fprintf('Elapsed time for calibration under Q %g s with error %1.3e\n',...
        ctimeCalQ,errCalQ);
%% Simulation with calibrated parameters
% % simulation under P
disp('Simulation under P')
ticSimP=tic;
RcalP = gEMP(aCal,bCal,sigmaCal,t,1:1:length(t),dWP);
ctimeSimP=toc(ticSimP);
fprintf('Elapsed time for simulation under P %g s\n',...
        ctimeSimP);
disp('Simulation under Q')
ticSimQ=tic;
[RcalQ,AQ] = gEMQ(aCal,bCal,sigmaCal,h,t,1:1:length(t),dWQ,comType);
ctimeSimQ=toc(ticSimQ);
fprintf('Elapsed time for simulation under Q %g s\n',...
        ctimeSimQ);
%%
% [figRDP,histCalP]=plotRatingDist(RcalP(:,:,tInd,:),Rgan(:,:,:,1:M),[],...
%                     'backgroundColor',backgroundColor,...
%                     'textColor',textColor);
% figRDQ=plotRatingDist(RcalQ(:,:,tInd,:),[],histCalP,...
%                     'backgroundColor',backgroundColor,...
%                     'textColor',textColor);
%%
% figTraP=plotTrajectories(t,tInd,RcalP,Rgan,[],...
%                     'backgroundColor',backgroundColor,...
%                     'textColor',textColor);
% figTraQ=plotTrajectories(t,tInd,RcalQ,[],RcalP,...
%                     'backgroundColor',backgroundColor,...
%                     'textColor',textColor);
%% Rating properties
% [mDCgan,sDDgan,dMLgan,iRSgan,rSOgan]=ratingProperties(Rgan,months);
% [mDCcalP,sDDcalP,dMLcalP,iRScalP,rSOcalP]=ratingProperties(RcalP(:,:,tInd,:),months);
% [mDCcalQ,sDDcalQ,dMLcalQ,iRScalQ,rSOcalQ]=ratingProperties(RcalQ(:,:,tInd,:),months);

%% Nested SSA
Nssa=100*12+1; % for intermediate transitions we need a finer grid for the nested SSA than for the simulation 
dtSSA=T/(Nssa-1);
dt=T./(N-1);
M1=10;
M2=10000;
XQ=zeros(K-1,Nssa,M2,M1,'uint8');
tSSA=linspace(0,T,Nssa);
tIssa=zeros(size(months));
for i=1:1:length(tIssa)
    tIssa(i)=find(tSSA>=months(i)/12,1,'first');
end
tic;
parfor i0=1:K-1
XQ(i0,:,:,:)=nestedSSA(AQ./dt,t,i0,M1,M2,dtSSA);
end
toc;
i0=3;
% figure();
% plot(t,reshape(X,length(t),[]))
RXQ=zeros(K-1,K,length(tIssa),M1);
for i=1:1:K-1
    for j=1:1:K
        RXQ(i,j,:,:)=permute(mean(XQ(i,tIssa,:,:)==j,3),[1 3 2 4]);
    end
end
mean(abs(RcalQ(1:K-1,:,tInd,1:M1)-RXQ),4)

