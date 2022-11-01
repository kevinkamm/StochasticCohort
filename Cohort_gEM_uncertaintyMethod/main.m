clear all; close all; fclose('all'); rng(0);
pool=gcp('nocreate');
if isempty(pool)
%     pool=parpool('local');
    pool=parpool('threads');
end
availableGPUs = gpuDeviceCount('available');
if availableGPUs > 0
    gpuDevice([]); % clears GPU
    gpuDevice(1); % selects first GPU, change for multiple with spmd
end
%% Load data
% parameters of TimeGAN code in Python:
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

% parameters for this code
% months of time series
months=[1,3,6,12];
T = months(end)/12;

ticLoadData=tic;
year=2022;
Cohortdataset=4;
[Rcohort,Rrec]=loadData(year,Cohortdataset);
% K=size(cohort,1);

% Probability of Default, 0=very similar, 1=realistic, 2=unrealistically high
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
comType='JLT';
% comType='Exp';
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
% simulation under P
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
%% Plots
[figRDP,histCalP]=plotRatingDist(RcalP(:,:,tInd(end),:),[],[],...
                    'backgroundColor',backgroundColor,...
                    'textColor',textColor);
figRDQ=plotRatingDist(RcalQ(:,:,tInd(end),:),[],histCalP,...
                    'backgroundColor',backgroundColor,...
                    'textColor',textColor);
%%
figTraP=plotTrajectories(t,tInd(end),RcalP,Rrec(:,:,end),[],...
                    'backgroundColor',backgroundColor,...
                    'textColor',textColor);
figTraQ=plotTrajectories(t,tInd(end),RcalQ,[],RcalP,...
                    'backgroundColor',backgroundColor,...
                    'textColor',textColor);
%% Rating properties
[mDCrec,sDDrec,dMLrec,iRSrec,rSOrec]=ratingProperties(Rrec,months);
[mDCcalP,sDDcalP,dMLcalP,iRScalP,rSOcalP]=ratingProperties(RcalP(:,:,tInd,:),months);
[mDCcalQ,sDDcalQ,dMLcalQ,iRScalQ,rSOcalQ]=ratingProperties(RcalQ(:,:,tInd,:),months);
%% Output
compileLatex=false;
output(fileName,...
       N,M,...
       ctimeCal,ctimeCalQ,...
       errCal,errCalQ,...
       paramTable,h,...
       mDCrec,sDDrec,dMLrec,iRSrec,rSOrec,...
       mDCcalP,sDDcalP,dMLcalP,iRScalP,rSOcalP,...
       mDCcalQ,sDDcalQ,dMLcalQ,iRScalQ,rSOcalQ,...
       figRDP,figRDQ,figTraP,figTraQ,...
       'backgroundColor',backgroundColor,...
       'textColor',textColor,...
       'compileLatex',compileLatex)
disp('Done')
%% Misc
% replace(latex(sym(h')),' &',',\, ')
% latex(sym(distCohort))
% latex(sym(Rcertain))