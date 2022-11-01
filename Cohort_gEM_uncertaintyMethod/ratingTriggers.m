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
ratings=arrayfun(@(x)char(x+64),[1:K],'UniformOutput',false);
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
[RcalP,AP] = gEMP(aCal,bCal,sigmaCal,t,1:1:length(t),dWP);
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
%%
fileDir=['Results/Pdf/GEM_',optimizer,'RatingTriggers','_PD_',num2str(PDdataset)];
% delDir(fileDir);
mkDir(fileDir);
%% Nested SSA
Nssa=100*12+1; % for intermediate transitions we need a finer grid for the nested SSA than for the simulation 
dtSSA=T/(Nssa-1);
dt=T./(N-1);
M1=100;
M2=100;
XQ=zeros(K-1,Nssa,M2,M1,'uint8');
XP=zeros(K-1,Nssa,M2,M1,'uint8');
tSSA=linspace(0,T,Nssa);
tIssa=zeros(size(months));
for i=1:1:length(tIssa)
    tIssa(i)=find(tSSA>=months(i)/12,1,'first');
end
tic;
for i0=1:K-1
    XQ(i0,:,:,:)=nestedSSA(AQ./dt,t,i0,M1,M2,dtSSA);
end
toc;
XQ=reshape(XQ,K-1,Nssa,M1*M2);
tic;
for i0=1:K-1
    XP(i0,:,:,:)=nestedSSA(AP./dt,t,i0,M1,M2,dtSSA);
end
toc;
XP=reshape(XP,K-1,Nssa,M1*M2);
%% Predefault Distribution
% Q
[Q,numOfDefaultsQ]=preDefaultDistribution(XQ,K);
preDefaultPlotsQ=plotPreDefault(Q,numOfDefaultsQ,M1*M2,ratings);
saveFigures(preDefaultPlotsQ,fileDir,'preDefaultPlotsQ',backgroundColor);
% P
[P,numOfDefaultsP]=preDefaultDistribution(XP,K);
preDefaultPlotsP=plotPreDefault(P,numOfDefaultsP,M1*M2,ratings);
saveFigures(preDefaultPlotsP,fileDir,'preDefaultPlotsP',backgroundColor);
%% Collateral and bilateral CVA
scale=1e6; % million euros
V=scale.*portfolio(0,T,dtSSA,25,M1*M2);
day=linspace(1,T*365,365)./365;
ti=zeros(size(day));

for i=1:1:length(ti)
    ti(i)=find(tSSA<=day(i),1,'last');
end
m=0; %minimal transfer amount
r=0; %interest rate
RI=squeeze(XQ(1,:,:));
RC=squeeze(XQ(floor(length(ratings)/2),:,:));
LGDI=0.6;
LGDC=0.6;
%% without collateralization
thresholdsIUC=scale.*1000000.*ones(length(ratings),1);%disable tresholds by making it very large
thresholdsCUC=scale.*1000000.*ones(length(ratings),1);
[CUC,CIUC,CCUC,Vadjusted]=collateral(ti,tSSA,V,m,RI,RC,thresholdsIUC,thresholdsCUC,r,length(ratings));
[cbvaUC,cdvaUC,ccvaUC]=CBVA(ti,tSSA,Vadjusted,RI,RC,CUC,r,length(ratings),LGDI,LGDC);
cvaPlotsUC=plotCBVA(tSSA,ti,Vadjusted,CUC,cbvaUC,cdvaUC,ccvaUC);
collateralPlotsUC=plotCollateral(tSSA,ti,CIUC,CCUC,CUC,Vadjusted,RI,RC,thresholdsIUC,thresholdsCUC,ratings);
fprintf('Without collateralization:\n\tCBVA=%3.3f\n\tCDVA=%3.3f\n\tCCVA=%3.3f\n',...
    cbvaUC(end),cdvaUC(end),ccvaUC(end));
%% with collateralization depending on rating triggers
thresholdsIRT=[scale.*10,scale.*5,0,0]';
thresholdsCRT=[scale.*10,scale.*5,0,0]';
% thresholdsIRT=zeros(length(ratings),1);
% thresholdsIRT(1:floor(length(ratings)./2))=scale.*10;
% thresholdsIRT(floor(length(ratings)./2)+1:end-2)=scale.*5;
% thresholdsCRT=zeros(length(ratings),1);
% thresholdsCRT(1:floor(length(ratings)./2))=scale.*10;
% thresholdsCRT(floor(length(ratings)./2)+1:end-2)=scale.*5;
[CRT,CIRT,CCRT,Vadjusted]=collateral(ti,tSSA,V,m,RI,RC,thresholdsIRT,thresholdsCRT,r,length(ratings));
[cbvaRT,cdvaRT,ccvaRT]=CBVA(ti,tSSA,Vadjusted,RI,RC,CRT,r,length(ratings),LGDI,LGDC);
cvaPlotsRT=plotCBVA(tSSA,ti,Vadjusted,CRT,cbvaRT,cdvaRT,ccvaRT);
collateralPlotsRT=...
    plotCollateral(tSSA,ti,CIRT,CCRT,CRT,Vadjusted,RI,RC,thresholdsIRT,thresholdsCRT,ratings);
fprintf('With rating triggers:\n\tCBVA=%3.3f\n\tCDVA=%3.3f\n\tCCVA=%3.3f\n',...
    cbvaRT(end),cdvaRT(end),ccvaRT(end));
saveFigures(collateralPlotsRT,fileDir,'collateralPlotsRT',backgroundColor);
%% with perfect collateralization
thresholdsIPC=zeros(length(ratings),1);
thresholdsCPC=zeros(length(ratings),1);
[CPC,CIPC,CCPC,Vadjusted]=collateral(ti,tSSA,V,m,RI,RC,thresholdsIPC,thresholdsCPC,r,length(ratings));
[cbvaPC,cdvaPC,ccvaPC]=CBVA(ti,tSSA,Vadjusted,RI,RC,CPC,r,length(ratings),LGDI,LGDC);
cvaPlotsPC=plotCBVA(tSSA,ti,Vadjusted,CPC,cbvaPC,cdvaPC,ccvaPC);
collateralPlotsPC=...
    plotCollateral(tSSA,ti,CIPC,CCPC,CPC,Vadjusted,RI,RC,thresholdsIPC,thresholdsCPC,ratings);
fprintf('Perfect collateralization:\n\tCBVA=%3.3f\n\tCDVA=%3.3f\n\tCCVA=%3.3f\n',...
    cbvaPC(end),cdvaPC(end),ccvaPC(end));
%% Plot Simulations
ratingPlotsP=plotRatingModel(tSSA,XP,floor(length(ratings)/2),ratings);
saveFigures(ratingPlotsP,fileDir,'ratingPlotsP',backgroundColor);
ratingPlotsQ=plotRatingModel(tSSA,XQ,floor(length(ratings)/2),ratings);
saveFigures(ratingPlotsQ,fileDir,'ratingPlotsQ',backgroundColor);
%% Output
% compileLatex=false;
% output(fileName,...
%        N,M,...
%        ctimeCal,ctimeCalQ,...
%        errCal,errCalQ,...
%        paramTable,h,...
%        mDCgan,sDDgan,dMLgan,iRSgan,rSOgan,...
%        mDCcalP,sDDcalP,dMLcalP,iRScalP,rSOcalP,...
%        mDCcalQ,sDDcalQ,dMLcalQ,iRScalQ,rSOcalQ,...
%        figRDP,figRDQ,figTraP,figTraQ,...
%        'backgroundColor',backgroundColor,...
%        'textColor',textColor,...
%        'compileLatex',compileLatex)
% disp('Done')
function delDir(dir)
    if exist(dir)==7
        rmdir(dir,'s');
    end
end
function mkDir(dir)
    if exist(dir)==0
        mkdir(dir);
    end
end
function saveFigures(figures,saveAt,figName,backgroundColor)
    for i=1:1:length(figures)
        picName=sprintf('%s_%d',figName,i);
        picPath = [saveAt,'/',picName,'.pdf'];
        if iscell(figures)
            exportgraphics(figures{i},picPath,'BackgroundColor',backgroundColor)
        else
            exportgraphics(figure(figures(i)),picPath,'BackgroundColor',backgroundColor)
        end
    end
end