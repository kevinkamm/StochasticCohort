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
PDdataset=2;
PD=defaultProbabilityGenerator(Rrec,PDdataset);

ctimeLoadData=toc(ticLoadData);
fprintf('Elapsed time for loading data %1.3f\n',ctimeLoadData);
% order of Lie Algebra basis
[~,basisOrder]=lieAlgebraBasis(K);
%% Figure settings
backgroundColor='w';
textColor = 'k';
figureRatio = 'fullScreen';
visible='on';
%% Uncertainty measure
distCohort=abs(Rrec(:,:,end)-Rcohort(:,:,end));
Rcertain=Rcohort(:,:,end)+(distCohort./sum(distCohort,2)).*distCohort;
% (Rrec(:,:,end)-Rcertain)./(sum(Rrec(:,:,end)-Rcertain,2))
% Rcertain=Rcertain+(Rrec(:,:,end)-Rcertain).^2./(sum(Rrec(:,:,end)-Rcertain,2))
%% Calibration under JLT
comType='JLT';
% comType='Exp';
optimizer='lsqnonlin';
M=1000;
% M=100;
% N=50*12+1;
N=25*12+1;

fileNameJLT = sprintf('GEM_%s_N%d_M%d_Cohort%d',optimizer,N,M,Cohortdataset);
saveCal = true;
loadCal = true;

disp('Start calibration under P')
[paramsJLT,errCalJLT,dWPJLT,tJLT,tIndJLT,ctimeCalJLT]=calibrateP(months,T,N,M,Rrec,Rcertain,...
                                    'Optimizer',optimizer,...
                                    'saveCal',saveCal,...
                                    'loadCal',loadCal,...
                                    'fileNameCal',fileNameJLT);
fprintf('Elapsed time for calibration under P %g s with error %1.3e\n',...
        ctimeCalJLT,errCalJLT);
aCalJLT=paramsJLT(1:9);
bCalJLT=paramsJLT(10:18);
sigmaCalJLT=paramsJLT(19:27);
paramTableJLT = array2table([aCalJLT,bCalJLT,sigmaCalJLT],...
                         'VariableNames',{'a','b','sigma'},...
                         'RowNames',basisOrder);
disp('Parameters after calibration')
disp(paramTableJLT)

fileNameJLT = sprintf('GEM_%s_%s_N%d_M%d_%d',optimizer,comType,N,M,PDdataset);
disp('Start calibration under Q')
[hJLT,errCalQJLT,ctimeCalQJLT,dWQJLT]=calibrateQ(months,T,tJLT,tIndJLT,paramsJLT,dWPJLT,PD,comType,...
                                'Optimizer',optimizer,...
                                'saveCal',saveCal,...
                                'loadCal',loadCal,...
                                'fileNameCal',fileNameJLT);
fprintf('Elapsed time for calibration under Q %g s with error %1.3e\n',...
        ctimeCalQJLT,errCalQJLT);

% simulation under P
disp('Simulation under P')
ticSimPJLT=tic;
RcalPJLT = gEMP(aCalJLT,bCalJLT,sigmaCalJLT,tJLT,1:1:length(tJLT),dWPJLT);
ctimeSimPJLT=toc(ticSimPJLT);
fprintf('Elapsed time for simulation under P %g s\n',...
        ctimeSimPJLT);
% simulation under Q
disp('Simulation under Q')
ticSimQJLT=tic;
RcalQJLT = gEMQ(aCalJLT,bCalJLT,sigmaCalJLT,hJLT,tJLT,1:1:length(tJLT),dWQJLT,comType);
ctimeSimQJLT=toc(ticSimQJLT);
fprintf('Elapsed time for simulation under Q %g s\n',...
        ctimeSimQJLT);

%% Calibration under Exp
% comType='JLT';
comType='Exp';
optimizer='lsqnonlin';
M=1000;
% M=100;
% N=50*12+1;
N=25*12+1;

fileNameExp = sprintf('GEM_%s_N%d_M%d_Cohort%d',optimizer,N,M,Cohortdataset);
saveCal = true;
loadCal = true;

disp('Start calibration under P')
[paramsExp,errCalExp,dWPExp,tExp,tIndExp,ctimeCalExp]=calibrateP(months,T,N,M,Rrec,Rcertain,...
                                    'Optimizer',optimizer,...
                                    'saveCal',saveCal,...
                                    'loadCal',loadCal,...
                                    'fileNameCal',fileNameExp);
fprintf('Elapsed time for calibration under P %g s with error %1.3e\n',...
        ctimeCalExp,errCalExp);
aCalExp=paramsExp(1:9);
bCalExp=paramsExp(10:18);
sigmaCalExp=paramsExp(19:27);
paramTableExp = array2table([aCalExp,bCalExp,sigmaCalExp],...
                         'VariableNames',{'a','b','sigma'},...
                         'RowNames',basisOrder);
disp('Parameters after calibration')
disp(paramTableExp)

fileNameExp = sprintf('GEM_%s_%s_N%d_M%d_%d',optimizer,comType,N,M,PDdataset);
disp('Start calibration under Q')
[hExp,errCalQExp,ctimeCalQExp,dWQExp]=calibrateQ(months,T,tExp,tIndExp,paramsExp,dWPExp,PD,comType,...
                                'Optimizer',optimizer,...
                                'saveCal',saveCal,...
                                'loadCal',loadCal,...
                                'fileNameCal',fileNameExp);
fprintf('Elapsed time for calibration under Q %g s with error %1.3e\n',...
        ctimeCalQExp,errCalQExp);

% simulation under P
disp('Simulation under P')
ticSimPExp=tic;
RcalPExp = gEMP(aCalExp,bCalExp,sigmaCalExp,tExp,1:1:length(tExp),dWPExp);
ctimeSimPExp=toc(ticSimPExp);
fprintf('Elapsed time for simulation under P %g s\n',...
        ctimeSimPExp);
% simulation under Q
disp('Simulation under Q')
ticSimQExp=tic;
RcalQExp = gEMQ(aCalExp,bCalExp,sigmaCalExp,hExp,tExp,1:1:length(tExp),dWQExp,comType);
ctimeSimQExp=toc(ticSimQExp);
fprintf('Elapsed time for simulation under Q %g s\n',...
        ctimeSimQExp);
%% Figures
fileDir=['Results/Pdf/GEM_',optimizer,'comComp'];
% delDir(fileDir);
mkDir(fileDir);
figName=['GEM_',optimizer,'_comComp','_',num2str(PDdataset)];

fig=newFigure('backgroundColor',backgroundColor,...
              'textColor',textColor,...
              'figureRatio',figureRatio,...
              'visible',visible);
mP=mean(RcalPJLT,4);
mQJLT=mean(RcalQJLT,4);
mQExp=mean(RcalQExp,4);

for ri = 1:1:size(RcalPJLT,1)-1
    for rj = 1:1:size(RcalPJLT,2)
        nexttile;hold on;
        plot(tJLT,squeeze(mP(ri,rj,:)),'LineStyle','--','Color',[0.9290 0.6940 0.1250]);
        plot(tJLT,squeeze(mQJLT(ri,rj,:)),'LineStyle','-','Color',[0 0.4470 0.7410]);
        plot(tExp,squeeze(mQExp(ri,rj,:)),'LineStyle','-','Color',[0.6350 0.0780 0.1840]);
    end
    plot(tJLT(end),PD(ri,end),'rx','MarkerSize',12);
end
lgdCell={'Mean SDE under P','Mean SDE under Q JLT','Mean SDE under Q Exp','Market Default Prob.'};
lgd = legend(lgdCell,'NumColumns',2);
lgd.Layout.Tile = 'south';
saveFigures(fig,fileDir,figName,backgroundColor);
%% Rating properties
% [mDCcalP,sDDcalP,dMLcalP,iRScalP,rSOcalP]=ratingProperties(RcalP(:,:,tInd,:),months);
% [mDCcalQJLT,sDDcalQJLT,dMLcalQJLT,iRScalQJLT,rSOcalQJLT]=ratingProperties(RcalQJLT(:,:,tIndJLT,:),months);
% [mDCcalQExp,sDDcalQExp,dMLcalQExp,iRScalQExp,rSOcalQExp]=ratingProperties(RcalQExp(:,:,tIndExp,:),months);


% dark blue: [0 0.4470 0.7410]
% orange: [0.8500 0.3250 0.0980]
% yellow: [0.9290 0.6940 0.1250]
% purple: [0.4940 0.1840 0.5560]
% green: [0.4660 0.6740 0.1880]
% light blue: [0.3010 0.7450 0.9330]
% red: [0.6350 0.0780 0.1840]
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
            exportgraphics(figures(i),picPath,'BackgroundColor',backgroundColor)
        end
    end
end