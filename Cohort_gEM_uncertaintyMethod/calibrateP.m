function [params,err,dW,t,tInd,ctimeCal]=calibrateP(months,T,N,M,Rrec,Rcohort,varargin)

optimizer='lsqnonlin';
saveCal=true;
loadCal=true;
saveDir='Results/Mat/Calibration';
fileNameCal='defaultName';
for iV=1:1:length(varargin)
    switch varargin{iV}
        case 'optimizer'
            optimizer=varargin{iV+1};
        case 'saveCal'
            saveCal=varargin{iV+1};
        case 'loadCal'
            saveCal=varargin{iV+1};
        case 'fileNameCal'
            fileNameCal=varargin{iV+1};
    end
end
configStr = ['_',sprintf('%d',months),...
             sprintf('_%1.2f_%d_%d_',T,N,M),'P'];
fileCal=[saveDir,'/',fileNameCal,configStr,'.mat'];

K=size(Rrec,1);
t=linspace(0,T,N);
tInd=zeros(size(months));
for i=1:1:length(tInd)
    tInd(i)=find(t>=months(i)/12,1,'first');
end

if loadCal && exist(fileCal,'file')
    tempLoad = load(fileCal);
    err = tempLoad.err;
    params = tempLoad.params;
    dW = tempLoad.dW;
    ctimeCal = tempLoad.ctimeCal;
else
    ticCal=tic;
    dW = sqrt(t(end)/(N+1)).*randn((K-1)^2,N-1,M);
%     [MuGAN,SigmaGAN,SkewGAN,KurtGAN]=ratingMoments(Rgan(:,:,end,:));
    
    switch optimizer
        case 'fmincon'
            options = optimoptions('fmincon',...
                                   'Display','iter',...
                                   'StepTolerance',1e-10,...
                                   'MaxFunctionEvaluations',10000,...
                                   'UseParallel',true);
            lb=zeros(18,1);
            ub=1.*ones(18,1);
            x0=(ub+lb)./2;
            [params,err]=fmincon(@(x)objectiveFminconP(x,t,tInd(end),dW,...
                                 Rrec(:,:,end),Rcohort(:,:,end)),...
                                 x0,[],[],[],[],...
                                 lb,ub,...
                                 [],options);
    
        case 'lsqnonlin'
            lb=1e-4.*ones(27,1);
%             ub=2.*ones(27,1);
            ub=3.*ones(27,1);
            x0=(ub+lb)./2;

            % very sensitive on intial datum, mean is minimizer
%             x0(1:length(x0)/3)=1.5;
%             x0(length(x0)/3+1:2*length(x0)/3)=.1;
%             x0(2*length(x0)/3+1:end)=1e-4;
%             x0=lb;
            options = optimoptions('lsqnonlin',...
                                   'Display','iter',...
                                   'UseParallel',true);
%             options = optimoptions('lsqnonlin',...
%                                    'Display','iter',...
%                                    'UseParallel',false);
%             optionsGa = optimoptions('ga',...
%                                    'Display','iter',...
%                                    'UseParallel',true);
%             x0=ga(@(x)sum(objectiveLsqnonlinP(x,t,tInd(end),dW,...
%                             Rrec(1:end-1,:,end),Rcohort(1:end-1,:,end)).^2,'all'),...
%                             length(lb),[],[],[],[],lb,ub,[],[],optionsGa);
            [params,err] = lsqnonlin(@(x)objectiveLsqnonlinP(x,t,tInd(end),dW,...
                            Rrec(1:end-1,:,end),Rcohort(1:end-1,:,end)),x0,lb,ub,options);
        otherwise
            error('Unknown optimizer')
    end
    ctimeCal=toc(ticCal);
    if saveCal
        mkDir(saveDir);
        save(fileCal,'params','err','dW','ctimeCal');
    end
end
end
function delFile(file)
    if exist(file)
        delete(file);
    end
end
function delDir(dir)
    if exist(dir)==7
        rmdir(dir,'s');
    end
end
function cleanDir(mdir,except)
    except{end+1}='.';
    except{end+1}='..';
    for d = dir(mdir).'
      if ~any(strcmp(d.name,except))
          if d.isdir
              rmdir([d.folder,'/',d.name],'s');
          else
              delete([d.folder,'/',d.name]);
          end
      end
    end
end
function mkDir(dir)
    if exist(dir)==0
        mkdir(dir);
    end
end