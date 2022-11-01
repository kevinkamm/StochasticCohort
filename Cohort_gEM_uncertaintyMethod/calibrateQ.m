function [h,errCalQ,ctimeCalQ,dWQ]=calibrateQ(months,T,t,tInd,params,dWP,PD,comType,varargin)

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
K=sqrt(size(dWP,1))+1;
N=size(dWP,2)+1;
M=size(dWP,3);

configStr = ['_',sprintf('%d',months),...
             sprintf('_%1.2f_%d_%d_',T,N,M),...
             sprintf('%1.2f_',params),'Q'];
fileCal=[saveDir,'/',fileNameCal,configStr,'.mat'];


if loadCal && exist(fileCal,'file')
    tempLoad = load(fileCal);
    errCalQ = tempLoad.errCalQ;
    h = tempLoad.h;
    dWQ = tempLoad.dWQ;
    ctimeCalQ = tempLoad.ctimeCalQ;
else
    ticCalQ=tic;
    dWQ = sqrt(t(end)/(N+1)).*randn((K-1)^2,N-1,M);
    switch optimizer
%         case 'fmincon'
%             options = optimoptions('fmincon',...
%                                    'Display','iter',...
%                                    'StepTolerance',1e-10,...
%                                    'MaxFunctionEvaluations',10000,...
%                                    'UseParallel',true);
%             lb=zeros(18,1);
%             ub=1.*ones(18,1);
%             x0=(ub+lb)./2;
%             [params,errCalQ]=fmincon(@(x)objectiveFminconP(x,t,tInd(end),dW,...
%                                  MuGAN,SigmaGAN,SkewGAN,KurtGAN),...
%                                  x0,[],[],[],[],...
%                                  lb,ub,...
%                                  [],options);
    
        case 'lsqnonlin'
%             lb=-10.*ones((K-1),length(months));
%             ub=10.*ones((K-1),length(months));
%             x0=(ub+lb)./2;
%             lb=1e-4.*ones(K-1,1);
%             ub=2.*ones(K-1,1);
%             x0=(ub+lb)./2;
        switch comType
            case 'Exp'
                lb=1e-4.*ones(K-1,1);
                ub=100.*ones(K-1,1);
%                 ub=200.*ones(K-1,1);
                x0=50*ones(K-1,1);
            case 'JLT'
                lb=[];
                ub=[];
                x0=zeros(K-1,1);
        end
            options = optimoptions('lsqnonlin',...
                                   'Display','iter',...
                                   'FunctionTolerance',1e-6,...
                                   'UseParallel',true);
           
            [h,errCalQ] = lsqnonlin(@(x)objectiveLsqnonlinQ(x,params,t,tInd,dWQ,...
                            PD,comType),x0,lb,ub,options);
        otherwise
            error('Unknown optimizer')
    end
    ctimeCalQ=toc(ticCalQ);
    if saveCal
        mkDir(saveDir);
        save(fileCal,'h','errCalQ','ctimeCalQ','dWQ');
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