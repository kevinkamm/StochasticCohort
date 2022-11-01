function Figures=plotTrajectories(t,tInd,Rcal,Rgan,RcalP,varargin)
backgroundColor='w';
textColor = 'k';
figureRatio = 'fullScreen';
visible='on';
showTitle=false;
for iV=1:2:length(varargin)
    switch varargin{iV}
        case 'backgroundColor'
            backgroundColor = varargin{iV+1};
        case 'figureRatio'
            figureRatio = varargin{iV+1};
        case 'textColor'
            textColor  = varargin{iV+1};
        case 'visible'
            visible  = varargin{iV+1};
        case 'showTitle'
            showTitle  = varargin{iV+1};
    end
end

[N,M]=size(Rcal,[3,4]);
wi=1;
Figures={};
legendEntries=beginFigure();
xLabel='t';
yLabel='';
titleStr='';
for iR=1:1:size(Rcal,1)-1
    for jR=1:1:size(Rcal,2)
        nexttile;hold on;
        plotRcal(iR,jR);
        if ~isempty(Rgan)
            plotRgan(iR,jR);
        end
        if ~isempty(RcalP)
            plotRcalP(iR,jR);
        end
        tileLables(xLabel,yLabel,titleStr);
    end 
end
if ~isempty(Rgan)
    legendEntries={'Paths of SDE','1 Path of SDE','Mean of SDE','Value of Rec'};
else
    legendEntries={'Paths of SDE','1 Path of SDE','Mean of SDE'};
end
if ~isempty(RcalP)
    legendEntries{end+1}='Mean of SDE under P';
end
endFigure(legendEntries);

    function plotRcal(iR,jR)
        tempR=squeeze(Rcal(iR,jR,:,:));
        tempR(end,:)=nan;
        patch(reshape(t,[],1).*ones(1,M),...
                 tempR,...
                 .1*ones(size(tempR)),...
                 'EdgeColor',[108,110,107]./255,'EdgeAlpha',.02,...
                 'LineWidth',.1);

        plot(t,squeeze(Rcal(iR,jR,:,wi)),'LineStyle','-','Color',[0.9290 0.6940 0.1250]);

        m=mean(Rcal(iR,jR,:,:),4);
        plot(t,squeeze(m),'LineStyle','-','Color',[0 0.4470 0.7410]);

    end
    function plotRcalP(iR,jR)
        m=mean(RcalP(iR,jR,:,:),4);
        plot(t,squeeze(m),'LineStyle','--','Color',[0.4940 0.1840 0.5560]);

    end
    function plotRgan(iR,jR)
        m=mean(Rgan(iR,jR,:,:),4);
        plot(t(tInd),squeeze(m),'rx');
    end
    function legendEntries=beginFigure()
        legendEntries={};
        Figures{end+1}=newFigure('backgroundColor',backgroundColor,...
                                 'figureRatio',figureRatio,...
                                 'textColor',textColor,...
                                 'visible',visible);
        tiledlayout(size(Rcal,1)-1,size(Rcal,2));
    end
    function tileLables(xLabel,yLabel,titleStr)
        if ~strcmp(xLabel,'')
            xlabel(xLabel, 'fontweight', 'bold','Color',textColor,'Interpreter','latex')
        end
        if ~strcmp(yLabel,'')
            ylabel(yLabel, 'fontweight', 'bold','Color',textColor,'Interpreter','latex')
        end
        if ~strcmp(titleStr,'') && showTitle
            title(titleStr,'Color',textColor,'Interpreter','latex')
        end
    end
    function endFigure(legendEntries)      
        lgd=legend(legendEntries,...
          'NumColumns',4,...
          'Interpreter','latex',...
          'TextColor',textColor); 
        lgd.Layout.Tile = 'south';

    end
end
% dark blue: [0 0.4470 0.7410]
% orange: [0.8500 0.3250 0.0980]
% yellow: [0.9290 0.6940 0.1250]
% purple: [0.4940 0.1840 0.5560]
% green: [0.4660 0.6740 0.1880]
% light blue: [0.3010 0.7450 0.9330]
% red: [0.6350 0.0780 0.1840]