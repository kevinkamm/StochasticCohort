function figures=plotRatingModel(t,R,ri,ratings)
ncolors=4;
cmap=jet(ncolors);
linspecs.R.linestyle = {'-','--',':','-.'};
linspecs.R.marker = repmat({'none'},1,ncolors);
linspecs.R.color = num2cell(cmap',1);
figures=[];
t=reshape(t,[],1);
for i=1:1:length(ri)
fig= figure('units','normalized',...
              'outerposition',[0 0 1 1]); hold on;
fig.WindowState = 'minimized';
figure_properties(fig);
figures(end+1)=fig;
legendEntries = {};
plots = [];

plot_trajectoryCloud(i)
plot_Highlights(i)

legend(plots,legendEntries,...
      'Location','southoutside',...
      'NumColumns',3,...
      'Interpreter','latex');
allAxesInFigure = findall(fig,'type','axes');
Ax1=allAxesInFigure(1);
Ax2=allAxesInFigure(2);
Ax1.Position(2)=Ax2.Position(2);
Ax1.Position(4)=Ax2.Position(4);
end
    function plot_trajectoryCloud(i)
        Rtemp=double(squeeze(R(ri(i),:,:)));
        Rtemp(end,:)=NaN;
        plt=patch(t.*ones(1,size(Rtemp,2)),...
              Rtemp,...
             .1*ones(size(Rtemp)),...
             'EdgeColor',[108,110,107]./255,'EdgeAlpha',.1,...
             'LineWidth',1);
        plots(end+1)=plt(1);
        legendEntries{end+1}=sprintf('All trajectories of Rating model starting in %s',ratings{ri(i)});
    end
    function plot_Highlights(i)
        Rtemp=squeeze(R(ri(i),:,:));
        temp=sum(abs(diff(Rtemp(:,:),1,1))>0,1);
        [~,ia1,~] = unique(temp);
        wMax=ia1(end);
        wMean=ia1(ceil(length(ia1)/2));
        plots(end+1)=plot(t,Rtemp(:,wMax),...
             'LineStyle',linspecs.R.linestyle{1},...
             'Marker',linspecs.R.marker{1},...
             'Color',linspecs.R.color{1});
        legendEntries{end+1}=sprintf('Trajectory with max no. of transitions');
        plots(end+1)=plot(t,Rtemp(:,wMean),...
             'LineStyle',linspecs.R.linestyle{2},...
             'Marker',linspecs.R.marker{2},...
             'Color',linspecs.R.color{2});
        legendEntries{end+1}=sprintf('Trajectory with med no. of transitions');
        wFirst=-1;
        wLast=-1;
        for kk=1:1:size(Rtemp,1)
            if any(Rtemp(kk,:)==length(ratings))
                wFirst=find(Rtemp(kk,:)==length(ratings),1,'first');
                break;
            end
        end
        for kk=size(Rtemp,1):-1:2
            if any(Rtemp(kk,:)==length(ratings) & Rtemp(kk-1,:)~=length(ratings))
                wLast=find(Rtemp(kk,:)==length(ratings) &...
                           Rtemp(kk-1,:)~=length(ratings),1,'first');
                break;
            end
        end
        if wFirst > 0
        plots(end+1)=plot(t,Rtemp(:,wFirst),...
             'LineStyle',linspecs.R.linestyle{3},...
             'Marker',linspecs.R.marker{3},...
             'Color',linspecs.R.color{3});
        legendEntries{end+1}=sprintf('First to default');
        end
        if wLast > 0
        plots(end+1)=plot(t,Rtemp(:,wLast),...
             'LineStyle',linspecs.R.linestyle{4},...
             'Marker',linspecs.R.marker{4},...
             'Color',linspecs.R.color{4});
        legendEntries{end+1}=sprintf('Last to default');
        end
        xlabel('Time')
        ylabel('Rating')
        ylim([1 length(ratings)])
        yTicks=unique(ceil(yticks));
%         yTicks(1)=1;
%         yTicks(end)=length(ratings);
%         yTicks=unique(yTicks);
        yticks(yTicks);
        yticklabels(ratings(yTicks));
%         ax2 = axes('XAxisLocation','top',...
%                      'YAxisLocation','right',...
%                      'Color','none',...
%                      'XColor','b','YColor','b',...
%                      'XDir','reverse');
        ax2 = axes('XAxisLocation','top',...
                     'YAxisLocation','right',...
                     'Color','none',...
                     'XColor',[0 0.4470 0.7410],'YColor','none');
        H=zeros(length(ratings),1);
        for j=1:1:length(ratings)
            H(j)=sum(Rtemp(end,:)==j);
        end
        H=H./size(Rtemp,2);
        plots(end+1)=line(H,1:1:length(ratings),...
                          'Parent',ax2,'Color',...
                          [0 0.4470 0.7410]);
        legendEntries{end+1}=sprintf('Probability distribution at final time');
        for j=1:1:length(ratings)
            text(H(j),j,sprintf('%2.2g',H(j)),...
                'HorizontalAlignment','left');
        end
        xlim([0 1]);
        xPos=ax2.Position;
        x3=9*xPos(3)/10;
        xPos(3)=xPos(3)-x3;
        xPos(1)=xPos(1)+x3+xPos(3);
        ax2.Position=xPos;
        ylim([1 length(ratings)])
        xlabel('Probability')
        ylabel('Rating')
        
    end
end
function figure_properties(fig)
    fontsize=22;
    linewidth=2;
    set(gca,'FontSize',fontsize)
    set(fig,'defaultlinelinewidth',linewidth)
    set(fig,'defaultaxeslinewidth',linewidth)
    set(fig,'defaultpatchlinewidth',linewidth)
    set(fig,'defaultAxesFontSize',fontsize)
end