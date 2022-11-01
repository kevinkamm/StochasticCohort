function figures=plotCollateral(t,ti,CI,CC,C,V,RI,RC,thresholdsI,thresholdsC,ratings)

linspecs.V.linestyle = {'-','-'};
linspecs.V.marker = {'none','none'};
linspecs.V.color = {'k','k'};
% linspecs.C.linestyle = {'-','none'};
% linspecs.C.marker = {'none','o'};
% linspecs.C.color = {'b','b'};
linspecs.C.linestyle = {'-','--'};
linspecs.C.marker = {'none','none'};
linspecs.C.color = {'b','b'};
linspecs.CC.linestyle = {'--','none'};
linspecs.CC.marker = {'none','.'};
linspecs.CC.color = {'r','r'};
linspecs.CI.linestyle = {'--','none'};
linspecs.CI.marker = {'none','.'};
linspecs.CI.color = {'g','g'};
linspecs.RC.linestyle = {'--','-'};
linspecs.RC.marker = {'none','none'};
linspecs.RC.color = {'r','r'};
linspecs.RI.linestyle = {'--','-'};
linspecs.RI.marker = {'none','none'};
linspecs.RI.color = {'g','g'};
linspecs.TC.linestyle = {'--','--'};
linspecs.TC.marker = {'none','none'};
linspecs.TC.color = {'r','r'};
linspecs.TI.linestyle = {'--','--'};
linspecs.TI.marker = {'none','none'};
linspecs.TI.color = {'g','g'};
figures=[];
% cmap=jet(3);
% linspecs.C.linestyle=num2cell(cmap',1)

fig= figure('units','normalized',...
              'outerposition',[0 0 1 1]); hold on;
fig.WindowState = 'minimized';
figure_properties(fig);
figures(end+1)=fig;
legendEntries = {};
plots = [];

plot_collateral_mean();

legend(plots,legendEntries,...
      'Location','southoutside',...
      'NumColumns',3,...
      'Interpreter','latex'); 
xlabel('Time')
ylabel('Collateral')

% select an interesting trajectory

% find a lot of transitions of counterparty
[m,~]=max(sum(diff(RC(:,:),1,1)>0,1),[],2);
wInd=sum(diff(RC(:,:),1,1)>0,1)==m;
temp=1:1:size(RC,2);
wCandidate=temp(wInd);
RCtemp=RC(:,wCandidate);

% non default of counterparty
wInd2=RCtemp(end,:)<length(ratings);
wCandidate2=wCandidate(wInd2);
RItemp2=RI(:,wCandidate2);

% non default of investor
wInd3=RItemp2(end,:)<length(ratings);
wCandidate3=wCandidate2(wInd3);
% RCtemp3=RC(:,wCandidate3);

% most positive exposure 
Vtemp=V(:,wCandidate3);
[~,wind4]=max(sum(Vtemp>0,1),[],2);
% wind4=4;
wi=wCandidate3(wind4);

if isempty(wi)
    wi=1;
end


fig= figure('units','normalized',...
              'outerposition',[0 0 1 1]); hold on;
% fig.WindowState = 'maximized';
fig.WindowState = 'minimized';
figure_properties(fig);
figures(end+1)=fig;

legendEntries = {};
plots = [];

tlo = tiledlayout(3,1);
ax=nexttile;hold on;
plot_collateral_trajectory();
% legend(plots,legendEntries,...
%       'Location','southoutside',...
%       'NumColumns',3,...
%       'Interpreter','latex'); 
% xlabel('Time')
ylabel('Collateral')

ax=nexttile;hold on;
% legendEntries = {};
% plots = [];

plot_ratings_trajectory();
% xlabel('Time')
ylabel('Rating')
ylim([min( min(RI(:,wi)) , min(RC(:,wi)) ) max( max(RI(:,wi)) , max(RC(:,wi)) )])
yTicks=unique(ceil(yticks));
% yTicks(1)=min( min(RI(:,wi)) , min(RC(:,wi)) );
% yTicks(end)=max( max(RI(:,wi)) , max(RC(:,wi)) );
yticks(yTicks);
yticklabels(ratings(yTicks));

ax=nexttile;hold on;
% legendEntries = {};
% plots = [];

plot_tresholds_trajectory()
xlabel('Time')
ylabel('Threshold')
lg=legend(ax,plots,legendEntries,...
      'Orientation','Horizontal',...
      'NumColumns',2,...
      'Interpreter','latex'); 
lg.Layout.Tile = 'south';
    function plot_collateral_mean()
        plots(end+1)=plot(t(ti),mean(C(2:end,:),2),...
                          'LineStyle',linspecs.C.linestyle{1},...
                          'Marker',linspecs.C.marker{1},...
                          'Color',linspecs.C.color{1});
        legendEntries{end+1}=sprintf('Mean of collateral account');
        plots(end+1)=plot(t(ti),mean(CI(1:end,:),2),...
                          'LineStyle',linspecs.CI.linestyle{1},...
                          'Marker',linspecs.CI.marker{1},...
                          'Color',linspecs.CI.color{1});
        legendEntries{end+1}=sprintf('Mean of bank''s collateral postings');
        plots(end+1)=plot(t(ti),mean(CC(1:end,:),2),...
                          'LineStyle',linspecs.CC.linestyle{1},...
                          'Marker',linspecs.CC.marker{1},...
                          'Color',linspecs.CC.color{1});
        legendEntries{end+1}=sprintf('Mean of counterparty''s collateral postings');
    end
    function plot_collateral_trajectory()
        plots(end+1)=plot(t,V(:,wi),...
                          'LineStyle',linspecs.V.linestyle{2},...
                          'Marker',linspecs.V.marker{2},...
                          'Color',linspecs.V.color{2});
        legendEntries{end+1}=sprintf('Trajectory of exposure');
        plots(end+1)=plot(t(ti),C(2:end,wi),...
                          'LineStyle',linspecs.C.linestyle{2},...
                          'Marker',linspecs.C.marker{2},...
                          'Color',linspecs.C.color{2});
        legendEntries{end+1}=sprintf('Trajectory of collateral account');
        plots(end+1)=plot(t(ti),CI(1:end,wi),...
                          'LineStyle',linspecs.CI.linestyle{2},...
                          'Marker',linspecs.CI.marker{2},...
                          'Color',linspecs.CI.color{2});
        legendEntries{end+1}=sprintf('Trajectory of bank''s collateral postings');
        plots(end+1)=plot(t(ti),CC(1:end,wi),...
                          'LineStyle',linspecs.CC.linestyle{2},...
                          'Marker',linspecs.CC.marker{2},...
                          'Color',linspecs.CC.color{2});
        legendEntries{end+1}=sprintf('Trajectory of counterparty''s collateral postings');
    end
    function plot_ratings_trajectory()
        plots(end+1)=plot(t,RI(:,wi),...
                          'LineStyle',linspecs.RI.linestyle{2},...
                          'Marker',linspecs.RI.marker{2},...
                          'Color',linspecs.RI.color{2});
        legendEntries{end+1}=sprintf('Trajectory of bank''s rating');
        plots(end+1)=plot(t,RC(:,wi),...
                          'LineStyle',linspecs.RC.linestyle{2},...
                          'Marker',linspecs.RC.marker{2},...
                          'Color',linspecs.RC.color{2});
        legendEntries{end+1}=sprintf('Trajectory of counterparty''s rating');
    end
    function plot_tresholds_trajectory()
        plots(end+1)=plot(t,thresholdsI(RI(:,wi)),...
                          'LineStyle',linspecs.TI.linestyle{2},...
                          'Marker',linspecs.TI.marker{2},...
                          'Color',linspecs.TI.color{2});
        legendEntries{end+1}=sprintf('Bank''s current threshold');
        plots(end+1)=plot(t,thresholdsC(RC(:,wi)),...
                          'LineStyle',linspecs.TC.linestyle{2},...
                          'Marker',linspecs.TC.marker{2},...
                          'Color',linspecs.TC.color{2});
        legendEntries{end+1}=sprintf('Counterparty''s current threshold');
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