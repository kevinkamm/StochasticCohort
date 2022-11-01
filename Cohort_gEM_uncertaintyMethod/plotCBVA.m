function figures=plotCBVA(t,ti,V,C,cbva,cdva,ccva)

linspecs.V.linestyle = {'-','-'};
linspecs.V.marker = {'none','none'};
linspecs.V.color = {'k','k'};
linspecs.C.linestyle = {'-','none'};
linspecs.C.marker = {'none','o'};
linspecs.C.color = {'b','b'};
linspecs.CC.linestyle = {'--','none'};
linspecs.CC.marker = {'none','x'};
linspecs.CC.color = {'r','r'};
linspecs.CI.linestyle = {'--','none'};
linspecs.CI.marker = {'none','*'};
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
linspecs.CBVA.linestyle = {'-','-'};
linspecs.CBVA.marker = {'none','none'};
linspecs.CBVA.color = {'m','m'};
linspecs.CDVA.linestyle = {'-','-'};
linspecs.CDVA.marker = {'none','none'};
linspecs.CDVA.color = {'y','y'};
linspecs.CCVA.linestyle = {'-','-'};
linspecs.CCVA.marker = {'none','none'};
linspecs.CCVA.color = {'c','c'};
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
plot_cva();

legend(plots,legendEntries,...
      'Location','southoutside',...
      'NumColumns',3,...
      'Interpreter','latex'); 
xlabel('Time')
ylabel('Collateral')
% 
% [m,wi]=max(sum(diff(RC(:,:),1,1)>0,1),[],2);
% 
% fig= figure('units','normalized',...
%               'outerposition',[0 0 1 1]); hold on;
% % fig.WindowState = 'maximized';
% fig.WindowState = 'minimized';
% figure_properties(fig);
% figures(end+1)=fig;
% 
% legendEntries = {};
% plots = [];
% 
% tlo = tiledlayout(3,1);
% ax=nexttile;hold on;
% plot_collateral_trajectory();
% % legend(plots,legendEntries,...
% %       'Location','southoutside',...
% %       'NumColumns',3,...
% %       'Interpreter','latex'); 
% % xlabel('Time')
% ylabel('Collateral')
% 
% ax=nexttile;hold on;
% % legendEntries = {};
% % plots = [];
% 
% plot_ratings_trajectory();
% % xlabel('Time')
% ylabel('Rating')
% yTicks=unique(ceil(yticks));
% yTicks(1)=min( min(RI(:,wi)) , min(RC(:,wi)) );
% yTicks(end)=max( max(RI(:,wi)) , max(RC(:,wi)) );
% yticks(yTicks);
% yticklabels(ratings(yTicks));
% 
% ax=nexttile;hold on;
% % legendEntries = {};
% % plots = [];
% 
% plot_tresholds_trajectory()
% xlabel('Time')
% ylabel('Treshold')
% lg=legend(ax,plots,legendEntries,...
%       'Orientation','Horizontal',...
%       'NumColumns',2,...
%       'Interpreter','latex'); 
% lg.Layout.Tile = 'south';
    function plot_collateral_mean()
        plots(end+1)=plot(t(ti),mean(C(2:end,:),2),...
                          'LineStyle',linspecs.C.linestyle{1},...
                          'Marker',linspecs.C.marker{1},...
                          'Color',linspecs.C.color{1});
        legendEntries{end+1}=sprintf('Mean of collateral account');
        plots(end+1)=plot(t,mean(V,2),...
                          'LineStyle',linspecs.V.linestyle{1},...
                          'Marker',linspecs.V.marker{1},...
                          'Color',linspecs.V.color{1});
        legendEntries{end+1}=sprintf('Mean of Exposure');
%         plots(end+1)=plot(t(ti),mean(CI(1:end,:),2),...
%                           'LineStyle',linspecs.CI.linestyle{1},...
%                           'Marker',linspecs.CI.marker{1},...
%                           'Color',linspecs.CI.color{1});
%         legendEntries{end+1}=sprintf('Mean of investor''s collateral postings');
%         plots(end+1)=plot(t(ti),mean(CC(1:end,:),2),...
%                           'LineStyle',linspecs.CC.linestyle{1},...
%                           'Marker',linspecs.CC.marker{1},...
%                           'Color',linspecs.CC.color{1});
%         legendEntries{end+1}=sprintf('Mean of counterparty''s collateral postings');
    end
    function plot_cva()
        plots(end+1)=plot(t,cbva,...
                          'LineStyle',linspecs.CBVA.linestyle{1},...
                          'Marker',linspecs.CBVA.marker{1},...
                          'Color',linspecs.CBVA.color{1});
        legendEntries{end+1}=sprintf('CBVA');
        plots(end+1)=plot(t,cdva,...
                          'LineStyle',linspecs.CDVA.linestyle{1},...
                          'Marker',linspecs.CDVA.marker{1},...
                          'Color',linspecs.CDVA.color{1});
        legendEntries{end+1}=sprintf('CDVA');
        plots(end+1)=plot(t,ccva,...
                          'LineStyle',linspecs.CCVA.linestyle{1},...
                          'Marker',linspecs.CCVA.marker{1},...
                          'Color',linspecs.CCVA.color{1});
        legendEntries{end+1}=sprintf('CCVA');
    end
%     function plot_collateral_trajectory()
%         plots(end+1)=plot(t(ti),V(ti,wi),...
%                           'LineStyle',linspecs.V.linestyle{2},...
%                           'Marker',linspecs.V.marker{2},...
%                           'Color',linspecs.V.color{2});
%         legendEntries{end+1}=sprintf('Trajectory of exposure');
%         plots(end+1)=plot(t(ti),C(2:end,wi),...
%                           'LineStyle',linspecs.C.linestyle{2},...
%                           'Marker',linspecs.C.marker{2},...
%                           'Color',linspecs.C.color{2});
%         legendEntries{end+1}=sprintf('Trajectory of collateral account');
%         plots(end+1)=plot(t(ti),CI(1:end,wi),...
%                           'LineStyle',linspecs.CI.linestyle{2},...
%                           'Marker',linspecs.CI.marker{2},...
%                           'Color',linspecs.CI.color{2});
%         legendEntries{end+1}=sprintf('Trajectory of investor''s collateral postings');
%         plots(end+1)=plot(t(ti),CC(1:end,wi),...
%                           'LineStyle',linspecs.CC.linestyle{2},...
%                           'Marker',linspecs.CC.marker{2},...
%                           'Color',linspecs.CC.color{2});
%         legendEntries{end+1}=sprintf('Trajectory of counterparty''s collateral postings');
%     end
%     function plot_ratings_trajectory()
%         plots(end+1)=plot(t(ti),RI(ti,wi),...
%                           'LineStyle',linspecs.RI.linestyle{2},...
%                           'Marker',linspecs.RI.marker{2},...
%                           'Color',linspecs.RI.color{2});
%         legendEntries{end+1}=sprintf('Trajectory of investor''s rating');
%         plots(end+1)=plot(t(ti),RC(ti,wi),...
%                           'LineStyle',linspecs.RC.linestyle{2},...
%                           'Marker',linspecs.RC.marker{2},...
%                           'Color',linspecs.RC.color{2});
%         legendEntries{end+1}=sprintf('Trajectory of counterparty''s rating');
%     end
%     function plot_tresholds_trajectory()
%         plots(end+1)=plot(t(ti),tresholdsI(RI(ti,wi)),...
%                           'LineStyle',linspecs.TI.linestyle{2},...
%                           'Marker',linspecs.TI.marker{2},...
%                           'Color',linspecs.TI.color{2});
%         legendEntries{end+1}=sprintf('Investor''s current treshold');
%         plots(end+1)=plot(t(ti),tresholdsC(RC(ti,wi)),...
%                           'LineStyle',linspecs.TC.linestyle{2},...
%                           'Marker',linspecs.TC.marker{2},...
%                           'Color',linspecs.TC.color{2});
%         legendEntries{end+1}=sprintf('Counterparty''s current treshold');
%     end

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