function figures=plotPreDefault(Q,numOfDefaults,M,ratings)
%%PLOTPREDEFAULT plots the predefault rating distribution as bar diagram
%   Input:
%       Q (K-1xK-1 array): contains the pre-default distribution for each
%                      rating.
%       numOfDefaults (int): contains the number of defaults
%       ratings (1xK cell array): contains names of all ratings
%   Output:
%       figures (Kx1 array of figurehandles): the figurehandles

figures=[];
cmap=jet(size(Q,1));
for i=1:1:size(Q,1)
fig= figure('units','normalized',...
              'outerposition',[0 0 1 1]); hold on;
fig.WindowState = 'minimized';
figure_properties(fig);
figures(end+1)=fig;
legendEntries = {};
plots = [];

plot_bars(i);

legend(plots,legendEntries,...
      'Location','southoutside',...
      'NumColumns',3,...
      'Interpreter','latex'); 
xlabel('Ratings')
ylabel('Probability')
xticks(1:1:length(ratings)-1)
xticklabels(ratings(1:end-1))
ylim([0 1])
% xtickangle(45)
end
fig= figure('units','normalized',...
              'outerposition',[0 0 1 1]); hold on;
fig.WindowState = 'minimized';
figure_properties(fig);
figures(end+1)=fig;
legendEntries = {};

plots=bar(Q'./numOfDefaults,'stacked');

for i=1:1:size(Q,1)
    plots(i).FaceColor='flat';
    plots(i).CData = cmap(i,:);
    legendEntries{end+1}=sprintf('initial rating %s',ratings{i});
end

legend(plots,legendEntries,...
      'Location','southoutside',...
      'NumColumns',3,...
      'Interpreter','latex'); 
xlabel('Ratings')
ylabel('Probability')
xticks(1:1:length(ratings)-1)
xticklabels(ratings(1:end-1))
ylim([0 1])
xLim=xlim;
text(xLim(1),1,sprintf('Total number of defaults %d using M=%d simulations',numOfDefaults,size(Q,1)*M),...
    "HorizontalAlignment","left","VerticalAlignment","top","FontSize",22);
% xtickangle(45)

    function plot_bars(i)
        plots(end+1)=bar(Q(i,:)./numOfDefaults,'FaceColor',cmap(i,:));
%         plots(end).FaceColor='flat';
%         plots(end).CData = cmap(i,:);
        legendEntries{end+1}=sprintf('Probability of pre-default ratings, starting at %s',ratings{i});
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