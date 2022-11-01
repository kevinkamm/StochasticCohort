function [cohort,rec]=loadData(year,dataset)
    cohort=loadCSV('CohortData');
    rec=loadCSV('MLData');
    function C=loadCSV(dataPath)
        files = dir([dataPath,'/TimeGAN_',num2str(year),'_',num2str(dataset),'*']);
        C={};
        for i=1:1:length(files)
            temp=readtable([files(i).folder,'/',files(i).name]);
            C{end+1}=temp{1:end,2:end};
        end
        C=cat(3,C{:});
    end
end