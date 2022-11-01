function [PD,lambda,R]=defaultProbabilityLoader(defaultProbabilityFolder,...
                                                defaultAgency,...
                                                defaultProbabilityDataset,...
                                                defaultProbabilityYears)
%%DEFAULTPROBABILITYLOADER loads default probabilities from destination 
% folder with given data set and array of years
%   Input:
%       defaultProbabilityFolder (str): contains the path to the folder of 
%                                       the rating matrices
%       defaultAgencey (str): contains the name of the rating agency
%       defaultProbabilityDataset (int): contains the number of the data 
%                                        set
%       defaultProbabilityYears (1xp array): contains the years of the 
%                                            rating matrices
%   Output:
%       PD (Kxp array): contains the default probabilities in decimals
%       lambda (Kxp array): contains default intensities
%       R (1xK cell array): contains the names of the ratings
PD=[];
lambda=[];
for i=1:1:length(defaultProbabilityYears)
    defaultProbabilityName=sprintf('%s_%d_%2.2f*y.*',...
                             defaultAgency,...
                             defaultProbabilityDataset,...
                             defaultProbabilityYears(i));
    d=dir([pwd,'/',defaultProbabilityFolder,'/',defaultAgency,...
           '/',defaultProbabilityName]);
    table=readtable([defaultProbabilityFolder,'/',defaultAgency,'/',d.name],...
                             'VariableNamingRule','preserve');
    headers = table.Properties.VariableNames;
    data = table2array(table(1:end,2:end));
    if strcmp(headers{1},'%')
        data=data./100;
    end
    PD(:,i)=data;
    lambda(:,i)=-log(1-PD(:,i));
end
R=headers(2:end);
end