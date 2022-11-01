function [Q,varargout]=preDefaultDistribution(X,K)
%%PREDEFAULTDISTRIBUTION computes the distribution of defaults one
%%time-step prior to default. Includes the initial rating default.
%   Input:
%       X (KxNxM): contains the simulated paths of the PHCTMC for all
%                  initial ratings including default
%   Output:
%       Q (K,K): contains the distribution of predefault ratings
%       varargout (cell array):
%           varargout{1} (int): number of defaults

Q=zeros(size(X,1),K-1);
numOfDefaults=0;
for ri=1:1:size(X,1)
    for mi=1:1:size(X,3)
        ti=find(X(ri,:,mi)==K,1,'first');
        if ~isempty(ti)
            numOfDefaults=numOfDefaults+1;
            if ti>1
                j=X(ri,ti-1,mi);
                Q(ri,j)=Q(ri,j)+1;
            elseif ti==1
                j=X(ri,ti,mi);
                Q(ri,j)=Q(ri,j)+1;
            end
        end
    end
    
end

if nargout>1
    varargout{1}=numOfDefaults;
end

end