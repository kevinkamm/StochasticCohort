function [C,CI,CC,varargout]=collateral(ti,t,V,m,RI,RC,thresholdsI,thresholdsC,r,K)
%%COLLATERAL computes the collateral postings over time for investor and
% couterparty with rating triggers and perfect collateralization (C=V)
%   Input:
%       ti (1xn array): contains the indices of points in time when the
%                       collateral shall be posted, e.g. each day
%       t (1xN array): contains the time grid
%       V (NxM array): contains the M simulated paths of the portfolio
%       m (double): contains the minimal transfer amount of collateral
%       RI (NxM array): contains the simulated paths of investor's rating
%       RC (NxM array): contains the simulated paths of counterparty's
%                       rating
%       threshholdI (Kx1 array): contains investor's thresholds assuming
%                               that each rating has a treshold which might
%                               be the same for several ratings
%       threshholdC (Kx1 array): contains counterparty's thresholds assuming
%                               that each rating has a treshold which might
%                               be the same for several ratings
%       r (double): short rate
%       K (int): default state
%   Output:
%       C (n+1xM array): contains the trajectories of the collateral account
%       CI (nxM array): contains the investor's collateral postings for
%                       each time ti
%       CC (nxM array): contains the counterparty's collateral postings for
%                       each time ti
    t=reshape(t,1,[]);
    CI=zeros(length(ti),size(RI,2));
    CC=zeros(length(ti),size(RC,2));
    C=zeros(length(ti)+1,size(RI,2));
    Vadjusted=V;
    betaI=ones(1,size(RI,2));
    betaC=ones(1,size(RC,2));
    for i=1:1:length(ti)
        rhoI=ratingTrigger(RI(ti(i),:),thresholdsI)';
        tempI=min(V(ti(i),:)+rhoI,0)-...
              min(C(i,:)./(exp( r.* ( t(ti(i))-t(betaI) ) ) ),0);
        indI=abs(tempI)>m;
        betaI(indI)=ti(i);
        CI(i,:)=tempI;
        
        rhoC=ratingTrigger(RC(ti(i),:),thresholdsC)';
        tempC=max(V(ti(i),:)-rhoC,0)-...
              max(C(i,:)./(exp( r.* ( t(ti(i))-t(betaC) ) ) ),0);
        indC=abs(tempI)>m;
        betaC(indC)=ti(i);
        CC(i,:)=tempC;
      
        C(i+1,:)=C(i,:)+tempI+tempC;
    end
    for wi=1:1:size(V,2)
        tiI=find(RI(:,wi)==K,1,'first');
        tiC=find(RC(:,wi)==K,1,'first');
        tInd=myMin(tiI,tiC);
        if ~isempty(tInd) 
            Vadjusted(tInd+1:end,wi)=0;
            tIndColl=find(t(ti)>=t(tInd),1,'first');
            C(1+tIndColl:end,wi)=0;
            CI(tIndColl:end,wi)=0;
            CC(tIndColl:end,wi)=0;
        end
    end
    if nargout>3
        varargout{1}=Vadjusted;
    end
end
function rho=ratingTrigger(R,threshold)
    rho=threshold(R);
end
function m=myMin(a,b)
    if isempty(a) && isempty(b)
        m=[];
    elseif isempty(a) && ~isempty(b)
        m=b;
    elseif isempty(b) && ~isempty(a)
        m=a;
    else
        m=min(a,b);
    end
end