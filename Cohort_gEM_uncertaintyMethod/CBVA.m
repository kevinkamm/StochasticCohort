function [cbva,cdva,ccva]=CBVA(ti,t,V,RI,RC,C,r,K,LGDI,LGDC)
%%CBVA computes the bilateral credit valuation adjustment with
% collateralization including rating tresholds without re-hypothecation
%   Input:
%       ti (1xn array): contains the indices of points in time when the
%                       collateral shall be posted, e.g. each day
%       t (1xN array): contains the time grid
%       V (NxM array): contains the M simulated paths of the portfolio
%       RI (NxM array): contains the simulated paths of investor's rating
%       RC (NxM array): contains the simulated paths of counterparty's
%                       rating
%       r (double): short rate
%       K (int): contains the integer corresponding to the default state
%       LGDI (double): Investor's Loss-Given-Default in decimal
%       LGDC (double): Counterparty's Loss-Given-Default in decimal
%   Output:
%       CBVA (Nx1 array): contains the collateral-inclusive bilateral CVA

t=reshape(t,[],1);
dFactor=exp(-r.*t);

cbva=zeros(size(t));
cdva=zeros(size(t));
ccva=zeros(size(t));
for wi=1:1:size(V,2)
tiI=find(RI(:,wi)==K,1,'first');
tiC=find(RC(:,wi)==K,1,'first');
tInd=myMin(tiI,tiC);
    if ~isempty(tInd)
        tIndColl=find(t(ti)<=t(tInd),1,'last');
        if isempty(tIndColl)
            tIndColl=0;
        end
        cdva(tInd)=cdva(tInd)-...
            dFactor(tInd).*LGDI.*min(min(V(tInd,wi),0)-min(C(1+tIndColl,wi),0),0);
        ccva(tInd)=ccva(tInd)+...
            dFactor(tInd).*LGDC.*max(max(V(tInd,wi),0)-max(C(1+tIndColl,wi),0),0);
        cbva(tInd)=cbva(tInd)-...
                    dFactor(tInd).*LGDC.*max(max(V(tInd,wi),0)-max(C(1+tIndColl,wi),0),0)-...
                    dFactor(tInd).*LGDI.*min(min(V(tInd,wi),0)-min(C(1+tIndColl,wi),0),0);
    end
end
cbva=cumsum(cbva./size(V,2),1);
cdva=cumsum(cdva./size(V,2),1);
ccva=cumsum(ccva./size(V,2),1);
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