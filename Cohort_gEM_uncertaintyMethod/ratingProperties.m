function [mDC,sDD,dML,iRS,rSO]=ratingProperties(R,months)
    tStr=strsplit(sprintf('t=%1.2g;',months./12.'),';');
    tStr=tStr(1:end-1);
    dtStr = cell(1,length(tStr)-1);
    for ti=1:1:length(dtStr)
        dtStr{ti}=['dt=',replace([tStr{ti+1},'-',tStr{ti}],'t=','')];
    end
    ratings=strsplit(sprintf('%d;',1:1:size(R,1)),';');
    ratings=ratings(1:end-1);
    dRatings=cell(length(ratings)-1,1);
    for ri=1:1:length(dtStr)
        dRatings{ri}=[ratings{ri},'-',ratings{ri+1}];
    end
    mDC=monotoneDefaultColumn();
    sDD=stronglyDiagonalDominant();
    dML=downMoreLikely();
    iRS=increasingRatingSpread();
    rSO=rowSumOne();
    function mDC=monotoneDefaultColumn()
        mDC=array2table(squeeze(mean(diff(R(:,end,:,:),1,1)>=0,4)),...
                'VariableNames',tStr,...
                'RowNames',dRatings);
    end
    function sDD=stronglyDiagonalDominant()
        ind=repmat(eye(size(R,1),'logical'),[1 1 size(R,[3,4])]);
        valueDiag=reshape(R(ind),size(R,[2,3,4]));
        valueOffDiag=R;
        valueOffDiag(ind)=0;
        valueOffDiag=squeeze(sum(valueOffDiag,2));
        sDD = array2table(mean(valueDiag-valueOffDiag>0,3),...
                'VariableNames',tStr,...
                'RowNames',ratings);
    end
    function dML=downMoreLikely()
        indUpper=repmat(triu(ones(size(R,[1,2]),'logical'),1),[1 1 size(R,[3,4])]);
        indLower=repmat(tril(ones(size(R,[1,2]),'logical'),-1),[1 1 size(R,[3,4])]);
        d=[(size(R,1)*size(R,2)-size(R,1))/2,size(R,[3,4])];
        dML=array2table(squeeze(mean(sum(reshape(R(indUpper),d),1)-sum(reshape(R(indLower),d),1)>0,3)),...
                'VariableNames',tStr);
    end
    function iRS=increasingRatingSpread()
        ind=repmat(eye(size(R,1),'logical'),[1 1 size(R,[3,4])]);
        valueDiag=reshape(R(ind),size(R,[2,3,4]));
        iRS=array2table(squeeze(mean(diff(valueDiag,1,2)<=0,3)),...
                'VariableNames',dtStr,...
                'RowNames',ratings);
    end
    function rSO=rowSumOne()
        rSO=array2table(squeeze(mean(sum(R,2),4)),...
                'VariableNames',tStr,...
                'RowNames',ratings);
    end
end