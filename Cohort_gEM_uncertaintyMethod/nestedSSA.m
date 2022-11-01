% function X=nestedSSA(A,t,i0,M1,M2)   
%     X=zeros(length(t),M2,M1);
%     for iM1=1:1:M1
%         X(:,:,iM1)=ssa(A(:,:,:,iM1),t,i0,M2);
%     end
% end
function X=nestedSSA(A,t,i0,M1,M2,dtSSA)   
    X=zeros(t(end)/dtSSA+1,M2,M1,'uint8');
    parfor iM1=1:M1
        X(:,:,iM1)=ssa(A(:,:,:,iM1),t(2:end),i0,M2,dtSSA);
    end
end