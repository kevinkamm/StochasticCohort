function X=ssa(A,tMarket,i0,M,dt)
%%SSA computes the Gillespie Stochastic Simulation Algorithmn of a
%%piecewise homogeneous CTMC for given generator, time partition and
%%starting state.
%   Input:
%       A (KxKxn array): contains the generator of the ICTMC
%       tMarket (1xn array): contains the time partition, when the
%                            generator changes
%       i0 (int): contains the initial state
%       M (int): number of simulations
%       dt (double): time discretization for output process
%   Output:
%       X (NxM array): contains the trajectories of an ICTMC starting in i0
t=linspace(0,tMarket(end),tMarket(end)/dt+1);
X=zeros(length(t),M,'uint8');
time=cat(2,0,tMarket);
for m=1:1:M
    i=i0;
    for k=1:1:length(tMarket)
        tau=time(k);
        ti=find(t<=tau,1,'last');
        while tau<time(k+1)
            if A(i,i,k)==0
                tau=time(k+1);
            else
                r=rand(2,1);
                temp1=log(r(1))/A(i,i,k);
%                 temp=exprnd(-1/A(i,i,k));
                tau=tau+temp1;
                if tau>=time(k+1)
                    break;
                end
                tiNew=find(t<=tau,1,'last');
                X(ti:1:tiNew,m)=i;
                temp2=A(i,[1:1:i-1,i+1:1:end],k)./(-A(i,i,k));
                temp2=cumsum(temp2);
                j=find(temp2>=r(2),1,'first');
                if j<i
                    i=j;
                else
                    i=j+1;
                end
                ti=tiNew+1;
            end
        end
        tiFin=find(t<=time(k+1),1,'last');
        X(ti:tiFin,m)=i;
    end
end
% figure();hold on;
% plot(t,X(:,:))
end