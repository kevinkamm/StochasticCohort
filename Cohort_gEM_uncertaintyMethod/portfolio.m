function V=portfolio(V0,T,dt,n,M)
%%PORTFOLIO simulates a portfolio of n cash-flows with M trajectories in
% the time period [0,T] for N grid points. At least one cash-flow is 
% assumed to survive till T.
%   Input:
%       V0 (double): initial value of portfolio
%       T (double): finite time horizon
%       dt (double): mesh size of time grid
%       n (int): number of cash-flows
%       M (int): numer of trajectories
%   Output:
%       V (NxM array): contains the simulated paths of aggregated
%                      cash-flows with (n-1) random life-times and at least
%                      one surviving

% time grid
% dt=T/(N-1);
N=floor((T/dt)+1);
t=linspace(0,T,N);

% n independent Brownian motions
Wn=zeros(N,M,n);
dWn=randn(N-1,M,n);
Wn(2:end,:,:)=sqrt(dt).*dWn;
Wn=cumsum(Wn,1);

% life-time of cash-flows
Un=T.*rand(n,1); %t0+(T-t0).*rand(M,n)
Un(1)=T; % at least one cashflow survives till the end

cashFlows=zeros(N,M,n);
sigma=10.*randn(n,1);

% simulate cash-flows with finite life-time
for i=1:1:n
%     for j=1:1:M
        tInd=t<=Un(i);
        cashFlows(tInd,:,i)=sigma(i).*Wn(tInd,:,i);
%     end
end

% calculate portfolio
V=V0+sum(cashFlows,3);
end