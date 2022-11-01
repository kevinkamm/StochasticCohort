function PD=defaultProbabilityGenerator(Rrec,dataset)
    K=size(Rrec,1);
    N=size(Rrec,3);
    factors=ones(K,1);
    
    switch dataset
        case 0
            factors(1:3)=1.05;
        case 1
            factors(1)=10;
            factors(2)=5;
            factors(3)=1.05;
        case 2
            factors(1)=50;
            factors(2)=25;
            factors(3)=2.5;
        otherwise
            factors=ones(K,1);
            warning('Unknown dataset, set factors to 1, i.e. measure P=Q')
    end

    PD=zeros(K,N);
    for iT=1:1:N
        PD(:,iT)=Rrec(:,end,iT).*factors;
    end
end