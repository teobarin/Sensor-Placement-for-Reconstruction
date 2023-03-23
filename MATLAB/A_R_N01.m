%Z=zeros(1,r);
Z=[];
c=0;
while size(Z,2)<r
    Y = exprnd(1)
    U_tilde = rand(1);
    if U_tilde <= exp(-(Y-1)^2)/2
        U=rand(1);
        if U<=1/2
            %Z(i)=-Y;
            Z=[Z,-Y];
        else
            %Z(i)=Y;
            Z=[Z,Y];
        end
    end
    c=c+1;
end
 hist(Z)