close all
clear
clc

%train and test sets
train = csvread('train.dat');
test = csvread('test.dat');

% parameters
iter   = 1000;       %iteration count
w1 = rand(10,22); 
w2 = rand(1,11); 
rate   = 0.25;       %step size 

for i=1:iter
    for j=1:size(train, 1)
        P = zeros(1,10);
        z = zeros(1,10);
        for jj=1:10
            for ll=1:21
                P(jj) = P(jj) + train(j,ll)*w1(jj,ll);   %P1 = i1*w1,1 + i2*w1,2 + w1,0;
            end
            P(jj) = P(jj) + w1(jj,22);
            z(jj) = 2/(1+exp(-2*P(jj)))-1;                   %f(P1)=z1
        end
     
        P_3 = 0;
        for jj=1:10
            P_3 = P_3+z(jj)*w2(jj);
        end
        P_3 = P_3+w2(11);
        z3 = 2/(1+exp(-2*P_3))-1;        %f(P3)=z3
        
        d_z = train(j,22)-z3;             %delta P over f(z)

  
        delta = zeros(10,22);
        for kk=1:10
            for ii= 1:21
                delta(kk,ii)=d_z*(1-z3*z3)*w2(kk)*(1-z(kk)*z(kk))*train(j,ii);
            end
            delta(kk,22)=d_z*(1-z3*z3)*w2(kk)*(1-z(kk)*z(kk));
        end
        %update first node's weights
        for kk=1:10
            w1(kk,:) = w1(kk,:) + rate*delta(kk);
        end
        
        %update output node's weights
        delta3 = zeros(1,11);
        for kk=1:10
            delta3(kk) = d_z*(1-z3*z3)*z(kk);
        end
        delta3(11) = d_z*(1-z3*z3);
        for kk=1:10
            w2(kk) = w2(kk) + rate*delta3(kk);
        end
    end
end 

err = 0;
result = zeros(size(train,1),3);
for k=1:size(train,1)
    P = zeros(1,10);
    z = zeros(1,10);
    for jj=1:10
        for ll=1:21
            P(jj) = P(jj) + train(k,ll)*w1(jj,ll);   %P1 = i1*w1,1 + i2*w1,2 + w1,0;
        end
        P(jj) = P(jj) + w1(jj,22);
        z(jj) = 1/(1+exp(-P(jj)));                   %f(P1)=z1
    end

    P_3 = 0;
    for jj=1:10
        P_3 = P_3+z(jj)*w2(jj);
    end
    P_3 = P_3+w2(11);
    z3 = 2/(1+exp(-2*P_3))-1;        %f(P3)=z3
    
    result(k,1)=z3;
    result(k,2)=train(k,22);
    result(k,3)=abs(train(k,22)-z3);
    err = err + result(k,3);
end
err/size(train,1)
