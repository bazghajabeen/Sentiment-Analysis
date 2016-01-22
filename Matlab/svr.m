close all
clear
clc

%train and test sets
train = csvread('train.dat');
test = csvread('test.dat');

epsilon=0.25; 
c=400; 

train=[train(:,1) train(:,6) train(:,8) train(:,9) train(:,18) train(:,19) train(:,20) train(:,22)];

svrobj = svr_trainer(train(:,1:7),train(:,8),c,epsilon,'gaussian',0.5);
y = svrobj.predict(train(:,1:7));
y = [y train(:,8) zeros(size(train(:,7),1),1)];

error = 0;
for i=1:size(train(:,8),1)
    y(i,3) = abs(y(i,1)-y(i,2));
    error = error + abs(y(i,1)-y(i,2));
end
error = error/size(y,1)

correl = corr(y(:,2), y(:,1));

figure
scatter(y(:,2), y(:,1), '+b');
title(strcat('correlation of ', num2str(correl)))
xlabel('Label');
ylabel('Estimated Label');
hold on