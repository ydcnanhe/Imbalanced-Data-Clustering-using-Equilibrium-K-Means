clear
clc
rng(0) %
replicate=5000;
%% load data
num_class=2;
num_feature=7;
num_instance=14987;
load("./pulsar/pulsar.mat");
true_idx=X(:,end);
X(:,end)=[];
% normalization
for p=1:num_feature
    X(:,p)=X(:,p)-mean(X(:,p));
    X(:,p)=X(:,p)/std(X(:,p));
end
% reference scatter diagram
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('Pulsar Cleaned')
xlabel('Normalized feature 1')
ylabel('Normalized feature 2')
legend off
saveas(gcf,'./pulsar/ground_truth.jpg');
hold off;
% cv0
label=unique(true_idx);
for k=1:num_class
    Ns(k)=sum(true_idx==label(k));
end
cv0=std(Ns)/mean(Ns);
%
alpha='dvariance';
output='./pulsar';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);