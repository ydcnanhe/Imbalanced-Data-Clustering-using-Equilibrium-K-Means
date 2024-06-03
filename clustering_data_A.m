clear
clc
rng(0)
replicate=5000;
%% load data
num_class=3;
num_feature=2;
load('./data-A/Data-A.mat')
true_idx=data(:,end);
X=data(:,1:num_feature);
num_instance=size(X,1);
% normalization
for p=1:num_feature
    X(:,p)=X(:,p)-mean(X(:,p));
    X(:,p)=X(:,p)/std(X(:,p));
end
% reference scatter diagram
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('Data-A','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./data-A/ground_truth.jpg');
hold off;
% cv0
label=unique(true_idx);
for k=1:num_class
    Ns(k)=sum(true_idx==label(k));
end
cv0=std(Ns)/mean(Ns);
%
alpha=1;
output='./data-A';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);