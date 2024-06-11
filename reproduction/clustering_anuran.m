clear
clc
rng(0)
replicate=5000;
%% load data
num_class=10;
num_feature=22;
num_instance=7195;
load("./anuran/anuran.mat");
true_idx=X(:,25);
X(:,[23 24 25 26])=[]; % column 23,24,25,26 are labels, here we choose column 25 as the target label
% normalization
for p=1:num_feature
    X(:,p)=X(:,p)-mean(X(:,p));
    X(:,p)=X(:,p)/std(X(:,p));
end
% reference scatter diagram
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('Anuran Calls','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./anuran/ground_truth.jpg');
hold off;
% calculate cv0
for k=1:num_class
    Ns(k)=sum(true_idx==k);
end
cv0=std(Ns)/mean(Ns);
%
alpha='dvariance';
output='./anuran';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);