clear
clc
rng(0)
replicate=5000;
%% load data
num_class=6;
num_feature=9;
num_instance=214;
load("./glass/glass.mat")
X(:,1)=[];
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
title('glass','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./glass/ground_truth.jpg');
hold off;
%
alpha='dvariance';
output='./glass';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);