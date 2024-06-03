clear
clc
rng(0)
replicate=5000;
%% load data
num_class=2;
num_feature=9;
num_instance=6321;
load("./shill/shill.mat");
true_idx=X(:,end);
X(:,end)=[];
% normalization
for p=1:num_feature
    X(:,p)=X(:,p)-mean(X(:,p));
    X(:,p)=X(:,p)/std(X(:,p));
end
% % % dimensional reduction by PCA
% [coeff,score,latent,tsquared,explained] = pca(X);
% X=X*coeff';
% X=X(:,1:2);
% Ground Truth
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('Shill Bidding','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./shill/ground_truth.jpg');
hold off;
% cv0
label=unique(true_idx);
for k=1:num_class
    Ns(k)=sum(true_idx==label(k));
end
cv0=std(Ns)/mean(Ns);
%
alpha='dvariance';
output='./shill';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);