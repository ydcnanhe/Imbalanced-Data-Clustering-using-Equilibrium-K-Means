clear
clc
rng(0) %
replicate=5000;
%% load data
num_class=2;
num_feature=768;
pca_comp=5;% 5
num_instance=5572;
load("./spam/spam.mat");
X=table2array(X);
X(any(isnan(X), 2), :) = [];
true_idx=X(:,1);
X(:,1)=[];
% dimensional reduction by PCA
[coeff,score,latent,tsquared,explained] = pca(X);
X=X*coeff(:,1:pca_comp);
% normalization
for p=1:pca_comp
    X(:,p)=X(:,p)-mean(X(:,p));
    X(:,p)=X(:,p)/std(X(:,p));
end
% reference scatter diagram
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('Bert-Embedded Spam','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./spam/ground_truth.jpg');
hold off;
% cv0
label=unique(true_idx);
for k=1:num_class
    Ns(k)=sum(true_idx==label(k));
end
cv0=std(Ns)/mean(Ns);
%
alpha='dvariance';
output='./spam';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);