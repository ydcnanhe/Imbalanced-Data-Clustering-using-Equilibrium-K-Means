clear
clc
rng(0)
replicate=5000;
%% load data
num_class=3;
num_feature=7;
num_instance=210;
load("./seeds/seeds.mat");
X(any(isnan(X), 2), :) = [];
true_idx=X(:,end);
X(:,end)=[];

% reference scatter diagram
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('Seeds','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./seeds/ground_truth.jpg');
hold off;
% cv0
label=unique(true_idx);
for k=1:num_class
    Ns(k)=sum(true_idx==label(k));
end
cv0=std(Ns)/mean(Ns);
%
alpha='dvariance';
output='./seeds';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);