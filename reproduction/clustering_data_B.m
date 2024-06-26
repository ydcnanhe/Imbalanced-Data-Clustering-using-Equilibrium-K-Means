clear
clc
rng(0)
replicate=5000;
%% load data
num_class=3;
num_feature=2;
load('./data-B/Data-B.mat')
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
title('Data-B','FontSize',15)
xlabel('Normalized feature 1')
ylabel('Normalized feature 2')
legend off
saveas(gcf,'./data-B/ground_truth.jpg');
hold off;
%
% cv0
label=unique(true_idx);
for k=1:num_class
    Ns(k)=sum(true_idx==label(k));
end
cv0=std(Ns)/mean(Ns);
%
alpha=1;
output='./data-B';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);