clear
clc
rng(0)
replicate=5000;
%% load data
num_class=8;
num_feature=7;
num_instance=336;
fid = fopen('./ecoli/ecoli.data');
data = textscan(fid,'%s %f32 %f %f %f %f %f %f %s');
fclose(fid);
X=zeros(num_instance,num_feature);
for p=1:num_feature
    X(:,p)=data{p+1};
end
true_idx_cell=data{end};
true_idx = zeros(size(X,1),1);
true_idx(true_idx_cell=="cp")=1;
true_idx(true_idx_cell=="im")=2;
true_idx(true_idx_cell=="pp")=3;
true_idx(true_idx_cell=="imU")=4;
true_idx(true_idx_cell=="om")=5;
true_idx(true_idx_cell=="omL")=6;
true_idx(true_idx_cell=="imL")=7;
true_idx(true_idx_cell=="imS")=8;
% normalization
for p=1:num_feature
    X(:,p)=X(:,p)-mean(X(:,p));
    X(:,p)=X(:,p)/std(X(:,p));
end

% Ground Truth
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('Ecoli','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./ecoli/ground_truth.jpg');
hold off;
% cv0
label=unique(true_idx);
for k=1:num_class
    Ns(k)=sum(true_idx==label(k));
end
cv0=std(Ns)/mean(Ns);
%
alpha='dvariance';
output='./ecoli';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);