clear
clc
rng(0)
replicate=5000;
%% load data
num_class=7;
num_feature=19;
num_instance=2310;
fid = fopen('./IS/segmentation.data');
data = textscan(fid,['%s %f ' ...
    '%f %f %f %f %f %f %f %f %f %f' ...
    '%f %f %f %f %f %f %f %f'],'Delimiter',{',','\n'});
fclose(fid);
X=zeros(num_instance,num_feature);
for p=1:num_feature
    X(:,p)=data{p+1};
end 
X(:,3)=[]; num_feature=num_feature-1; % feature 3 is zero for all instances
true_idx_cell=data{1};
true_idx = zeros(size(X,1),1);
true_idx(true_idx_cell=="BRICKFACE")=1;
true_idx(true_idx_cell=="SKY")=2;
true_idx(true_idx_cell=="FOLIAGE")=3;
true_idx(true_idx_cell=="CEMENT")=4;
true_idx(true_idx_cell=="WINDOW")=5;
true_idx(true_idx_cell=="PATH")=6;
true_idx(true_idx_cell=="GRASS")=7;

% normalization
for p=1:num_feature
    X(:,p)=X(:,p)-mean(X(:,p));
    X(:,p)=X(:,p)/std(X(:,p));
end

% reference scatter diagram
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('IS','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./IS/ground_truth.jpg');
hold off;

% cv0
label=unique(true_idx);
for k=1:num_class
    Ns(k)=sum(true_idx==label(k));
end
cv0=std(Ns)/mean(Ns);
%
alpha='dvariance';
output='./IS';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);