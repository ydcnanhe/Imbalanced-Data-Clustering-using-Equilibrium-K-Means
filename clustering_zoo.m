clear
clc
rng(0)
replicate=5000;
%% load data
num_class=7;
num_feature=16;
num_instance=101;
fid = fopen('./zoo/zoo.data');
data = textscan(fid,'%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d','Delimiter',{',','\n'});
fclose(fid);
X=zeros(num_instance,num_feature);
for p=1:num_feature
    X(:,p)=data{p+1};
end
true_idx=data{end};
true_idx=double(true_idx);
% normalization
for p=1:num_feature
    X(:,p)=X(:,p)-mean(X(:,p));
    X(:,p)=X(:,p)/std(X(:,p));
end
% reference scatter diagram
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('Zoo','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./zoo/ground_truth.jpg');
hold off;
%
alpha='dvariance';
output='./zoo';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);