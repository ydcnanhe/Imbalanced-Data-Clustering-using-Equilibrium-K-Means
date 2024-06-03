clear
clc
rng(0)
replicate=5000;
%% load data
num_class=2;
num_instance=569;
num_feature=30;
fid = fopen('./WDBC/wdbc.data');
data = textscan(fid,['%f %c ' ...
    '%f %f %f %f %f %f %f %f %f %f' ...
    '%f %f %f %f %f %f %f %f %f %f' ...
    '%f %f %f %f %f %f %f %f %f %f'],'Delimiter',{',','\n'});
fclose(fid);
X=zeros(num_instance,num_feature);
for p=1:num_feature
    X(:,p)=data{p+2};
end
true_idx_cell=data{2};
true_idx = zeros(size(X,1),1);
true_idx(true_idx_cell=='B')=1;
true_idx(true_idx_cell=='M')=2;
% normalization
for p=1:num_feature
    X(:,p)=X(:,p)-mean(X(:,p));
    X(:,p)=X(:,p)/std(X(:,p));
end
% cv0
label=unique(true_idx);
for k=1:num_class
    Ns(k)=sum(true_idx==label(k));
end
cv0=std(Ns)/mean(Ns);
%
figure;
gscatter(X(:,1), X(:,2), true_idx);
title('WDBC','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
saveas(gcf,'./WDBC/ground_truth.jpg');
hold off;
%
alpha='dvariance';
output='./WDBC';
is_plot=1;
is_saveplot=1;
eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot);