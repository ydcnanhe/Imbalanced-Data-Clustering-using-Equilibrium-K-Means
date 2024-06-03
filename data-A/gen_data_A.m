addpath('../')
% clustering of 2D balls
rng(0);
K=3; % number of balls
center=[-2 2;2 -2; 4 4];
Q=[2000 50 200]'; % data-A
X=[];
% noisy points
NoiQ=0; % 
%
figure;
hold on
for k=1:K
    tmp=randn(2,Q(k))+center(k,:)';
    X=[X tmp];
end
X=X';
true_idx=ones(sum(Q),1);
true_idx(Q(1)+1:Q(1)+Q(2))=2;
true_idx(Q(1)+Q(2)+1:end)=3;
gscatter(X(:,1),X(:,2),true_idx);

save('./Data-A.mat',"X");
save('./true_idx.mat',"true_idx");