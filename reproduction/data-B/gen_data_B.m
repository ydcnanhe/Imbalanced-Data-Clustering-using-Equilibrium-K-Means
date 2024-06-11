% clustering of 2D square balls
rng(0);
K=3; % number of balls
center=[-2 2;2 -2; 4 4];
Q=[2000 50 200]'; % data-B
X=[];
%
figure;
hold on
for k=1:K
    tmp=3.5*rand(2,Q(k))+center(k,:)';
    scatter(tmp(1,:), tmp(2,:),"o");
    X=[X tmp];
end
X=X';
true_idx=ones(sum(Q),1);
true_idx(Q(1)+1:Q(1)+Q(2))=2;
true_idx(Q(1)+Q(2)+1:Q(1)+Q(2)+Q(3))=3;
gscatter(X(:,1),X(:,2),true_idx);

data=[X true_idx];
save('./Data-B.mat',"data");