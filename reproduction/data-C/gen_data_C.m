rng(0);

N1=5000;
N2=200;

ball = [randn(N1,2) zeros(N1,1)];
stick=[3+5*rand(N2,1) rand(N2,1) ones(N2,1)];

X=[ball;stick];
true_idx=X(:,3);
X(:,3)=[];

gscatter(X(:,1),X(:,2),true_idx);

data=[X true_idx];
save("./Data-C.mat","data");