rng(0);

N1 = 5000;
c1 = [0 0];

N2 = 50;
c2 = [5 5];

N3 = 50;
c3 = [-5 5];

N4 = 50;
c4 = [-5 -5];

N5 = 50;
c5 = [5 -5];

N6 = 50;
c6 = [10 0];

N7 = 50;
c7 = [-10 0];

N8 = 50;
c8 = [0 10];

N9 = 50;
c9 = [0 -10];


ball1 = [3*(rand(N1,2)-0.5)+c1 zeros(N1,1)];

ball2 = [randn(N2,2)+c2 zeros(N2,1)+1];

ball3 = [randn(N3,2)+c3 zeros(N3,1)+2];


ball4 = [randn(N4,2)+c4 zeros(N4,1)+3];

ball5 = [randn(N5,2)+c5 zeros(N5,1)+4];

ball6 = [randn(N6,2)+c6 zeros(N6,1)+5];

ball7 = [randn(N7,2)+c7 zeros(N7,1)+6];

ball8 = [randn(N8,2)+c8 zeros(N8,1)+7];

ball9 = [randn(N9,2)+c9 zeros(N9,1)+8];


data = [ball1; ball2;ball3;ball4;ball5;ball6;ball7;ball8;ball9];

X=data(:,1:2);
true_idx=data(:,3)+1;

gscatter(X(:,1),X(:,2),true_idx);

save("./Data-D.mat","data");