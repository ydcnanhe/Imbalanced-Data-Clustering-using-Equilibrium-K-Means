# Repository Description
This repository contains datasets and codes for reproducing results in our paper [Imbalanced Data Clustering Using Equilibrium K-Means](https://arxiv.org/abs/2402.14490v3).

# Equilibrium K-Means: A K-Means type Clustering Algorithm for Imbalanced Data
## The objective of EKM:

$$\min_{c_1,\cdots,c_K}\sum_{n=1}^N \left(\sum_{i=1}^K d_{in} e^{-\alpha d_{in}}\right)/\left(\sum_{i=1}^K e^{-\alpha d_{in}}\right),$$
where $c_1,\cdots,c_K$ are K centroids as centers of clusters, $d_{in}$ is the distance (usually the Euclidean distance) between $n$-th data point and $i$-th centroid, and $\alpha$ is a parameter to be tuned.

## Optimization by a two-step iteration algorithm:

Step 1. Computing weights:

$$w_{kn}^{(\tau)}=\frac{e^{-\alpha d_{kn}^{(\tau)}}}{\sum_i e^{-\alpha d_{in}^{(\tau)}}} [1-\alpha(d_{kn}^{(\tau)}-\frac{\sum_i d_{in}^{(\tau)}e^{-\alpha d_{in}^{(\tau)}}}{\sum_i e^{-\alpha d_{in}^{(\tau)}}})]$$

Step 2. Computing weighted centroids:

$$c_k^{(\tau+1)}=\frac{\sum_n w_{kn}^{(\tau)} x_n}{\sum_n w_{kn}^{(\tau)}},$$
where $x_n$ is the $n$-th data point.

EKM converges when centroids cease to change or the maximum number of iterations is reached. The time complexity of one iteration of the above two steps is $O(NK^2P)$ with $P$ being the data dimension.

# Examples
![image](https://github.com/ydcnanhe/Imbalanced-Data-Clustering-using-Equilibrium-K-Means/assets/52923246/8798b7d4-6935-457a-b926-d5118899b9f7)
![image](https://github.com/ydcnanhe/Imbalanced-Data-Clustering-using-Equilibrium-K-Means/assets/52923246/f62bf735-0936-4f6a-9bc9-72b4c7115c8a)
![image](https://github.com/ydcnanhe/Imbalanced-Data-Clustering-using-Equilibrium-K-Means/assets/52923246/9ac03f80-d7d6-4a84-b2b9-f90bfc78259c)
![image](https://github.com/ydcnanhe/Imbalanced-Data-Clustering-using-Equilibrium-K-Means/assets/52923246/552dccda-5f80-4810-a258-fbd56f4ea041)

# How To Use
Install Matlab 2022a (or the latest version), and download this repository to your local directory.

# Clustering a Dataset
EKM is embbeded in the function "smooth_kmeans.m" which you can find in "algorithms". To implement EKM clustering, you shall set the parameter 'SmoothMethod' to 'Boltzmann' and set the parameter 'SmoothCoefficient' properly (this parameter is the $\alpha$ in our paper).

Below is an example of using EKM to cluster the iris dataset.

```
rng(0) % for reproducibility
addpath("./algorithms")
addpath("./metrics")
data = iris_dataset; % load the iris dataset
data = data(1:2,:); % only use the first two features for clustering_Ecoli
data=data';

% normalization
for p=1:2
    data(:,p)=data(:,p)-mean(data(:,p));
    data(:,p)=data(:,p)/std(data(:,p));
end

% create labels of iris dataset
label_iris = ones(150,1);
label_iris(51:100)=2;
label_iris(101:150)=3;

% scatter plot of Iris
figure;
gscatter(data(:,1), data(:,2), label_iris);
title('Iris','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off

% clustering by EKM
alpha=1;
[label_ekm,C,~,~,~,~,~,~]=smooth_kmeans(data,3,'Replicates',1,'SmoothMethod','Boltzmann','SmoothCoefficient',alpha);

% scatter diagram of EKM clustering
figure;
gscatter(data(:,1), data(:,2), label_ekm);
hold on
plot(C(:,1),C(:,2),'k+','MarkerSize',15,'LineWidth',3)
title('EKM clustering for Iris','FontSize',15)
xlabel('Normalized feature 1','FontSize',15)
ylabel('Normalized feature 2','FontSize',15)
legend off
```

# Reproducing the Experiments
To replicate the experiments in the original paper, first put "reproduction", "algorithm", and "metrics" in the same directory (e.g., D:/git-EKM), open Matlab 2022a (or the latest version) and specify the working directory  as "D:/git-EKM", then:
```
addpath("./algorithms")
addpath("./metrics")
```

Specify your Matlab working directory as "D:/git-EKM/reproduction"

If you want to replicate the clustering result on the "Ecoli" dataset, type the following code and enter in the command window
```
clustering_Ecoli
```
After the program is finished, you shall see the generated result files and folders in "./ecoli".

If you want to see the average of clustering quality, run
```
log_avg_best
```
If you want to see the average implementation time and the number of iterations, run
```
log_time
```
