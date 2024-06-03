function [idx, C, numit, smoothness, W, sumd, D, J] =  smooth_kmeans(X,K,options)
%
% Version 0.5
% Created at June, 2024
% Author: Yudong He
% Email: yhebh@connect.ust.hk
%
%% Description:
%
% Inputs:
% X: numeric matrix. N-by-P data matrix containing N samples with P features.
% K: positive integer. number of clusters.
% options:
% - Distance: 'sqeuclidean' (default) | 'cosine'. Method for defining similarity.
%   'sqeuclidean': squared euclidean distance. 'cosine': cosine
%
% - SmoothMethod: 'wcss' (default) | 'logsumexp' | 'p-norm' | 'Boltzmann'. Method for specifying smoothing functions.
%   'wcss': within-sum-of-squared (hard k-means). 'logsumexp': the logsumexp function (MEFC). 
%   'p-norm': the p-norm function (fuzzy  k-means). 'Boltzmann': the Boltzmann operator (EKM).
%
% - SmoothCoefficient: double (default 1) | "dvariance". Smoothing parameter or method to calculate the smoothing parameter.
%   'dvariance': data variance.
%
% - MaxIter: 500 (default) | positive integer. Maximum number of iterations.
%
% - Eta: 1e-3 | positive double. Tolerance for convergence
% 
% - Replicates: 1 (default) | positive integer. Replicate number. The replication with the lowest objetive value will be chosen as the final outcome.
%
% - Start: 'plus' (default) | numeric matrix. Method for specifying centroids initialization | K-by-P matrix containing initial centroids
%   'plus': the k-means++ algorithm
%
% Outputs:
% idx: numeric column vector. the cluster indices in an N-by-1 vector
% C: numeric matrix. Estimated centroids in an K-by-P matrix
% W: numeric matrix. Weights in N-by-K matrix.
% sumd: numeric column vector. Within-cluster sums of point-to-centroid distances in the K-by-1 vector
% D: numeric matrix. distances from each point to every centroid in the N-by-K matrix.
% J: scalar number. The lowest loss value over replications.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2024 Yudong He
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  Arguments validation
    arguments
        X (:,:) double {mustBeNonempty,mustBeFinite,mustBeReal}
        K double {mustBeInteger,mustBePositive} 
        options.Distance char {mustBeText} = 'sqeuclidean'
        options.SmoothMethod char {mustBeText} = 'wcss'
        options.SmoothCoefficient = 1
        options.MaxIter double {mustBeInteger,mustBePositive} = 500
        options.Eta double {mustBePositive} = 1e-3 % for convergence
        options.Replicates double {mustBeInteger,mustBePositive} = 1
        options.Start = 'plus'
%         options.EmptyAction char {mustBeText} ='singleton' 
    end
    
%% 
    if isnumeric(options.Start)
        R=size(options.Start,3);
    else
        R=options.Replicates;
    end
    % Define distance function
    switch options.Distance
        case 'sqeuclidean'
            dist=@sqeuclidean;
        case 'cosine'
            dist=@cosine;
    end
    % Define weight function
    switch options.SmoothMethod
        case 'wcss'
            membership=@wcss_membership;
        case 'logsumexp'
            membership=@lse_membership;
        case 'p-norm'
            membership=@pn_membership;
        case 'Boltzmann'
            membership=@boltzmann_membership;
    end
    if ischar(options.SmoothCoefficient)
        switch options.SmoothCoefficient
            case 'dvariance'
                options.SmoothCoefficient = 2/(mean(dist(X,mean(X))));
%             case 'you can add more methods to determine an appropriate smoothing parameter here'
        end
    end
    SmoothCoefficient=options.SmoothCoefficient;
    J_set=zeros(R,1);
    C_set=randn(K,size(X,2),R);
    numit_set=zeros(R,1);
    SmoothCoefficient_set=zeros(R,1);
%% Start replicates
    for r=1:R
        % Initialize centroid
        if isnumeric(options.Start)
            C=squeeze(options.Start(:,:,r)); % K-by-P matrix
        else
            switch options.Start
                case 'plus' % K-means++
                    tmp=X'; % P-by-Q matrix
                    C = tmp(:,1+round(rand*(size(tmp,2)-1)));
                    L = ones(1,size(tmp,2));
                    for i = 2:K
                        D = tmp-C(:,L);
                        D = cumsum(sqrt(dot(D,D,1)));
                        if D(end) == 0, C(:,i:K) = tmp(:,ones(1,K-i+1)); return; end
                        C(:,i) = tmp(:,find(rand < D/D(end),1));
                        [~,L] = max(bsxfun(@minus,2*real(C'*tmp),dot(C,C,1).'));
                    end
                    C=C'; % K-by-P matrix
%                 case 'you can add more initialization methods here'
            end
        end
%% Find Centroid
        C_old=C;
        it=1;
        W=ones(size(X,1),K);
        while 1
            D=dist(X,C);
            W=membership(D,SmoothCoefficient);
            for k=1:K
                  % calculate mag of partial derivative w.r.t k-th centroid
%                 partial_J_norm(k)=norm(sum(W(:,k).*(X-C(k,:))));
                sum_Wk=sum(W(:,k));
                if sum_Wk==0 % which means there is one centroid too far from data so all membership to that centroid is zero
                    sum_Wk=eps; % to prevent NaN
                end
                C(k,:)=sum(W(:,k).*X,1)/sum_Wk; % update k-th centroid
            end

            if norm(C-C_old,'fro')/norm(C,'fro')<options.Eta 
                break;
            end

            if it+1>options.MaxIter
                disp(['Failed to converge in ' num2str(options.MaxIter) ' iterations during replicate ' ...
                    num2str(r) ' for ' options.SmoothMethod ' with ' num2str(K) ' clusters'])
                break;
            else
                C_old=C;
                it=it+1;
            end
        end
%% Calculate objectives
        D=dist(X,C);
        switch options.SmoothMethod
            case 'wcss'
                J=sum(min(D,[],2));
            case 'logsumexp'
                lambda=SmoothCoefficient;
                J=-1/lambda*sum(log(sum(exp(-lambda*D),2)));
            case 'p-norm'
                p=SmoothCoefficient;
                J=sum(sum(D.^-p,2).^(-1/p));
            case 'Boltzmann'
                alpha=SmoothCoefficient;
                J=sum(sum(D.*exp(-alpha*D),2)./sum(exp(-alpha*D),2));
        end
        J_set(r)=J;
        C_set(:,:,r)=C;
        numit_set(r)=it;
        SmoothCoefficient_set(r)=SmoothCoefficient;
    end
    [~,best_id]=min(J_set);
    J=J_set(best_id);
    C=C_set(:,:,best_id);
    numit=numit_set(best_id);
    smoothness=SmoothCoefficient_set(best_id);
    D=dist(X,C);
    [~,idx]=min(D,[],2);
    sumd=zeros(K,1);
    for k=1:K
        sumd(k)=sum(D(idx==k,k));
    end
    W=membership(D,SmoothCoefficient);
end

function D=sqeuclidean(X,C)
% Input:
    % X: N-by-P matrix containing N instances.
    % C: K-by-P matrix containing K centroids.
% Output
    % D: N-by-K matrix containing distances between data points and
    % centroids

% function code
    Q=size(X,1);
    K=size(C,1);
    D=zeros(Q,K);
    for k=1:K
        c=squeeze(C(k,:));
        D(:,k)=0.5*vecnorm(X-c,2,2).^2;
    end
end

function D=cosine(X,C)
% Input:
    % X: N-by-P matrix containing N instances.
    % C: K-by-P matrix containing K centroids.
% Output
    % D: N-by-K matrix containing distances between data points and
    % centroids
    Q=size(X,1);
    K=size(C,1);
    D=zeros(Q,K);
    for k=1:K
        c=squeeze(C(k,:));
        D(:,k)=1-X*c'./(vecnorm(X,2,2)*norm(c,2));
    end
end

function W=wcss_membership(D,lambda)
% Input:
% D: N-by-K matrix containing defined distance matrix (From N data points to K centeroids).
% lambda: no use.
% Output:
% W: N-by-K  matrix containing weights

% function code  
    K=size(D,2);
    [~,pos]=min(D,[],2);
    W = (pos==1:K);
end

function W=lse_membership(D,lambda)
% Input:
% D: N-by-K matrix containing defined distance matrix (From N data points to K centeroids).
% lambda: smoothing parameter
% Output:
% W: N-by-K  matrix containing weights

% function code  
    K=size(D,2);
    W=zeros(size(D));
    for k=1:K
        W(:,k)=exp(-lambda.*D(:,k))./sum(exp(-lambda.*D),2);
    end
    % prevent all 0 membership because of numerical resolution
    zero_idx=find(sum(W,2)==0);
    [~,pos]=min(D(zero_idx,:),[],2);
    W(zero_idx,:) = (pos==1:K);
end

function W=pn_membership(D,p)
% Input:
% D: N-by-K matrix containing defined distance matrix (From N data points to K centeroids).
% p: smoothing parameter
% Output:
% W: N-by-K  matrix containing weights

% function code  
    K=size(D,2);
    W=zeros(size(D));
    D=D+eps; % preventing NAN
    for k=1:K
        W(:,k)=D(:,k).^(-p-1)./(sum(D.^-p,2).^(1/p+1));
    end
    % prevent all 0 membership because of numerical resolution
    zero_idx=find(sum(W,2)==0);
    [~,pos]=min(D(zero_idx,:),[],2);
    W(zero_idx,:) = (pos==1:K);
end

function W=boltzmann_membership(D,alpha)
% Input:
% D: N-by-K matrix containing defined distance matrix (From N data points to K centeroids).
% alpha: smoothing parameter
% Output:
% W: N-by-K  matrix containing weights

% function code  
    K=size(D,2);
    W=zeros(size(D));
    J=sum(D.*exp(-alpha*D),2)./sum(exp(-alpha*D),2); % objectives contributed by Q points individually
    for k=1:K
        W(:,k)=exp(-alpha*D(:,k))./sum(exp(-alpha*D),2).*(1-alpha*(D(:,k)-J));
    end
    % prevent all 0 membership because of numerical resolution
    zero_idx=find(sum(W,2)==0);
    [~,pos]=min(D(zero_idx,:),[],2);
    W(zero_idx,:) = (pos==1:K);
end

    