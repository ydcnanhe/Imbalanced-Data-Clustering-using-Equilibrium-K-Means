function [idx, C, numit, W, U, sumd, D, J] = ekm(X, K, options)
%
% Version 1
% Created at June, 2024
% Last modified at June, 2025
% Author: Yudong He
% Email: yhebh@connect.ust.hk
% Tested Matlab version: R2022a
%% Description:
%
% Inputs:
% X : N-by-P numeric matrix. Data matrix containing N samples with P features.
% K : positive integer. Number of clusters.
% options : Struct containing optional parameters that govern the algorithm's behavior.
% - Distance: 'euclidean' (default) | 'manhattan'
%   'euclidean': euclidean distance or l2 norm.
%   'manhattan': manhattan distance or l1 norm.
%
% - alpha: double (default 0.5) | "dvariance". Smoothing parameter or method to calculate the smoothing parameter.
%   'dvariance': data variance.
%
% - MaxIter: 500 (default) | positive integer. Maximum number of iterations.
%
% - Eta: 1e-3 | positive double. Tolerance for convergence
% 
% - Replicates: 1 (default) | positive integer. Replicate number. The replication with the lowest objetive value will be chosen as the final outcome.
%
% - Start: 'plus' (default) | numeric matrix. Method for initializing centroid or K-by-P matrix containing initial centroids
%   'plus': the k-means++ algorithm
%
% Outputs:
% idx : N-by-1 numeric column vector. The cluster indices for each sample.
% C : K-by-P numeric matrix. Estimated centroids of the clusters.
% W : N-by-K numeric matrix. Weights for each sample and cluster.
% sumd : K-by-1 numeric column vector. Within-cluster sums of point-to-centroid distances.
% D : N-by-K numeric matrix. Distances from each data point to each centroid.
% J : scalar number. The lowest loss value over replications.
% 
% Note: This function implements an enhanced K-means algorithm, named equilibrium K-means (EKM). EKM is robust to imbalanced data clustering.
%
%% Example:
% [idx,C] = ekm(X,K) returns estimated indices and centroids given instances X and the number of clusters K.
% [idx,C] = ekm(X,K,"alpha",1) specifies the only one hyperparameter, alpha, in EKM.
% [idx,C] = ekm(X,K,"Replicate",100) specifies the number
% of replications. The K-means++ is used to intialize centroid for each
% replication

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference
% He Yudong, An Equilibrium Approach to Clustering: Surpassing Fuzzy C-Means
% on Imbalanced Data, IEEE Transactions on Fuzzy Systems, 2025.

% He Yudong, Imbalanced Data Clustering Using Equilibrium K-Means, arXiv, 2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2024 Yudong He
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Arguments validation
    arguments
        X (:,:) double {mustBeNonempty, mustBeFinite, mustBeReal}
        K double {mustBeInteger, mustBePositive}
        options.metric char {mustBeText} = 'euclidean'
        options.alpha = 0.5
        options.MaxIter double {mustBeInteger, mustBePositive} = 500
        options.Eta double {mustBePositive} = 1e-3 % for convergence
        options.Replicates double {mustBeInteger, mustBePositive} = 1
        options.Start = 'plus'
    end
    
    %% Initialize necessary variables
    if isnumeric(options.Start)
        R = size(options.Start, 3);
    else
        R = options.Replicates;
    end
   
    % Define distance function based on user selection
    switch options.metric
        case 'euclidean'
            dist = @euclidean;
        case 'manhattan'
            dist = @manhattan;
        otherwise
            error('Unsupported distance metric specified.');
    end
    
    % Handle the alpha parameter based on its type
    if ischar(options.alpha)
        switch options.alpha
            case 'dvariance'
                options.alpha = 2/(mean(dist(X,mean(X)).^2)); % determine alpha based on data variance
            otherwise
                error('Unsupported alpha type specified.');
        end
    end
    
    %% Prepare for multiple replicates
    J_set = zeros(R, 1);
    C_set = zeros(K, size(X, 2), R); % Preallocate for centroids
    numit_set = zeros(R, 1);
    
    %% Start replicates
    for r = 1:R
        % Initialize centroid based on specified method
        if isnumeric(options.Start)
            C = squeeze(options.Start(:, :, r)); % K-by-P matrix
        else
            switch options.Start
                case 'plus' % K-means++ initialization
                    C = kmeans_plus_init(X, K);
                otherwise
                    error('Unsupported initialization method specified.');
            end
        end
        
        %% Find Centroids and Assign Weights
        C_old = C; % Initialize old centroids
        it = 1;    % Iteration counter
        while true
            D = dist(X, C); % Calculate distances
            W = calc_weight(D, options.alpha); % Calculate weights
            
            % Update centroids
            for k = 1:K
                sum_Wk = sum(W(:, k));
                if sum_Wk == 0
                    sum_Wk = eps; % Prevent NaN for empty clusters
                end
                C(k, :) = sum(W(:, k) .* X, 1) / sum_Wk; % Update centroid
            end
            
            % Check for convergence
            if norm(C - C_old, 'fro') / norm(C, 'fro') < options.Eta
                break; % Convergence achieved
            end
            
            % Check if maximum iterations reached
            if it >= options.MaxIter
                fprintf('Failed to converge in %d iterations during replicate %d for EKM with %d clusters\n', ...
                        options.MaxIter, r, K);
                break;
            else
                C_old = C; % Update old centroid for next iteration
                it = it + 1; % Increment iteration count
            end
        end
        
        %% Calculate objectives
        D = dist(X, C);
        J = sum(sum(D.^2 .* exp(-options.alpha * D.^2), 2) ./ (sum(exp(-options.alpha * D.^2), 2) + eps));
        J_set(r) = J;
        C_set(:, :, r) = C; % Store centroids
        numit_set(r) = it; % Store number of iterations
    end
    
    %% Final Results Selection
    [~, best_id] = min(J_set);
    J = J_set(best_id);
    C = C_set(:, :, best_id);
    numit = numit_set(best_id);
    D = dist(X, C); % Final distance calculation
    
    % Assign cluster indices
    [~, idx] = min(D, [], 2);
    
    % Calculate the within-cluster sums of point-to-centroid distances
    sumd = arrayfun(@(k) sum(D(idx == k, k)), 1:K)'; % Vectorized for each cluster
    
    % Compute weights
    W = calc_weight(D, options.alpha);
    % Compute membership
    U = exp(-options.alpha .* D.^2) ./ (sum(exp(-options.alpha .* D.^2), 2) + eps); % Normalized memberships
end

function C = kmeans_plus_init(X, K)
    % K-means++ initialization for centroid selection
    tmp=X'; % P-by-N matrix
    C = tmp(:,1+round(rand*(size(tmp,2)-1))); % randomly select the first centroid
    L = ones(1,size(tmp,2));
    for i = 2:K
        D = tmp-C(:,L);
        D = cumsum(sqrt(dot(D,D,1)));
        if D(end) == 0, C(:,i:K) = tmp(:,ones(1,K-i+1)); return; end
        C(:,i) = tmp(:,find(rand < D/D(end),1));
        [~,L] = max(bsxfun(@minus,2*real(C'*tmp),dot(C,C,1).'));
    end
    C=C'; % K-by-P matrix
end

function D = euclidean(X, C)
    % Euclidean distance computation
    % X: N-by-P matrix; C: K-by-P matrix
    N=size(X,1);
    K=size(C,1);
    D = zeros(N, K);
    for k = 1:K
        D(:, k) = vecnorm(X - C(k, :), 2, 2); % Calculate euclideaan distance vector
    end
end

function D = manhattan(X, C)
    % Manhattan distance computation
    % X: N-by-P matrix; C: K-by-P matrix
    N=size(X,1);
    K=size(C,1);
    D = zeros(N, K);
    for k = 1:K
        D(:, k) = sum(abs(X - C(k, :)), 2); % Calculate euclideaan distance vector
    end
end

function W = calc_weight(D, alpha)
    % Calculate weights based on distances D and smoothing parameter alpha
    K=size(D,2);
    J=sum(D.^2.*exp(-alpha*D.^2),2)./(sum(exp(-alpha*D.^2),2)+eps); % objectives contributed by N points individually
    W=exp(-alpha*D.^2)./repmat(sum(exp(-alpha*D.^2),2)+eps,1,K).*(1-alpha*(D.^2-repmat(J,1,K)));
    % prevent all 0 membership because of numerical precision
    zero_idx=find(sum(W,2)==0);
    [~,pos]=min(D(zero_idx,:),[],2);
    W(zero_idx,:) = (pos==1:K);
end