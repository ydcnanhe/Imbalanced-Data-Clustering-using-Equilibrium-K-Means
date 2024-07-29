function [idx, C, numit, W, D, J] =  csiFCM(X,K,m,tol,maxit,R)
%
% I implement csiFKM according to the following reference:
% J. Noordam, W. Van Den Broek, and L. Buydens, “Multivariate image
% segmentation with cluster size insensitive fuzzy c-means,” Chemometrics
% and intelligent laboratory systems, vol. 64, no. 1, pp. 65–78, 2002.
%
if nargin<3
    m=2;
    tol=1e-3;
    maxit=500;
    R=1;
end

J_set=zeros(R,1);
C_set=randn(K,size(X,2),R);
numit_set=zeros(R,1);
idx_set=zeros(size(X,1),R);
W_set=zeros(size(X,1),K,R);
%% Start replicates
    for r=1:R
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
%% Find Centroid
        C_old=C;
        it=1;
        Rho=ones(size(X,1),1);
        D=sqeuclidean(X,C);
        W=membership(D,m,Rho);
        S=zeros(K,1);
        while 1
            [~,idx]=max(W,[],2);
            for k=1:K
                S(k)=sum(idx==k);
            end
            S=S./size(X,1);
            for i=1:size(X,1)
                Rho(i)=(1-S(idx(i)))/(max(1-S)+eps);
            end
            for k=1:K
                C(k,:)=sum(W(:,k).^m.*X,1)/(sum(W(:,k).^m)+eps); % update k-th centroid
            end

            D=sqeuclidean(X,C);
            W=membership(D,m,Rho);
            if norm(C-C_old,'fro')/norm(C,'fro')<tol
                break;
            end
            if it+1>maxit
                disp(['Failed to converge in ' num2str(maxit) ' iterations during replicate ' ...
                    num2str(r) ' for csiFCM with ' num2str(K) ' clusters'])
                break;
            else
                it=it+1;
                C_old=C;
            end
        end
%% Calculate objectives
        J=sum(sum(W.^m.*D));
        J_set(r)=J;
        C_set(:,:,r)=C;
        [~,idx]=max(W,[],2);
        idx_set(:,r)=idx;
        numit_set(r)=it;
        W_set(:,:,r)=W;
    end
    [~,best_id]=min(J_set);
    J=J_set(best_id);
    C=C_set(:,:,best_id);
    D=sqeuclidean(X,C);
    numit=numit_set(best_id);
    idx=idx_set(:,best_id);
    W=W_set(:,:,best_id);
end

function D=sqeuclidean(X,C)
% Input:
    % X: N-by-P matrix containing Q samples.
    % C: K-by-P centroid.
% Output
    % D: N-by-K matrix containing distances between data points and centroids

% function code
    Q=size(X,1);
    K=size(C,1);
    D=zeros(Q,K);
    for k=1:K
        c=squeeze(C(k,:));
        D(:,k)=vecnorm(X-c,2,2).^2;
    end
end

function W=membership(D,m,Rho)
% Input:
% D: N-by-K matrix containing defined distance matrix (From N data points to K centeroids).
% m: smoothing parameter
% Rho: N-1 matrix containing cluster size coefficient
% Output:
% W: N-by-K membership or weight

% function code  
    K=size(D,2);
    W=zeros(size(D));
    D=D+eps; % preventing NAN
    for k=1:K
        tmp=0;
        for k1=1:K
            tmp=tmp+(D(:,k)./D(:,k1)).^(1/(m-1));
        end
        W(:,k)=Rho.*tmp.^-1;
    end
    % prevent all 0 membership because of numerical resolution
    zero_idx=find(sum(W,2)==0);
    [~,pos]=min(D(zero_idx,:),[],2);
    W(zero_idx,:) = (pos==1:K);
end