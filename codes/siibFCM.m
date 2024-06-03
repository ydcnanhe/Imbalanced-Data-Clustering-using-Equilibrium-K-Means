function [idx, C, numit,W, D, J] =  siibFCM(X,K,m,tol,maxit,R)
%
% I implement siibFKM according to the following reference:
% P.-L. Lin, P.-W. Huang, C.-H. Kuo, and Y. Lai, “A size-insensitive
% integrity-based fuzzy c-means method for data clustering,” Pattern
% Recognition, vol. 47, no. 5, pp. 2042–2056, 2014.
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
        tmp=X'; % P-by-N matrix
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
        U=ones(size(X,1),K);
        Mu=ones(K,1);
        D=sqeuclidean(X,C);
        W=membership(D,m,U);
        S=zeros(K,1);
        Rho=zeros(size(X,1));
        while 1
            [~,idx]=max(W,[],2);
            for k=1:K
                Mu(k)=sum(sqrt(D(idx==k,k)))/(sum(idx==k)+1);
                Comp(k)=1-sqrt(sum(((sqrt(D(idx==k,k))-Mu(k)).^2))/(sum(idx==k)+1));% add one to avid zero denominator (not eps because one is the smallest value of cluster size)
            end
            p=zeros(K,size(X,1));
            for k=1:K
                tmp2=zeros(K,1);
                for k2=1:K
                    if k2==k
                        tmp2(k2)=realmax;
                    else
                        tmp2(k2)=vecnorm(C(k)-C(k2));
                    end
                end
                [~,j]=min(tmp2);
                for i=1:size(X,1)
                    p(k,i)=abs(vecnorm(X(i)-C(k))-vecnorm(X(i)-C(j)))/(vecnorm(C(k)-C(j))+eps);
                end
            end
            P=zeros(K,1);
            for k=1:K
                P(k)=sum(p(k,idx==k))/(sum(idx==k)+1); % add one to avid zero denominator (not eps because one is the smallest value of cluster size)
            end
            I=0.5*(Comp'+P);
            I_star=(I-min(I))./(max(I)-min(I)+eps);
            for k=1:K
                S(k)=sum(idx==k);
            end
            S=S./size(X,1);
            for i=1:size(X,1)
                Rho(i)=(1-S(idx(i)))/max(1-S);
            end
            U=zeros(size(X,1),K);
            for k=1:K
                for i=1:size(X,1)
                    p_star=exp((1-I_star(k))*p(k,i));
                    U(i,k)=p_star*Rho(i);
                end
            end
            for k=1:K
                sum_Wk=sum(W(:,k).^m);
                if sum_Wk==0 % which means there is one centroid too far from data so all membership to that centroid is zero
                    sum_Wk=eps; % to prevent NaN
                end
                C(k,:)=sum(W(:,k).^m.*X,1)/sum_Wk; % update k-th centroid
            end
            D=sqeuclidean(X,C);
            W=membership(D,m,U);
            if isnan(W)
                pause();
            end
            if norm(C-C_old,'fro')/norm(C,'fro')<tol
                break;
            end
            if it+1>maxit
                disp(['Failed to converge in ' num2str(maxit) ' iterations during replicate ' ...
                    num2str(r) ' for siibFCM with ' num2str(K) ' clusters'])
                break;
            else
                it=it+1;
                C_old=C;
            end
        end
%% Calculate objectives
        W_set(:,:,r)=W;
        J=sum(sum(W.^m.*D));
        J_set(r)=J;
        C_set(:,:,r)=C;
        [~,idx]=max(W,[],2);
        idx_set(:,r)=idx;
        numit_set(r)=it;
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
    % D: N-by-K matrix containing distances between data points and
    % centroids

% function code
    Q=size(X,1);
    K=size(C,1);
    D=zeros(Q,K);
    for k=1:K
        c=squeeze(C(k,:));
        D(:,k)=vecnorm(X-c,2,2).^2;
    end
end

function W=membership(D,m,U)
% Input:
% D: N-by-K matrix containing defined distance matrix (From N data points to K centeroids).
% m: smoothing parameter
% U: N-K matrix containing cluster size coefficient
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
        W(:,k)=U(:,k).*tmp.^-1;
    end
    % prevent all 0 membership because of numerical resolution
    zero_idx=find(sum(W,2)==0);
    [~,pos]=min(D(zero_idx,:),[],2);
    W(zero_idx,:) = (pos==1:K);
end