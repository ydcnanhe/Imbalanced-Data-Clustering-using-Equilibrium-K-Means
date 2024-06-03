function[idx,center,U,obj,iter]=fwpkm(data,cluster_n,m)
%
% I implement feature weighted possibilistic k-means according to the 
% following reference:
% M.-S. Yang and J. B. Benjamin, “Feature-weighted possibilistic c-means
% clustering with a feature-reduction framework,” IEEE Transactions on
% Fuzzy Systems, vol. 29, no. 5, pp. 1093–1106, 2020.
%
    % some parameters 
    min_impro=1e-3;
    max_iter=500;
    % calculate lambda
    beta=norm(data-mean(data),"fro")/size(data,1);
    lambda=beta/(m^2*cluster_n);
    %initialize centers by kmeans++
    tmp=data'; % P-by-Q matrix
    C = tmp(:,1+round(rand*(size(tmp,2)-1)));
    L = ones(1,size(tmp,2));
    for i = 2:cluster_n
        D = tmp-C(:,L);
        D = cumsum(sqrt(dot(D,D,1)));
        if D(end) == 0, C(:,i:cluster_n) = tmp(:,ones(1,cluster_n-i+1)); return; end
        C(:,i) = tmp(:,find(rand < D/D(end),1));
        [~,L] = max(bsxfun(@minus,2*real(C'*tmp),dot(C,C,1).'));
    end
    init_center=C'; % K-by-P matrix
    % Initial fuzzy partition and typicality matrix
    W=ones(size(data,2),1);
    W=W./sum(W);
    % initialize U
    U=calc_membership(data,init_center,W,m,lambda);
    %
    old_center=init_center;
    iter=1;
    obj_fwpkm=zeros(max_iter,1);
    while 1
        [W,center,obj_fwpkm(iter)] = stepfwpkm(data,U,m,lambda);
        % update U
        U=calc_membership(data,center,W,m,lambda);
    % check termination condition
        if norm(center-old_center,'fro')/norm(center,'fro')<min_impro
            break;
        end

        if iter+1>max_iter
            disp(['Failed to converge in ' num2str(max_iter) ' iterations during replicate '])
            break;
        else
            old_center=center;
            iter=iter+1;
        end
    end
    [~,idx]=max(U,[],2);
    obj=obj_fwpkm(iter);
end

function U=calc_membership(data,center,W,m,lambda)
    [no_data,no_feature]=size(data);
    [cluster_n,~]=size(center);
    U_comp=zeros(no_data,cluster_n,no_feature);
    for j=1:no_feature
        data_j=data(:,j); % N X 1
        Data_j=repmat(data_j,1,cluster_n); % N X K
        center_j=center(:,j)'; % 1 X K
        Center_j=repmat(center_j,no_data,1); % N X K
        U_comp(:,:,j)=W(j).*(Data_j-Center_j).^2; % N X K X J
    end
    U=(1+(1/lambda*sum(U_comp,3)).^(1/(m-1))).^(-1); % N X K
end

function [W,center,obj_fwpkm]=stepfwpkm(data,U,m,lambda)
    [~,no_feature]=size(data);
    [~,cluster_n]=size(U);
    center=zeros(cluster_n,no_feature); % K X J
    mf = U.^m; % MF matrix after exponential
    % update cluster center
    for j=1:no_feature
        data_j=data(:,j); % N X 1
        Data_j=repmat(data_j,1,cluster_n); % N X K
        center(:,j) = sum(mf.*Data_j,1)./sum(mf,1);
    end
    % update feature weight
    W=calc_fweigght(data,center,U,m,lambda);
    % compute objective
    obj_fwpkm=calc_fwpkm_obj(data,U,center,W,m,lambda);
end

function W=calc_fweigght(data,center,U,m,lambda)
    [no_data,no_feature]=size(data);
    [cluster_n,~]=size(center);
    W=zeros(no_feature,1);
    for j=1:no_feature
        data_j=data(:,j); % N X 1
        Data_j=repmat(data_j,1,cluster_n); % N X K
        center_j=center(:,j)'; % 1 X K
        Center_j=repmat(center_j,no_data,1); % N X K
        W(j)=exp(-cluster_n*(sum(U.^m.*(Data_j-Center_j).^2,"all")+lambda*sum((1-U).^m,"all"))/no_data);
    end
    W=W./sum(W);
end

function obj=calc_fwpkm_obj(data,U,center,W,m,lambda)
    [no_data,no_feature]=size(data);
    [cluster_n,~]=size(center);
    obj_1=0;
    obj_2=0;
    obj_3=0;
    for j=1:no_feature
        data_j=data(:,j); % N X 1
        Data_j=repmat(data_j,1,cluster_n); % N X K
        center_j=center(:,j)'; % 1 X K
        Center_j=repmat(center_j,no_data,1); % N X K
        obj_1=obj_1+sum(W(j)*U.^m.*(Data_j-Center_j).^2,"all");
        obj_2=obj_2+W(j)*sum((1-U).^m,"all");
        obj_3=obj_3+W(j)*log(W(j));
    end
    obj=obj_1+lambda*obj_2+no_data/cluster_n*obj_3;
end