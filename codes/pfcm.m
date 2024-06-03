function[idx,center,U,T,obj,iter]=pfcm(data,cluster_n,options)
%
% I implement possibilistic fuzzy k-means according to the following references:
% [1] N. R. Pal, K. Pal, J. M. Keller, and J. C. Bezdek, "A possibilistic fuzzy
% c-means clustering algorithm,” IEEE transactions on fuzzy systems,
% vol. 13, no. 4, pp. 517–530, 2005.
% [2] https://www.ijser.org/researchpaper/implementation-of-possibilistic-fuzzy-cmeans-clustering-algorithm-in-matlab.pdf
%
    if nargin ~= 2 && nargin ~= 3
        error('Too many or too few input arguments!');
    end
    % Change the following to set default options
    default_options = [2;500;1e-3;0;1;4;2];
    if nargin == 2
        options = default_options;
        else
    % If "options" is not fully specified, pad it with
    % default values.
        if length(options) < 7
            tmp = default_options;
            tmp(1:length(options)) = options;options = tmp;
        end
    % If some entries of "options" are nan's, replace them
    % with defaults.
        nan_index = find(isnan(options)==1);
        options(nan_index) = default_options(nan_index);
        if options(1) <= 1
            error('The exponent should be greater than 1!');
        end
    end
    expo = options(1);
    max_iter = options(2);
    min_impro = options(3);
    display = options(4);
    a=options(5);
    b=options(6);
    nc =options(7);
    % Run fkm to compute ni
    [~,C_fkm,~,~,~,~,~, ~]=smooth_kmeans(data,cluster_n,'Replicates',1,'SmoothMethod','p-norm','SmoothCoefficient',1);
    dist = distfcm(C_fkm, data);
    tmp = dist.^(-2/(expo-1));
    U_fkm=tmp./(ones(cluster_n,1)*sum(tmp));
    ni=sum(U_fkm.^expo.*dist.^2,2)./sum(U_fkm.^expo,2);
%     ni=0.08;
    obj_fcn = zeros(max_iter, 1);
    % Array for objective function
    % initialize centers by kmeans++
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
    dist = distfcm(init_center, data);
    tmp = dist.^(-2/(expo-1));
    U=tmp./(ones(cluster_n,1)*sum(tmp));
    U(isnan(U))=1;
    tmpt=((b./ni).*dist.^2).^(1/(nc-1));
    T = 1./(1+tmpt);
    old_center=init_center;
    iter=1;
    while 1
        [U,T,new_center,obj_fcn(iter)] = pstepfcm(data,U,T,cluster_n, expo,a,b,nc,ni);
        if display
           fprintf('Iteration count = %d, obj. fcn = %f\n', i,obj_fcn(i));
        end
    % check termination condition
        if norm(new_center-old_center,'fro')/norm(new_center,'fro')<min_impro
            break;
        end

        if iter+1>max_iter
            disp(['Failed to converge in ' num2str(options.MaxIter) ' iterations during replicate '])
            break;
        else
            old_center=new_center;
            iter=iter+1;
        end
    end
    center=new_center;
    [~,idx]=max(U,[],1);
    obj=obj_fcn(iter);
end

function [U_new,T_new,center_new,obj_fcn]=pstepfcm(data,U,T,cluster_n, expo,a,b,nc,ni)
    mf = U.^expo; % MF matrix after exponential
    %modification
    tf=T.^nc;
    tfo=(1-T).^nc;
    center_new = (a.*mf+b.*tf)*data./((ones(size(data,2), 1)*sum(a.*mf'+b.*tf'))');
    dist = distfcm(center_new, data);
    % fill the distance matrix
    obj_fcn=sum(sum((dist.^2).*(a.*mf+b.*tf)))+sum(ni.*sum(tfo,2));
    % objective function
    tmp = dist.^(-2/(expo-1));
    U_new=tmp./(ones(cluster_n,1)*sum(tmp));
    U_new(isnan(U_new))=1;
    tmpt=((b./ni).*dist.^2).^(1/(nc-1));
    T_new = 1./(1+tmpt);
end

function out = distfcm(center, data)
    out = zeros(size(center, 1), size(data, 1));
    if size(center, 2) > 1
        for k = 1:size(center, 1)
            out(k, :) = sqrt(sum(((data-ones(size(data,1),1)*center(k,:)).^2)'));
        end
    else
        for k = 1:size(center, 1)
            out(k, :) = abs(center(k)-data)';
        end
    end
end