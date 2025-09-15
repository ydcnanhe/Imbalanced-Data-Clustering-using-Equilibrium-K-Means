function eval_clustering(X,num_class,true_idx,replicate,alpha,output,is_plot,is_saveplot)

    if nargin <5
        error("Not enough input arguments");
    end

    if nargin<6
        output='./';
        is_plot=0;
        is_saveplot=0;
    end
    
    if nargin <7
        is_plot=0;
        is_saveplot=0;
    end

    if nargin <8
        is_saveplot=0;
    end

    num_feature=size(X,2);
    % calculate CV0
    Ns=zeros(num_class,1);
    label=unique(true_idx);
    for k=1:num_class
        Ns(k)=sum(true_idx==label(k));
    end
    cv0=std(Ns)/mean(Ns);
    % normalization
    for p=1:num_feature
        X(:,p)=X(:,p)-mean(X(:,p));
        X(:,p)=X(:,p)/std(X(:,p));
    end
    
    %% clustering by equilibrium k-means
    fprintf('EKM... \n');
    time_ekm=zeros(replicate,1);
    cv1_ekm=zeros(replicate,1);
    nmi_ekm=zeros(replicate,1);
    ari_ekm=zeros(replicate,1);
    acc_ekm=zeros(replicate,1);
    dcv_ekm=zeros(replicate,1);
    numit_ekm=zeros(replicate,1);
    J_ekm=zeros(replicate,1);
    idx_total=zeros(size(X,1),replicate);
    C_total=zeros(num_class,num_feature,replicate);
    parfor r=1:replicate
        tic
        [idx,C,numit,~,~,~,~, J]=ekm(X,num_class,'Replicates',1,'alpha',alpha);
        time_ekm(r)=toc;
        % calculate CV1
        Ns=zeros(num_class,1);
        for k=1:num_class
            Ns(k)=sum(idx==k);
        end
        % number of iteration
        numit_ekm(r)=numit;
        cv1_ekm(r)=std(Ns)/mean(Ns);
        % calculate nmi
        nmi_ekm(r)=nmi(true_idx,idx);
        % calculate ari
        ari_ekm(r)=rand_index(true_idx,idx,'adjusted');
        % calculate acc
        acc_ekm(r)=cluster_acc(true_idx,idx);
        % calculate dcv
        dcv_ekm(r)=cv0-cv1_ekm(r);
        % loss
        J_ekm(r)=J;
        %
        idx_total(:,r) = idx;
        C_total(:,:,r) = C;
    end
    [~,best_id]=min(J_ekm);
    if is_plot==1
        figure;
        gscatter(X(:,1), X(:,2), idx_total(:,best_id));
        hold on
        plot(C_total(:,1,best_id),C_total(:,2,best_id),'k+','MarkerSize',15,'LineWidth',3) 
        title('EKM','FontSize',15)
        xlabel('Normalized feature 1','FontSize',15)
        ylabel('Normalized feature 2','FontSize',15)
        legend off
     if is_saveplot==1
        saveas(gcf,[output '/EKM.jpg']);
     end
        hold off;
    end

    % save record
    mkdir([output '/ekm/']);
    save([output  '/ekm/time_ekm.mat'],"time_ekm");
    save([output  '/ekm/numit_ekm.mat'],"numit_ekm");
    save([output  '/ekm/cv1_ekm.mat'],"cv1_ekm");
    save([output  '/ekm/nmi_ekm.mat'],"nmi_ekm");
    save([output  '/ekm/ari_ekm.mat'],"ari_ekm");
    save([output  '/ekm/acc_ekm.mat'],"acc_ekm");
    save([output  '/ekm/dcv_ekm.mat'],"dcv_ekm");
    save([output  '/ekm/J_ekm.mat'],"J_ekm");

    fprintf('The replicate with the lowest objective and avg +- std of nmi for ekm: %.4f and %.4f +- %.4f \n',nmi_ekm(best_id),mean(nmi_ekm),std(nmi_ekm));
    fprintf('The replicate with the lowest objective and avg +- std of ari for ekm: %.4f and %.4f +- %.4f \n',ari_ekm(best_id),mean(ari_ekm),std(ari_ekm));
    fprintf('The replicate with the lowest objective and avg +- std of acc for ekm: %.4f and %.4f +- %.4f \n',acc_ekm(best_id),mean(acc_ekm),std(acc_ekm));
    fprintf('avg it for ekm: %.4f \n',mean(numit_ekm));
    fprintf('avg time for ekm: %.4f \n',mean(time_ekm));
    fprintf('\n')