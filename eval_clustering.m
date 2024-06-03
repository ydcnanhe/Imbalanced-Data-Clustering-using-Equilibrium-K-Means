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
    
    fprintf('HKM... \n');
    %% clustering by k-means
    time_hkm=zeros(replicate,1);
    cv1_hkm=zeros(replicate,1);
    nmi_hkm=zeros(replicate,1);
    ari_hkm=zeros(replicate,1);
    acc_hkm=zeros(replicate,1);
    dcv_hkm=zeros(replicate,1);
    numit_hkm=zeros(replicate,1);
    J_hkm=zeros(replicate,1);
    idx_total=zeros(size(X,1),replicate);
    C_total=zeros(num_class,num_feature,replicate);
    parfor r=1:replicate
        tic
        [idx,C,numit,~,~,~,~, J]=smooth_kmeans(X,num_class,'Replicates',1,'SmoothMethod','wcss','SmoothCoefficient',1);
        time_hkm(r)=toc;
        % number of iteration
        numit_hkm(r)=numit;
        % calculate CV1
        Ns=zeros(num_class,1);
        for k=1:num_class
            Ns(k)=sum(idx==k);
        end
        cv1_hkm(r)=std(Ns)/mean(Ns);
        % calculate nmi
        nmi_hkm(r)=nmi(true_idx,idx);
        % calculate ari
        ari_hkm(r)=rand_index(true_idx,idx,'adjusted');
        % calculate acc
        acc_hkm(r)=cluster_acc(true_idx,idx);
        % calculate dcv
        dcv_hkm(r)=cv0-cv1_hkm(r);
        % loss
        J_hkm(r)=J;
        %
        idx_total(:,r) = idx;
        C_total(:,:,r) = C;
    end

    [~,best_id]=min(J_hkm);
    if is_plot==1
        figure;
        gscatter(X(:,1), X(:,2), idx_total(:,best_id));
        hold on
        plot(C_total(:,1,best_id),C_total(:,2,best_id),'k+','MarkerSize',15,'LineWidth',3) 
        title('HKM','FontSize',15)
        xlabel('Normalized feature 1','FontSize',15)
        ylabel('Normalized feature 2','FontSize',15)
        legend off
    if is_saveplot==1
        saveas(gcf,[output '/HKM.jpg']);
    end
        hold off;
    end
        
    % save record
    mkdir([output '/hkm/']);
    save([output '/hkm/time_hkm.mat'],"time_hkm");
    save([output '/hkm/numit_hkm.mat'],"numit_hkm");
    save([output '/hkm/cv1_hkm.mat'],"cv1_hkm");
    save([output '/hkm/nmi_hkm.mat'],"nmi_hkm");
    save([output '/hkm/ari_hkm.mat'],"ari_hkm");
    save([output '/hkm/acc_hkm.mat'],"acc_hkm");
    save([output '/hkm/dcv_hkm.mat'],"dcv_hkm");
    save([output '/hkm/J_hkm.mat'],"J_hkm");

    fprintf('best and avg +- std of nmi for hkm: %.4f and %.4f +- %.4f \n',nmi_hkm(best_id),mean(nmi_hkm),std(nmi_hkm));
    fprintf('best and avg +- std of ari for hkm: %.4f and %.4f +- %.4f \n',ari_hkm(best_id),mean(ari_hkm),std(ari_hkm));
    fprintf('best and avg +- std of acc for hkm: %.4f and %.4f +- %.4f \n',acc_hkm(best_id),mean(acc_hkm),std(acc_hkm));
    fprintf('avg it for hkm: %.4f \n',mean(numit_hkm));
    fprintf('avg time for hkm: %.4f \n',mean(time_hkm));
    fprintf('\n')
    
    fprintf('FKM... \n');
    %% clustering by fuzzy c-means
    time_fkm=zeros(replicate,1);
    cv1_fkm=zeros(replicate,1);
    nmi_fkm=zeros(replicate,1);
    ari_fkm=zeros(replicate,1);
    acc_fkm=zeros(replicate,1);
    dcv_fkm=zeros(replicate,1);
    numit_fkm=zeros(replicate,1);
    J_fkm=zeros(replicate,1);
    idx_total=zeros(size(X,1),replicate);
    C_total=zeros(num_class,num_feature,replicate);
    parfor r=1:replicate
        tic
        [idx,C,numit,~,~,~,~, J]=smooth_kmeans(X,num_class,'Replicates',1,'SmoothMethod','p-norm','SmoothCoefficient',1);
        time_fkm(r)=toc;
        % number of iteration
        numit_fkm(r)=numit;
        % calculate CV1
        Ns=zeros(num_class,1);
        for k=1:num_class
            Ns(k)=sum(idx==k);
        end
        cv1_fkm(r)=std(Ns)/mean(Ns);
        % calculate nmi
        nmi_fkm(r)=nmi(true_idx,idx);
        % calculate ari
        ari_fkm(r)=rand_index(true_idx,idx,'adjusted');
        % calculate acc
        acc_fkm(r)=cluster_acc(true_idx,idx);
        % calculate dcv
        dcv_fkm(r)=cv0-cv1_fkm(r);
        % loss
        J_fkm(r)=J;
        %
        idx_total(:,r) = idx;
        C_total(:,:,r) = C;
    end
        
    [~,best_id]=min(J_fkm);
    if is_plot==1
        figure;
        gscatter(X(:,1), X(:,2), idx_total(:,best_id));
        hold on
        plot(C_total(:,1,best_id),C_total(:,2,best_id),'k+','MarkerSize',15,'LineWidth',3) 
        title('FKM','FontSize',15)
        xlabel('Normalized feature 1','FontSize',15)
        ylabel('Normalized feature 2','FontSize',15)
        legend off
    end
    if is_saveplot==1
        saveas(gcf,[output '/FKM.jpg']);
    end
    hold off;
    % save record
    mkdir([output '/fkm/']);
    save([output  '/fkm/time_fkm.mat'],"time_fkm");
    save([output  '/fkm/numit_fkm.mat'],"numit_fkm");
    save([output  '/fkm/cv1_fkm.mat'],"cv1_fkm");
    save([output  '/fkm/nmi_fkm.mat'],"nmi_fkm");
    save([output  '/fkm/ari_fkm.mat'],"ari_fkm");
    save([output  '/fkm/acc_fkm.mat'],"acc_fkm");
    save([output  '/fkm/dcv_fkm.mat'],"dcv_fkm");
    save([output  '/fkm/J_fkm.mat'],"J_fkm");

    fprintf('best and avg +- std of nmi for fkm: %.4f and %.4f +- %.4f \n',nmi_fkm(best_id),mean(nmi_fkm),std(nmi_fkm));
    fprintf('best and avg +- std of ari for fkm: %.4f and %.4f +- %.4f \n',ari_fkm(best_id),mean(ari_fkm),std(ari_fkm));
    fprintf('best and avg +- std of acc for fkm: %.4f and %.4f +- %.4f \n',acc_fkm(best_id),mean(acc_fkm),std(acc_fkm));
    fprintf('avg it for fkm: %.4f \n',mean(numit_fkm));
    fprintf('avg time for fkm: %.4f \n',mean(time_fkm));
    fprintf('\n')
    
    %% clustering by maximum-entropy fuzzy clustering
    fprintf('MEFC... \n');
    time_mefc=zeros(replicate,1);
    cv1_mefc=zeros(replicate,1);
    nmi_mefc=zeros(replicate,1);
    ari_mefc=zeros(replicate,1);
    acc_mefc=zeros(replicate,1);
    dcv_mefc=zeros(replicate,1);
    numit_mefc=zeros(replicate,1);
    J_mefc=zeros(replicate,1);
    idx_total=zeros(size(X,1),replicate);
    C_total=zeros(num_class,num_feature,replicate);
    parfor r=1:replicate
        tic
        [idx,C,numit,~,~,~,~, J]=smooth_kmeans(X,num_class,'Replicates',1,'SmoothMethod','logsumexp','SmoothCoefficient',1);
        time_mefc(r)=toc;
        % number of iteration
        numit_mefc(r)=numit;
        % calculate CV1
        Ns=zeros(num_class,1);
        for k=1:num_class
            Ns(k)=sum(idx==k);
        end
        cv1_mefc(r)=std(Ns)/mean(Ns);
        % calculate nmi
        nmi_mefc(r)=nmi(true_idx,idx);
        % calculate ari
        ari_mefc(r)=rand_index(true_idx,idx,'adjusted');
        % calculate acc
        acc_mefc(r)=cluster_acc(true_idx,idx);
        % calculate dcv
        dcv_mefc(r)=cv0-cv1_mefc(r);
        % loss
        J_mefc(r)=J;
        %
        idx_total(:,r) = idx;
        C_total(:,:,r) = C;
    end

    [~,best_id]=min(J_mefc);
    if is_plot==1
        figure;
        gscatter(X(:,1), X(:,2), idx_total(:,best_id));
        hold on
        plot(C_total(:,1,best_id),C_total(:,2,best_id),'k+','MarkerSize',15,'LineWidth',3) 
        title('MEFC','FontSize',15)
        xlabel('Normalized feature 1','FontSize',15)
        ylabel('Normalized feature 2','FontSize',15)
        legend off
    end
    if is_saveplot==1
        saveas(gcf,[output '/MEFC.jpg']);
    end
    hold off;
    % save record
    mkdir([output '/mefc/']);
    save([output  '/mefc/time_mefc.mat'],"time_mefc");
    save([output  '/mefc/numit_mefc.mat'],"numit_mefc");
    save([output  '/mefc/cv1_mefc.mat'],"cv1_mefc");
    save([output  '/mefc/nmi_mefc.mat'],"nmi_mefc");
    save([output  '/mefc/ari_mefc.mat'],"ari_mefc");
    save([output  '/mefc/acc_mefc.mat'],"acc_mefc");
    save([output  '/mefc/dcv_mefc.mat'],"dcv_mefc");
    save([output  '/mefc/J_mefc.mat'],"J_mefc");

    fprintf('best and avg +- std of nmi for mefc: %.4f and %.4f +- %.4f \n',nmi_mefc(best_id),mean(nmi_mefc),std(nmi_mefc));
    fprintf('best and avg +- std of ari for mefc: %.4f and %.4f +- %.4f \n',ari_mefc(best_id),mean(ari_mefc),std(ari_mefc));
    fprintf('best and avg +- std of acc for mefc: %.4f and %.4f +- %.4f \n',acc_mefc(best_id),mean(acc_mefc),std(acc_mefc));
    fprintf('avg it for mefc: %.4f \n',mean(numit_mefc));
    fprintf('avg time for mefc: %.4f \n',mean(time_mefc));
    fprintf('\n')
    
    %% clustering by smooth k-means (Boltzmann)
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
        [idx,C,numit,~,~,~,~, J]=smooth_kmeans(X,num_class,'Replicates',1,'SmoothMethod','Boltzmann','SmoothCoefficient',alpha);
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

    fprintf('best and avg +- std of nmi for ekm: %.4f and %.4f +- %.4f \n',nmi_ekm(best_id),mean(nmi_ekm),std(nmi_ekm));
    fprintf('best and avg +- std of ari for ekm: %.4f and %.4f +- %.4f \n',ari_ekm(best_id),mean(ari_ekm),std(ari_ekm));
    fprintf('best and avg +- std of acc for ekm: %.4f and %.4f +- %.4f \n',acc_ekm(best_id),mean(acc_ekm),std(acc_ekm));
    fprintf('avg it for ekm: %.4f \n',mean(numit_ekm));
    fprintf('avg time for ekm: %.4f \n',mean(time_ekm));
    fprintf('\n')
    
    %% clustering by Size-insensitive Fuzzy K-Means
    fprintf('CSIFKM... \n');
    time_csifkm=zeros(replicate,1);
    cv1_csifkm=zeros(replicate,1);
    nmi_csifkm=zeros(replicate,1);
    ari_csifkm=zeros(replicate,1);
    acc_csifkm=zeros(replicate,1);
    dcv_csifkm=zeros(replicate,1);
    numit_csifkm=zeros(replicate,1);
    J_csifkm=zeros(replicate,1);
    idx_total=zeros(size(X,1),replicate);
    C_total=zeros(num_class,num_feature,replicate);
    parfor r=1:replicate
        tic
        [idx, C, numit,~,~,J] =  csiFCM(X,num_class);
        time_csifkm(r)=toc;
        % number of iteration
        numit_csifkm(r)=numit;
        % calculate CV1
        Ns=zeros(num_class,1);
        for k=1:num_class
            Ns(k)=sum(idx==k);
        end
        cv1_csifkm(r)=std(Ns)/mean(Ns);
        % calculate nmi
        nmi_csifkm(r)=nmi(true_idx,idx);
        % calculate ari
        ari_csifkm(r)=rand_index(true_idx,idx,'adjusted');
        % calculate acc
        acc_csifkm(r)=cluster_acc(true_idx,idx);
        % calculate dcv
        dcv_csifkm(r)=cv0-cv1_csifkm(r);
        % loss
        J_csifkm(r)=J;
        %
        idx_total(:,r) = idx;
        C_total(:,:,r) = C;
    end

    [~,best_id]=min(J_csifkm);
    if is_plot==1
        figure;
        gscatter(X(:,1), X(:,2), idx_total(:,best_id));
        hold on
        plot(C_total(:,1,best_id),C_total(:,2,best_id),'k+','MarkerSize',15,'LineWidth',3) 
        title('csiFKM','FontSize',15)
        xlabel('Normalized feature 1','FontSize',15)
        ylabel('Normalized feature 2','FontSize',15)
        legend off
    end
    if is_saveplot==1
        saveas(gcf,[output '/csiFKM.jpg']);
    end
    hold off;
    % save record
    mkdir([output '/csifkm/']);
    save([output  '/csifkm/time_csifkm.mat'],"time_csifkm");
    save([output  '/csifkm/numit_csifkm.mat'],"numit_csifkm");
    save([output  '/csifkm/cv1_csifkm.mat'],"cv1_csifkm");
    save([output  '/csifkm/nmi_csifkm.mat'],"nmi_csifkm");
    save([output  '/csifkm/ari_csifkm.mat'],"ari_csifkm");
    save([output  '/csifkm/acc_csifkm.mat'],"acc_csifkm");
    save([output  '/csifkm/dcv_csifkm.mat'],"dcv_csifkm");
    save([output  '/csifkm/J_csifkm.mat'],"J_csifkm");

    fprintf('best and avg +- std of nmi for csifkm: %.4f and %.4f +- %.4f \n',nmi_csifkm(best_id),mean(nmi_csifkm),std(nmi_csifkm));
    fprintf('best and avg +- std of ari for csifkm: %.4f and %.4f +- %.4f \n',ari_csifkm(best_id),mean(ari_csifkm),std(ari_csifkm));
    fprintf('best and avg +- std of acc for csifkm: %.4f and %.4f +- %.4f \n',acc_csifkm(best_id),mean(acc_csifkm),std(acc_csifkm));
    fprintf('avg it for csifkm: %.4f \n',mean(numit_csifkm));
    fprintf('avg time for csifkm: %.4f \n',mean(time_csifkm));
    fprintf('\n')
    
    % Clustering by siibFCM
    fprintf('SIIBFKM... \n');
    time_siibfkm=zeros(replicate,1);
    cv1_siibfkm=zeros(replicate,1);
    nmi_siibfkm=zeros(replicate,1);
    ari_siibfkm=zeros(replicate,1);
    acc_siibfkm=zeros(replicate,1);
    dcv_siibfkm=zeros(replicate,1);
    numit_siibfkm=zeros(replicate,1);
    J_siibfkm=zeros(replicate,1);
    idx_total=zeros(size(X,1),replicate);
    C_total=zeros(num_class,num_feature,replicate);
    parfor r=1:replicate
        tic
        [idx,C,numit,~,~,J] =  siibFCM(X,num_class);
        time_siibfkm(r)=toc;
        number of iteration
        numit_siibfkm(r)=numit;
        calculate CV1
        Ns=zeros(num_class,1);
        for k=1:num_class
            Ns(k)=sum(idx==k);
        end
        cv1_siibfkm(r)=std(Ns)/mean(Ns);
        calculate nmi
        nmi_siibfkm(r)=nmi(true_idx,idx);
        calculate ari
        ari_siibfkm(r)=rand_index(true_idx,idx,'adjusted');
        calculate acc
        acc_siibfkm(r)=cluster_acc(true_idx,idx);
        calculate dcv
        dcv_siibfkm(r)=cv0-cv1_siibfkm(r);
        loss
        J_siibfkm(r)=J;
        
        idx_total(:,r) = idx;
        C_total(:,:,r) = C;
    end

    [~,best_id]=min(J_siibfkm);
    if options.plot==1
        figure;
        gscatter(X(:,1), X(:,2), idx_total(:,best_id));
        hold on
        plot(C_total(:,1,best_id),C_total(:,2,best_id),'k+','MarkerSize',15,'LineWidth',3) 
        title('siibFKM','FontSize',15)
        xlabel('Normalized feature 1','FontSize',15)
        ylabel('Normalized feature 2','FontSize',15)
        legend off
    end
    if is_saveplot==1
        saveas(gcf,[output '/siibFKM.jpg']);
    end
    hold off;
    
    % save record
    mkdir([output '/siibfkm/']);
    save([output  '/siibfkm/time_siibfkm.mat'],"time_siibfkm");
    save([output  '/siibfkm/numit_siibfkm.mat'],"numit_siibfkm");
    save([output  '/siibfkm/cv1_siibfkm.mat'],"cv1_siibfkm");
    save([output  '/siibfkm/nmi_siibfkm.mat'],"nmi_siibfkm");
    save([output  '/siibfkm/ari_siibfkm.mat'],"ari_siibfkm");
    save([output  '/siibfkm/acc_siibfkm.mat'],"acc_siibfkm");
    save([output  '/siibfkm/dcv_siibfkm.mat'],"dcv_siibfkm");
    save([output  '/siibfkm/J_siibfkm.mat'],"J_siibfkm");

    fprintf('best and avg +- std of nmi for siibfkm: %.4f and %.4f +- %.4f \n',nmi_siibfkm(best_id),mean(nmi_siibfkm),std(nmi_siibfkm));
    fprintf('best and avg +- std of ari for siibfkm: %.4f and %.4f +- %.4f \n',ari_siibfkm(best_id),mean(ari_siibfkm),std(ari_siibfkm));
    fprintf('best and avg +- std of acc for siibfkm: %.4f and %.4f +- %.4f \n',acc_siibfkm(best_id),mean(acc_siibfkm),std(acc_siibfkm));
    fprintf('avg it for siibfkm: %.4f \n',mean(numit_siibfkm));
    fprintf('avg time for siibfkm: %.4f \n',mean(time_siibfkm));
    fprintf('\n')

     %% Clustering by PFKM
    fprintf('PFKM... \n');
    time_pfkm=zeros(replicate,1);
    cv1_pfkm=zeros(replicate,1);
    nmi_pfkm=zeros(replicate,1);
    ari_pfkm=zeros(replicate,1);
    acc_pfkm=zeros(replicate,1);
    dcv_pfkm=zeros(replicate,1);
    numit_pfkm=zeros(replicate,1);
    J_pfkm=zeros(replicate,1);
    idx_total=zeros(size(X,1),replicate);
    C_total=zeros(num_class,num_feature,replicate);
    parfor r=1:replicate
        tic
        [idx,C,~,~,J,numit]=pfcm(X,num_class);
        time_pfkm(r)=toc;
        % number of iteration
        numit_pfkm(r)=numit;
        % calculate CV1
        Ns=zeros(num_class,1);
        for k=1:num_class
            Ns(k)=sum(idx==k);
        end
        cv1_pfkm(r)=std(Ns)/mean(Ns);
        % calculate nmi
        nmi_pfkm(r)=nmi(true_idx,idx);
        % calculate ari
        ari_pfkm(r)=rand_index(true_idx,idx,'adjusted');
        % calculate acc
        acc_pfkm(r)=cluster_acc(true_idx,idx);
        % calculate dcv
        dcv_pfkm(r)=cv0-cv1_pfkm(r);
        % loss
        J_pfkm(r)=J;
        %
        idx_total(:,r) = idx;
        C_total(:,:,r) = C;
    end

    [~,best_id]=min(J_pfkm);
    if options.plot==1
        figure;
        gscatter(X(:,1), X(:,2), idx_total(:,best_id));
        hold on
        plot(C_total(:,1,best_id),C_total(:,2,best_id),'k+','MarkerSize',15,'LineWidth',3) 
        title('PFKM','FontSize',15)
        xlabel('Normalized feature 1','FontSize',15)
        ylabel('Normalized feature 2','FontSize',15)
        legend off
    end
    if is_saveplot==1
        saveas(gcf,[output '/PFKM.jpg']);
    end
    hold off;
    % save record
    mkdir([output '/pfkm/']);
    save([output  '/pfkm/time_pfkm.mat'],"time_pfkm");
    save([output  '/pfkm/numit_pfkm.mat'],"numit_pfkm");
    save([output  '/pfkm/cv1_pfkm.mat'],"cv1_pfkm");
    save([output  '/pfkm/nmi_pfkm.mat'],"nmi_pfkm");
    save([output  '/pfkm/ari_pfkm.mat'],"ari_pfkm");
    save([output  '/pfkm/acc_pfkm.mat'],"acc_pfkm");
    save([output  '/pfkm/dcv_pfkm.mat'],"dcv_pfkm");
    save([output  '/pfkm/J_pfkm.mat'],"J_pfkm");

    fprintf('best and avg +- std of nmi for pfkm: %.4f and %.4f +- %.4f \n',nmi_pfkm(best_id),mean(nmi_pfkm),std(nmi_pfkm));
    fprintf('best and avg +- std of ari for pfkm: %.4f and %.4f +- %.4f \n',ari_pfkm(best_id),mean(ari_pfkm),std(ari_pfkm));
    fprintf('best and avg +- std of acc for pfkm: %.4f and %.4f +- %.4f \n',acc_pfkm(best_id),mean(acc_pfkm),std(acc_pfkm));
    fprintf('avg it for pfkm: %.4f \n',mean(numit_pfkm));
    fprintf('avg time for pfkm: %.4f \n',mean(time_pfkm));
    fprintf('\n')

     %% Clustering by FWPKM
    fprintf('FWPKM... \n');
    time_fwpkm=zeros(replicate,1);
    cv1_fwpkm=zeros(replicate,1);
    nmi_fwpkm=zeros(replicate,1);
    ari_fwpkm=zeros(replicate,1);
    acc_fwpkm=zeros(replicate,1);
    dcv_fwpkm=zeros(replicate,1);
    numit_fwpkm=zeros(replicate,1);
    J_fwpkm=zeros(replicate,1);
    idx_total=zeros(size(X,1),replicate);
    C_total=zeros(num_class,num_feature,replicate);
    parfor r=1:replicate
        tic
        [idx,C,~,J,numit]=fwpkm(X,num_class,2);
        time_fwpkm(r)=toc;
        % number of iteration
        numit_fwpkm(r)=numit;
        % calculate CV1
        Ns=zeros(num_class,1);
        for k=1:num_class
            Ns(k)=sum(idx==k);
        end
        cv1_fwpkm(r)=std(Ns)/mean(Ns);
        % calculate nmi
        nmi_fwpkm(r)=nmi(true_idx,idx);
        % calculate ari
        ari_fwpkm(r)=rand_index(true_idx,idx,'adjusted');
        % calculate acc
        acc_fwpkm(r)=cluster_acc(true_idx,idx);
        % calculate dcv
        dcv_fwpkm(r)=cv0-cv1_fwpkm(r);
        % loss
        J_fwpkm(r)=J;
        %
        idx_total(:,r) = idx;
        C_total(:,:,r) = C;
    end

    [~,best_id]=min(J_fwpkm);
    if options.plot==1
        figure;
        gscatter(X(:,1), X(:,2), idx_total(:,best_id));
        hold on
        plot(C_total(:,1,best_id),C_total(:,2,best_id),'k+','MarkerSize',15,'LineWidth',3) 
        title('FWPKM','FontSize',15)
        xlabel('Normalized feature 1','FontSize',15)
        ylabel('Normalized feature 2','FontSize',15)
        legend off
    end
    if is_saveplot==1
        saveas(gcf,[output '/FWPKM.jpg']);
    end
    hold off;
    % save record
    mkdir([output '/fwpkm/']);
    save([output  '/fwpkm/time_fwpkm.mat'],"time_fwpkm");
    save([output  '/fwpkm/numit_fwpkm.mat'],"numit_fwpkm");
    save([output  '/fwpkm/cv1_fwpkm.mat'],"cv1_fwpkm");
    save([output  '/fwpkm/nmi_fwpkm.mat'],"nmi_fwpkm");
    save([output  '/fwpkm/ari_fwpkm.mat'],"ari_fwpkm");
    save([output  '/fwpkm/acc_fwpkm.mat'],"acc_fwpkm");
    save([output  '/fwpkm/dcv_fwpkm.mat'],"dcv_fwpkm");
    save([output  '/fwpkm/J_fwpkm.mat'],"J_fwpkm");

    fprintf('best and avg +- std of nmi for fwpkm: %.4f and %.4f +- %.4f \n',nmi_fwpkm(best_id),mean(nmi_fwpkm),std(nmi_fwpkm));
    fprintf('best and avg +- std of ari for fwpkm: %.4f and %.4f +- %.4f \n',ari_fwpkm(best_id),mean(ari_fwpkm),std(ari_fwpkm));
    fprintf('best and avg +- std of acc for fwpkm: %.4f and %.4f +- %.4f \n',acc_fwpkm(best_id),mean(acc_fwpkm),std(acc_fwpkm));
    fprintf('avg it for fwpkm: %.4f \n',mean(numit_fwpkm));
    fprintf('avg time for fwpkm: %.4f \n',mean(time_fwpkm));
    fprintf('\n')
