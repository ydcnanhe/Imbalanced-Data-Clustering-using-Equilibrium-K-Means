clear
clc
names=strsplit('data-A,data-B,data-C,data-D,IS,seeds,heart,wine,rice,WDBC,zoo,glass,ecoli,htru2,shill,anuran,occupancy_detection,machine_failure,pulsar,spam',',');
replicates=5000; % this is the total number of replicates for each algorithm. 5000=50*100 where 50 is the trial number and 100 is the replicate number of each trial
trial_size=100; % the replicate number of each trial
split_flag=0:trial_size:replicates; % find the lowest objective value every 100 replicates
for i=1:length(names)
    disp(['dataset: ',names{i}]);
    fprintf('\n')
    % hkm
    load(['./' names{i} '/hkm/nmi_hkm.mat']);
    load(['./' names{i} '/hkm/ari_hkm.mat']);
    load(['./' names{i} '/hkm/acc_hkm.mat']);
    load(['./' names{i} '/hkm/J_hkm.mat']);
    for j=1:length(split_flag)-1
        [~,best_id]=min(J_hkm(split_flag(j)+1:split_flag(j+1))); % here "best" means the one with the lowest objective value
        best_id=best_id+split_flag(j);
        nmi_hkm_best(j)=nmi_hkm(best_id);
        ari_hkm_best(j)=ari_hkm(best_id);
        acc_hkm_best(j)=acc_hkm(best_id);
    end
    fprintf('avg nmi for hkm_best: %.4f \n',mean(nmi_hkm_best));
    fprintf('std nmi for hkm_best: %.4f \n\n',std(nmi_hkm_best));
    fprintf('avg ari for hkm_best: %.4f \n',mean(ari_hkm_best));
    fprintf('std ari for hkm_best: %.4f \n\n',std(ari_hkm_best));
    fprintf('avg acc for hkm_best: %.4f \n',mean(acc_hkm_best));
    fprintf('std acc for hkm_best: %.4f \n\n',std(acc_hkm_best));

    % fkm
    load(['./' names{i} '/fkm/nmi_fkm.mat']);
    load(['./' names{i} '/fkm/ari_fkm.mat']);
    load(['./' names{i} '/fkm/acc_fkm.mat']);
    load(['./' names{i} '/fkm/J_fkm.mat']);
    for j=1:length(split_flag)-1
        [~,best_id]=min(J_fkm(split_flag(j)+1:split_flag(j+1)));
        best_id=best_id+split_flag(j);
        nmi_fkm_best(j)=nmi_fkm(best_id);
        ari_fkm_best(j)=ari_fkm(best_id);
        acc_fkm_best(j)=acc_fkm(best_id);
    end
    fprintf('avg nmi for fkm_best: %.4f \n',mean(nmi_fkm_best));
    fprintf('std nmi for fkm_best: %.4f \n\n',std(nmi_fkm_best));
    fprintf('avg ari for fkm_best: %.4f \n',mean(ari_fkm_best));
    fprintf('std ari for fkm_best: %.4f \n\n',std(ari_fkm_best));
    fprintf('avg acc for fkm_best: %.4f \n',mean(acc_fkm_best));
    fprintf('std acc for fkm_best: %.4f \n\n',std(acc_fkm_best));

    % mefc
    load(['./' names{i} '/mefc/nmi_mefc.mat']);
    load(['./' names{i} '/mefc/ari_mefc.mat']);
    load(['./' names{i} '/mefc/acc_mefc.mat']);
    load(['./' names{i} '/mefc/J_mefc.mat']);
    for j=1:length(split_flag)-1
        [~,best_id]=min(J_mefc(split_flag(j)+1:split_flag(j+1)));
        best_id=best_id+split_flag(j);
        nmi_mefc_best(j)=nmi_mefc(best_id);
        ari_mefc_best(j)=ari_mefc(best_id);
        acc_mefc_best(j)=acc_mefc(best_id);
    end
    fprintf('avg nmi for mefc_best: %.4f \n',mean(nmi_mefc_best));
    fprintf('std nmi for mefc_best: %.4f \n\n',std(nmi_mefc_best));
    fprintf('avg ari for mefc_best: %.4f \n',mean(ari_mefc_best));
    fprintf('std ari for mefc_best: %.4f \n\n',std(ari_mefc_best));
    fprintf('avg acc for mefc_best: %.4f \n',mean(acc_mefc_best));
    fprintf('std acc for mefc_best: %.4f \n\n',std(acc_mefc_best));

    % pfkm
    load(['./' names{i} '/pfkm/nmi_pfkm.mat']);
    load(['./' names{i} '/pfkm/ari_pfkm.mat']);
    load(['./' names{i} '/pfkm/acc_pfkm.mat']);
    load(['./' names{i} '/pfkm/J_pfkm.mat']);
    for j=1:length(split_flag)-1
        [~,best_id]=min(J_pfkm(split_flag(j)+1:split_flag(j+1)));
        best_id=best_id+split_flag(j);
        nmi_pfkm_best(j)=nmi_pfkm(best_id);
        ari_pfkm_best(j)=ari_pfkm(best_id);
        acc_pfkm_best(j)=acc_pfkm(best_id);
    end
    fprintf('avg nmi for pfkm_best: %.4f \n',mean(nmi_pfkm_best));
    fprintf('std nmi for pfkm_best: %.4f \n\n',std(nmi_pfkm_best));
    fprintf('avg ari for pfkm_best: %.4f \n',mean(ari_pfkm_best));
    fprintf('std ari for pfkm_best: %.4f \n\n',std(ari_pfkm_best));
    fprintf('avg acc for pfkm_best: %.4f \n',mean(acc_pfkm_best));
    fprintf('std acc for pfkm_best: %.4f \n\n',std(acc_pfkm_best));

    % csifkm
    load(['./' names{i} '/csifkm/nmi_csifkm.mat']);
    load(['./' names{i} '/csifkm/ari_csifkm.mat']);
    load(['./' names{i} '/csifkm/acc_csifkm.mat']);
    load(['./' names{i} '/csifkm/J_csifkm.mat']);
    for j=1:length(split_flag)-1
        [~,best_id]=min(J_csifkm(split_flag(j)+1:split_flag(j+1)));
        best_id=best_id+split_flag(j);
        nmi_csifkm_best(j)=nmi_csifkm(best_id);
        ari_csifkm_best(j)=ari_csifkm(best_id);
        acc_csifkm_best(j)=acc_csifkm(best_id);
    end
    fprintf('avg nmi for csifkm_best: %.4f \n',mean(nmi_csifkm_best));
    fprintf('std nmi for csifkm_best: %.4f \n\n',std(nmi_csifkm_best));
    fprintf('avg ari for csifkm_best: %.4f \n',mean(ari_csifkm_best));
    fprintf('std ari for csifkm_best: %.4f \n\n',std(ari_csifkm_best));
    fprintf('avg acc for csifkm_best: %.4f \n',mean(acc_csifkm_best));
    fprintf('std acc for csifkm_best: %.4f \n\n',std(acc_csifkm_best));

    % siibfkm
    load(['./' names{i} '/siibfkm/nmi_siibfkm.mat']);
    load(['./' names{i} '/siibfkm/ari_siibfkm.mat']);
    load(['./' names{i} '/siibfkm/acc_siibfkm.mat']);
    load(['./' names{i} '/siibfkm/J_siibfkm.mat']);
    for j=1:length(split_flag)-1
        [~,best_id]=min(J_siibfkm(split_flag(j)+1:split_flag(j+1)));
        best_id=best_id+split_flag(j);
        nmi_siibfkm_best(j)=nmi_siibfkm(best_id);
        ari_siibfkm_best(j)=ari_siibfkm(best_id);
        acc_siibfkm_best(j)=acc_siibfkm(best_id);
    end
    fprintf('avg nmi for siibfkm_best: %.4f \n',mean(nmi_siibfkm));
    fprintf('std nmi for siibfkm_best: %.4f \n\n',std(nmi_siibfkm));
    fprintf('avg ari for siibfkm_best: %.4f \n',mean(ari_siibfkm));
    fprintf('std ari for siibfkm_best: %.4f \n\n',std(ari_siibfkm));
    fprintf('avg acc for siibfkm_best: %.4f \n',mean(acc_siibfkm));
    fprintf('std acc for siibfkm_best: %.4f \n\n',std(acc_siibfkm));

    % fwpkm
    load(['./' names{i} '/fwpkm/nmi_fwpkm.mat']);
    load(['./' names{i} '/fwpkm/ari_fwpkm.mat']);
    load(['./' names{i} '/fwpkm/acc_fwpkm.mat']);
    load(['./' names{i} '/fwpkm/J_fwpkm.mat']);
    for j=1:length(split_flag)-1
        [~,best_id]=min(J_fwpkm(split_flag(j)+1:split_flag(j+1)));
        best_id=best_id+split_flag(j);
        nmi_fwpkm_best(j)=nmi_fwpkm(best_id);
        ari_fwpkm_best(j)=ari_fwpkm(best_id);
        acc_fwpkm_best(j)=acc_fwpkm(best_id);
    end
    fprintf('avg nmi for fwpkm_best: %.4f \n',mean(nmi_fwpkm_best));
    fprintf('std nmi for fwpkm_best: %.4f \n\n',std(nmi_fwpkm_best));
    fprintf('avg ari for fwpkm_best: %.4f \n',mean(ari_fwpkm_best));
    fprintf('std ari for fwpkm_best: %.4f \n\n',std(ari_fwpkm_best));
    fprintf('avg acc for fwpkm_best: %.4f \n',mean(acc_fwpkm_best));
    fprintf('std acc for fwpkm_best: %.4f \n\n',std(acc_fwpkm_best));

    % ekm
    load(['./' names{i} '/ekm/nmi_ekm.mat']);
    load(['./' names{i} '/ekm/ari_ekm.mat']);
    load(['./' names{i} '/ekm/acc_ekm.mat']);
    load(['./' names{i} '/ekm/J_ekm.mat']);
    for j=1:length(split_flag)-1
        [J(j),best_id]=min(J_ekm(split_flag(j)+1:split_flag(j+1)));
        best_id=best_id+split_flag(j);
        nmi_ekm_best(j)=nmi_ekm(best_id);
        ari_ekm_best(j)=ari_ekm(best_id);
        acc_ekm_best(j)=acc_ekm(best_id);
    end
    fprintf('avg nmi for ekm_best: %.4f \n',mean(nmi_ekm_best));
    fprintf('std nmi for ekm_best: %.4f \n\n',std(nmi_ekm_best));
    fprintf('avg ari for ekm_best: %.4f \n',mean(ari_ekm_best));
    fprintf('std ari for ekm_best: %.4f \n\n',std(ari_ekm_best));
    fprintf('avg acc for ekm_best: %.4f \n',mean(acc_ekm_best));
    fprintf('std acc for ekm_best: %.4f \n\n',std(acc_ekm_best));
end

