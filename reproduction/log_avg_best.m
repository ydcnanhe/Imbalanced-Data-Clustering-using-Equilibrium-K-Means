clear
clc
% names=strsplit('data-A,data-B,data-C,data-D,IS,seeds,heart,wine,rice,WDBC,zoo,glass,ecoli,htru2,shill,anuran,occupancy_detection,machine_failure,pulsar,spam',',');
names=strsplit('ecoli');
replicates=5000; % this is the total number of replicates for each algorithm. 5000=50*100 where 50 is the trial number and 100 is the replicate number of each trial
trial_size=100; % the replicate number of each trial
split_flag=0:trial_size:replicates; % find the lowest objective value every 100 replicates
for i=1:length(names)
    disp(['dataset: ',names{i}]);
    fprintf('\n')
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

