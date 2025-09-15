clear
clc
% names=strsplit('data-A,data-B,data-C,data-D,IS,seeds,heart,wine,rice,WDBC,zoo,glass,ecoli,htru2,shill,anuran,occupancy_detection,machine_failure,pulsar,spam',',');
names=strsplit('ecoli');
for i=1:length(names)
    disp(['dataset: ',names{i}]);
    fprintf('\n')
    % ekm
    load(['./' names{i} '/ekm/numit_ekm.mat']);
    load(['./' names{i} '/ekm/time_ekm.mat']);
    % removing outliers to ensure a small std
    [time_ekm,TFrm]=rmoutliers(time_ekm);
    numit_ekm(TFrm)=[];
    %
    fprintf('ekm: avg it : %.1f and std it : %.1f \n',mean(numit_ekm),std(numit_ekm));
    fprintf('ekm: avg time : %.3f and std time : %.4f \n\n',mean(time_ekm),std(time_ekm));
end

