clear
clc
names=strsplit('data-A,data-B,data-C,data-D,IS,seeds,heart,wine,rice,WDBC,zoo,glass,ecoli,htru2,shill,anuran,occupancy_detection,machine_failure,pulsar,spam',',');
stat_it=zeros(8,length(names));
stat_time=zeros(8,length(names));
for i=1:length(names)
    disp(['dataset: ',names{i}]);
    fprintf('\n')
    % hkm
    load(['./' names{i} '/hkm/numit_hkm.mat']);
    load(['./' names{i} '/hkm/time_hkm.mat']);
    % removing outliers (because of bad initialization) to ensure a small std.
    [time_hkm,TFrm]=rmoutliers(time_hkm);
    numit_hkm(TFrm)=[];
    %
    fprintf('hkm: avg it : %.1f and std it : %.1f \n',mean(numit_hkm),std(numit_hkm));
    fprintf('hkm: avg time : %.3f and std time : %.4f \n\n',mean(time_hkm),std(time_hkm));
    stat_it(1,i)=mean(numit_hkm);
    stat_time(1,i)=mean(time_hkm);
    % fkm
    load(['./' names{i} '/fkm/numit_fkm.mat']);
    load(['./' names{i} '/fkm/time_fkm.mat']);
    % removing outliers to ensure a small std
    [time_fkm,TFrm]=rmoutliers(time_fkm);
    numit_fkm(TFrm)=[];
    %
    fprintf('fkm: avg it : %.1f and std it : %.1f \n',mean(numit_fkm),std(numit_fkm));
    fprintf('fkm: avg time : %.3f and std time : %.4f \n\n',mean(time_fkm),std(time_fkm));
    stat_it(2,i)=mean(numit_fkm);
    stat_time(2,i)=mean(time_fkm);
    % mefc
    load(['./' names{i} '/mefc/numit_mefc.mat']);
    load(['./' names{i} '/mefc/time_mefc.mat']);
    % removing outliers to ensure a small std
    [time_mefc,TFrm]=rmoutliers(time_mefc);
    numit_mefc(TFrm)=[];
    %
    fprintf('mefc: avg it : %.1f and std it : %.1f \n',mean(numit_mefc),std(numit_mefc));
    fprintf('mefc: avg time : %.3f and std time : %.4f \n\n',mean(time_mefc),std(time_mefc));
    stat_it(3,i)=mean(numit_mefc);
    stat_time(3,i)=mean(time_mefc);
    % pfkm
    load(['./' names{i} '/pfkm/numit_pfkm.mat']);
    load(['./' names{i} '/pfkm/time_pfkm.mat']);
    % removing outliers to ensure a small std
    [time_pfkm,TFrm]=rmoutliers(time_pfkm);
    numit_pfkm(TFrm)=[];
    %
    fprintf('pfkm: avg it : %.1f and std it : %.1f \n',mean(numit_pfkm),std(numit_pfkm));
    fprintf('pfkm: avg time : %.3f and std time : %.4f \n\n',mean(time_pfkm),std(time_pfkm));
    stat_it(4,i)=mean(numit_pfkm);
    stat_time(4,i)=mean(time_pfkm);
    % csifkm
    load(['./' names{i} '/csifkm/numit_csifkm.mat']);
    load(['./' names{i} '/csifkm/time_csifkm.mat']);
    % removing outliers to ensure a small std
    [time_csifkm,TFrm]=rmoutliers(time_csifkm);
    numit_csifkm(TFrm)=[];
    %
    fprintf('csifkm: avg it : %.1f and std it : %.1f \n',mean(numit_csifkm),std(numit_csifkm));
    fprintf('csifkm: avg time : %.3f and std time : %.4f \n\n',mean(time_csifkm),std(time_csifkm));
    stat_it(5,i)=mean(numit_csifkm);
    stat_time(5,i)=mean(time_csifkm);
    % siibfkm
    load(['./' names{i} '/siibfkm/numit_siibfkm.mat']);
    load(['./' names{i} '/siibfkm/time_siibfkm.mat']);
    % removing outliers to ensure a small std
    [time_siibfkm,TFrm]=rmoutliers(time_siibfkm);
    numit_siibfkm(TFrm)=[];
    %
    fprintf('siibfkm: avg it : %.1f and std it : %.1f \n',mean(numit_siibfkm),std(numit_siibfkm));
    fprintf('siibfkm: avg time : %.3f and std time : %.4f \n\n',mean(time_siibfkm),std(time_siibfkm));
    stat_it(6,i)=mean(numit_siibfkm);
    stat_time(6,i)=mean(time_siibfkm);
    % fwpkm
    load(['./' names{i} '/fwpkm/numit_fwpkm.mat']);
    load(['./' names{i} '/fwpkm/time_fwpkm.mat']);
    % removing outliers to ensure a small std
    [time_fwpkm,TFrm]=rmoutliers(time_fwpkm);
    numit_fwpkm(TFrm)=[];
    %
    fprintf('fwpkm: avg it : %.1f and std it : %.1f \n',mean(numit_fwpkm),std(numit_fwpkm));
    fprintf('fwpkm: avg time : %.3f and std time : %.4f \n\n',mean(time_fwpkm),std(time_fwpkm));
    stat_it(7,i)=mean(numit_fwpkm);
    stat_time(7,i)=mean(time_fwpkm);
    % ekm
    load(['./' names{i} '/ekm/numit_ekm.mat']);
    load(['./' names{i} '/ekm/time_ekm.mat']);
    % removing outliers to ensure a small std
    [time_ekm,TFrm]=rmoutliers(time_ekm);
    numit_ekm(TFrm)=[];
    %
    fprintf('ekm: avg it : %.1f and std it : %.1f \n',mean(numit_ekm),std(numit_ekm));
    fprintf('ekm: avg time : %.3f and std time : %.4f \n\n',mean(time_ekm),std(time_ekm));
    fprintf('\n')
    stat_it(8,i)=mean(numit_ekm);
    stat_time(8,i)=mean(time_ekm);
    % order
    [~,I]=sort(stat_time(:,i));
    rank=zeros(8,1);
    for j=1:8
        rank(j)=find(I==j);
    end
    fprintf('time ranking is: %d \n\n',rank);
end
% AVK
AVK_it=mean(stat_it,2);
AVK_time=mean(stat_time,2);
[~,I]=sort(AVK_time);
for j=1:8
    rank(j)=find(I==j);
end
fprintf('AVK it is: %.1f \n',AVK_it);
fprintf('AVK time is: %.3f \n',AVK_time);
fprintf('AVK ranking is: %d \n\n',rank);
% MDK
MED_it=median(stat_it,2);
MED_time=median(stat_time,2);
[~,I]=sort(MED_time);
for j=1:8
    rank(j)=find(I==j);
end
fprintf('MDK it is: %.1f \n',MED_it);
fprintf('MDK time is: %.3f \n',MED_time);
fprintf('MDK is: %d \n\n',rank);

