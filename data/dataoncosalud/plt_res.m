close all;clear;clc;

clear;close all;clc;
addpath('../../code/functions/');

fcell={'breast\','ganglio\'};

windowsize=16;
const=pi/2/sqrt(6);

for k=1:length(fcell)
    f=fcell{k};
    lstfiles=ls([f,'*.rf']);
    tot=size(lstfiles,1);
    for i=1:tot
        fname=strtrim(lstfiles(i,:));
        for j=1:5
            % clear data;
            nameBmode=[f,'bmode\',fname(1:end-3),'_m',num2str(j),'.mat'];
            data=load(nameBmode);
            % info = whos('-file', nameBmode);
            % variableNames = {info.name};

            % if exist(nameBmode)==0;keyboard;end
            figure;imagesc(data.feat.x,data.feat.z_interp,data.comp_env_interp);colormap gray;colorbar;
            
        end
    end
end