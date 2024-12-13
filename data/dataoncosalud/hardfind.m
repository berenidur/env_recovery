clear;close all;clc;
addpath('../../code/functions/');

fcell={'breast\','ganglio\'};

windowsize=16;
const=2*sqrt(6)/pi;

for k=1:length(fcell)
    f=fcell{k};
    lstfiles=ls([f,'*.rf']);
    tot=size(lstfiles,1);
    for i=1:tot
        fname=strtrim(lstfiles(i,:));
        for j=1:5
            clear s;
            nameBmode=[f,'bmode\',fname(1:end-3),'_m',num2str(j),'.mat'];
            s=load(nameBmode);
            % info = whos('-file', nameBmode);
            % variableNames = {info.name};

            % if exist(nameBmode)==0;keyboard;end
            % figure;imagesc(feat.x,feat.z_interp,comp_env_interp);colormap gray;colorbar;
            s.windowsize=windowsize;
            
            s.std_map=zeros(length(s.feat.z_interp)-windowsize+1,length(s.feat.x)-windowsize+1);
            s.err_map=zeros(length(s.feat.z_interp)-windowsize+1,length(s.feat.x)-windowsize+1);
            [n,m]=size(s.comp_env_interp);

            for ni=1:n-windowsize+1
                for mi=1:m-windowsize+1
                    roi=s.comp_env_interp(ni:ni+windowsize-1,mi:mi+windowsize-1);
                    stdroi=std(roi(:));
                    s.std_map(ni,mi)=stdroi;
                    s.err_map(ni,mi)=(stdroi*const-s.a_0)/s.a_0;

                end
            end
            save(nameBmode,'-struct','s');
        end
    end
end