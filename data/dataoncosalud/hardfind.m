clear;close all;clc;
addpath('../../code/functions/');

fcell={'breast\','ganglio\'};

% windowsize=16;
const=2*sqrt(6)/pi;

for k=1:length(fcell)
% for k=2
    f=fcell{k};
    lstfiles=ls([f,'*.rf']);
    tot=size(lstfiles,1);
    % for i=1:tot
    for i=1:10
        fname=strtrim(lstfiles(i,:));
        disp(['Processing ',f(1:end-1),': ',num2str(i),'/',num2str(tot)]);
        for j=1:5
            clear s std_map err_map;
            nameBmode=[f,'bmode\',fname(1:end-3),'_m',num2str(j),'.mat'];
            s=load(nameBmode);
            if isfield(s,'windowsize');s = rmfield(s,{'windowsize','std_map','err_map'});end
            if isfield(s,'maps');s = rmfield(s,{'maps'});end
            for windowsize=16:32:min(size(s.comp_env_interp))
                % info = whos('-file', nameBmode);
                % variableNames = {info.name};

                % if exist(nameBmode)==0;keyboard;end
                % figure;imagesc(feat.x,feat.z_interp,comp_env_interp);colormap gray;colorbar;
                s.maps.(['w',num2str(windowsize)]).windowsize=windowsize;

                mean_map=zeros(length(s.feat.z_interp)-windowsize+1,length(s.feat.x)-windowsize+1);
                std_map=zeros(length(s.feat.z_interp)-windowsize+1,length(s.feat.x)-windowsize+1);
                sk_map=zeros(length(s.feat.z_interp)-windowsize+1,length(s.feat.x)-windowsize+1);
                err_map=zeros(length(s.feat.z_interp)-windowsize+1,length(s.feat.x)-windowsize+1);
                [n,m]=size(s.comp_env_interp);

                for ni=1:n-windowsize+1
                    for mi=1:m-windowsize+1
                        roi=s.comp_env_interp(ni:ni+windowsize-1,mi:mi+windowsize-1);
                        meanroi=mean(roi(:));
                        stdroi=std(roi(:));
                        skroi=skewness(roi(:));
                        mean_map(ni,mi)=meanroi;
                        std_map(ni,mi)=stdroi;
                        sk_map(ni,mi)=skroi;
                        err_map(ni,mi)=(stdroi*const-s.a_0)/s.a_0;

                    end
                end
                s.maps.(['w',num2str(windowsize)]).mean_map=mean_map;
                s.maps.(['w',num2str(windowsize)]).std_map=std_map;
                s.maps.(['w',num2str(windowsize)]).sk_map=sk_map;
                s.maps.(['w',num2str(windowsize)]).err_map=err_map;
            end
            save(nameBmode,'-struct','s');
        end
        % keyboard;
    end
end