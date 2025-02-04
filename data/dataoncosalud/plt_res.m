close all;clear;clc;

clear;close all;clc;
addpath('../../code/functions/');
g=false;

fcell={'breast\','ganglio\'};

windowsize=16;
const=2*sqrt(6)/pi;

% plotear=1;    % error
% plotear=2;    % mean
plotear=3;    % R
% plotear=4;    % S

for k=1:length(fcell)
% for k=2
    if k==1;totsel=[1,9];else;totsel=[1,4];end
    f=fcell{k};
    lstfiles=ls([f,'*.rf']);
    tot=size(lstfiles,1);
    % for i=1:tot
    for i=totsel
        fname=strtrim(lstfiles(i,:));
        figure;

        for j=1:5
            clear data;
            nameBmode=[f,'bmode\',fname(1:end-3),'_m',num2str(j),'.mat'];
            data=load(nameBmode);
            % info = whos('-file', nameBmode);
            % variableNames = {info.name};

            % if exist(nameBmode)==0;keyboard;end
            % figure;imagesc(data.feat.x,data.feat.z_interp,data.comp_env_interp);
            % colormap gray;colorbar;clim([0 255]);

            ax=subplot(2,3,j+1);
            windowsize=16;

            mean_map = data.maps.(['w',num2str(windowsize)]).mean_map;
            std_map = data.maps.(['w',num2str(windowsize)]).std_map;
            sk_map = data.maps.(['w',num2str(windowsize)]).sk_map;
            err_map = data.maps.(['w',num2str(windowsize)]).err_map;

            switch plotear
                case 1
                    var=100*err_map;
                    t1=['Error (%) (case ',num2str(j),')'];
                case 2
                    var=mean_map;
                    t1=['Mean (case ',num2str(j),')'];
                case 3
                    var=mean_map./std_map;
                    t1=['R (case ',num2str(j),')'];
                case 4
                    var=sk_map;
                    t1=['Skewness (case ',num2str(j),')'];
            end
                    

            imagesc(1e3*data.feat.x(windowsize/2:end-windowsize/2), ...
                    1e3*data.feat.z_interp(windowsize/2:end-windowsize/2), ...
                    var);axis image;
            ylabel('Depth (mm)');xlabel('Lateral distance (mm)');
            gridsize=(data.feat.x(2)-data.feat.x(1));
            title(t1, ...
                  ['a=',num2str(data.a_0),', windowsize=',num2str(windowsize*gridsize*1e3,3),'x',num2str(gridsize*1e3,3),' mm']);
            switch plotear
                case 1
                    colormap(ax,slanCM(103));colorbar;
                    % clim([-1,1]*max(abs(100*err_map(:))));
                    clim([-1,1]*20);
                case 2
                    colorbar;
                case 3
                    colorbar;clim([0 2]);
                case 4
                    colorbar;clim([2 4]);
            end
            
        end
        ax=subplot(2,3,1);
        imagesc(1e3*data.feat.x, ...
                1e3*data.feat.z_interp, ...
                data.comp_env_interp);
        colormap(ax,'gray');colorbar;clim([0 255]);axis image;
        ylabel('Depth (mm)');xlabel('Lateral distance (mm)');title('B-mode');
        % keyboard;
        if g
            saveas(gcf,['res_img\',f,path2fname(nameBmode),'.png']);
        end
        % close;
        % keyboard;
    end
end