close all;clear;clc;

clear;close all;clc;
addpath('../../code/functions/');
g=false;

fcell={'breast\','ganglio\'};

windowsize=16;
const=2*sqrt(6)/pi;

for k=1:length(fcell)
% for k=2
    f=fcell{k};
    lstfiles=ls([f,'*.rf']);
    tot=size(lstfiles,1);
    for i=1:tot
        fname=strtrim(lstfiles(i,:));
        figure;

        for j=1:5
            % clear data;
            nameBmode=[f,'bmode\',fname(1:end-3),'_m',num2str(j),'.mat'];
            data=load(nameBmode);
            % info = whos('-file', nameBmode);
            % variableNames = {info.name};

            % if exist(nameBmode)==0;keyboard;end
            % figure;imagesc(data.feat.x,data.feat.z_interp,data.comp_env_interp);
            % colormap gray;colorbar;clim([0 255]);

            ax=subplot(2,3,j+1);
            imagesc(1e3*data.feat.x(windowsize/2:end-windowsize/2), ...
                    1e3*data.feat.z_interp(windowsize/2:end-windowsize/2), ...
                    100*data.err_map);axis image;
            ylabel('Depth (mm)');xlabel('Lateral distance (mm)');
            gridsize=(data.feat.x(2)-data.feat.x(1))*1e4;
            title(['Error (%) (case ',num2str(j),')'], ...
                  ['a=',num2str(data.a_0),', windowsize=',num2str(gridsize,3),'x',num2str(gridsize,3),' m^{-4}']);
            colormap(ax,slanCM(103));colorbar;
            % clim([-1,1]*max(abs(100*data.err_map(:))));
            clim([-1,1]*20);
        end
        ax=subplot(2,3,1);
        imagesc(1e3*data.feat.x, ...
                1e3*data.feat.z_interp, ...
                data.comp_env_interp);
        colormap(ax,'gray');colorbar;clim([0 255]);axis image;
        ylabel('Depth (mm)');xlabel('Lateral distance (mm)');title('B-mode');
        keyboard;
        if g
            saveas(gcf,['res_img\',f,path2fname(nameBmode),'.png']);
        end
        close;

    end
end