close all;clear;clc;
addpath('../../code/functions/');

fcell={'breast\','ganglio\'};

for k=1:length(fcell)
f=fcell{k};
lstfiles=ls([f,'*.rf']);
tot=size(lstfiles,1);

for i=1:tot
    fname=strtrim(lstfiles(i,:));
    [rf,feat] = RPread([f,fname]);
    env_rf=abs(hilbert(rf));

    z=(1540/(2*feat.sf))*feat.h;
    x=0.038;
    z=linspace(0,z,size(rf,1));
    x=linspace(0,x,size(rf,2));
    feat.z=z;
    feat.x=x;

    res=x(2)-x(1);
    z_interp=z(1):res:z(end);
    feat.z_interp=z_interp;

    [X,Z]=meshgrid(x,z);
    [X_interp,Z_interp]=meshgrid(x,z_interp);

for j=1:5
    [a_0,b_0,comp_env]=getab(env_rf,j);
    comp_env_interp = interp2(X,Z,comp_env,X_interp,Z_interp);

    namesave=[f,'bmode\',fname(1:end-3),'_m',num2str(j),'.mat'];
    save(namesave,'comp_env','comp_env_interp','a_0','b_0','feat');
    % disp([f,':',num2str(i),'/',num2str(tot),' Saved ',namesave]);

    % figure;
    % subplot(2,2,1);imagesc(x,z,comp_env);colormap gray;colorbar;axis image;clim([0 255])
    % subplot(2,2,2);imagesc(x,z_interp,comp_env_interp);colormap gray;colorbar;axis image;clim([0 255])
    % subplot(2,2,[3,4]);histogram(comp_env);
    % keyboard
end
disp([f(1:end-1),':',num2str(i),'/',num2str(tot),' Saved']);
% keyboard;close all;
end
end

% [rf1,feat1] = RPread('breast\09-57-58.rf');
% [rf2,feat2] = RPread('breast\09-58-14.rf');
% 
% isequal(rf1,rf2)