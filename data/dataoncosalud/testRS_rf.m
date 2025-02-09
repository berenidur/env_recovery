close all;clear;clc;
addpath('../../code/functions/');

fcell={'breast\','ganglio\'};

for kcell=1:length(fcell)
f=fcell{kcell};
lstfiles=ls([f,'*.rf']);
tot=size(lstfiles,1);

% for i=1:tot
for i=[1 3]
    fname=strtrim(lstfiles(i,:));
    [rf,feat] = RPread([f,fname]);
    env_rf=abs(hilbert(rf));

    [x, z, z_interp, X, Z, X_interp, Z_interp] = xz_interp_grid(rf, feat);

for j=1:5
    [a_0,b_0,comp_env]=getab(env_rf,j);
    % comp_env=20*log10(env_rf/max(env_rf(:)));

    comp_env_interp = interp2(X,Z,comp_env,X_interp,Z_interp);
    figure;
    subplot(2,2,1);
    imagesc(100*x,100*z_interp,comp_env);colormap gray;colorbar;axis image;%clim([-50 0]);
    title('B-mode');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % keyboard;break;

    env_rf_interp = interp2(X,Z,env_rf,X_interp,Z_interp);
    % env_rf_interp = env_rf;

    [rows, cols] = size(env_rf_interp);
    n=57;
    m=n;

    % Preallocate output matrix
    output_rows = rows - n + 1;
    output_cols = cols - m + 1;
    S_matrix = zeros(output_rows, output_cols);
    R_matrix = zeros(output_rows, output_cols);
    validRS = zeros(size(S_matrix));

    % Loop through each window
    for ni = 1:output_rows
        for mi = 1:output_cols
            % Extract the n x m window
            window = env_rf_interp(ni:ni+n-1, mi:mi+m-1);
            S_matrix(ni, mi) = skewness(window(:));
            R_matrix(ni, mi) = mean(window(:))/std(window(:));
            try
                [k, beta] = getkbeta(R_matrix(ni, mi), S_matrix(ni, mi));
                if 0<=k && k<1 && 0<=beta && beta<0.25
                    validRS(ni,mi)=1;
                end
            catch exception
            end
        end
    end

    % figure;imagesc(x,z,env_rf_interp);colormap gray;colorbar;axis image;
    ax=subplot(2,2,3);
    imagesc(100*x(round(n/2):round(end-n/2)),100*z_interp(round(m/2):round(end-m/2)),S_matrix);
    colormap(ax,slanCM(114));colorbar;clim([1 3]);axis image;
    title('S');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');

    ax=subplot(2,2,4);
    imagesc(100*x(round(n/2):round(end-n/2)),100*z_interp(round(m/2):round(end-m/2)),R_matrix);
    colormap(ax,slanCM(114));colorbar;clim([1 3]);axis image;
    title('R');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % keyboard;

    sgtitle(['Windows ',num2str(n*res*1e3,3),'mm x',num2str(m*res*1e3,3),'mm'])
    
    % Iterate through the output_matrix to apply the criteria
    % tol=0.25;
    % idealR=1;
    % idealS=2;
    visualized_image=zeros(size(comp_env_interp));
    patches.(['val',num2str(j)])=[];
    patches.(['a',num2str(j)])=a_0;
    patches.(['comp_env',num2str(j)])=comp_env_interp;
    patches.(['validRS_original',num2str(j)])=validRS;
    
    for ni = 1:output_rows
        for mi = 1:output_cols
            % if S_matrix(ni, mi) > idealS-tol && S_matrix(ni, mi) < idealS+tol ...
            %         && R_matrix(ni, mi) > idealR-tol && R_matrix(ni, mi) < idealR+tol
            if logical(validRS(ni,mi))
                % Map output_matrix position back to input_matrix window
                visualized_image(ni:ni+n-1, mi:mi+m-1) = comp_env_interp(ni:ni+n-1, mi:mi+m-1);
                temp_window=comp_env_interp(ni:ni+n-1, mi:mi+m-1);
                patches.(['val',num2str(j)])=cat(3,patches.(['val',num2str(j)]),temp_window);
                % keyboard;
            end
        end
    end

    ax=subplot(2,2,2);
    imagesc(100*x,100*z_interp,visualized_image);colormap(ax,gray);colorbar;axis image;%clim([-50 0]);
    title('Speckle patches');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % keyboard;


    % figure;
    % subplot(2,2,1);imagesc(x,z,comp_env);colormap gray;colorbar;axis image;clim([0 255])
    % subplot(2,2,2);imagesc(x,z_interp,comp_env_interp);colormap gray;colorbar;axis image;clim([0 255])
    % subplot(2,2,[3,4]);histogram(comp_env);
    % keyboard
end
% continue;
% save(['breast',fname(1:end-3),'_patches.mat'],'-struct','patches');
disp([f(1:end-1),':',num2str(i),'/',num2str(tot),' saved']);
clear patches;
keyboard;
% close all;
end
end

% [rf1,feat1] = RPread('breast\09-57-58.rf');
% [rf2,feat2] = RPread('breast\09-58-14.rf');
% 
% isequal(rf1,rf2)