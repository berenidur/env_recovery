close all;clear;clc;
addpath('../../code/functions/');

lstfiles=ls('*.mat');

tot=size(lstfiles,1);

for i=1:tot
    fname=strtrim(lstfiles(i,:));
    disp(['Processing ',fname]);
    patches=load(fname);
    % if isfield(patches,'a_est1')
    %     continue
    % end

for j=1:5
    % disp(j);
    % % windows = patches.(['val',num2str(j)]);
    a_0 = patches.(['a',num2str(j)]);
    comp_env_interp = patches.(['comp_env',num2str(j)]);
    validRS_original = patches.(['validRS_original',num2str(j)]);
    
    [rows, cols] = size(comp_env_interp);
    n=57;
    m=n;
   
    % Preallocate output matrix
    output_rows = rows - n + 1;
    output_cols = cols - m + 1;
    
    a_matrix=patches.(['a_est',num2str(j)]);
    a_matrix_show=zeros(size(a_matrix));
    a_matrix_valid=[];
    a_matrix_valid_uncomp=[];
    validRS_uncomp=patches.(['validRS_uncomp',num2str(j)]);
    validRS_corr=patches.(['validRS_uncomp_corr',num2str(j)]);
    
    visualized_image_original=zeros(size(comp_env_interp));
    visualized_image_uncomp=zeros(size(comp_env_interp));
    visualized_image_uncomp_corr=zeros(size(comp_env_interp));
    for ni = 1:output_rows
        for mi = 1:output_cols
            if logical(validRS_original(ni,mi))
                visualized_image_original(ni:ni+n-1, mi:mi+m-1) = comp_env_interp(ni:ni+n-1, mi:mi+m-1);
                a_matrix_show(ni,mi)=a_matrix(ni,mi);
                a_matrix_valid=[a_matrix_valid,a_matrix(ni,mi)];
            end
            if logical(validRS_uncomp(ni,mi))
                visualized_image_uncomp(ni:ni+n-1, mi:mi+m-1) = comp_env_interp(ni:ni+n-1, mi:mi+m-1);
                a_matrix_valid_uncomp=[a_matrix_valid_uncomp,a_matrix(ni,mi)];
                % a_matrix_show(ni,mi)=a_matrix(ni,mi);
                % a_matrix_valid=[a_matrix_valid,a_matrix(ni,mi)];
            end
            if logical(validRS_corr(ni,mi))
                visualized_image_uncomp_corr(ni:ni+n-1, mi:mi+m-1) = comp_env_interp(ni:ni+n-1, mi:mi+m-1);
                % a_matrix_show(ni,mi)=a_matrix(ni,mi);
                % a_matrix_valid=[a_matrix_valid,a_matrix(ni,mi)];
            end
        end
    end
    figure;
    subplot(2,2,1);
    imagesc(comp_env_interp);colormap gray;colorbar;axis image;
    title('B-mode');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');

    subplot(2,2,2);
    imagesc(visualized_image_original);colormap gray;colorbar;axis image;
    title('Ground truth');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');

    subplot(2,2,3);
    imagesc(visualized_image_uncomp);colormap gray;colorbar;axis image;
    title('Uncompressed');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');

    subplot(2,2,4);
    imagesc(visualized_image_uncomp_corr);colormap gray;colorbar;axis image;
    title('Uncompressed corrected');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    
    % err=(a_matrix-a_0)/a_0*100;
    % figure;imagesc(err);colormap turbo; colorbar; axis image;
    % title('error %');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % clim([-1,1]*10);
    
    figure;imagesc(a_matrix_show);colormap gray; colorbar; axis image;
    title(['Estimated (gt=',num2str(a_0),')']);ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    mean(a_matrix_valid)
    mean(a_matrix_valid_uncomp)
    a_0
    keyboard;close all;
end
% % save(fname,'-struct','patches');
disp([fname,' saved']);
clear patches;close all;
% keyboard;
end

%cuando hay menos muestras es peor