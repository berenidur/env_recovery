close all;clear;clc;
addpath('../../code/functions/');

lstfiles=ls('*.mat');

tot=size(lstfiles,1);

for i=1:tot
    fname=strtrim(lstfiles(i,:));
    disp(['Processing ',fname]);
    patches=load(fname);
    if isfield(patches,'a_est1')
        continue
    end

for j=1:5
    disp(j);
    % windows = patches.(['val',num2str(j)]);
    a_0 = patches.(['a',num2str(j)]);
    comp_env = patches.(['comp_env',num2str(j)]);
    validRS_original = patches.(['validRS_original',num2str(j)]);

    [rows, cols] = size(comp_env);
    n=57;
    m=n;

    % Preallocate output matrix
    output_rows = rows - n + 1;
    output_cols = cols - m + 1;
    a_matrix = zeros(output_rows, output_cols);
    S_matrix = zeros(output_rows, output_cols);
    R_matrix = zeros(output_rows, output_cols);
    validRS_uncomp = zeros(size(S_matrix));
    validRS_corr = zeros(size(S_matrix));

    % Loop through each window
    % tic
    for ni = 1:output_rows
        for mi = 1:output_cols
            % Extract the n x m window
            window = comp_env(ni:ni+n-1, mi:mi+m-1);
            [a_prager,uncomp_env_window]=est_prager(window,a_0,0,false);

            a_matrix(ni, mi) = a_prager;
            S_matrix(ni, mi) = skewness(uncomp_env_window(:));
            R_matrix(ni, mi) = mean(uncomp_env_window(:))/std(uncomp_env_window(:));
            try
                [k, beta] = getkbeta(R_matrix(ni, mi), S_matrix(ni, mi));
                if 0<=k && k<1 && 0<=beta && beta<0.25
                    validRS_uncomp(ni,mi)=1;
                end
            catch exception
            end
            try
                [k, beta] = getkbeta(R_matrix(ni, mi)-0.07, S_matrix(ni, mi)-0.95);
                if 0<=k && k<1 && 0<=beta && beta<0.25
                    validRS_corr(ni,mi)=1;
                end
            catch exception
            end
        end
    end
    % toc
    patches.(['a_est',num2str(j)])=a_matrix;
    patches.(['validRS_uncomp',num2str(j)])=validRS_uncomp;
    patches.(['validRS_uncomp_corr',num2str(j)])=validRS_corr;
    
    visualized_image_original=zeros(size(comp_env));
    visualized_image_uncomp=zeros(size(comp_env));
    visualized_image_uncomp_corr=zeros(size(comp_env));
    for ni = 1:output_rows
        for mi = 1:output_cols
            if logical(validRS_original(ni,mi))
                visualized_image_original(ni:ni+n-1, mi:mi+m-1) = comp_env(ni:ni+n-1, mi:mi+m-1);
            end
            if logical(validRS_uncomp(ni,mi))
                visualized_image_uncomp(ni:ni+n-1, mi:mi+m-1) = comp_env(ni:ni+n-1, mi:mi+m-1);
            end
            if logical(validRS_corr(ni,mi))
                visualized_image_uncomp_corr(ni:ni+n-1, mi:mi+m-1) = comp_env(ni:ni+n-1, mi:mi+m-1);
            end
        end
    end
    % figure;
    % subplot(2,2,1);
    % imagesc(comp_env);colormap gray;colorbar;axis image;
    % title('B-mode');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % 
    % subplot(2,2,2);
    % imagesc(visualized_image_original);colormap gray;colorbar;axis image;
    % title('Ground truth');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % 
    % subplot(2,2,3);
    % imagesc(visualized_image_uncomp);colormap gray;colorbar;axis image;
    % title('Uncompressed');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % 
    % subplot(2,2,4);
    % imagesc(visualized_image_uncomp_corr);colormap gray;colorbar;axis image;
    % title('Uncompressed corrected');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % keyboard;
end
save(fname,'-struct','patches');
disp([fname,' saved']);
clear patches;
% keyboard;
end

% [rf1,feat1] = RPread('breast\09-57-58.rf');
% [rf2,feat2] = RPread('breast\09-58-14.rf');
% 
% isequal(rf1,rf2)

% falta comparar: valores de R, S y a