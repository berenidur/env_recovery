close all;clear;clc;
addpath('../../code/functions/');

load('res_valid\classification_only.mat');

tot=length(accepted_files);
results = struct();

parfor_progress(tot);

parfor i=1:tot
    fpath=accepted_files{i};
    [rf,feat] = RPread(fpath);
    env_rf=abs(hilbert(rf));

    [x, z, z_interp, X, Z, X_interp, Z_interp] = xz_interp_grid(rf, feat);

% for j=1
    % [a_0,b_0,comp_env]=getab(env_rf,j);
    % comp_env=20*log10(env_rf/max(env_rf(:)));

    % comp_env_interp = interp2(X,Z,comp_env,X_interp,Z_interp);
    env_rf_interp = interp2(X,Z,env_rf,X_interp,Z_interp);

    [rows, cols] = size(env_rf_interp);
    n=57;
    m=n;

    % Preallocate output matrix
    output_rows = rows - n + 1;
    output_cols = cols - m + 1;

    % metrics
    % k_matrix = zeros(output_rows, output_cols);
    % beta_matrix = zeros(output_rows, output_cols);
    validRS = false(output_rows, output_cols);
    % validRS_uncomp = false(size(k_matrix));
    % validRS_uncomp_corr = false(size(k_matrix));
    % a_matrix_valid=[];
    % a_matrix_valid_uncomp=[];
    % a_matrix_valid_uncomp_corr=[];
    
    % plot
    % visualized_image=zeros(size(comp_env_interp));
    % visualized_image_uncomp=zeros(size(comp_env_interp));
    % visualized_image_uncomp_corr=zeros(size(comp_env_interp));

    % coeff
    a_matrix = zeros(output_rows, output_cols);

    %% Loop through each window
    % tic;
    for ni = 1:output_rows
        for mi = 1:output_cols
            % envelope
            window = env_rf_interp(ni:ni+n-1, mi:mi+m-1);
            window_v=window(:);
            S = skewness(window_v);
            R = mean(window_v)/std(window_v);

            % compressed
            % window_comp=comp_env_interp(ni:ni+n-1, mi:mi+m-1);
            % window_comp_v=window_comp(:);
            % [a_prager,uncomp_env_window]=est_prager(window_comp_v,a_0,0,false);
            % a_matrix(ni, mi) = a_prager;
            % 
            % S_uncomp = skewness(uncomp_env_window);
            % R_uncomp = mean(uncomp_env_window)/std(uncomp_env_window);

            %% k beta calculation

            % original envelope
            try
                [k, beta] = getkbeta(R, S);
                % k_matrix(ni, mi)=k;
                % beta_matrix(ni, mi)=beta;
                if 0<=k && k<1 && 0<=beta && beta<0.25
                    validRS(ni,mi)=1;
                    % visualized_image(ni:ni+n-1, mi:mi+m-1) = comp_env_interp(ni:ni+n-1, mi:mi+m-1);
                    % a_matrix_valid=[a_matrix_valid,a_matrix(ni,mi)];
                end
            catch exception
            end

            % compressed
            % try
            %     [k, beta] = getkbeta(R_uncomp, S_uncomp);
            %     % k_matrix(ni, mi)=k;
            %     % beta_matrix(ni, mi)=beta;
            %     if 0<=k && k<1 && 0<=beta && beta<0.25
            %         validRS_uncomp(ni,mi)=1;
            %         % visualized_image_uncomp(ni:ni+n-1, mi:mi+m-1) = comp_env_interp(ni:ni+n-1, mi:mi+m-1);
            %         % a_matrix_valid_uncomp=[a_matrix_valid_uncomp,a_matrix(ni,mi)];
            %     end
            % catch exception
            % end

            % compressed ("corrected")
            % try
            %     [k, beta] = getkbeta(R_uncomp-0.07, S_uncomp-0.95);
            %     % k_matrix(ni, mi)=k;
            %     % beta_matrix(ni, mi)=beta;
            %     if 0<=k && k<1 && 0<=beta && beta<0.25
            %         validRS_uncomp_corr(ni,mi)=1;
            %         % visualized_image_uncomp_corr(ni:ni+n-1, mi:mi+m-1) = comp_env_interp(ni:ni+n-1, mi:mi+m-1);
            %         % a_matrix_valid_uncomp_corr=[a_matrix_valid_uncomp_corr,a_matrix(ni,mi)];
            %     end
            % catch exception
            % end
        end
    end
    % t_tot=toc;

    % ploteo y disp
    % 
    % figure;
    % ax=subplot(2,2,1);
    % imagesc(100*x,100*z_interp,comp_env_interp);colormap(ax,gray);colorbar;axis image;
    % title('B-mode');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % 
    % ax=subplot(2,2,2);
    % imagesc(100*x,100*z_interp,visualized_image);colormap(ax,gray);colorbar;axis image;
    % title('Envelope');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % 
    % ax=subplot(2,2,3);
    % imagesc(100*x,100*z_interp,visualized_image_uncomp);colormap(ax,gray);colorbar;axis image;
    % title('Uncompressed');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % 
    % ax=subplot(2,2,4);
    % imagesc(100*x,100*z_interp,visualized_image_uncomp_corr);colormap(ax,gray);colorbar;axis image;
    % title('"corrected"');ylabel('Depth (cm)');xlabel('Lateral distance (cm)');
    % 
    % sgtitle(['Windows ',num2str(n*res*1e3,3),'mm x',num2str(m*res*1e3,3),'mm']);
    % keyboard;
    % 
    % disp([f(1:end-1),':',num2str(i),'/',num2str(tot),'(',num2str(j),') : ',num2str(t_tot),'s']);
    % disp(['Ground truth coeff: ',num2str(a_0)]);
    % disp('Envelope:');
    % [a_prager_fin,~]=est_prager(comp_env_interp(visualized_image~=0),a_0,0,false);
    % disp(['Junto   : ',num2str(a_prager_fin), ...
    %     ' (',num2str(calc_error_p(a_0,a_prager_fin)),'%)']);
    % disp(['Separado: ',num2str(mean(a_matrix_valid)), ...
    %     ' (',num2str(calc_error_p(a_0,mean(a_matrix_valid))),'%)']);
    % 
    % disp('Uncompressed:');
    % a=visualized_image_uncomp~=0;
    % [a_prager_fin,~]=est_prager(comp_env_interp(a),a_0,0,false);
    % disp(['Junto1  : ',num2str(a_prager_fin), ...
    %     ' (',num2str(calc_error_p(a_0,a_prager_fin)),'%)']);
    % a(1:61,:)=0;
    % [a_prager_fin,~]=est_prager(comp_env_interp(a),a_0,0,false);
    % disp(['Junto2  : ',num2str(a_prager_fin), ...
    %     ' (',num2str(calc_error_p(a_0,a_prager_fin)),'%)']);
    % disp(['Separado: ',num2str(mean(a_matrix_valid_uncomp)), ...
    %     ' (',num2str(calc_error_p(a_0,mean(a_matrix_valid_uncomp))),'%)']);
    % 
    % disp('Corrected:');
    % [a_prager_fin,~]=est_prager(comp_env_interp(visualized_image_uncomp_corr~=0),a_0,0,false);
    % disp(['Junto   : ',num2str(a_prager_fin), ...
    %     ' (',num2str(calc_error_p(a_0,a_prager_fin)),'%)']);
    % disp(['Separado: ',num2str(mean(a_matrix_valid_uncomp_corr)), ...
    %     ' (',num2str(calc_error_p(a_0,mean(a_matrix_valid_uncomp_corr))),'%)']);
    
% end
results(i).filename = fpath;
results(i).validRS = validRS;

% n_processed=n_processed+1;
parfor_progress;
end
parfor_progress(0);

save('res_valid\processed_classification_results.mat', 'results');