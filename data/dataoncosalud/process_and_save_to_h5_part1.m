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
    j=1;
    [a_0,b_0,comp_env]=getab(env_rf,j);
    % comp_env=20*log10(env_rf/max(env_rf(:)));

    comp_env_interp_1 = interp2(X,Z,comp_env,X_interp,Z_interp);
    env_rf_interp = interp2(X,Z,env_rf,X_interp,Z_interp);

    [rows, cols] = size(env_rf_interp);
    n=57;
    m=n;

    % Preallocate output matrix
    output_rows = rows - n + 1;
    output_cols = cols - m + 1;

    % metrics
    k_matrix = NaN(output_rows, output_cols);
    beta_matrix = NaN(output_rows, output_cols);
    S_matrix = NaN(output_rows, output_cols);
    R_matrix = NaN(output_rows, output_cols);
    validRS = false(output_rows, output_cols);

    %% Loop through each window
    % tic;
    for ni = 1:output_rows
        for mi = 1:output_cols
            % envelope
            window = env_rf_interp(ni:ni+n-1, mi:mi+m-1);
            window_v=window(:);
            S = skewness(window_v);
            R = mean(window_v)/std(window_v);

            S_matrix(ni,mi)=S;
            R_matrix(ni,mi)=R;

            %% k beta calculation
            try
                [k, beta] = getkbeta(R, S);
                k_matrix(ni, mi)=k;
                beta_matrix(ni, mi)=beta;
                if 0<=k && k<1 && 0<=beta && beta<0.25
                    validRS(ni,mi)=1;
                end
            catch exception
            end

            % window_comp=comp_env_interp_1(ni:ni+n-1, mi:mi+m-1);
            % [a_prager,uncomp_env_window]=est_prager(window_comp,a_0,0,false);
            
        end
    end
    % toc
    % keyboard

results(i).filename = fpath;
results(i).env_rf_interp = env_rf_interp;
results(i).comp_env_interp_1 = comp_env_interp_1;
results(i).a_0 = a_0;
results(i).b_0 = b_0;
results(i).n = n;

results(i).R_matrix = R_matrix;
results(i).S_matrix = S_matrix;

results(i).k_matrix = k_matrix;
results(i).beta_matrix = beta_matrix;
results(i).validRS = validRS;

parfor_progress;
end
parfor_progress(0);

save('res_valid\processed_classification_results.mat', 'results');