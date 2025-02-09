close all; clear; clc;
addpath('../../code/functions/');

files = dir(fullfile('../dataoncosalud_recovery/', '**', '*.rf'));
tot = numel(files);

threshold = 0.08;

% Load existing results if available
if exist('res_valid\classification_results.mat', 'file')
    load('res_valid\classification_results.mat', 'results', 'accepted_files', 'rejected_files');
    processed_files = {results.filename};
else
    results = struct();
    accepted_files = {};
    rejected_files = {};
    processed_files= {};
end

for i = 1:tot
    clc;disp([num2str(i),'/',num2str(tot)]);

    rfpath = fullfile(files(i).folder, files(i).name);

    % Skip already processed files
    if ismember(rfpath, processed_files)
        continue;
    end
    
    [rf, feat] = RPread(rfpath);
    env_rf = abs(hilbert(rf));

    z = (1540 / (2 * feat.sf)) * feat.h;
    x = 0.038;
    z = linspace(0, z, size(rf, 1));
    x = linspace(0, x, size(rf, 2));
    feat.z = z;
    feat.x = x;

    res = x(2) - x(1);
    z_interp = z(1):res * 2:z(end);
    feat.z_interp = z_interp;

    [X, Z] = meshgrid(x, z);
    [X_interp, Z_interp] = meshgrid(x, z_interp);

    comp_env = 20 * log10(env_rf / max(env_rf(:)));
    comp_env_interp = interp2(X, Z, comp_env, X_interp, Z_interp);

    comp_env_interp_filt = imgaussfilt(comp_env_interp, 1);

    edges = edge(rescale(comp_env_interp_filt, 0, 1), 'Canny');
    result = mean(edges(ceil(end / 2):end, :), 'all');
    
    results(i).filename = rfpath;
    results(i).comp_env_interp = comp_env_interp; % Save processed data
    results(i).edge_mean = result;
    
    % Extract relative path for title
    relative_path = slicePath(rfpath,4);
    
    % Generate and save plot
    figure('visible','off');
    imagesc(x * 1e3, z_interp * 1e3, comp_env_interp);
    colormap gray;
    clim([-50, 0]);
    colorbar;
    title(relative_path, 'Interpreter', 'none');
    xlabel('x (mm)');
    ylabel('z (mm)');
    axis image;
    
    if result >= threshold
        accepted_files{end + 1} = rfpath;
        saveas(gcf, fullfile('res_valid\accepted', [replace(relative_path(1:end-3), '/', '_'), '.png']));
    else
        rejected_files{end + 1} = rfpath;
        saveas(gcf, fullfile('res_valid\rejected', [replace(relative_path(1:end-3), '/', '_'), '.png']));
    end
    close(gcf);
    % Save progress periodically
    if mod(i, 10) == 0
        save('res_valid\classification_results.mat', 'results', 'accepted_files', 'rejected_files');
    end
end

save('res_valid\classification_results.mat', 'results', 'accepted_files', 'rejected_files');

% To reclassify with a different threshold without reloading RF data:
% load('classification_results.mat');
% new_threshold = 0.75; % Change as needed
% new_accepted = {}; new_rejected = {};
% for i = 1:numel(results)
%     if results(i).edge_mean >= new_threshold
%         new_accepted{end + 1} = results(i).filename;
%     else
%         new_rejected{end + 1} = results(i).filename;
%     end
% end
% save('classification_results.mat', 'results', 'new_accepted', 'new_rejected');
