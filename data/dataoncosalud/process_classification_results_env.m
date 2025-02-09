close all; clear; clc;
addpath('../../code/functions/');

load('res_valid/classification_only.mat');
load('res_valid/processed_classification_results.mat');

tot = length(accepted_files);
h5_filename = 'res_valid/comp_env_data.h5';

% Create an empty HDF5 file if it doesn't exist
if ~isfile(h5_filename)
    fid = H5F.create(h5_filename);
    H5F.close(fid);
end

for i = 1:tot
    dataset_name = sprintf('/file_%d', i);
    
    % Check if this file already exists in the HDF5
    if is_h5_group_exists(h5_filename, dataset_name)
        fprintf('Skipping file %d/%d (already processed)\n', i, tot);
        continue; % Skip if already processed
    end

    % Process the file
    fpath = accepted_files{i};
    [rf, feat] = RPread(fpath);
    env_rf = abs(hilbert(rf));
    [x, z, z_interp, X, Z, X_interp, Z_interp] = xz_interp_grid(rf, feat);

    % Create the validRS dataset so attributes can be attached
    validRS_name = sprintf('%s/validRS', dataset_name);

    validRS = double(results(i).validRS);
    validRS = padarray(validRS,[28 28],0,'both');   % zero padding to match
    validRS = validRS.';                            % row-major

    h5create(h5_filename, validRS_name, size(validRS), 'Datatype', class(validRS));
    h5write(h5_filename, validRS_name, validRS);

    % Now write the file path as an attribute
    h5writeatt(h5_filename, dataset_name, 'filepath', fpath);

    for j = 1:5
        data_name = sprintf('%s/comp_env_interp_%d', dataset_name, j);

        % Check if this dataset already exists
        if is_h5_dataset_exists(h5_filename, data_name)
            fprintf('Skipping file %d/%d, j=%d (already processed)\n', i, tot, j);
            continue;
        end

        % Compute and store comp_env_interp
        [a_0, b_0, comp_env] = getab(env_rf, j);
        comp_env_interp = interp2(X, Z, comp_env, X_interp, Z_interp);
        
        
        comp_env_interp = comp_env_interp.';    % row major

        % Write data
        h5create(h5_filename, data_name, size(comp_env_interp), 'Datatype', 'double');
        h5write(h5_filename, data_name, comp_env_interp);
    end

    fprintf('Finished processing file %d/%d\n', i, tot);
end

disp('Processing complete.');


function exists = is_h5_dataset_exists(filename, dataset)
    try
        info = h5info(filename, dataset);
        exists = ~isempty(info);
    catch
        exists = false; % Dataset does not exist
    end
end

function exists = is_h5_group_exists(filename, group)
    try
        info = h5info(filename);
        group_names = {info.Groups.Name};
        exists = any(strcmp(group, group_names));
    catch
        exists = false; % Group does not exist
    end
end