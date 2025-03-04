close all; clear; clc;
addpath('../../code/functions/');

load('res_valid/processed_classification_results.mat');

tot = length(results);
h5_filename = 'res_valid/comp_env_data.h5';

% Create an empty HDF5 file if it doesn't exist
if ~isfile(h5_filename)
    fid = H5F.create(h5_filename);
    H5F.close(fid);
end

for i = 1:tot
    dataset_name = sprintf('/file_%04d', i);
    
    % Check if this file already exists in the HDF5
    % if is_h5_group_exists(h5_filename, dataset_name)
    %     fprintf('Skipping file %d/%d (already processed)\n', i, tot);
    %     continue; % Skip if already processed
    % end

    % Write numerical data
    h5create(h5_filename, [dataset_name '/env_rf_interp'], size(results(i).env_rf_interp.'));
    h5write(h5_filename, [dataset_name '/env_rf_interp'], results(i).env_rf_interp.');

    h5create(h5_filename, [dataset_name '/comp_env_interp_1'], size(results(i).comp_env_interp_1.'));
    h5write(h5_filename, [dataset_name '/comp_env_interp_1'], results(i).comp_env_interp_1.');
    
    h5create(h5_filename, [dataset_name '/a_0'], size(results(i).a_0));
    h5write(h5_filename, [dataset_name '/a_0'], results(i).a_0);

    h5create(h5_filename, [dataset_name '/b_0'], size(results(i).b_0));
    h5write(h5_filename, [dataset_name '/b_0'], results(i).b_0);
    
    h5create(h5_filename, [dataset_name '/n'], size(results(i).n));
    h5write(h5_filename, [dataset_name '/n'], results(i).n);
    
    h5create(h5_filename, [dataset_name '/R_matrix'], size(results(i).R_matrix.'));
    h5write(h5_filename, [dataset_name '/R_matrix'], results(i).R_matrix.');
    
    h5create(h5_filename, [dataset_name '/S_matrix'], size(results(i).S_matrix.'));
    h5write(h5_filename, [dataset_name '/S_matrix'], results(i).S_matrix.');

    h5create(h5_filename, [dataset_name '/k_matrix'], size(results(i).k_matrix.'));
    h5write(h5_filename, [dataset_name '/k_matrix'], results(i).k_matrix.');
    
    h5create(h5_filename, [dataset_name '/beta_matrix'], size(results(i).beta_matrix.'));
    h5write(h5_filename, [dataset_name '/beta_matrix'], results(i).beta_matrix.');

    h5create(h5_filename, [dataset_name '/validRS'], size(results(i).validRS.'));
    h5write(h5_filename, [dataset_name '/validRS'], double(results(i).validRS.'));

    % Write filename (string data needs a separate approach)
    h5writeatt(h5_filename, dataset_name, 'filename', results(i).filename);

    fprintf('Finished processing file %d/%d\n', i, tot);
end

disp('Processing complete.');