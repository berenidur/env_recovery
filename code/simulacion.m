% Simulating B-mode Ultrasound Images Example

clearvars; close all; clc;
addpath('functions/');

% para guardar el código que se utilizó. Si da error, solo comentar
puntom=guardarpuntom(mfilename('fullpath'));

% verificar ruta de bgmaps en línea 50 o Ctrl+F generarbgmap2D

% simulation settings
% addpath(genpath([pwd,'/k-wave-toolbox-version-1.3']))

% att_sam = 0.6;            % att sample values
% att_ref = 0.5;            % att reference value
% 
% a_pow = 1.05;             % medium.alpha_power

source_strength_vector = [600]*1e3;

n_slices_sam = 0;

% grid size
X = 1024;
Y = 760;
% Z = 200;

use_gpu = true;

if use_gpu
    DATA_CAST       = 'gpuArray-single';     % set to 'single' or 'gpuArray-single' to speed up computations
else
    DATA_CAST       = 'single';
end

if n_slices_sam==0
    bona_sam=[];
end

% =========================================================================
% DEFINE THE K-WAVE GRID
% =========================================================================

% set the size of the perfectly matched layer (PML)
pml_x_size = 40;                % [grid points]
pml_y_size = 10;                % [grid points]
% pml_z_size = 10;                % [grid points]

bgmappath='D:/Rodolfo/matlab/bgmaps/';
generarbgmap2D(X,Y,[pml_x_size,pml_y_size],n_slices_sam,2,bgmappath);
disp('Background maps OK')
pause(3);

% set total number of grid points not including the PML
Nx = X - 2 * pml_x_size;      % [grid points]
Ny = Y - 2 * pml_y_size;      % [grid points]
% Nz = Z - 2 * pml_z_size;      % [grid points]

% calculate the spacing between the grid points
% dx = 0.11e-3;       % max freq 7
% dx = 0.10e-3;       % max freq 7.7
dx = 0.09e-3;       % max freq 8.56
% dx = 0.08e-3;       % max freq 9.63
dy = dx;                        % [m]
% dz = dx;                        % [m]

% create the k-space grid
% kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% =========================================================================
% DEFINE THE MEDIUM PARAMETERS
% =========================================================================

% define the properties of the propagation medium
c0 = 1540;                      % [m/s]
rho0 = 1000;                    % [kg/m^3]

% create the time array
t_end = (Nx * dx) * 2.2 / c0;   % [s]
kgrid.makeTime(c0, [], t_end);

% =========================================================================
% DEFINE THE INPUT SIGNAL
% =========================================================================

% define properties of the input signal
tone_burst_freq = 3.5e6;        % [Hz]
tone_burst_cycles = 10;

for bonaa=[0,bona_sam]
    disp([newline,'BonA=',num2str(bonaa)]);
    for ss = 1:length(source_strength_vector)
        % append input signal used to drive the transducer
        source_strength =source_strength_vector(ss);          % [Pa]
        input_signal_norm = toneBurst(1/kgrid.dt, tone_burst_freq, tone_burst_cycles);
        input_signal = source_strength * input_signal_norm;

        % =========================================================================
        % DEFINE THE ULTRASOUND TRANSDUCER
        % =========================================================================
        number_elements = 128;       % number of elements
        element_width   = 0.46e-3;     % width [m]
        % element_length  = 13.5e-3;    % elevation height [m]
        element_length  = dx;    % elevation height [m] = 1 dx
        element_pitch   = 0.508e-3;     % pitch [m]
        radius          = 49.57e-3;

        karray = kWaveArray;
        positions = 0 - (number_elements * element_pitch / 2 - element_pitch / 2) + (0:number_elements-1) * element_pitch;
        for ind = 1:number_elements
            rotation = 90;
            karray.addRectElement([kgrid.x_vec(8), positions(ind)], element_width, element_length, rotation);
        end
        
        % assign binary mask
        source.p_mask = karray.getArrayBinaryMask(kgrid);
        input_signal=repmat(input_signal,number_elements,1);
        source.p = karray.getDistributedSourceSignal(kgrid, input_signal);
        sensor.mask = karray.getArrayBinaryMask(kgrid);

        % sensor.mask=zeros(Nx,Ny);
        % dist_indx=57:56:Nx;
        % dist_indy=zeros(8,1);
        % for i = 1:numel(dist_indy)
        %     [~, closest_index] = min(abs(kgrid.y_vec - positions(16*i-8))); % Find index of closest element
        %     dist_indy(i) = closest_index;
        % end
        % dist_indy=[dist_indy;Ny/2];
        % sensor.mask(dist_indx,dist_indy)=1;

        % check mask positions
        % indices = find(sensor.mask);
        % [subscriptsX, subscriptsY, subscriptsZ] = ind2sub(size(sensor.mask), indices);
        % uniqueValues = unique(subscriptsX);
        % disp(uniqueValues');
        % uniqueValues = unique(subscriptsZ);
        % disp(uniqueValues');
        % uniqueValues = unique(subscriptsY);
        % disp(uniqueValues(1));
        % disp(uniqueValues(end));
        % keyboard
        % 
        % figure;imagesc(sensor.mask);axis image;title('yz');colormap gray;colorbar;

        % =========================================================================
        % DEFINE THE MEDIUM PROPERTIES
        % =========================================================================

        % BonA_map = bonaa;
        % att_map = att_sam;

        % if bonaa==0
        %     BonA_map=bona_ref;
        %     att_map=att_ref;
        % end

        medium.sound_speed_ref = 1540;
        medium.sound_speed = 1540;

        % =========================================================================
        % RUN THE SIMULATION
        % =========================================================================

        % set the input settings
        input_args = {...
            'PMLInside', false, 'PMLSize', [pml_x_size, pml_y_size], ...
            'DataCast', DATA_CAST, 'DataRecast', true, 'PlotSim', false};

        % loop through the scan lines
        % aa=0;
        % while aa<=n_slices_sam
        if n_slices_sam==0
            n_slices_sam=1;
            onlyref=true;
        else
            onlyref=false;
        end
        for aa = 1:n_slices_sam
            disp('');  

            % if any(BonA_map)
            %     medium.BonA = BonA_map;
            % end
            % medium.alpha_coeff = att_map;
            % medium.alpha_power = a_pow;

            if bonaa==0
                load([bgmappath,'/bgmap2D_',num2str(X),'_',num2str(Y),'_',num2str(bonaa),'.mat']);
            else
                if onlyref
                    break;
                end
                load([bgmappath,'/bgmap2D_',num2str(X),'_',num2str(Y),'_',num2str(aa),'.mat']);
            end

            scatterer_radius=c0/tone_burst_freq*0.8;
            scatterer_density=0.005;

            background_map=generate_scatterers(background_map, scatterer_radius, scatterer_density,kgrid);

            density_map = rho0 * background_map(:,:,1);
            medium.density = density_map;

            % figure;imagesc(density_map);axis image;colormap(gray);colorbar;keyboard;
            % figure;imagesc(kgrid.y_vec,kgrid.x_vec-kgrid.x_vec(1),density_map);axis image;colormap(gray);colorbar;keyboard;

            % run the simulation
            disp(['Computing scan line PLANE WAVE...',num2str(aa)]);
            if use_gpu
                sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, input_args{:});
            else
                sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
            end
            combined_sensor_data_aa = karray.combineSensorData(kgrid, sensor_data);clear sensor_data;
            combined_sensor_data(:,:,aa) = combined_sensor_data_aa'; clear combined_sensor_data_aa;
            
            fs = 1/kgrid.dt;
            if bonaa==0
                break;
            end
            % aa=aa+1;
        end
        if bonaa==0
            pre='ref2D';
        else
            pre='sam2D';
        end
        % file_out = ['avance13/',pre,'_puntos_rect_bona',num2str(BonA_map),'_coeff',num2str(att_map),'_power',num2str(a_pow),'_',num2str(tone_burst_freq*1e-6),'MHz_',num2str(source_strength*1e-3),'.mat'];
        file_out = [pre,'_ray_rect_',num2str(tone_burst_freq*1e-6),'MHz_',num2str(source_strength*1e-3),'.mat'];
        mascara=sensor.mask;
        % if exist('puntom')
        % save(file_out,'sensor_data','fs','c0','tone_burst_freq','positions','BonA_map','mascara','puntom');
        % else
        % save(file_out,'sensor_data','fs','c0','tone_burst_freq','positions','BonA_map','mascara');
        % end
        if exist('puntom')
        save(file_out,'combined_sensor_data','fs','c0','tone_burst_freq','positions','background_map','mascara','puntom');
        else
        save(file_out,'combined_sensor_data','fs','c0','tone_burst_freq','positions','background_map','mascara');
        end
    end
end


function background_map0 = generate_scatterers(background_map0, scatterer_radius, scatterer_density,kgrid)
    % Define the size of the grid
    [Nx, Ny] = size(background_map0);
    dx=kgrid.dx;
    scattering_rho0 = 1+10/100*randn([Nx, Ny]);
    
    % Calculate the total number of scatterers
    num_scatterers = round(scatterer_density * Nx * Ny);
    
    % Initialize the scatterer array
    scatterers = zeros(Nx, Ny);
    
    % Randomly place scatterers in the grid
    for i = 1:num_scatterers
        x = randi([1, Nx]);
        y = randi([1, Ny]);
        scattering_region = makeDisc(Nx, Ny, x, y, round(scatterer_radius/dx));
        background_map0(scattering_region == 1) = scattering_rho0(scattering_region == 1);
    end
end

function [inc_bgmap,scattering_region]=make_inc(background_map0,kgrid)

% =========================================================================
% scatterers
% =========================================================================

% define a random distribution of scatterers for the highly scattering region
[Nx, Ny]=size(background_map0);
dx=kgrid.dx;
scattering_rho0 = 1+10/100*randn([Nx, Ny]);

% define properties
newmap = background_map0;

% define a sphere for a highly scattering region
radius = 5e-3;
scattering_region1 = makeDisc(Nx, Ny, round(Nx/4), round(Ny/4), round(radius/dx));
scattering_region2 = makeDisc(Nx, Ny, round(Nx/2), round(Ny/2), round(radius/dx));
scattering_region3 = makeDisc(Nx, Ny, round(3*Nx/4), round(3*Ny/4), round(radius/dx));
scattering_region = scattering_region1|scattering_region2|scattering_region3;

newmap(scattering_region == 1) = scattering_rho0(scattering_region == 1);

inc_bgmap = newmap;


% check
% figure;imagesc(density_map);axis image;colormap(gray);colorbar;keyboard;
% figure;imagesc(kgrid.y_vec,kgrid.x_vec-kgrid.x_vec(1),density_map);axis image;colormap(gray);colorbar;keyboard;

end