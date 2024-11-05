%% Chengdu Ultimedical, Inc.
%  Purpose  : read rf data from .txt file and display the image (Linear Probe RF Data Demo)
%% Load Data
clc;
clear all;
close all;
addpath('../data_adq/')
% filename = 'C:\Users\u_imagenes\Documents\MATLAB\Code\RF Function Instruction\RF Function Instruction\rf_capture_7.txt';
% filename = 'C:\Users\u_imagenes\Documents\MATLAB\Code\RF Function Instruction\RF Function Instruction\Try\rf_capture_6.txt';
filename = '../data_adq/data_adq/rf_capture_0.txt';

data  = load(filename);

%% Data Transform
lines  = 256;
points = 3584;

rfdata = reshape(data,lines,points);

%% Get Data 
fs      = 40e6;         % [Hz]   --  fixed parameter -- system sampling frequency is 40MHz in MU1L(Linear probe system)
depth   = 50*1e-3;      % [m]    --  adjustable parameter -- change the depth parameter, if the depth is larger than 69mm, the rf data is unvalid
c       = 1540;         % [m/s]


points_valid    = round(2*depth/c*fs); % sample points
if(points_valid>points)
    points_valid = points;
end
        
lines_valid     = 256;                  % line number

rfdata_valid    = rfdata(1:lines_valid,1:points_valid);

%% Show Image  please note this script is only for most premitive viewing
%% of the data. 
figure;
rfdata_valid=rfdata_valid(:,600:end);
env=abs(hilbert(rfdata_valid.'));
env_norm=env/max(env(:));
imagesc(20*log10(env_norm))
colormap(gray);colorbar;clim([-50 0])
title('Linear RF Image')