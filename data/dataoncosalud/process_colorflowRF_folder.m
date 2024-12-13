function [Properties,sono,Data]=process_colorflowRF_folder(DataPath,SaveToPath,phi,dB)
% process_colorflowRF_folder read the RF data in order to obtain
% sonoelastography and B-mode images. The RF data is then demodulated in 
% order to have the in-phase and quadrature signals (IQ). After that, the 
% IQ information is processing using autocorrelation-based spectral 
% variance estimation to finally obtain the sonoelastography data.
% 
% The scripts to read the RF data was supplied by:
% Copyright Ultrasonix Medical corporation
% Author: Reza Zahiri Azar  
% SaveToPath, DataPath

%Demodular y downsampling
BaseLocation = pwd;

cd(DataPath) %Cambia el directorio al directorio actual
D(:,1) = dir('*.crf');
D(:,2)= dir('*.rf');

cd(BaseLocation)
% Gain=0.01;
% pitch = 3.08e-04;
% fs = 1/pitch;
wf = [1 0];
%wf = [0.525 -0.475];
NumImges=size(D,1);
for ii=1:size(D,1)
    disp(['Processing file ' num2str(ii) ' of ' num2str(size(D,1))]);
    name_file=D(ii,1).name;                % Name of CRF file
    save_name = name_file(1:end-4);
    name_file2=D(ii,2).name;               % Name of RF file
    %      name_file3=D(ii,3).name;
    f=str2double(name_file(1:3));          % VIibration frequency
    file=[DataPath name_file];        % Path of the CRF file
    file2=[DataPath name_file2];      % Path of the RF file
    %      file3=[Location2 '\' name_file3];
    [Data, Properties] = RPread(file);     % Acquiring CRF data
    [Data2,Properties2] = RPread(file2);   % Acquiring RF data
    %      [Data3,Properties3] = RPread(file3);
    
    if isempty(Data) || isempty(Data2)
        ii=ii+1;
    else
        Properties.PRF=Properties.dr;          % PRF
        Properties.FrameRate=Properties2.dr;  % Frame rate

        IQ=process_IQ(Data,Properties);        % IQ demodulation of CRF
        sono = process_colorflowIQ_m(IQ,Properties,wf);   % Sonoelastography
        Depth_B=(1540/(2*Properties.sf))*Properties2.h;
        Width_B=0.038;
        Properties.Depth_B=linspace(0,Depth_B,size(Data2,1));
        Properties.Width_B=linspace(0,Width_B,size(Data2,2));
        inicio_Depth=0.249/1000;
        fin_Depth=39.04/1000;
        inicio_Width=0.01/1000;
        fin_Width=37.82/1000;
        Properties.Depth_S=linspace(inicio_Depth,fin_Depth,size(sono,1));
        Properties.Width_S=linspace(inicio_Width,fin_Width,size(sono,2));
        Properties.pitch=3.08e-04;
        Properties.VibFreq=f;
        Properties.VibFreqOffset=phi;
        disp('   Saving ...');
        Bmode=imgUS(Data2(:,:,1),Properties.Width_B,Properties.Depth_B,dB,f);
        Properties.Bmode=Bmode;
        
        save([SaveToPath save_name],'sono','Properties','Bmode')
    end
end
cd(BaseLocation)

end