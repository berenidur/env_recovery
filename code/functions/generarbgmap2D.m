function generarbgmap2D(X,Y,pml,n_slices,d,folder)

% la ubicación por defecto es pwd\bgmaps2d

if nargin<6
    folder='bgmaps2D';
end

folder=strrep(folder, '\', '/');

[status, ~, ~] = mkdir(folder);
if status==1
    disp(['bgmaps folder: ',pwd,'\',folder])
end

% set total number of grid points not including the PML
Nx = X - 2 * pml(1);      % [grid points]
Ny = Y - 2 * pml(2);      % [grid points]

n=0:n_slices;
for i=1:length(n)
    file_out=[folder,'/bgmap2D_',num2str(X),'_',num2str(Y),'_',num2str(n(i)),'.mat'];
    if isfile(file_out)
        fprintf(['\t',file_out,' already exists.\n'])
        continue
    else
        background_map = 1 + d/100 * randn([Nx, Ny]);
        save(file_out,'background_map','pml','-v7.3');
        fprintf(['\t',file_out,' saved.\n'])
    end
end

end