function [a_prager,y] = est_prager(z,a_0,b_0,plotear)
if nargin < 3
        b_0 = 0; % Default value
end
if nargin < 4
        plotear=true; % Default value
end

D0=30;
% D0=a_0;
% con 50 y hruska funciona bien

n=[0.25 0.5 1.5 2 2.5 3];   % prager
% n=[0.72 0.88];   % hruska
nmom_exp=gamma(n+1);

err_vect_0=optimizarD(D0, z, nmom_exp, n);

intensity=exp(z/a_0);
nmom_real=norm_moment(intensity,n);

D_prager = lsqnonlin(@(D) optimizarD(D, z, nmom_exp, n), D0,[],[],optimoptions('lsqnonlin','Display','none'));

intensity=exp(z/D_prager);
nmom_est=norm_moment(intensity,n);
err_vect=nmom_est-nmom_exp;

if plotear
    figure;hold on;grid on;
    plot(nmom_exp ,Marker='.',MarkerSize=25,LineStyle='none');
    plot(nmom_est,Marker='.',MarkerSize=20,LineStyle='none');
    plot(nmom_real,Marker='.',MarkerSize=15,LineStyle='none');
    % xlim([0.5 6.5]);yticks(0:0.5:6.5);xticks(1:6);set(gca,'XTickLabel',n);
    xlim([0.5,length(n)+0.5]);xticks(1:length(n));set(gca,'XTickLabel',n);
    xlabel('n');ylabel('raw moment')
    legend('Theoretical','Estimated','Real',Location='best');
    
    disp(['Mean      : ',num2str(mean(err_vect))]);
    disp(['Std       : ',num2str(std(err_vect))]);
    disp(['Estimated : ',num2str(D_prager)]);
    disp(['Real value: ',num2str(a_0)]);
    disp(['Error     : ',num2str((D_prager-a_0)/a_0*100,3),'%']);
end

a_prager=D_prager;
y=exp((z-b_0)/a_prager);

end


function mu_n = raw_moment(x,n)
    x=x(:);
    % n: the order of the moment
    % x_values: vector of observed values of the random variable X    

    % % anterior implementación:
    % [N,edges] = histcounts(x,Normalization='pdf',NumBins=1e5);
    % edge=edges(2)-edges(1);
    % 
    % mu_n=sum(N.*edges(1:end-1).^(n.')*edge,2).';
    
    % al final resultó ser igual que
    % x=x-mean(x);  % en caso que sea momento central
    mu_n=mean(abs(x).^n);
end

function mu_nn = norm_moment(x,n)
    x=x(:);
    % n: the order of the moment
    % x_values: vector of observed values of the random variable X
    
    rawmoment=raw_moment(x,n);
    mu_nn=rawmoment./(mean(x).^n);
end

function err_vect=optimizarD(D, z, nmom_exp, n)
    intensity=exp(z/D);
    nmom_data=norm_moment(intensity,n);
    err_vect=nmom_data-nmom_exp;
end