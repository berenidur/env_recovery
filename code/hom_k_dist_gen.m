clear;close all;clc;

% s=1:10;
% sigma=1;
% k=s/sigma;
% 
% mu=1:12;
% beta=1./mu;
% 
% [K,Beta]=meshgrid(k,beta);
% R=(K.^2+2) ./ (2*sqrt(K.^2+2*Beta+1));                              % mean/std
% S=((K.^2+2*Beta+1).*(6*Beta+3)-1) ./ (2*sqrt(K.^2+2*Beta+1)).^1.5;  % skewness

% figure;imagesc(R);colorbar;
% figure;imagesc(S);colorbar;

% Mean of gamma is k*theta
% Std of gamma is  sqrt(k*theta^2)=sqrt(k)*theta
% R of gamma is    sqrt(k)
% shape of gamma is k
% shape = R^2
% no funciona

% PDF generation
% Parameters
% s=1:10;
% sigma=1;
% mu=1:12;
s=2;
sigma=1;
mu=8;
% [sigma,mu]=meshgrid(sigma,mu);
A=linspace(0, 20, 1e5);

pdf_values = zeros(length(A),length(s),length(mu));
% tic
for j = 1:length(s)
for k = 1:length(mu)
for i = 1:length(A)
    integrand = @(x) x.* besselj(0, s(j)*x) .* besselj(0, A(i)*x) .* (1 + (x.^2 * sigma^2) / (2 * mu(k))).^(-mu(k));
    pdf_values(i,j,k) = A(i) * integral(integrand, 0, Inf);
end
end
end
% toc

% tic
% for j = 1:length(s)
% for k = 1:length(mu)
% % for i = 1:length(A)
%     integrand = @(x) x.* besselj(0, s(j)*x) .* besselj(0, A.*x) .* (1 + (x.^2 * sigma^2) / (2 * mu(k))).^(-mu(k));
%     pdf_values(:,j,k) = A .* integral(integrand, 0, Inf,'ArrayValued',true);
% % end
% end
% end
% toc

% save('pdf_values_HKdist_0a20','pdf_values','s','sigma','mu','A');
% load('pdf_values_HKdist')
%%
% Plotting the PDF
figure;
% plot(A, pdf_values(:,1,1), 'LineWidth', 2);
plot(A, pdf_values(:,1,1), 'LineWidth', 2);
xlabel('A');
ylabel('p_A(A)');
title('Homodyned K Distribution');
grid on;
%%

% pdf=pdf_values(:,1,5);
pdf=pdf_values;
x = A;

% Compute the CDF by integrating the PDF
cdf = cumtrapz(x, pdf);
figure;plot(x,cdf);grid on;

% Ensure that CDF values do not exceed 1 and are unique
cdf(cdf > 1) = 1; % Cap the CDF at 1
[cdf_unique, unique_indices] = unique(cdf);
x_unique = x(unique_indices); % Corresponding x values

% Generate uniform random samples
num_samples = 1e5; % Specify the number of samples you want
u = rand(num_samples, 1); % Uniformly distributed samples in (0,1)

% Clip uniform samples to the range of the CDF
u = min(max(u, 0), 1); % Ensure u is within [0, 1]

% Use interpolation to find the corresponding x values from the unique CDF
data = interp1(cdf_unique, x_unique, u, 'linear');
%%
% Plot the generated data
figure;
histogram(data, 'Normalization', 'pdf', 'BinWidth', 0.01); % Adjust BinWidth as needed
hold on;
plot(x, pdf, 'r-', 'LineWidth', 2); % Overlay original PDF
title('Generated Data vs Original PDF');
xlabel('x');
ylabel('Density');
legend('Generated Data', 'Original PDF');
grid on;

% datarescaled=rescale(data.^2,0,255,InputMin=0,InputMax=20);
% figure;
% histogram(datarescaled,Normalization='pdf',BinWidth=1);xlim([-10 265])