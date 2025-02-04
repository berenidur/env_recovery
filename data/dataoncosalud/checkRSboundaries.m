clear;close all;clc;
R=0:0.01:5;
S=0:0.01:5;
a=zeros(length(R),length(S));
k_mat=zeros(length(R),length(S));
beta_mat=zeros(length(R),length(S));

for i=1:length(R)
    if ~logical(mod(i,100)),disp(i);end
for j=1:length(S)
    try 
        [k, beta] = getkbeta(R(i), S(j));
        a(i,j)=1;
        k_mat(i,j)=k;
        beta_mat(i,j)=beta;
    catch exception
        a(i,j)=-1;
    end
end
end
%%
close all;
figure;imagesc(S,R,a);colorbar;%axis image;
xlabel('S');ylabel('R');axis xy;title('R,S with solution')
% set(gca, 'YDir','normal');

validkb=zeros(size(k_mat));
for n=1:size(k_mat,1)
for m=1:size(k_mat,2)
    k=k_mat(n,m);beta=beta_mat(n,m);
    if 0<=k && k<1 && 0<=beta && beta<0.25
        validkb(n,m)=1;
    end
end
end

figure;imagesc(S,R,validkb);colorbar;%axis image;
xlabel('S');ylabel('R');axis xy;title('Valid k,\beta');


figure;imagesc(S,R,validkb.*a);colorbar;%axis image;
xlabel('S');ylabel('R');axis xy;title('Final valid values of R,S');

