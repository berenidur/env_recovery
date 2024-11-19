function [a_seabra2008,y] = est_seabra2008(z,a_0,b_0)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
sigma=std(z(:));
% mu=mean(z(:));
% gamma=double(eulergamma);
% psi=7000;

a_seabra2008=sqrt(24/pi^2*sigma^2);
% beta=mu-alpha/2*(log(2*psi)-gamma);
beta=b_0;

y=exp((z-beta)/a_seabra2008);
end