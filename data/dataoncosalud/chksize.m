close all;clear;clc;
addpath('../../code/functions/');

fcell={'breast\bmode\','ganglio\bmode\'};

H=[];

for k=1:length(fcell)
f=fcell{k};
lstfiles=ls([f,'*.mat']);

for i=1:size(lstfiles,1)
fname=strtrim(lstfiles(i,:));
load([f,fname]);
h=size(comp_env,1);
H=[H,h];
end
end
%%
disp(max(H)); % 5216
disp(min(H)); % 1200