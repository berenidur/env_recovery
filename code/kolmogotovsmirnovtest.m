close all;clear;clc;

sample1 = randn(50, 1);  % First sample (from normal distribution)
sample2 = randn(50, 1);  % Second sample (shifted normal distribution)

h=kstest2(sample1,sample2);
disp(h);