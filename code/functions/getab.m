function [a,b,comp_env] = getab(env_rf)
%getab Get a and b.
%   The function gets a and b for envelope log compression: a*log(env_rf)+b
% 
%   Syntax
%     [a,b,comp_env] = getab(env_rf)
%   
%   Output
%     a,b - Coefficients
%     comp_env - Log compressed envelope
complog=log(env_rf);
complog_v=complog(:);
% a=255/(max(complog_v)-min(complog_v));
% a=255/(max(complog)-(quantile(complog, 0.25) - 1.5 * iqr(complog)));
a=255/(max(complog_v)-graythresh(complog_v/max(complog_v))*max(complog_v));
% a=255/(max(complog_v)-min(complog_v)*0.5);
b=255-a*max(complog_v);
comp_env=a*complog+b;
end