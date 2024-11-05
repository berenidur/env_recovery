function [a,b] = getab(env_rf)
%getab Get a and b.
%   The function gets a and b.
complog=log(env_rf);
complog=complog(:);
a=255/(max(complog)-min(complog));
% a=255/(max(complog)-(quantile(complog, 0.25) - 1.5 * iqr(complog)));
b=255-a*max(complog);
end