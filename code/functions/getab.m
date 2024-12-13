function [a,b,comp_env] = getab(env_rf,method)
%getab Get a and b.
%   The function gets a and b for envelope log compression: a*log(env_rf)+b
% 
%   Syntax
%     [a,b,comp_env] = getab(env_rf)
%   
%   Output
%     a,b - Coefficients
%     comp_env - Log compressed envelope
if nargin<2
    method=1;
end
complog=log(env_rf);
complog_v=complog(:);

maxcl=max(complog_v);mincl=min(complog_v);
switch method
    case 1
        lowthresh=mincl;
    case 2
        lowthresh=(quantile(complog_v, 0.25) - 1.5 * iqr(complog_v));
    case 3
        lowthresh=graythresh(complog_v/maxcl)*maxcl;
    case 4
        lowthresh=mincl*0.5;
    case 5
        % histogram
        [N,edges] = histcounts(complog_v);
        [M,I] = max(N);
        N=N(1:I);edges=edges(1:I);

        [~,I] = min(abs(N-0.05*M));
        lowthresh = edges(I);
    otherwise
        ME = MException('MyComponent:noSuchVariable', ...
            'Method %s not found',method);
        throw(ME)
end



a=255/(maxcl-lowthresh);
b=255-a*maxcl;
comp_env=a*complog+b;
end