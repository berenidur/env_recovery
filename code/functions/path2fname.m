function fname = path2fname(path2file)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fname = path2file(find(path2file == '/' | path2file == '\', 1, 'last')+1:end);
end