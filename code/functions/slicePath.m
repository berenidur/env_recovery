function result = slicePath(inputStr, numBackslashes)
    % Set default value for numBackslashes if not provided
    if nargin < 2
        numBackslashes = 1;
    end
    
    % Replace '\' with '/' in the string
    inputStr = strrep(inputStr, '\', '/');
    
    % Find positions of '/'
    idx = strfind(inputStr, '/');
    
    % Check if there are enough slashes
    if length(idx) >= numBackslashes
        % Get the position of the nth last '/'
        pos = idx(end - numBackslashes + 1);
        % Extract the substring after this position
        result = inputStr(pos+1:end);
    else
        % If there are fewer slashes than requested, return the whole string
        result = inputStr;
    end
end
