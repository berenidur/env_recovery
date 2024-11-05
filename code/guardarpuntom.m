function puntom = guardarpuntom(mfilePath)
%retorna el código del .m
if contains(mfilePath,'LiveEditorEvaluationHelper')
    mfilePath = matlab.desktop.editor.getActiveFilename;
end
if mfilePath(end-1:end) ~= ".m"
    mfilePath=[mfilePath,'.m'];
end
puntom = fileread(mfilePath);
end