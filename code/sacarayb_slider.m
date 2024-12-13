% env_rf ya se tiene
global env_rf
% a_0 = 20/log(10);                      % Initial value for a
% b_0 = 0;                      % Initial value for b

% Create figure and initial plot
fig = figure;
y = a_0 * log(env_rf) + b_0;
subplot(1,2,1);imgHandle = imagesc(y);colorbar;colormap gray;
subplot(1,2,2);histHandle = histogram(y, 'Normalization', 'pdf');xlim([0 Inf])
% xlabel('z');
% ylabel('y = a * ln(z) + b');
title('Interactive imagesc of y = a * ln(z) + b');

% Slider for 'a'
aSlider = uicontrol('Style', 'slider', 'Min', -100, 'Max', 100, 'Value', a_0, ...
    'Units', 'normalized', 'Position', [0.2, 0.01, 0.25, 0.05]);
aLabel = uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.2, 0.02, 0.25, 0.02], ...
    'String', ['a = ' num2str(a_0)]);

% Slider for 'b'
bSlider = uicontrol('Style', 'slider', 'Min', -100, 'Max', 200, 'Value', b_0, ...
    'Units', 'normalized', 'Position', [0.55, 0.01, 0.25, 0.05]);
bLabel = uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.55, 0.02, 0.25, 0.02], ...
    'String', ['b = ' num2str(b_0)]);

guidata(fig, struct('aSlider', aSlider, 'bSlider', bSlider, 'aLabel', aLabel, 'bLabel', bLabel, 'histHandle', histHandle, 'imgHandle', imgHandle));

% Callback function to update the plot when sliders are moved
function updateFigure(src, ~)
    global env_rf
    % Access stored UI controls
    data = guidata(src);
    
    % Get current values of a and b from sliders
    a = data.aSlider.Value;
    b = data.bSlider.Value;
    
    % Update label texts
    data.aLabel.String = ['a = ' num2str(a)];
    data.bLabel.String = ['b = ' num2str(b)];
    
    % Recalculate y and update histogram
    y = a * log(env_rf) + b;
    set(data.histHandle, 'Data', y); % Update the histogram data
    set(data.imgHandle, 'CData', y); % Update the image data
end

% Set the callback functions for the sliders
aSlider.Callback = @updateFigure;
bSlider.Callback = @updateFigure;

keyboard;