classdef GUIControllerManager < handle
    properties (Access = private)
        % GUI要素
        mainFigure
        controlPanel
        visualPanel
        paramPanel
        labelPanel
        
        % コントロールボタン
        startButton
        stopButton
        pauseButton
        modeToggle
        
        % ラベル関連コンポーネント
        labelButtons
        
        % 表示パネル
        statusText
        timeDisplay
        signalDisplay
        
        % 表示制御
        displayControls
        
        % データ可視化
        plotHandles
        
        % 設定とコールバック
        params
        callbacks
        
        % 状態管理
        isRunning = false
        isPaused = false
        startTime
        updateTimer
        
        % 表示更新用
        lastUpdateTime
    end
    
    methods (Access = public)
        function obj = GUIControllerManager(params)
            obj.params = params;
            obj.initializeGUI();
            obj.setupTimers();
            obj.lastUpdateTime = 0;
        end
        
        function delete(obj)
            try
                if ~isempty(obj.updateTimer) && isvalid(obj.updateTimer)
                    stop(obj.updateTimer);
                    delete(obj.updateTimer);
                end
                if isvalid(obj.mainFigure)
                    delete(obj.mainFigure);
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function setCallbacks(obj, callbacks)
            obj.callbacks = callbacks;
        end
        
        function updateStatus(obj, status)
            set(obj.statusText, 'String', ['Status: ' status]);
            obj.updateButtonStates();
        end
        
        function updateMode(obj, mode)
            set(obj.modeToggle, 'Value', strcmpi(mode, 'online'));
            if strcmpi(mode, 'online')
                set(obj.modeToggle, 'String', 'Offline Mode');
            else
                set(obj.modeToggle, 'String', 'Online Mode');
            end
            obj.updateParameterPanel(mode);
        end
        
        function updateResults(obj, data)
            try
                if ~isempty(data)
                    % 各表示の有効/無効に基づいて更新
                    if isfield(data, 'rawData') && ...
                            obj.params.gui.display.visualization.enable.rawData
                        obj.updateRawDataPlot(data.rawData);
                    end
                    
                    if isfield(data, 'processedData') && ...
                            obj.params.gui.display.visualization.enable.processedData
                        obj.updateProcessedDataPlot(data.processedData);
                    end
                    
                    if isfield(data, 'spectrum') && ...
                            obj.params.gui.display.visualization.enable.spectrum
                        obj.updateSpectrumPlot(data.spectrum);
                    end
                    
                    if isfield(data, 'ersp') && ...
                            obj.params.gui.display.visualization.enable.ersp
                        obj.updateERSPPlot(data.ersp);
                    end
                    
                    drawnow;
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function showError(~, title, message)
            errordlg(message, title);
        end
    end
    
    methods (Access = private)
        function initializeGUI(obj)
            obj.mainFigure = figure('Name', 'EEG Acquisition Control', ...
                'Position', [100 100 1200 800], ...
                'MenuBar', 'none', ...
                'ToolBar', 'none', ...
                'NumberTitle', 'off', ...
                'CloseRequestFcn', @(~,~) obj.closeGUI());
            
            obj.createControlPanel();
            obj.createVisualizationPanel();
            obj.createParameterPanel();
            obj.createLabelPanel();
        end
        
        function createControlPanel(obj)
            obj.controlPanel = uipanel(obj.mainFigure, ...
                'Title', 'Control', ...
                'Position', [0.02 0.8 0.96 0.18]);
            
            % 基本コントロールボタン
            obj.startButton = uicontrol(obj.controlPanel, ...
                'Style', 'pushbutton', ...
                'String', 'Start', ...
                'Position', [20 80 100 30], ...
                'Callback', @obj.startButtonCallback);
            
            obj.stopButton = uicontrol(obj.controlPanel, ...
                'Style', 'pushbutton', ...
                'String', 'Stop', ...
                'Position', [130 80 100 30], ...
                'Enable', 'off', ...
                'Callback', @obj.stopButtonCallback);
            
            obj.pauseButton = uicontrol(obj.controlPanel, ...
                'Style', 'pushbutton', ...
                'String', 'Pause', ...
                'Position', [240 80 100 30], ...
                'Enable', 'off', ...
                'Callback', @obj.pauseButtonCallback);
            
            obj.modeToggle = uicontrol(obj.controlPanel, ...
                'Style', 'togglebutton', ...
                'String', 'Online Mode', ...
                'Position', [350 80 100 30], ...
                'Callback', @obj.modeToggleCallback);
            
            % 状態表示
            obj.statusText = uicontrol(obj.controlPanel, ...
                'Style', 'text', ...
                'String', 'Status: Ready', ...
                'Position', [20 20 200 30]);
            
            % 時間表示
            obj.timeDisplay = uicontrol(obj.controlPanel, ...
                'Style', 'text', ...
                'String', 'Time: 00:00:00', ...
                'Position', [230 20 200 30]);
        end
        
        function createVisualizationPanel(obj)
            obj.visualPanel = uipanel(obj.mainFigure, ...
                'Title', 'Visualization', ...
                'Position', [0.02 0.3 0.96 0.48]);
            
            obj.initializePlots();
        end
        
        
        function createParameterPanel(obj)
            % パラメータ設定パネルの作成
            obj.paramPanel = uipanel(obj.mainFigure, ...
                'Title', 'Display Control', ...
                'Position', [0.02 0.02 0.96 0.26]);

            % 左側：表示制御チェックボックスの作成
            displayNames = {'Raw Data', 'Processed Data', 'Spectrum', 'ERSP'};
            displayFields = {'rawData', 'processedData', 'spectrum', 'ersp'};
            obj.displayControls = struct();

            % チェックボックスの配置（位置を下に調整）
            for i = 1:length(displayNames)
                obj.displayControls.(displayFields{i}) = uicontrol(obj.paramPanel, ...
                    'Style', 'checkbox', ...
                    'String', displayNames{i}, ...
                    'Value', obj.params.gui.display.visualization.enable.(displayFields{i}), ...
                    'Position', [20 170-i*30 150 25], ...
                    'Callback', @(src,~) obj.displayControlChanged(displayFields{i}, src.Value));
            end

            % 中央：基本設定
            obj.displayControls.autoScale = uicontrol(obj.paramPanel, ...
                'Style', 'checkbox', ...
                'String', 'Auto Scaling', ...
                'Value', obj.params.gui.display.visualization.scale.auto, ...
                'Position', [200 140 150 25], ...
                'Callback', @(src,~) obj.autoScaleChanged(src.Value));

            obj.displayControls.showBands = uicontrol(obj.paramPanel, ...
                'Style', 'checkbox', ...
                'String', 'Show Frequency Bands', ...
                'Value', obj.params.gui.display.visualization.showBands, ...
                'Position', [200 110 150 25], ...
                'Callback', @(src,~) obj.showBandsChanged(src.Value));

            % 右側：詳細パラメータ設定
            paramNames = {'Time Window (s)', 'Freq Range (Hz)', 'Raw Scale (μV)', ...
                         'Power Scale (μV²/Hz)', 'ERSP Time (s)', 'ERSP Freq (Hz)'};
            paramFields = {'timeWindow', 'freqRange', 'rawScale', ...
                          'powerScale', 'erspTime', 'erspFreq'};
                      
            defaultValues = {...
                obj.params.gui.display.visualization.scale.displaySeconds, ...  % Time Window
                obj.params.gui.display.visualization.scale.freq, ...               % Freq Range
                obj.params.gui.display.visualization.scale.raw, ...               % Raw Scale
                obj.params.gui.display.visualization.scale.power, ...             % Power Scale
                obj.params.gui.display.visualization.ersp.time, ...              % ERSP Time
                obj.params.gui.display.visualization.ersp.freqRange ...          % ERSP Freq
            };

            % 入力フィールドの作成部分を修正
            for i = 1:length(paramNames)
                % ラベル
                uicontrol(obj.paramPanel, ...
                    'Style', 'text', ...
                    'String', paramNames{i}, ...
                    'Position', [380 165-i*25 120 20]);

                if strcmp(paramFields{i}, 'timeWindow')
                    % 時間窓は1つの入力フィールドのみ
                    obj.displayControls.timeWindow = uicontrol(obj.paramPanel, ...
                        'Style', 'edit', ...
                        'String', num2str(obj.params.gui.display.visualization.scale.displaySeconds), ...
                        'Position', [510 165-i*25 50 20], ...
                        'Callback', @(src,~) obj.timeWindowChanged(str2double(src.String)));
                else
                    % その他のパラメータは最小値と最大値の入力フィールド
                    obj.displayControls.([paramFields{i} 'Min']) = uicontrol(obj.paramPanel, ...
                        'Style', 'edit', ...
                        'String', num2str(defaultValues{i}(1)), ...
                        'Position', [510 165-i*25 50 20], ...
                        'Callback', @(src,~) obj.parameterValueChanged(paramFields{i}, 'min', str2double(src.String)));

                    obj.displayControls.([paramFields{i} 'Max']) = uicontrol(obj.paramPanel, ...
                        'Style', 'edit', ...
                        'String', num2str(defaultValues{i}(2)), ...
                        'Position', [570 165-i*25 50 20], ...
                        'Callback', @(src,~) obj.parameterValueChanged(paramFields{i}, 'max', str2double(src.String)));
                end
            end

            % カラーマップ設定
            uicontrol(obj.paramPanel, ...
                'Style', 'text', ...
                'String', 'Colormap', ...
                'Position', [650 150 80 20]);

            obj.displayControls.colormapType = uicontrol(obj.paramPanel, ...
                'Style', 'popupmenu', ...
                'String', {'jet', 'parula', 'hot', 'cool'}, ...
                'Position', [650 120 80 25], ...
                'Callback', @(src,~) obj.colormapTypeChanged(src.String{src.Value}));

            obj.displayControls.colormapReverse = uicontrol(obj.paramPanel, ...
                'Style', 'checkbox', ...
                'String', 'Reverse', ...
                'Value', obj.params.gui.display.visualization.ersp.colormap.reverse, ...
                'Position', [650 80 80 25], ...
                'Callback', @(src,~) obj.colormapReverseChanged(src.Value));
        end
        
        function createLabelPanel(obj)
            obj.labelPanel = uipanel(obj.mainFigure, ...
                'Title', 'Labels', ...
                'Position', [0.8 0.02 0.18 0.2]);
            
            mappings = obj.params.udp.receive.triggers.mappings;
            mappingFields = fieldnames(mappings);
            numMappings = length(mappingFields);
            
            labelNames = cell(numMappings, 1);
            labelValues = zeros(numMappings, 1);
            for i = 1:numMappings
                labelNames{i} = mappings.(mappingFields{i}).text;
                labelValues(i) = mappings.(mappingFields{i}).value;
            end
            
            buttonHeight = 25;
            buttonSpacing = 5;
            totalButtonsHeight = numMappings * (buttonHeight + buttonSpacing);
            
            panelHeight = totalButtonsHeight + 40;
            obj.labelPanel.Position = [0.8 0.02 0.18 panelHeight/800];
            
            obj.labelButtons = cell(numMappings, 1);
            for i = 1:numMappings
                yPos = totalButtonsHeight - (i-1)*(buttonHeight + buttonSpacing);
                obj.labelButtons{i} = uicontrol(obj.labelPanel, ...
                    'Style', 'pushbutton', ...
                    'String', labelNames{i}, ...
                    'Position', [10 yPos-buttonHeight 140 buttonHeight], ...
                    'Callback', @(~,~)obj.onLabelButtonClick(labelValues(i)));
            end
        end
        
        % 時間窓変更用の新しいコールバック関数
        function timeWindowChanged(obj, value)
            try
                % 時間窓の制限（0.1秒から30秒）
                value = max(0.1, min(30, value));

                % 値を更新
                obj.params.gui.display.visualization.scale.displaySeconds = value;

                % 入力フィールドを更新
                set(obj.displayControls.timeWindow, 'String', num2str(value));

                % 表示を更新
                obj.updateDisplay();

            catch ME
                % エラーメッセージを表示
                errordlg(['Invalid input: ' ME.message], 'Parameter Error');
                % 入力フィールドを前の有効な値に戻す
                set(obj.displayControls.timeWindow, 'String', ...
                    num2str(obj.params.gui.display.visualization.scale.displaySeconds));
            end
        end
        
        function parameterValueChanged(obj, field, type, value)
            try
                % 入力値の検証
                switch field
                    case 'timeWindow'
                        % 時間窓の制限（例：最小0.1秒，最大30秒）
                        if strcmp(type, 'min')
                            value = max(0.1, value);  % 最小値を0.1秒に制限
                            value = min(value, obj.params.gui.display.visualization.scale.displaySeconds);  % 最大値以下に制限
                        else
                            value = max(value, 0.1);  % 最小0.1秒
                            value = max(value, obj.params.gui.display.visualization.scale.displaySeconds);  % 最小値以上
                        end

                    case {'freqRange', 'erspFreq'}
                        % 周波数の制限（例：0-500Hz）
                        value = max(0, value);  % 負の周波数を防止
                        value = min(500, value);  % 最大周波数を制限

                    case 'rawScale'
                        % スケールの制限（例：±1000μV）
                        value = max(-1000, min(1000, value));

                    case 'powerScale'
                        % パワーの制限（常に正の値）
                        value = max(0.000001, value);  % 最小値を制限

                    case 'erspTime'
                        % ERSP時間の制限
                        value = max(-10, min(10, value));  % 例：±10秒に制限
                end

                % 検証済みの値を適用
                switch field
                    case 'timeWindow'
                        if strcmp(type, 'min')
                            % 開始時間は固定で0にする
                            set(obj.displayControls.timeWindowMin, 'String', '0');
                            return;
                        else
                            obj.params.gui.display.visualization.scale.displaySeconds = value;
                        end
                    case 'freqRange'
                        obj.params.gui.display.visualization.scale.freq(strcmp(type, 'max')+1) = value;
                    case 'rawScale'
                        obj.params.gui.display.visualization.scale.raw(strcmp(type, 'max')+1) = value;
                    case 'powerScale'
                        obj.params.gui.display.visualization.scale.power(strcmp(type, 'max')+1) = value;
                    case 'erspTime'
                        obj.params.gui.display.visualization.ersp.time(strcmp(type, 'max')+1) = value;
                    case 'erspFreq'
                        obj.params.gui.display.visualization.ersp.freqRange(strcmp(type, 'max')+1) = value;
                end

                % 入力フィールドを更新された値で更新
                if strcmp(type, 'min')
                    set(obj.displayControls.([field 'Min']), 'String', num2str(value));
                else
                    set(obj.displayControls.([field 'Max']), 'String', num2str(value));
                end

                % 表示を更新
                obj.updateDisplay();

            catch ME
                % エラーメッセージを表示
                errordlg(['Invalid input: ' ME.message], 'Parameter Error');
                % 入力フィールドを前の有効な値に戻す
                if strcmp(type, 'min')
                    set(obj.displayControls.([field 'Min']), 'String', num2str(obj.getLastValidValue(field, 'min')));
                else
                    set(obj.displayControls.([field 'Max']), 'String', num2str(obj.getLastValidValue(field, 'max')));
                end
            end
        end
        
        function value = getLastValidValue(obj, field, type)
            switch field
                case 'timeWindow'
                    if strcmp(type, 'min')
                        value = 0;
                    else
                        value = obj.params.gui.display.visualization.scale.displaySeconds;
                    end
                case 'freqRange'
                    value = obj.params.gui.display.visualization.scale.freq(strcmp(type, 'max')+1);
                case 'rawScale'
                    value = obj.params.gui.display.visualization.scale.raw(strcmp(type, 'max')+1);
                case 'powerScale'
                    value = obj.params.gui.display.visualization.scale.power(strcmp(type, 'max')+1);
                case 'erspTime'
                    value = obj.params.gui.display.visualization.ersp.time(strcmp(type, 'max')+1);
                case 'erspFreq'
                    value = obj.params.gui.display.visualization.ersp.freqRange(strcmp(type, 'max')+1);
            end
        end
        
        function initializePlots(obj)
            obj.plotHandles.rawData = subplot(2, 2, 1, 'Parent', obj.visualPanel);
            title(obj.plotHandles.rawData, 'Raw EEG');
            xlabel(obj.plotHandles.rawData, 'Time (s)');
            ylabel(obj.plotHandles.rawData, 'Amplitude (μV)');
            grid(obj.plotHandles.rawData, 'on');
            
            obj.plotHandles.processed = subplot(2, 2, 2, 'Parent', obj.visualPanel);
            title(obj.plotHandles.processed, 'Processed Signal');
            xlabel(obj.plotHandles.processed, 'Time (s)');
            ylabel(obj.plotHandles.processed, 'Amplitude (μV)');
            grid(obj.plotHandles.processed, 'on');
            
            obj.plotHandles.spectrum = subplot(2, 2, 3, 'Parent', obj.visualPanel);
            title(obj.plotHandles.spectrum, 'Power Spectrum');
            xlabel(obj.plotHandles.spectrum, 'Frequency (Hz)');
            ylabel(obj.plotHandles.spectrum, 'Power (μV²/Hz)');
            grid(obj.plotHandles.spectrum, 'on');
            
            obj.plotHandles.ersp = subplot(2, 2, 4, 'Parent', obj.visualPanel);
            title(obj.plotHandles.ersp, 'ERSP');
            xlabel(obj.plotHandles.ersp, 'Time (s)');
            ylabel(obj.plotHandles.ersp, 'Frequency (Hz)');
            colorbar(obj.plotHandles.ersp);
            grid(obj.plotHandles.ersp, 'on');
        end
        
        function setupTimers(obj)
            obj.updateTimer = timer(...
                'ExecutionMode', 'fixedRate', ...
                'Period', obj.params.gui.display.visualization.refreshRate, ...
                'TimerFcn', @(~,~) obj.updateDisplay());
            start(obj.updateTimer);
        end
        
        function updateDisplay(obj)
            if obj.isRunning
                elapsed = toc(obj.startTime);
                timeStr = sprintf('Time: %02d:%02d:%02d', ...
                    floor(elapsed/3600), ...
                    mod(floor(elapsed/60), 60), ...
                    mod(floor(elapsed), 60));
                set(obj.timeDisplay, 'String', timeStr);
            end
        end
        
        function updateRawDataPlot(obj, data)
            if ~isempty(data) && obj.params.gui.display.visualization.enable.rawData
                displaySeconds = obj.params.gui.display.visualization.scale.displaySeconds;
                windowSize = displaySeconds * obj.params.device.sampleRate;
                
                if size(data, 2) > windowSize
                    displayData = data(:, end-windowSize+1:end);
                else
                    displayData = data;
                end
                
                timeAxis = (0:size(displayData,2)-1) / obj.params.device.sampleRate;
                
                plot(obj.plotHandles.rawData, timeAxis, displayData(1,:));
                xlabel(obj.plotHandles.rawData, 'Time (s)');
                ylabel(obj.plotHandles.rawData, 'Amplitude (μV)');
                
                if obj.params.gui.display.visualization.scale.auto
                    axis(obj.plotHandles.rawData, 'auto');
                else
                    ylim(obj.plotHandles.rawData, obj.params.gui.display.visualization.scale.raw);
                end
                
                xlim(obj.plotHandles.rawData, [0 displaySeconds]);
                grid(obj.plotHandles.rawData, 'on');
                title(obj.plotHandles.rawData, sprintf('Raw EEG (Channel 1) - Window: %ds', displaySeconds));
            end
        end
        
        function updateProcessedDataPlot(obj, data)
            if ~isempty(data) && obj.params.gui.display.visualization.enable.processedData
                displaySeconds = obj.params.gui.display.visualization.scale.displaySeconds;
                windowSize = displaySeconds * obj.params.device.sampleRate;
                
                if iscell(data)
                    displayData = data{end};
                else
                    if size(data, 2) > windowSize
                        displayData = data(:, end-windowSize+1:end);
                    else
                        displayData = data;
                    end
                end
                
                timeAxis = (0:size(displayData,2)-1) / obj.params.device.sampleRate;
                
                plot(obj.plotHandles.processed, timeAxis, displayData(1,:));
                xlabel(obj.plotHandles.processed, 'Time (s)');
                ylabel(obj.plotHandles.processed, 'Amplitude (μV)');
                
                if obj.params.gui.display.visualization.scale.auto
                    axis(obj.plotHandles.processed, 'auto');
                else
                    ylim(obj.plotHandles.processed, obj.params.gui.display.visualization.scale.processed);
                end
                
                xlim(obj.plotHandles.processed, [0 displaySeconds]);
                grid(obj.plotHandles.processed, 'on');
                title(obj.plotHandles.processed, sprintf('Processed EEG (Channel 1) - Window: %ds', displaySeconds));
            end
        end
        
        function updateSpectrumPlot(obj, data)
            if ~isempty(data) && obj.params.gui.display.visualization.enable.spectrum
                if isfield(data, 'pxx') && isfield(data, 'f')
                    semilogy(obj.plotHandles.spectrum, data.f, data.pxx);
                    xlabel(obj.plotHandles.spectrum, 'Frequency (Hz)');
                    ylabel(obj.plotHandles.spectrum, 'Power (μV²/Hz)');
                    
                    if obj.params.gui.display.visualization.scale.auto
                        axis(obj.plotHandles.spectrum, 'auto');
                    else
                        xlim(obj.plotHandles.spectrum, obj.params.gui.display.visualization.scale.freq);
                        if isfield(obj.params.gui.display.visualization.scale, 'power')
                            ylim(obj.plotHandles.spectrum, obj.params.gui.display.visualization.scale.power);
                        end
                    end
                    
                    grid(obj.plotHandles.spectrum, 'on');
                    title(obj.plotHandles.spectrum, 'Power Spectrum');
                    
                    if obj.params.gui.display.visualization.showBands
                        hold(obj.plotHandles.spectrum, 'on');
                        bands = obj.params.feature.power.bands;
                        bandNames = obj.params.feature.power.bands.names;
                        
                        for i = 1:length(bandNames)
                            bandName = bandNames{i};
                            if isfield(bands, bandName)
                                bandRange = bands.(bandName);
                                plot(obj.plotHandles.spectrum, ...
                                    [bandRange(1) bandRange(1)], get(obj.plotHandles.spectrum, 'YLim'), '--k');
                                plot(obj.plotHandles.spectrum, ...
                                    [bandRange(2) bandRange(2)], get(obj.plotHandles.spectrum, 'YLim'), '--k');
                                text(mean(bandRange), max(get(obj.plotHandles.spectrum, 'YLim'))*0.9, ...
                                    bandName, 'HorizontalAlignment', 'center');
                            end
                        end
                        hold(obj.plotHandles.spectrum, 'off');
                    end
                end
            end
        end
        
        function updateERSPPlot(obj, data)
            try
                if ~isempty(data) && obj.params.gui.display.visualization.enable.ersp
                    if isfield(data, 'ersp')
                        erspParams = obj.params.gui.display.visualization.ersp;
                        
                        imagesc(obj.plotHandles.ersp, data.times, data.freqs, data.ersp);
                        
                        if obj.params.gui.display.visualization.scale.auto
                            clim('auto');
                        else
                            clim(obj.plotHandles.ersp, erspParams.scale);
                        end
                        
                        colormap(obj.plotHandles.ersp, erspParams.colormap.type);
                        if erspParams.colormap.reverse
                            colormap(obj.plotHandles.ersp, flipud(colormap(obj.plotHandles.ersp)));
                        end
                        
                        set(obj.plotHandles.ersp, 'Color', erspParams.colormap.background);
                        
                        axis(obj.plotHandles.ersp, 'xy');
                        if ~obj.params.gui.display.visualization.scale.auto
                            xlim(obj.plotHandles.ersp, erspParams.time);
                            ylim(obj.plotHandles.ersp, erspParams.freqRange);
                        end
                        
                        colorbar(obj.plotHandles.ersp);
                        xlabel(obj.plotHandles.ersp, 'Time (s)');
                        ylabel(obj.plotHandles.ersp, 'Frequency (Hz)');
                        title(obj.plotHandles.ersp, 'Event-Related Spectral Perturbation');
                        
                        if obj.params.gui.display.visualization.showBands
                            obj.drawFrequencyBands(obj.plotHandles.ersp, erspParams);
                        end
                    end
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        % コールバック関数
        function drawFrequencyBands(obj, axHandle, erspParams)
            hold(axHandle, 'on');
            bands = obj.params.feature.power.bands;
            bandNames = obj.params.feature.power.bands.names;
            
            xRange = get(axHandle, 'XLim');
            lineColor = 'w';
            textColor = 'w';
            if strcmp(erspParams.colormap.background, 'white')
                lineColor = 'k';
                textColor = 'k';
            end
            
            for i = 1:length(bandNames)
                bandName = bandNames{i};
                if isfield(bands, bandName)
                    bandRange = bands.(bandName);
                    plot(axHandle, xRange, [bandRange(1) bandRange(1)], ...
                        '--', 'Color', lineColor, 'LineWidth', 0.5);
                    plot(axHandle, xRange, [bandRange(2) bandRange(2)], ...
                        '--', 'Color', lineColor, 'LineWidth', 0.5);
                    text(xRange(2), mean(bandRange), [' ' bandName], ...
                        'Color', textColor, ...
                        'HorizontalAlignment', 'left', ...
                        'VerticalAlignment', 'middle');
                end
            end
            hold(axHandle, 'off');
        end
        
        function colormapTypeChanged(obj, type)
            % カラーマップタイプの変更を処理
            obj.params.gui.display.visualization.ersp.colormap.type = type;
            % ERSPの表示を更新
            if obj.params.gui.display.visualization.enable.ersp
                obj.updateDisplay();
            end
        end

        function colormapReverseChanged(obj, value)
            % カラーマップの反転設定を処理
            obj.params.gui.display.visualization.ersp.colormap.reverse = value;
            % ERSPの表示を更新
            if obj.params.gui.display.visualization.enable.ersp
                obj.updateDisplay();
            end
        end
        
        function displayControlChanged(obj, field, value)
            obj.params.gui.display.visualization.enable.(field) = value;
            obj.updateDisplay();
        end
        
        function autoScaleChanged(obj, value)
            obj.params.gui.display.visualization.scale.auto = value;
            obj.updateDisplay();
        end
        
        function showBandsChanged(obj, value)
            obj.params.gui.display.visualization.showBands = value;
            obj.updateDisplay();
        end
        
        function startButtonCallback(obj, ~, ~)
            if ~obj.isRunning && isfield(obj.callbacks, 'onStart')
                obj.isRunning = true;
                obj.isPaused = false;
                obj.startTime = tic;
                obj.updateButtonStates();
                obj.callbacks.onStart();
            end
        end
        
        function stopButtonCallback(obj, ~, ~)
            if obj.isRunning && isfield(obj.callbacks, 'onStop')
                obj.isRunning = false;
                obj.isPaused = false;
                obj.updateButtonStates();
                obj.callbacks.onStop();
            end
        end
        
        function pauseButtonCallback(obj, ~, ~)
            if obj.isRunning
                if obj.isPaused && isfield(obj.callbacks, 'onResume')
                    obj.isPaused = false;
                    obj.updateButtonStates();
                    obj.callbacks.onResume();
                elseif ~obj.isPaused && isfield(obj.callbacks, 'onPause')
                    obj.isPaused = true;
                    obj.updateButtonStates();
                    obj.callbacks.onPause();
                end
            end
        end
        
        function modeToggleCallback(obj, ~, ~)
            if get(obj.modeToggle, 'Value')
                mode = 'online';
                set(obj.modeToggle, 'String', 'Offline Mode');
            else
                mode = 'offline';
                set(obj.modeToggle, 'String', 'Online Mode');
            end
            
            if isfield(obj.callbacks, 'onModeChange')
                obj.callbacks.onModeChange(mode);
            end
        end
        
        function onLabelButtonClick(obj, labelValue)
            if obj.isRunning && ~obj.isPaused && ...
                    ~isempty(obj.callbacks) && isfield(obj.callbacks, 'onLabel')
                currentTime = toc(obj.startTime);
                trigger = struct(...
                    'value', labelValue, ...
                    'time', uint64(currentTime * 1000), ...
                    'sample', []);

                % トリガー値に対応するテキストを取得
                mappings = obj.params.udp.receive.triggers.mappings;
                fields = fieldnames(mappings);
                triggerText = '';

                % マッピングから対応するテキストを検索
                for i = 1:length(fields)
                    if mappings.(fields{i}).value == labelValue
                        triggerText = mappings.(fields{i}).text;
                        break;
                    end
                end

                % デバッグ用の出力（UDPManagerと同じスタイル）
                fprintf('Label button pressed: %s (value: %d)\n', triggerText, labelValue);

                % コールバック実行
                obj.callbacks.onLabel(trigger);
            end
        end
        
        function updateButtonStates(obj)
            if obj.isRunning
                set(obj.startButton, 'Enable', 'off');
                set(obj.stopButton, 'Enable', 'on');
                set(obj.pauseButton, 'Enable', 'on');
                set(obj.modeToggle, 'Enable', 'off');
            else
                set(obj.startButton, 'Enable', 'on');
                set(obj.stopButton, 'Enable', 'off');
                set(obj.pauseButton, 'Enable', 'off');
                set(obj.modeToggle, 'Enable', 'on');
            end
            
            if obj.isPaused
                set(obj.pauseButton, 'String', 'Resume');
            else
                set(obj.pauseButton, 'String', 'Pause');
            end
        end
        
        function closeGUI(obj)
            if obj.isRunning
                obj.stopButtonCallback();
            end
            
            if ~isempty(obj.updateTimer) && isvalid(obj.updateTimer)
                stop(obj.updateTimer);
                delete(obj.updateTimer);
            end
            
            delete(obj.mainFigure);
        end
        
        function updateParameterPanel(obj, mode)
            if strcmpi(mode, 'online')
                set(obj.paramPanel, 'Title', 'Online Display Control');
            else
                set(obj.paramPanel, 'Title', 'Offline Display Control');
            end
        end
    end
end