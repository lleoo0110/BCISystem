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
       
        % モード表示用のテキスト
        modeDisplay

        % ラベル関連コンポーネント
        labelButtons

        % 表示パネル
        statusText
        timeDisplay

        % スライダー
        sliderFigure
        slider
        sliderValue
        sliderText

        % 表示制御
        displayControls

        % データ可視化
        plotHandles
        displayParams     % 表示パラメータ

        % 設定とコールバック
        params
        callbacks

        % 状態管理
        isRunning = false
        isPaused = false
        startTime
        updateTimer
    end
    
    methods (Access = public)
        function obj = GUIControllerManager(params)
            obj.params = params;

            % plotHandlesの初期化を最初に行う
            obj.plotHandles = struct('rawData', [], 'emgData', [], 'spectrum', [], 'ersp', []);

            % GUIの初期化
            obj.initializeGUI();

            % プロットの初期化を明示的に行う
            obj.initializePlots();

            % その他の初期化
            obj.setupTimers();
            
            % スライダーGUIの初期化（有効な場合のみ）
            if obj.params.gui.slider.enable
                obj.createSliderWindow();
            end
        end
        
        function delete(obj)
            try
                % タイマーの停止と削除
                if ~isempty(obj.updateTimer)
                    if isa(obj.updateTimer, 'timer') && isvalid(obj.updateTimer)
                        stop(obj.updateTimer);
                        delete(obj.updateTimer);
                    end
                    obj.updateTimer = [];
                end

                % メインウィンドウの削除
                if ~isempty(obj.mainFigure) && isvalid(obj.mainFigure)
                    delete(obj.mainFigure);
                end

                % スライダーウィンドウの削除
                if ~isempty(obj.sliderFigure) && isvalid(obj.sliderFigure)
                    delete(obj.sliderFigure);
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
        
        function updateDisplayData(obj, displayData)
            if ~isvalid(obj.mainFigure) || ~ishandle(obj.mainFigure)
                return;  % GUIが既に閉じられている場合は更新をスキップ
            end

            try
                if isempty(displayData)
                    return;
                end

                % プロットハンドルの有効性チェック
                if obj.params.gui.display.visualization.enable.rawData && ...
                   isfield(displayData, 'rawData') && ...
                   isvalid(obj.plotHandles.rawData)
                    obj.updateRawDataPlot(displayData.rawData);
                end

                if obj.params.gui.display.visualization.enable.emgData && ...
                   isfield(displayData, 'emgData') && ...
                   isvalid(obj.plotHandles.emg)
                    obj.updateEMGPlot(displayData.emgData);
                end

                if obj.params.gui.display.visualization.enable.spectrum && ...
                   isfield(displayData, 'spectrum') && ...
                   isvalid(obj.plotHandles.spectrum)
                    obj.updateSpectrumPlot(displayData.spectrum);
                end

                if obj.params.gui.display.visualization.enable.ersp && ...
                   isfield(displayData, 'ersp') && ...
                   isvalid(obj.plotHandles.ersp)
                    obj.updateERSPPlot(displayData.ersp);
                end

                drawnow limitrate;

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end

        function showError(~, title, message)
            errordlg(message, title);
        end
        
        function closeAllWindows(obj)
            try
                % タイマーの停止と削除
                if ~isempty(obj.updateTimer) && isvalid(obj.updateTimer)
                    stop(obj.updateTimer);
                    delete(obj.updateTimer);
                end

                % メインウィンドウの削除
                if ~isempty(obj.mainFigure) && isvalid(obj.mainFigure)
                    % GUIコールバックを無効化
                    set(obj.mainFigure, 'CloseRequestFcn', []);
                    controls = findobj(obj.mainFigure, 'Type', 'uicontrol');
                    set(controls, 'Callback', []);

                    % メインウィンドウを削除
                    delete(obj.mainFigure);
                end

                % スライダーウィンドウの削除
                if ~isempty(obj.sliderFigure) && isvalid(obj.sliderFigure)
                    delete(obj.sliderFigure);
                end

                % プロットハンドルのリセット
                if isfield(obj, 'plotHandles') && isstruct(obj.plotHandles)
                    fields = fieldnames(obj.plotHandles);
                    for i = 1:length(fields)
                        if ishandle(obj.plotHandles.(fields{i}))
                            delete(obj.plotHandles.(fields{i}));
                        end
                    end
                    obj.plotHandles = struct();
                end

            catch ME
                warning(ME.identifier, 'Error in closeAllWindows: %s', ME.message);
            end
        end
        
        function createSliderWindow(obj)
            if ~obj.params.gui.slider.enable || (~isempty(obj.sliderFigure))
                return;  % すでにウィンドウが存在する場合は作成しない
            end

            % スライダーウィンドウの作成
            obj.sliderFigure = figure('Name', obj.params.gui.slider.title, ...
                'Position', obj.params.gui.slider.position, ...
                'MenuBar', 'none', ...
                'ToolBar', 'none', ...
                'NumberTitle', 'off');

            % スライダーの作成
            stepSize = 1/(obj.params.gui.slider.steps-1);
            obj.slider = uicontrol(obj.sliderFigure, ...
                'Style', 'slider', ...
                'Min', obj.params.gui.slider.minValue, ...
                'Max', obj.params.gui.slider.maxValue, ...
                'Value', obj.params.gui.slider.defaultValue, ...
                'Position', [50 80 300 20], ...
                'SliderStep', [stepSize stepSize], ...
                'Callback', @obj.sliderCallback);

            % スライダー値の表示テキスト
            obj.sliderText = uicontrol(obj.sliderFigure, ...
                'Style', 'text', ...
                'Position', [150 120 100 30], ...
                'String', num2str(obj.params.gui.slider.defaultValue));

            % 「不快」のラベル（左端）
            uicontrol(obj.sliderFigure, ...
                'Style', 'text', ...
                'Position', [20 80 30 20], ...
                'String', '不快（非覚醒）', ...
                'HorizontalAlignment', 'left');

            % 「快」のラベル（右端）
            uicontrol(obj.sliderFigure, ...
                'Style', 'text', ...
                'Position', [350 80 30 20], ...
                'String', '快（覚醒）', ...
                'HorizontalAlignment', 'right');

            % スライダー値の初期化
            obj.sliderValue = obj.params.gui.slider.defaultValue;
        end

        function value = getSliderValue(obj)
            if obj.params.gui.slider.enable
                value = round(obj.sliderValue);
            else
                value = [];  % スライダーが無効な場合は空を返す
            end
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
            
            obj.modeDisplay = uicontrol(obj.controlPanel, ...
                'Style', 'text', ...
                'String', sprintf('Mode: %s', upper(obj.params.acquisition.mode)), ...
                'Position', [350 75 100 30]);
            
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
            try
                obj.visualPanel = uipanel(obj.mainFigure, ...
                    'Title', 'Visualization', ...
                    'Position', [0.02 0.3 0.96 0.48]);
                
            catch ME
                fprintf('Error in createVisualizationPanel: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        
        function createParameterPanel(obj)
            % パラメータ設定パネルの作成
            obj.paramPanel = uipanel(obj.mainFigure, ...
                'Title', 'Display Control', ...
                'Position', [0.02 0.02 0.96 0.26]);
            
            % 左側：表示制御チェックボックスの作成
            displayNames = {'Raw Data', 'EMG Data', 'Spectrum', 'ERSP'};
            displayFields = {'rawData', 'emgData', 'spectrum', 'ersp'};
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
            
            % 安静状態の閾値調整セクション
            uicontrol(obj.paramPanel, ...
                'Style', 'text', ...
                'String', 'Classifier Threshold', ...
                'Position', [380 40 120 20]);

            % 閾値入力ボックス
            obj.displayControls.restThreshold = uicontrol(obj.paramPanel, ...
                'Style', 'edit', ...
                'String', num2str(obj.params.classifier.threshold), ...
                'Position', [510 40 50 20], ...
                'Callback', @(src,~) obj.restThresholdChanged(str2double(src.String)));
            
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
                        value = max(-1000, min(5000, value));
                        
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
            % plotHandlesの初期化確認
            if ~isfield(obj, 'plotHandles') || ~isstruct(obj.plotHandles)
                obj.plotHandles = struct();
            end

            try
                % 生脳波データのプロット
                obj.plotHandles.rawData = subplot(2, 2, 1, 'Parent', obj.visualPanel);
                if ~ishandle(obj.plotHandles.rawData)
                    error('Failed to create raw data plot handle');
                end
                title(obj.plotHandles.rawData, 'Raw EEG');
                xlabel(obj.plotHandles.rawData, 'Time (s)');
                ylabel(obj.plotHandles.rawData, 'Amplitude (μV)');
                grid(obj.plotHandles.rawData, 'on');

                % EMGデータのプロット
                obj.plotHandles.emg = subplot(2, 2, 2, 'Parent', obj.visualPanel);
                if ~ishandle(obj.plotHandles.emg)
                    error('Failed to create EMG signal plot handle');
                end
                title(obj.plotHandles.emg, 'EMG Signal');
                xlabel(obj.plotHandles.emg, 'Time (s)');
                ylabel(obj.plotHandles.emg, 'Amplitude (μV)');
                grid(obj.plotHandles.emg, 'on');

                % パワースペクトルのプロット
                obj.plotHandles.spectrum = subplot(2, 2, 3, 'Parent', obj.visualPanel);
                if ~ishandle(obj.plotHandles.spectrum)
                    error('Failed to create spectrum plot handle');
                end
                title(obj.plotHandles.spectrum, 'Power Spectrum');
                xlabel(obj.plotHandles.spectrum, 'Frequency (Hz)');
                ylabel(obj.plotHandles.spectrum, 'Power (μV²/Hz)');
                grid(obj.plotHandles.spectrum, 'on');

                % ERSPのプロット
                obj.plotHandles.ersp = subplot(2, 2, 4, 'Parent', obj.visualPanel);
                if ~ishandle(obj.plotHandles.ersp)
                    error('Failed to create ERSP plot handle');
                end
                title(obj.plotHandles.ersp, 'ERSP');
                xlabel(obj.plotHandles.ersp, 'Time (s)');
                ylabel(obj.plotHandles.ersp, 'Frequency (Hz)');
                colorbar(obj.plotHandles.ersp);
                grid(obj.plotHandles.ersp, 'on');

            catch ME
                fprintf('Error in plot initialization: %s\n', ME.message);
                rethrow(ME);
            end
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
            if ~ishandle(obj.plotHandles.rawData) || ~isvalid(obj.plotHandles.rawData)
                return;
            end
            try
                % 表示チャンネルの選択
                displayChannels = obj.params.gui.display.visualization.channels.eeg.display;
                data = data(displayChannels, :);

                % チャンネル名をデバイス設定から取得
                channelNames = obj.params.device.channels(displayChannels);

                % 表示用データの準備
                displaySecondsEEG = obj.params.gui.display.visualization.scale.displaySeconds;
                timeAxisEEG = (0:size(data,2)-1) / obj.params.device.sampleRate;

                % プロットの更新
                axes(obj.plotHandles.rawData);
                cla(obj.plotHandles.rawData);
                hold(obj.plotHandles.rawData, 'on');

                % カラーマップの生成
                colors = lines(size(data,1));

                % 各チャンネルをプロット
                legendEntries = cell(1, size(data,1));
                for ch = 1:size(data,1)
                    plot(obj.plotHandles.rawData, timeAxisEEG, data(ch,:), ...
                        'Color', colors(ch,:), 'LineWidth', 1);
                    legendEntries{ch} = channelNames{ch};
                end

                % 凡例の追加
                legend(obj.plotHandles.rawData, legendEntries, 'Location', 'eastoutside');
                hold(obj.plotHandles.rawData, 'off');

                % 軸の設定
                if obj.params.gui.display.visualization.scale.auto
                    axis(obj.plotHandles.rawData, 'auto');
                    yLims = ylim(obj.plotHandles.rawData);
                    yRange = diff(yLims);
                    ylim(obj.plotHandles.rawData, [yLims(1) - 0.1*yRange, yLims(2) + 0.1*yRange]);
                else
                    ylim(obj.plotHandles.rawData, obj.params.gui.display.visualization.scale.raw);
                end
                xlim(obj.plotHandles.rawData, [0 displaySecondsEEG]);

                % プロットの装飾
                title(obj.plotHandles.rawData, 'Raw EEG Signal');
                xlabel(obj.plotHandles.rawData, 'Time (s)');
                ylabel(obj.plotHandles.rawData, 'Amplitude (μV)');
                grid(obj.plotHandles.rawData, 'on');

                drawnow limitrate;

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end

        function updateEMGPlot(obj, data)
            if ~ishandle(obj.plotHandles.emg) || ~isvalid(obj.plotHandles.emg)
                return;
            end
            try
                % 表示チャンネルの選択
                displayChannels = obj.params.gui.display.visualization.channels.emg.display;
                data = data(displayChannels, :);

                % 表示用データの準備
                displaySecondsEMG = obj.params.gui.display.visualization.scale.displaySeconds;
                timeAxisEMG = (0:size(data,2)-1) / obj.params.acquisition.emg.sampleRate;

                % プロットの更新
                axes(obj.plotHandles.emg);
                cla(obj.plotHandles.emg);
                hold(obj.plotHandles.emg, 'on');

                % カラーマップの生成
                colors = lines(size(data,1));

                % EMGチャンネルのプロット
                legendEntries = cell(1, size(data,1));
                for ch = 1:size(data,1)
                    plot(obj.plotHandles.emg, timeAxisEMG, data(ch,:), ...
                        'Color', colors(ch,:), 'LineWidth', 1);
                    legendEntries{ch} = obj.params.acquisition.emg.channels.names{displayChannels(ch)};
                end

                % 凡例の追加
                legend(obj.plotHandles.emg, legendEntries, 'Location', 'eastoutside');
                hold(obj.plotHandles.emg, 'off');

                % Y軸の設定
                if obj.params.gui.display.visualization.scale.auto
                    axis(obj.plotHandles.emg, 'auto');
                    yLims = ylim(obj.plotHandles.emg);
                    yRange = diff(yLims);
                    ylim(obj.plotHandles.emg, [yLims(1) - 0.1*yRange, yLims(2) + 0.1*yRange]);
                else
                    ylim(obj.plotHandles.emg, obj.params.gui.display.visualization.scale.emg);
                end

                % X軸の設定
                xlim(obj.plotHandles.emg, [0 displaySecondsEMG]);

                % プロットの装飾
                title(obj.plotHandles.emg, 'EMG Signal');
                xlabel(obj.plotHandles.emg, 'Time (s)');
                ylabel(obj.plotHandles.emg, 'Amplitude (μV)');
                grid(obj.plotHandles.emg, 'on');

                drawnow limitrate;

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function updateSpectrumPlot(obj, spectrumData)
            if ~ishandle(obj.plotHandles.spectrum) || ~isvalid(obj.plotHandles.spectrum)
                return;
            end
            try
                axes(obj.plotHandles.spectrum);
                cla(obj.plotHandles.spectrum);

                % パワースペクトル表示
                semilogy(obj.plotHandles.spectrum, spectrumData.f, spectrumData.pxx);

                % 軸の設定
                if obj.params.gui.display.visualization.scale.auto
                    axis(obj.plotHandles.spectrum, 'auto');
                    xlim(obj.plotHandles.spectrum, [0 50]);  % 表示範囲を50Hzまでに制限
                else
                    xlim(obj.plotHandles.spectrum, obj.params.gui.display.visualization.scale.freq);
                    ylim(obj.plotHandles.spectrum, obj.params.gui.display.visualization.scale.power);
                end

                title(obj.plotHandles.spectrum, 'Power Spectrum');
                grid(obj.plotHandles.spectrum, 'on');
                
                drawnow limitrate;

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end

        function updateERSPPlot(obj, erspData)
            if ~ishandle(obj.plotHandles.ersp) || ~isvalid(obj.plotHandles.ersp)
                return;
            end
            try
                axes(obj.plotHandles.ersp);

                % ERSPデータの表示
                imagesc(obj.plotHandles.ersp, erspData.times, erspData.freqs, erspData.ersp);

                % カラーマップの設定
                if obj.params.gui.display.visualization.scale.auto
                    caxis(obj.plotHandles.ersp, 'auto');
                else
                    caxis(obj.plotHandles.ersp, obj.params.gui.display.visualization.ersp.scale);
                end

                colormap(obj.plotHandles.ersp, obj.params.gui.display.visualization.ersp.colormap.type);
                if obj.params.gui.display.visualization.ersp.colormap.reverse
                    colormap(obj.plotHandles.ersp, flipud(colormap(obj.plotHandles.ersp)));
                end

                % 軸の設定
                axis(obj.plotHandles.ersp, 'xy');
                if ~obj.params.gui.display.visualization.scale.auto
                    xlim(obj.plotHandles.ersp, obj.params.gui.display.visualization.ersp.time);
                    ylim(obj.plotHandles.ersp, obj.params.gui.display.visualization.ersp.freqRange);
                end

                title(obj.plotHandles.ersp, 'ERSP');
                colorbar(obj.plotHandles.ersp);
                
                drawnow limitrate;
                
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
            if strcmp(erspParams.colormap.background, 'white')
                lineColor = 'k';
            end
            
            for i = 1:length(bandNames)
                bandName = bandNames{i};
                if isfield(bands, bandName)
                    bandRange = bands.(bandName);
                    plot(axHandle, xRange, [bandRange(1) bandRange(1)], ...
                        '--', 'Color', lineColor, 'LineWidth', 0.5);
                    plot(axHandle, xRange, [bandRange(2) bandRange(2)], ...
                        '--', 'Color', lineColor, 'LineWidth', 0.5);
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
        
        function restThresholdChanged(obj, value)
            try
                % 閾値の制限（0-1の範囲）
                value = max(0, min(1, value));

                % パラメータの更新
                obj.params.classifier.threshold = value;

                % 入力フィールドの更新
                set(obj.displayControls.restThreshold, 'String', num2str(value));

                % コールバックの呼び出し
                if isfield(obj.callbacks, 'onParamChange')
                    obj.callbacks.onParamChange('threshold', value);
                end
            catch ME
                errordlg(['Invalid threshold value: ' ME.message], 'Parameter Error');
                set(obj.displayControls.restThreshold, 'String', ...
                    num2str(obj.params.classifier.threshold));
            end
        end
        
        function startButtonCallback(obj, ~, ~)
            if ~obj.isRunning && isfield(obj.callbacks, 'onStart')
                obj.isRunning = true;
                obj.isPaused = false;
                obj.startTime = tic;
                obj.updateButtonStates();
                obj.callbacks.onStart();
                drawnow;
            end
        end
        
        function stopButtonCallback(obj, ~, ~)
            if obj.isRunning && isfield(obj.callbacks, 'onStop')
                obj.isRunning = false;
                obj.isPaused = false;
                obj.updateButtonStates();
                obj.callbacks.onStop();
                drawnow;
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
                    drawnow;
                end
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
                drawnow;
            end
        end
        
        function updateButtonStates(obj)
            if obj.isRunning
                set(obj.startButton, 'Enable', 'off');
                set(obj.stopButton, 'Enable', 'on');
                set(obj.pauseButton, 'Enable', 'on');
            else
                set(obj.startButton, 'Enable', 'on');
                set(obj.stopButton, 'Enable', 'off');
                set(obj.pauseButton, 'Enable', 'off');
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
            
            if ~isempty(obj.updateTimer)
                stop(obj.updateTimer);
                delete(obj.updateTimer);
            end
            
            delete(obj.mainFigure);
        end
        
        function sliderCallback(obj, src, ~)
            if ~obj.params.gui.slider.enable
                return;
            end

            % スライダー値を整数に丸める
            value = round(get(src, 'Value'));
            set(src, 'Value', value);
            set(obj.sliderText, 'String', num2str(value));
            obj.sliderValue = value;
            drawnow;
        end
    end
end