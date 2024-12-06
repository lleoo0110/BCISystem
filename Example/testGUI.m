classdef testGUI < handle  % handleクラスを継承することで、オブジェクトの変更が可能になります
    % 使用例：test = testGUI()
    properties (Access = private)
        guiController
        params
        simulatedData
        updateTimer
        currentIndex
        isRunning
    end
    
    methods
        function obj = testGUI()
            % テスト用パラメータの設定
            obj.params = obj.createTestParams();
            
            % GUIControllerManagerの初期化
            obj.guiController = GUIControllerManager(obj.params);
            
            % コールバックの設定
            callbacks = struct(...
                'onStart', @() obj.startCallback(), ...
                'onStop', @() obj.stopCallback(), ...
                'onPause', @() obj.pauseCallback(), ...
                'onResume', @() obj.resumeCallback(), ...
                'onModeChange', @(mode) obj.onModeChange(mode), ...  % 名前を変更
                'onParamChange', @(param, value) obj.onParamChange(param, value));  % 名前を変更
            
            obj.guiController.setCallbacks(callbacks);
            
            % 擬似データの生成
            obj.generateSimulatedData();
            
            % タイマーの初期化
            obj.updateTimer = timer(...
                'ExecutionMode', 'fixedRate', ...
                'Period', 0.1, ... % 100ms間隔で更新
                'TimerFcn', @(~,~) obj.updateDisplay(), ...
                'ErrorFcn', @(~,~) disp('Timer error occurred'));
            
            obj.currentIndex = 1;
            obj.isRunning = false;
        end

        % モード変更コールバック（名前を変更）
        function onModeChange(~, mode)
            fprintf('Mode changed to: %s\n', mode);
        end
        
        % パラメータ変更コールバック（名前を変更）
        function onParamChange(~, param, value)
            fprintf('Parameter %s changed to: %f\n', param, value);
        end
        
        function startCallback(obj)
            try
                obj.isRunning = true;
                obj.currentIndex = 1;
                start(obj.updateTimer);
                fprintf('Started simulation\n');
            catch ME
                fprintf('Error in start: %s\n', ME.message);
            end
        end
        
        function stopCallback(obj)
            try
                obj.isRunning = false;
                stop(obj.updateTimer);
                obj.currentIndex = 1;
                % 最後のデータ表示をクリア
                obj.guiController.updateResults(struct('data', zeros(size(obj.simulatedData, 1), obj.params.device.sampleRate)));
                fprintf('Stopped simulation\n');
            catch ME
                fprintf('Error in stop: %s\n', ME.message);
            end
        end
        
        function pauseCallback(obj)
            try
                if obj.isRunning
                    obj.isRunning = false;
                    stop(obj.updateTimer);
                    fprintf('Paused simulation\n');
                end
            catch ME
                fprintf('Error in pause: %s\n', ME.message);
            end
        end
        
        function resumeCallback(obj)
            try
                obj.isRunning = true;
                start(obj.updateTimer);
                fprintf('Resumed simulation\n');
            catch ME
                fprintf('Error in resume: %s\n', ME.message);
            end
        end
        
        function updateDisplay(obj)
            try
                if obj.isRunning
                    % データの取得（100ms分）
                    windowSize = round(obj.params.device.sampleRate * 0.1); % 100msのデータ
                    if obj.currentIndex + windowSize <= size(obj.simulatedData, 2)
                        displayData = obj.simulatedData(:, obj.currentIndex:obj.currentIndex+windowSize-1);
                        
                        % 表示データの作成
                        plotData = struct();
                        plotData.rawData = displayData;
                        
                        % スペクトル計算（プロットのため）
                        [pxx, f] = pwelch(displayData(1,:), [], [], [], obj.params.device.sampleRate);
                        plotData.spectrum = struct('pxx', pxx, 'f', f);
                        
                        % GUIの更新
                        obj.guiController.updateResults(plotData);
                        
                        % インデックスの更新
                        obj.currentIndex = obj.currentIndex + windowSize;
                        
                        % データの最後まで到達した場合は最初に戻る
                        if obj.currentIndex >= size(obj.simulatedData, 2)
                            obj.currentIndex = 1;
                        end
                    end
                end
            catch ME
                fprintf('Error in updateDisplay: %s\n', ME.message);
            end
        end
        
        function delete(obj)
            try
                % クリーンアップ
                if ~isempty(obj.updateTimer)
                    stop(obj.updateTimer);
                    delete(obj.updateTimer);
                end
            catch ME
                fprintf('Error in cleanup: %s\n', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function params = createTestParams(~)
            % テスト用のパラメータ構造体を作成
            params = struct();
            
            % デバイス設定
            params.device.sampleRate = 250;  % Hz
            params.device.channelCount = 8;
            
            % 信号処理設定
            params.signal.window.analysis = 1.0;  % 秒
            params.signal.window.stimulus = 4.0;  % 秒
            params.signal.epoch.overlap = 0.5;    % 50%オーバーラップ
            
            % フィルタ設定
            params.signal.filter.fir.enabled = true;
            params.signal.filter.fir.low = 1;    % Hz
            params.signal.filter.fir.high = 40;   % Hz
            
            % 表示設定
            params.display.updateRate = 10;  % Hz
            
            return;
        end
        
        function generateSimulatedData(obj)
            try
                % 擬似的な脳波データの生成（10秒分）
                fs = obj.params.device.sampleRate;
                duration = 10;  % 10秒分のデータ
                t = 0:1/fs:duration-1/fs;
                nChannels = obj.params.device.channelCount;
                
                % 基本波形の生成（複数の周波数成分を含む）
                obj.simulatedData = zeros(nChannels, length(t));
                
                % 各チャンネルでの信号生成
                for ch = 1:nChannels
                    % アルファ波 (8-13 Hz)
                    alpha = 10 * sin(2*pi*10*t + rand*2*pi);
                    
                    % ベータ波 (13-30 Hz)
                    beta = 5 * sin(2*pi*20*t + rand*2*pi);
                    
                    % シータ波 (4-8 Hz)
                    theta = 7 * sin(2*pi*6*t + rand*2*pi);
                    
                    % ノイズの追加
                    noise = 2 * randn(1, length(t));
                    
                    % 信号の合成
                    obj.simulatedData(ch,:) = alpha + beta + theta + noise;
                end
                
                fprintf('Simulated data generated: %d channels, %d samples\n', ...
                    size(obj.simulatedData,1), size(obj.simulatedData,2));
                
            catch ME
                fprintf('Error in generateSimulatedData: %s\n', ME.message);
            end
        end
    end
end