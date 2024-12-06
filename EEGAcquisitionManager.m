classdef EEGAcquisitionManager < handle    
    properties (Access = private)
        % 各種マネージャーインスタンス
        lslManager
        udpManager
        signalProcessor
        powerExtractor
        featureExtractor
        classifier
        guiController
        dataManager
        
        % 設定とデータ管理
        params
        rawData
        labels
        processedData
        processedLabel
        processingInfo
        cspfilters
        cspfeatures
        svmclassifier
        results
        
        % 状態管理
        isRunning
        isPaused
        isOnlineMode
        currentTotalSamples
        
        % データバッファ
        dataBuffer
        bufferSize
        processingWindow    % 処理ウィンドウサイズ（サンプル数）
        slidingStep        % スライディングステップ（サンプル数）
        
        % タイマー
        acquisitionTimer
        processingTimer
        
        % データ管理用
        tempDataFiles
        fileIndex
        lastSaveTime
        lastSavedFilePath    % 最後に保存したファイルのパス
        
        % サンプル数管理用の変数を追加
        totalSampleCount    % 累積サンプル数
        lastResetSampleCount % 最後のリセット時のサンプル数
    end
    
    methods (Access = public)
        function obj = EEGAcquisitionManager(params)
            obj.params = params;
            obj.isRunning = false;
            obj.isPaused = false;
            obj.isOnlineMode = strcmpi(params.acquisition.mode, 'online');
            
            % 初期化
            obj.initializeManagers();
            obj.initializeDataBuffers();
            obj.setupTimers();
            
            % 結果保存用構造体の初期化を追加
            obj.results = struct('predict', [], 'power', [], 'faa', [], 'erd', []);
            
            % 保存ディレクトリの作成
            if ~exist(params.acquisition.save.path, 'dir')
                mkdir(params.acquisition.save.path);
            end
            
            % データ管理の初期化
            obj.tempDataFiles = {};
            obj.fileIndex = 1;
            
            % サンプル数カウンタの初期化
            obj.totalSampleCount = 0;
            obj.lastResetSampleCount = 0;
        end
        
        function delete(obj)
            % デストラクタ
            try
                % タイマーの停止と削除
                if ~isempty(obj.acquisitionTimer) && isvalid(obj.acquisitionTimer)
                    stop(obj.acquisitionTimer);
                    delete(obj.acquisitionTimer);
                end
                if ~isempty(obj.processingTimer) && isvalid(obj.processingTimer)
                    stop(obj.processingTimer);
                    delete(obj.processingTimer);
                end

                % UDPManagerのクリーンアップ
                if ~isempty(obj.udpManager)
                    delete(obj.udpManager);
                end

                % 一時ファイルの削除
                for i = 1:length(obj.tempDataFiles)
                    if exist(obj.tempDataFiles{i}, 'file')
                        delete(obj.tempDataFiles{i});
                    end
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function setMode(obj, mode)
            % モード切替（オンライン/オフライン）
            if ~obj.isRunning
                obj.reconfigureForMode();
                obj.guiController.updateMode(mode);
            end
        end
        
        function start(obj)
            if ~obj.isRunning
                obj.isRunning = true;
                obj.isPaused = false;
                
                % データ収集の初期化
                obj.rawData = [];
                obj.labels = [];
                obj.tempDataFiles = {};
                obj.fileIndex = 1;               
                
                % タイマー関連の初期化
                obj.lastSaveTime = uint64(tic);  % uint64として保存
                obj.processingTimer = tic;  % 計測開始時間の記録
                
                start(obj.acquisitionTimer);
         
                obj.guiController.updateStatus('Recording');
            end
        end
        
        function stop(obj)
            if obj.isRunning
                obj.isRunning = false;
                obj.isPaused = false;

                % 最後のデータを一時保存
                if ~isempty(obj.rawData)
                    obj.saveTemporaryData();
                end
                
                % データの統合と処理
                [mergedData, mergedLabels] = obj.mergeAndSaveData();

                if ~obj.isOnlineMode
                    if ~isempty(mergedData)
                        obj.rawData = mergedData;
                        obj.labels = mergedLabels;
                        obj.executeDataProcessing();
                    end
                end
                
                % タイマー停止
                stop(obj.acquisitionTimer);

                % UDPManagerのクリーンアップ
                if ~isempty(obj.udpManager)
                    delete(obj.udpManager);
                end       
                    obj.guiController.updateStatus('Stopped');
            end
        end
        
        function pause(obj)
            if obj.isRunning && ~obj.isPaused
                obj.isPaused = true;
                obj.guiController.updateStatus('Paused');
            end
        end
        
        function resume(obj)
            if obj.isRunning && obj.isPaused
                obj.isPaused = false;
                obj.guiController.updateStatus('Recording');
            end
        end
        
        function acquireData(obj)
            try
                if obj.isRunning
                    [data] = obj.lslManager.getData();
                    
                    if ~isempty(data)
                        obj.rawData = [obj.rawData, data];

                        % データバッファの更新
                        if obj.isOnlineMode
                            obj.updateDataBuffer(data);
                        end

                        % GUI更新用の情報構造体
                        updateInfo = struct();
                        updateInfo.rawData = obj.rawData;

                        % GUI更新
                        obj.guiController.updateResults(updateInfo);

                        % トリガー処理（Pause中は停止）
                        if ~obj.isPaused
                            trigger = obj.udpManager.receiveTrigger();
                            if ~isempty(trigger)
                                obj.currentTotalSamples = obj.totalSampleCount + size(obj.rawData, 2);
                                trigger.sample = obj.currentTotalSamples;
                                obj.labels = [obj.labels; trigger];
                            end
                        end

                        % 一時保存の確認
                        if toc(obj.lastSaveTime) >= obj.params.acquisition.save.saveInterval
                            obj.totalSampleCount = obj.totalSampleCount + size(obj.rawData, 2);
                            obj.saveTemporaryData();
                            obj.lastResetSampleCount = obj.totalSampleCount;
                        end
                    end
                end
            catch ME
                if ~isempty(obj.udpManager)
                    delete(obj.udpManager);
                end
                obj.guiController.showError('Data Acquisition Error', ME.message);
                obj.stop();
            end
        end
    end
   
    methods (Access = private)
        function initializeManagers(obj)
            % 各マネージャーの初期化
            try
                obj.lslManager = LSLManager(obj.params);
                obj.udpManager = UDPManager(obj.params);
                obj.signalProcessor = SignalProcessor(obj.params);
                obj.powerExtractor = PowerFeatureExtractor(obj.params);
                obj.featureExtractor = CSPFeatureExtractor(obj.params);
                obj.classifier = SVMClassifier(obj.params);
                obj.dataManager = DataManager(obj.params);
                
                % GUIControllerの初期化と各種コールバックの設定
                obj.guiController = GUIControllerManager(obj.params);
                obj.setupGUICallbacks();
                
                % オンラインモードならオンライン初期化
                if obj.isOnlineMode
                    obj.initializeOnline();
                end
            catch ME
                error('Failed to initialize managers: %s', ME.message);
            end
        end
        
        function reconfigureForMode(obj)
            try
                if obj.isOnlineMode
                    if ismethod(obj.signalProcessor, 'setMode')
                        obj.signalProcessor.setMode('online');
                    end
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
                fprintf('Mode reconfiguration failed: %s\n', ME.message);
            end
        end
        
        function setupGUICallbacks(obj)
            % 既存のコールバック設定に追加
            callbacks = struct(...
                'onStart', @() obj.start(), ...
                'onStop', @() obj.stop(), ...
                'onPause', @() obj.pause(), ...
                'onResume', @() obj.resume(), ...
                'onModeChange', @(mode) obj.setMode(mode), ...
                'onLabel', @(value) obj.handleLabel(value), ...
                'onParamChange', @(param, value) obj.updateParameter(param, value), ...
                'onVisualizationRequest', @(type) obj.visualizeData(type));
            
            obj.guiController.setCallbacks(callbacks);
        end
        
        function setupTimers(obj)
            % データ収集用タイマーの設定
            obj.acquisitionTimer = timer(...
                'ExecutionMode', 'fixedRate', ...　％ fixedRate or fixedSpacing
                'Period', 1/obj.params.device.sampleRate, ...  % サンプリング周期
                'TimerFcn', @(~,~) obj.acquireData());
        end
        
        function handleLabel(obj, trigger)
            if obj.isRunning && ~obj.isPaused
                % 現在の累積サンプル数を計算して設定
                obj.currentTotalSamples = obj.totalSampleCount + size(obj.rawData, 2);
                trigger.sample = obj.currentTotalSamples;

                % ラベル情報を保存
                obj.labels = [obj.labels; trigger];

                % GUI更新
                updateInfo = struct('label', trigger);
                obj.guiController.updateResults(updateInfo);
            end
        end
        
        function saveTemporaryData(obj)
            try
                if ~isempty(obj.rawData)
                    tempFilename = sprintf('%s_temp_%d.mat', obj.params.acquisition.save.name, obj.fileIndex);
                    tempFilePath = fullfile(obj.params.acquisition.save.path, tempFilename);
                    
                    % 保存用の一時変数作成
                    tempData = struct();
                    tempData.rawData = obj.rawData;
                    % ラベルデータの保存（構造体配列として保存）
                    if ~isempty(obj.labels)
                        tempData.labels = obj.labels;
                    else
                        tempData.labels = struct('value', [], 'time', [], 'sample', []);
                        tempData.labels = tempData.labels([]);  % 空の構造体配列として初期化
                    end
                    
                    % データの保存
                    save(tempFilePath, '-struct', 'tempData', '-v7.3');
                    
                    % ファイル名を記録
                    obj.tempDataFiles{end+1} = tempFilePath;
                    
                    % メモリクリア
                    obj.rawData = [];
                    obj.labels = [];
                    
                    % カウンタの更新
                    obj.fileIndex = obj.fileIndex + 1;
                    obj.lastSaveTime = tic;
                    
                    fprintf('Temporary data saved: %s\n', tempFilename);
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function executeDataProcessing(obj)
            try                
                % 信号処理の実行                
                if obj.params.signal.enable && ~isempty(obj.rawData)
                    [obj.processedData, obj.processedLabel, obj.processingInfo] = ...
                        obj.signalProcessor.process(obj.rawData, obj.labels);
                    % 処理済みデータが生成されたら即座に保存
                    if ~isempty(obj.processedData)
                        obj.mergeAndSaveData();
                        fprintf('Processed data and previous data saved\n');
                    end
                end

                % 処理済みデータの確認
                if ~isempty(obj.processedData)
                    fprintf('Processed data size: [%s], Labels: %d\n', ...
                        num2str(size(obj.processedData)), length(obj.processedLabel));
                end

                % 特徴抽出（CSP）の実行
                if obj.params.feature.csp.enable && ~isempty(obj.processedData) && ~isempty(obj.processedLabel)
                    obj.cspfilters = obj.featureExtractor.trainCSP(obj.processedData, obj.processedLabel);
                    obj.cspfeatures = obj.featureExtractor.extractFeatures(obj.processedData, obj.cspfilters);
                    
                    if ~isempty(obj.cspfeatures)
                        obj.mergeAndSaveData();
                        fprintf('CSP Features and previous data saved\n');
                    end
                end

                % 分類器の学習
                if obj.params.classifier.svm.enable && ~isempty(obj.cspfeatures)
                    obj.results = obj.classifier.trainOffline(obj.cspfeatures, obj.processedLabel);
                    obj.svmclassifier = obj.results.classifier;
                    
                    if ~isempty(obj.svmclassifier)
                        obj.mergeAndSaveData();
                        fprintf('Classifier training completed and data saved\n');
                    end
                end
                
            catch ME
                obj.guiController.showError('Processing Error', ME.message);
                fprintf('Error details: %s\n', ME.message);
            end
        end
        
        function cleanupTempFiles(obj)
            % 一時ファイルの削除
            for i = 1:length(obj.tempDataFiles)
                if exist(obj.tempDataFiles{i}, 'file')
                    delete(obj.tempDataFiles{i});
                end
            end
            obj.tempDataFiles = {};
        end

        function logError(obj, ME)
            % エラーログの保存
            errorLog = struct(...
                'message', ME.message, ...
                'identifier', ME.identifier, ...
                'stack', ME.stack, ...
                'time', datetime('now'), ...
                'dataInfo', struct(...
                    'rawDataSize', size(obj.rawData), ...
                    'labelsSize', size(obj.labels), ...
                    'tempFilesCount', length(obj.tempDataFiles)));

            % エラーログファイルの保存
            errorLogFile = fullfile(obj.params.acquisition.save.path, ...
                sprintf('error_log_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
            save(errorLogFile, 'errorLog');
        end

        % 一時ファイルの統合を行う補助関数
        function [mergedData, mergedLabels] = mergeTempFiles(obj)
            try
                totalSamples = 0;
                totalLabels = 0;

                % サイズの計算
                for i = 1:length(obj.tempDataFiles)
                    if exist(obj.tempDataFiles{i}, 'file')
                        fileInfo = load(obj.tempDataFiles{i});
                        totalSamples = totalSamples + size(fileInfo.rawData, 2);
                        if isfield(fileInfo, 'labels') && ~isempty(fileInfo.labels)
                            totalLabels = totalLabels + length(fileInfo.labels);
                        end
                    end
                end

                if totalSamples > 0  % データが存在する場合
                    % データ配列の初期化
                    mergedData = zeros(obj.params.device.channelCount, totalSamples);

                    % ラベルの初期化（ラベルが存在する場合のみ）
                    if totalLabels > 0
                        emptyStruct = struct('value', [], 'time', [], 'sample', []);
                        mergedLabels = repmat(emptyStruct, 1, totalLabels);
                    else
                        mergedLabels = struct('value', [], 'time', [], 'sample', []);
                        mergedLabels = mergedLabels([]);  % 空の構造体配列
                    end

                    currentSample = 1;
                    labelIndex = 1;

                    for i = 1:length(obj.tempDataFiles)
                        if exist(obj.tempDataFiles{i}, 'file')
                            fileData = load(obj.tempDataFiles{i});

                            % データの統合
                            sampleCount = size(fileData.rawData, 2);
                            if sampleCount > 0
                                mergedData(:, currentSample:currentSample+sampleCount-1) = fileData.rawData;
                                currentSample = currentSample + sampleCount;
                            end

                            % ラベルの統合（ラベルが存在する場合のみ）
                            if isfield(fileData, 'labels') && ~isempty(fileData.labels) && totalLabels > 0
                                for j = 1:length(fileData.labels)
                                    if labelIndex <= totalLabels
                                        mergedLabels(labelIndex) = fileData.labels(j);
                                        labelIndex = labelIndex + 1;
                                    end
                                end
                            end

                            % 一時ファイルの削除
                            delete(obj.tempDataFiles{i});
                        end
                    end

                    % 使用した分のみ保持
                    mergedData = mergedData(:, 1:currentSample-1);
                    if totalLabels > 0
                        mergedLabels = mergedLabels(1:labelIndex-1);
                    end

                    fprintf('Merged data size: [%s], Labels: %d\n', ...
                        num2str(size(mergedData)), length(mergedLabels));
                else
                    % データが無い場合
                    mergedData = zeros(obj.params.device.channelCount, 0);
                    mergedLabels = struct('value', [], 'time', [], 'sample', []);
                    mergedLabels = mergedLabels([]);
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
                mergedData = zeros(obj.params.device.channelCount, 0);
                mergedLabels = struct('value', [], 'time', [], 'sample', []);
                mergedLabels = mergedLabels([]);
            end
        end
        
        function [mergedData, mergedLabels] = mergeAndSaveData(obj)
           try
               % データサイズの計算と一時ファイルの統合
               [mergedData, mergedLabels] = obj.mergeTempFiles();

               % 保存データの構造体を直接作成
               saveData = struct();  

               saveData.params = obj.params;
               if ~isempty(mergedData)
                   saveData.rawData = mergedData;
                   saveData.labels = mergedLabels;
               else
                   saveData.rawData = obj.rawData;
                   saveData.labels = obj.labels;     
               end

               % メタデータの設定
               if obj.isOnlineMode
                   processingMode = 'online';
               else
                   processingMode = 'offline';
               end
               saveData.metadata = struct(...
                   'samplingRate', obj.params.device.sampleRate, ...
                   'channelCount', obj.params.device.channelCount, ...
                   'channelLabels', {obj.params.device.channels}, ...
                   'recordingTime', datetime('now'), ...
                   'deviceType', obj.params.device.name, ...
                   'processingMode', processingMode ...
               );
           
               if ~obj.isOnlineMode
                   % 処理済みデータがある場合は追加
                   if ~isempty(obj.processedData)
                       saveData.processedData = obj.processedData;
                       saveData.processedLabel = obj.processedLabel;
                       saveData.metadata.processingInfo = obj.processingInfo;
                   end

                   % CSPデータがある場合は追加
                   if ~isempty(obj.cspfeatures)
                       saveData.cspFilters = obj.cspfilters;
                       saveData.cspFeatures = obj.cspfeatures;
                   end

                   % 分類器データがある場合は追加
                   if ~isempty(obj.classifier) && ~isempty(obj.classifier.svmModel)
                       saveData.svmClassifier = obj.svmclassifier;
                       if ~isempty(obj.classifier.performance)
                           saveData.results.performance = obj.results.performance;
                       end
                   end  
               else
                   % オンライン処理の場合の保存
                   saveData.cspFilters = obj.cspfilters;
                   saveData.svmClassifier = obj.svmclassifier;
                   saveData.results = obj.results;
               end

               % データの保存とパスの取得
               obj.lastSavedFilePath = obj.dataManager.saveDataset(saveData);
               fprintf('Data saved to: %s\n', obj.lastSavedFilePath);

               % 保存したデータの概要を表示
               obj.displaySavedDataSummary(saveData);
           catch ME
               obj.guiController.showError('Save Error', ME.message);
               fprintf('Error details: %s\n', ME.message);
               % エラーログの保存
               obj.logError(ME);
           end
        end

        % 処理ステップリストを作成する補助関数
        function steps = createProcessingStepsList(obj)
            steps = {};
            if ~isempty(obj.rawData)
                steps{end+1} = 'Raw data acquisition';
            end
            if ~isempty(obj.signalProcessor) && ~isempty(obj.signalProcessor.processedData)
                steps{end+1} = 'Signal processing';
            end
            if ~isempty(obj.cspfeatures)
                steps{end+1} = 'Feature extraction';
            end
            if ~isempty(obj.classifier) && ~isempty(obj.classifier.svmModel)
                steps{end+1} = 'Classification';
            end
        end
        
        function initializeOnline(obj)
            try
                if obj.params.feature.csp.enable || obj.params.feature.erd.enable
                    % データ読み込みダイアログの表示
                    loadedData = obj.loadDataBrowser();

                    if obj.params.feature.csp.enable
                        obj.cspfilters = loadedData.cspFilters;
                        obj.svmclassifier = loadedData.svmClassifier;
                    end

                    if obj.params.feature.erd.enable
                        % ベースライン用の前処理済みデータを取得
                        if isfield(loadedData, 'processedData')
                            baselineData = loadedData.processedData;
                            % ベースラインパワーの計算
                            obj.powerExtractor.calculateBaseline(baselineData);
                            fprintf('Baseline data loaded and power calculated\n');
                        else
                            error('No processed data found for ERD baseline calculation');
                        end
                    end
                end

                obj.labels = [];        % トリガー情報
                obj.rawData = [];       % 生データの一時保存用

            catch ME
                obj.guiController.showError('Online processing failed', ME.message);
                obj.stop();
            end
        end
        
        
        function loadedData = loadDataBrowser(obj)
            try
                % ファイル選択ダイアログを表示
                [filename, pathname] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, ...
                    'Select EEG data file', obj.params.acquisition.save.path);

                if filename ~= 0  % ユーザーがファイルを選択した場合
                    fullpath = fullfile(pathname, filename);

                    % データの読み込み
                    loadedData = obj.dataManager.loadDataset(fullpath);

                    fprintf('Successfully loaded data from: %s\n', fullpath);
                    fprintf('Raw data size: [%s], Labels: %d\n', ...
                        num2str(size(obj.rawData)), length(obj.labels));
                end
            catch ME
                error('Failed to load data: %s', ME.message);
            end
        end

        function initializeDataBuffers(obj)
            obj.processingWindow = round(obj.params.signal.window.analysis * obj.params.device.sampleRate);
            obj.slidingStep = round(obj.params.signal.window.updateBuffer * obj.params.device.sampleRate);
            obj.bufferSize = round(obj.params.signal.window.bufferSize * obj.params.device.sampleRate);

            % バッファを0で初期化
            obj.dataBuffer = zeros(obj.params.device.channelCount, 0);  % 空の状態から開始
        end
        
        
        function updateDataBuffer(obj, newData)
            if isempty(newData)
                return;
            end

            try
                obj.dataBuffer = [obj.dataBuffer, newData];
                
                if size(obj.dataBuffer, 2) >= obj.bufferSize
                    shiftSize = obj.slidingStep;
                    obj.dataBuffer = obj.dataBuffer(:, shiftSize:end);
                    obj.processSegment();
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
                
        function processSegment(obj)
            try
                % 初期化
                preprocessedSegment = [];
                currentFeatures = [];
                label = [];
                score = [];
                faaValue = [];
                arousalState = '';
                bandPowers = struct();  % パワー値格納用の構造体を追加
                
                % timeとsampleの記録
                currentTime = toc(obj.processingTimer)*1000;
                obj.currentTotalSamples = obj.totalSampleCount + size(obj.rawData, 2);

                % 前処理
                if obj.params.signal.enable && ~isempty(obj.dataBuffer)
                    preprocessedBuffer = obj.signalProcessor.preprocess(obj.dataBuffer);

                    endIdx = size(preprocessedBuffer, 2);
                    startIdx = endIdx - obj.processingWindow + 1;
                    preprocessedSegment = preprocessedBuffer(:, startIdx:endIdx);
                end

                % パワー値計算
                if obj.params.feature.power.enable && ~isempty(preprocessedSegment)
                    % パワー値の計算
                    bandNames = obj.params.feature.power.bands.names;
                    if iscell(bandNames{1})
                        bandNames = bandNames{1};
                    end

                    % 各周波数帯のパワー値を計算
                    for i = 1:length(bandNames)
                        bandName = bandNames{i};
                        freqRange = obj.params.feature.power.bands.(bandName);
                        bandPower = obj.powerExtractor.calculatePower(preprocessedSegment, freqRange);
                        bandPowers.(bandName) = bandPower;
                    end

                    % パワー値の結果を保存
                    newPowerResult = struct(...
                        'bands', bandPowers, ...
                        'time', currentTime, ...
                        'sample', obj.currentTotalSamples ...
                    );
                    if isempty(obj.results.power)
                        obj.results.power = newPowerResult;
                    else
                        obj.results.power(end+1) = newPowerResult;
                    end
                else
                    % fprintf('Power analysis is disabled or no data available\n');
                end
                    
                % FAA計算と保存
                if obj.params.feature.faa.enable && ~isempty(preprocessedSegment)
                    [faaValue, arousalState] = obj.powerExtractor.calculateFAA(preprocessedSegment);
                    fprintf('FAA Value: %.4f, Arousal State: %s\n', faaValue, arousalState);

                    % FAA値の結果を保存
                    newFAAResult = struct(...
                        'faa', faaValue, ...
                        'arousal', arousalState, ...
                        'time', currentTime, ...
                        'sample', obj.currentTotalSamples ...
                    );
                    if isempty(obj.results.faa)
                        obj.results.faa = newFAAResult;
                    else
                        obj.results.faa(end+1) = newFAAResult;
                    end
                else
                    % fprintf('FAA analysis is disabled or no data available\n');
                end

                % ERD計算と保存
                if obj.params.feature.erd.enable && ~isempty(preprocessedSegment)
                    [erdValues, erdPercent] = obj.powerExtractor.calculateERD(preprocessedSegment);

                    % ERD値の結果表示（構造体のフィールドごとに処理）
                    bandNames = fieldnames(erdValues);
                    for i = 1:length(bandNames)
                        bandName = bandNames{i};
                        fprintf('%s band - ERD: %.2f%% (Value: %.4f)\n', ...
                            bandName, ...
                            mean(erdPercent.(bandName)), ...
                            mean(erdValues.(bandName)));
                    end

                    % ERD値の結果を保存
                    newERDResult = struct(...
                        'values', erdValues, ...
                        'percent', erdPercent, ...
                        'time', currentTime, ...
                        'sample', obj.currentTotalSamples ...
                    );

                    if isempty(obj.results.erd)
                        obj.results.erd = newERDResult;
                    else
                        obj.results.erd(end+1) = newERDResult;
                    end
                else
                    % fprintf('ERD analysis is disabled or no data available\n');
                end

                % CSP特徴抽出
                if obj.params.feature.csp.enable && ~isempty(preprocessedSegment)
                    currentFeatures = obj.featureExtractor.extractFeatures(...
                        preprocessedSegment, obj.cspfilters);
                end

                % SVM特徴分類
                if obj.params.classifier.svm.enable && ~isempty(currentFeatures)
                    [label, score] = obj.classifier.predictOnline(...
                        currentFeatures, obj.svmclassifier);

                    % 結果の保存
                    currentTime = toc(obj.processingTimer)*1000;
                    newPredictResult = struct('label', label, ...
                        'score', score, ...
                        'time', currentTime ...
                    );

                    if isempty(obj.results.predict)
                        obj.results.predict = newPredictResult;
                    else
                        obj.results.predict(end+1) = newPredictResult;
                    end
                end

                % UDP送信（FAA値と覚醒状態の送信）
                % 後に構造体で送信できるようにして計算した結果を全て送る（今は手動で選択）
                udpData = faaValue;
                obj.udpManager.sendTrigger(udpData);
                fprintf('UDP sent: udpData=%f \n', udpData);

                % GUI更新
                updateInfo.processedData = preprocessedSegment;
                obj.guiController.updateResults(updateInfo);

            catch ME
                warning(ME.identifier, 'Error in processSegment: %s\n', ME.message);
                fprintf('Error stack:\n');
                for k = 1:length(ME.stack)
                    fprintf('File: %s, Line: %d, Function: %s\n', ...
                        ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
                end
            end
        end
        
        function ready = isBufferReady(obj)
            % バッファが処理準備完了かチェック
            ready = size(obj.dataBuffer, 2) >= obj.bufferSize;
        end
        
        
        function updateParameter(obj, paramName, value)
            % パラメータの動的更新
            try
                switch lower(paramName)
                    case 'window size'
                        obj.params.signal.window.analysis = value;
                        obj.initializeDataBuffers();
                    case 'filter range'
                        obj.params.signal.filter.fir.frequency = value;
                    case 'threshold'
                        obj.params.classifier.svm.threshold = value;
                end
                
                % 必要に応じて処理パイプラインの再設定
                if obj.isOnlineMode
                    obj.reconfigureForMode();
                end
            catch ME
                obj.guiController.showError('Parameter Update Error', ME.message);
            end
        end
        
        function displaySavedDataSummary(~, saveData)
            fprintf('\nSaved Data Summary:\n');
            fprintf('------------------\n');

            % 基本データ情報
            if isfield(saveData, 'rawData')
                [rows, cols] = size(saveData.rawData);
                fprintf('Raw Data: %d channels x %d samples\n', rows, cols);
            end

            if isfield(saveData, 'labels')
                fprintf('Labels: %d entries\n', length(saveData.labels));
            end

            % 処理済みデータ情報
            if isfield(saveData, 'processedData')
                if iscell(saveData.processedData)
                    fprintf('Processed Data: %d epochs\n', length(saveData.processedData));
                else
                    [rows, cols, depth] = size(saveData.processedData);
                    fprintf('Processed Data: %d channels x %d samples x %d trials\n', ...
                        rows, cols, depth);
                end
            end

            % 特徴量情報
            if isfield(saveData, 'cspfeatures')
                if iscell(saveData.cspfeatures)
                    fprintf('CSP Features: %d sets\n', length(saveData.cspfeatures));
                else
                    [rows, cols] = size(saveData.cspfeatures);
                    fprintf('CSP Features: %d trials x %d features\n', rows, cols);
                end
            end

            % CSPフィルタ情報
            if isfield(saveData, 'cspFilters')
                fprintf('CSP Filters: Present\n');
            end

            % 分類器情報
            if isfield(saveData, 'classifierModel')
                fprintf('Classifier Model: Present\n');
                if isfield(saveData, 'classifierPerformance')
                    if isfield(saveData.classifierPerformance, 'accuracy')
                        fprintf('Classification Accuracy: %.2f%%\n', ...
                            saveData.classifierPerformance.accuracy * 100);
                    end
                end
            end

            fprintf('------------------\n');
        end
        
        function visualizeData(obj, type)
            % データの可視化
            try
                updateInfo = struct();
                switch lower(type)
                    case 'raw'
                        if ~isempty(obj.rawData)
                            updateInfo.rawData = obj.rawData;
                        end
                    case 'processed'
                        if ~isempty(obj.processedData)
                            updateInfo.processedData = obj.processedData;
                        end
                    case 'cspfeatures'
                        if ~isempty(obj.cspfeatures)
                            updateInfo.cspfeatures = obj.cspfeatures;
                        end
                end
                
                if ~isempty(fieldnames(updateInfo))
                    obj.guiController.updateResults(updateInfo);
                end
            catch ME
                obj.guiController.showError('Visualization Error', ME.message);
            end
        end

    end
    
    methods (Access = protected)
        function errorHandler(obj, ~, eventData)
            % エラーハンドリング
            try
                % UDPManagerのクリーンアップ
                if ~isempty(obj.udpManager)
                    delete(obj.udpManager);
                end
                % その他のクリーンアップ処理
                obj.stop();
                % エラーメッセージの表示
                obj.guiController.showError('System Error', eventData.message);
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
    end
end