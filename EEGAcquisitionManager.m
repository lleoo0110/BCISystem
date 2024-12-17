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
        normParams
        optimalThreshold
        
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
            obj.results = struct('predict', [], 'power', [], 'faa', [], 'erd', [], 'emotion', []);
            
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
            
            % スライダーウィンドウの作成
            obj.guiController.createSliderWindow();
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
            if ~obj.isRunning
                % 現在のパラメータを保持してモードを更新
                currentParams = obj.params;
                currentParams.acquisition.mode = mode;  % 新しいモードを設定
                obj.isOnlineMode = strcmpi(mode, 'online');  % モードフラグを更新
                
                % GUIを閉じる
                if ~isempty(obj.guiController)
                    obj.guiController.closeAllWindows();
                end
                
                % タイマーの停止
                if ~isempty(obj.acquisitionTimer) && isvalid(obj.acquisitionTimer)
                    stop(obj.acquisitionTimer);
                end
                
                % 全てのリソースを解放
                delete(obj);
                
                % システムの再初期化前に待機
                pause(0.5);
                
                % 新しいインスタンスを作成（新しいモードで初期化）
                EEGAcquisitionManager(currentParams);
            else
                % 記録中は切り替え不可のメッセージを表示
                warning('Cannot change mode while recording is in progress.');
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
                
                obj.guiController.updateStatus('Recording');
                start(obj.acquisitionTimer);
                
            end
        end
        
        function stop(obj)
            if obj.isRunning
                obj.isRunning = false;
                obj.isPaused = false;
                
                % 最後のデータを保存
                if ~isempty(obj.rawData)
                    obj.saveTemporaryData();
                end
                
                [mergedData, mergedLabels] = obj.mergeAndSaveData();
                
                % オフラインモードの場合の処理
                if ~obj.isOnlineMode
                    if ~isempty(mergedData)
                        obj.rawData = mergedData;
                        obj.labels = mergedLabels;
                        obj.executeDataProcessing();
                    end
                end
                
                % タイマーの停止
                stop(obj.acquisitionTimer);
                obj.guiController.updateStatus('Stopped');
                
                % 全てのウィンドウを閉じる
                if ~isempty(obj.guiController)
                    obj.guiController.closeAllWindows();
                end
                
                % 現在のパラメータを保持
                currentParams = obj.params;
                
                % 全てのリソースを解放
                delete(obj);
                
                % システムの再初期化
                pause(0.5);  % リソース解放のための短い待機
                EEGAcquisitionManager(currentParams);
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
                        
                        obj.updateDataBuffer(data);
                        
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
                % 時間とサンプル数の記録
                currentTime = toc(obj.processingTimer)*1000;
                obj.currentTotalSamples = obj.totalSampleCount + size(obj.rawData, 2);
                
                % 信号処理の実行
                if obj.params.signal.enable && ~isempty(obj.rawData)
                    [obj.processedData, obj.processedLabel, obj.processingInfo] = ...
                        obj.signalProcessor.process(obj.rawData, obj.labels);
                    
                    % パワー値の計算と保存
                    if obj.params.feature.power.enable && ~isempty(obj.processedData)
                        allPowers = obj.powerExtractor.calculatePowerForAllBands(obj.processedData);
                        
                        % 時間情報とサンプル数を追加
                        for i = 1:length(allPowers)
                            allPowers{i}.time = currentTime;
                            allPowers{i}.sample = obj.currentTotalSamples;
                        end
                        
                        obj.results.power = allPowers;
                    end
                    
                    % FAA計算と保存
                    if obj.params.feature.faa.enable && ~isempty(obj.processedData)
                        faaResults = obj.powerExtractor.calculateFAA(obj.processedData);
                        
                        % 時間情報とサンプル数を追加
                        for i = 1:length(faaResults)
                            faaResults{i}.time = currentTime;
                            faaResults{i}.sample = obj.currentTotalSamples;
                        end
                        
                        obj.results.faa = faaResults;
                    end
                    
                    % ERD計算と保存
                    if obj.params.feature.erd.enable && ~isempty(obj.processedData)
                        erdResults = obj.powerExtractor.calculateERD(obj.processedData);
                        
                        % 時間情報とサンプル数を追加
                        for i = 1:length(erdResults)
                            erdResults{i}.time = currentTime;
                            erdResults{i}.sample = obj.currentTotalSamples;
                        end
                        
                        obj.results.erd = erdResults;
                    end
                    
                    % 感情状態の分類と保存
                    if obj.params.feature.emotion.enable && ~isempty(obj.processedData)
                        emotionResults = obj.powerExtractor.classifyEmotion(obj.processedData);
                        
                        % 時間情報とサンプル数を追加
                        for i = 1:length(emotionResults)
                            emotionResults{i}.time = currentTime;
                            emotionResults{i}.sample = obj.currentTotalSamples;
                        end
                        
                        obj.results.emotion = emotionResults;
                    end
                    
                    % CSP特徴抽出の実行
                    if obj.params.feature.csp.enable && ~isempty(obj.processedData)
                        obj.cspfilters = obj.featureExtractor.trainCSP(...
                            obj.processedData, obj.processedLabel);
                        obj.cspfeatures = obj.featureExtractor.extractFeatures(...
                            obj.processedData, obj.cspfilters);
                        
                        if ~isempty(obj.cspfeatures)
                            obj.mergeAndSaveData();
                            fprintf('CSP Features and previous data saved\n');
                        end
                    end
                    
                    % 分類器の学習
                    if obj.params.classifier.svm.enable && ~isempty(obj.cspfeatures)
                        classifierResults = obj.classifier.trainOffline(...
                            obj.cspfeatures, obj.processedLabel);
                        obj.results.performance = classifierResults.performance;
                        obj.svmclassifier = classifierResults.classifier;
                        
                        if ~isempty(obj.svmclassifier)
                            obj.mergeAndSaveData();
                            fprintf('Classifier training completed and data saved\n');
                        end
                    end
                    
                    obj.mergeAndSaveData();
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
                
                % 基本パラメータの保存
                saveData.params = obj.params;
                
                % 生データとラベルの保存
                if ~isempty(mergedData)
                    saveData.rawData = mergedData;
                    saveData.labels = mergedLabels;
                else
                    saveData.rawData = obj.rawData;
                    saveData.labels = obj.labels;
                end
                
                % 処理済みデータの保存
                if ~isempty(obj.processedData)
                    saveData.processedData = obj.processedData;
                    saveData.processedLabel = obj.processedLabel;
                    saveData.processingInfo = obj.processingInfo;
                end
                
                % CSPデータの保存
                if ~isempty(obj.cspfeatures)
                    saveData.cspFilters = obj.cspfilters;
                    saveData.cspFeatures = obj.cspfeatures;
                end
                
                % 分類器データの保存
                if ~isempty(obj.svmclassifier)
                    saveData.svmClassifier = obj.svmclassifier;
                end
                
                % 解析結果の保存
                if ~isempty(obj.results)
                    % パワー値の保存
                    if isfield(obj.results, 'power')
                        saveData.results.power = obj.results.power;
                    end
                    
                    % FAA値の保存
                    if isfield(obj.results, 'faa')
                        saveData.results.faa = obj.results.faa;
                    end
                    
                    % ERD値の保存
                    if isfield(obj.results, 'erd')
                        saveData.results.erd = obj.results.erd;
                    end
                    
                    % 感情分類結果の保存
                    if isfield(obj.results, 'emotion')
                        saveData.results.emotion = obj.results.emotion;
                    end
                    
                    % 分類性能の保存
                    if isfield(obj.results, 'performance')
                        saveData.results.performance = obj.results.performance;
                    end
                    
                    % 予測結果の保存
                    if isfield(obj.results, 'predict')
                        saveData.results.predict = obj.results.predict;
                    end
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
                % オンライン処理用の学習済みCSP読み込み
                if obj.params.feature.csp.enable && obj.params.classifier.svm.enable
                    loadedData = DataLoading.loadDataBrowserWithPrompt('classifier');
                    obj.cspfilters = loadedData.cspFilters;
                    obj.svmclassifier = loadedData.svmClassifier;
                    
                    % 最適閾値情報の読み込み
                    if obj.params.classifier.svm.threshold.useOptimal && ...
                            isfield(loadedData, 'results') && ...
                            isfield(loadedData.results, 'performance') && ...
                            isfield(loadedData.results.performance, 'optimalThreshold')
                        obj.optimalThreshold = loadedData.results.performance.optimalThreshold;
                        fprintf('Loaded optimal threshold: %.3f\n', obj.optimalThreshold);
                    else
                        % デフォルトの安静状態閾値を使用
                        obj.optimalThreshold = obj.params.classifier.svm.threshold.rest;
                        fprintf('Using default threshold: %.3f\n', obj.optimalThreshold);
                    end
                end
                
                % ベースラインデータ読み込み
                if obj.params.feature.erd.enable
                    loadedData = DataLoading.loadDataBrowserWithPrompt('baseline');
                    baselineData = loadedData.processedData;
                    % ベースラインパワーの計算
                    obj.powerExtractor.calculateBaseline(baselineData);
                end
                
                % 正規化処理の実行
                if obj.params.signal.normalize.enabled
                    % データ読み込みダイアログの表示
                    loadedNormalizedData = DataLoading.loadDataBrowserWithPrompt('normalization');
                    
                    % 記録した正規化パラメータ情報を読み込む
                    obj.normParams = loadedNormalizedData.processingInfo.normalize.normParams;
                    
                    % 正規化パラメータのチェック
                    obj.validateNormalizationParams(obj.normParams);
                    
                    fprintf('正規化パラメータを初期化しました\n');
                    fprintf('正規化方法: %s\n', obj.params.signal.normalize.method);
                    
                    % 正規化パラメータの情報を表示
                    obj.displayNormalizationParams(obj.normParams);
                end
                
                obj.labels = [];        % トリガー情報
                obj.rawData = [];       % 生データの一時保存用
                
            catch ME
                obj.guiController.showError('Online processing failed', ME.message);
                obj.stop();
            end
        end
        
        function validateNormalizationParams(obj, normParams)
            % 正規化パラメータの妥当性チェック
            switch obj.params.signal.normalize.method
                case 'zscore'
                    if ~isfield(normParams, 'mean') || ~isfield(normParams, 'std')
                        error('z-score正規化に必要なパラメータ（mean, std）が不足しています．');
                    end
                    if any(normParams.std == 0)
                        error('標準偏差が0のチャンネルが存在します．');
                    end
                    
                case 'minmax'
                    if ~isfield(normParams, 'min') || ~isfield(normParams, 'max')
                        error('min-max正規化に必要なパラメータ（min, max）が不足しています．');
                    end
                    if any(normParams.max == normParams.min)
                        error('最大値と最小値が同じチャンネルが存在します．');
                    end
                    
                case 'robust'
                    if ~isfield(normParams, 'median') || ~isfield(normParams, 'mad')
                        error('ロバスト正規化に必要なパラメータ（median, mad）が不足しています．');
                    end
                    if any(normParams.mad == 0)
                        error('MADが0のチャンネルが存在します．');
                    end
                    
                otherwise
                    error('未知の正規化方法です: %s', obj.params.signal.normalize.method);
            end
        end
        
        function displayNormalizationParams(obj, normParams)
            % 正規化パラメータの情報を表示
            fprintf('\n正規化パラメータの情報:\n');
            fprintf('------------------------\n');
            
            switch obj.params.signal.normalize.method
                case 'zscore'
                    fprintf('平均値の範囲: [%.4f, %.4f]\n', min(normParams.mean), max(normParams.mean));
                    fprintf('標準偏差の範囲: [%.4f, %.4f]\n', min(normParams.std), max(normParams.std));
                    
                case 'minmax'
                    fprintf('最小値の範囲: [%.4f, %.4f]\n', min(normParams.min), max(normParams.min));
                    fprintf('最大値の範囲: [%.4f, %.4f]\n', min(normParams.max), max(normParams.max));
                    
                case 'robust'
                    fprintf('中央値の範囲: [%.4f, %.4f]\n', min(normParams.median), max(normParams.median));
                    fprintf('MADの範囲: [%.4f, %.4f]\n', min(normParams.mad), max(normParams.mad));
            end
            
            fprintf('------------------------\n');
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
                    % スライダーが有効で，かつシステムがpause状態でない場合のみラベルを記録
                    if ~obj.isPaused
                        sliderValue = obj.guiController.getSliderValue();
                        if ~isempty(sliderValue)  % スライダーが有効で値が存在する場合
                            currentTime = toc(obj.processingTimer)*1000;
                            obj.currentTotalSamples = obj.totalSampleCount + size(obj.rawData, 2);

                            % ラベル情報の作成と記録
                            trigger = struct(...
                                'value', sliderValue, ...
                                'time', uint64(currentTime), ...
                                'sample', obj.currentTotalSamples);
                            obj.labels = [obj.labels; trigger];
                        end
                    end

                    % オンライン処理の更新
                    if obj.isOnlineMode
                        obj.processOnline();
                    end

                    % GUI処理の更新
                    if obj.params.gui.display.visualization.enable.rawData || ...
                            obj.params.gui.display.visualization.enable.processedData || ...
                            obj.params.gui.display.visualization.enable.spectrum || ...
                            obj.params.gui.display.visualization.enable.ersp
                        obj.processGUI();
                    end

                    % データシフト
                    shiftSize = obj.slidingStep;
                    obj.dataBuffer = obj.dataBuffer(:, shiftSize:end);
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function processOnline(obj)
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

                    % 正規化処理
                    if obj.params.signal.normalize.enabled
                        preprocessedBuffer = obj.signalProcessor.normalizeOnline(preprocessedBuffer, obj.normParams);
                    end

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
                end

                % FAA計算と保存
                if obj.params.feature.faa.enable && ~isempty(preprocessedSegment)
                    % faaResultsを受け取るように変更
                    faaResults = obj.powerExtractor.calculateFAA(preprocessedSegment);
                    if ~isempty(faaResults) && iscell(faaResults)
                        faaResult = faaResults{1}; % オンライン処理では最初の結果のみ使用
                        faaValue = faaResult.faa;
                        arousalState = faaResult.arousal;
                    elseif isstruct(faaResults)
                        faaValue = faaResults.faa;
                        arousalState = faaResults.arousal;
                    end

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
                end
                
                % ERD計算と保存
                if obj.params.feature.erd.enable && ~isempty(preprocessedSegment)
                    % ERD結果の取得
                    erdResults = obj.powerExtractor.calculateERD(preprocessedSegment);

                    % セル配列の場合
                    if iscell(erdResults)
                        erdResult = erdResults{1};  % オンライン処理では最初の結果のみ使用
                    else
                        erdResult = erdResults;
                    end

                    % ERD値の結果表示
                    if isstruct(erdResult)
                        bandNames = fieldnames(erdResult.values);
                        for i = 1:length(bandNames)
                            bandName = bandNames{i};
                            fprintf('%s band - ERD: %.2f%% (Value: %.4f)\n', ...
                                bandName, ...
                                mean(erdResult.percent.(bandName)), ...
                                mean(erdResult.values.(bandName)));
                        end
                    end

                    % ERD結果の保存
                    newERDResult = struct(...
                        'values', erdResult.values, ...
                        'percent', erdResult.percent, ...
                        'time', currentTime, ...
                        'sample', obj.currentTotalSamples ...
                    );

                    if isempty(obj.results.erd)
                        obj.results.erd = newERDResult;
                    else
                        obj.results.erd(end+1) = newERDResult;
                    end
                end
                
                if obj.params.feature.emotion.enable && ~isempty(preprocessedSegment)
                    % 感情状態の分類を実行
                    emotionResults = obj.powerExtractor.classifyEmotion(preprocessedSegment);

                    % セル配列の場合
                    if iscell(emotionResults)
                        emotionResult = emotionResults{1};  % オンライン処理では最初の結果のみ使用
                    else
                        emotionResult = emotionResults;
                    end

                    % 結果表示
                    fprintf('Emotion State: %s\n', emotionResult.state);
                    fprintf('Valence: %.2f, Arousal: %.2f\n', ...
                        emotionResult.coordinates.valence, ...
                        emotionResult.coordinates.arousal);

                    % 感情状態の結果を保存
                    newEmotionResult = struct(...
                        'state', emotionResult.state, ...
                        'coordinates', emotionResult.coordinates, ...
                        'emotionCoords', emotionResult.emotionCoords, ...
                        'time', currentTime, ...
                        'sample', obj.currentTotalSamples ...
                    );

                    if isempty(obj.results.emotion)
                        obj.results.emotion = newEmotionResult;
                    else
                        obj.results.emotion(end+1) = newEmotionResult;
                    end
                end
                
                % CSP特徴抽出
                if obj.params.feature.csp.enable && ~isempty(preprocessedSegment)
                    currentFeatures = obj.featureExtractor.extractFeatures(...
                        preprocessedSegment, obj.cspfilters);
                end
                
                % SVM特徴分類
                if obj.params.classifier.svm.enable && ~isempty(currentFeatures)
                    % 閾値の取得（優先順位：最適閾値 > 手動設定閾値 > デフォルト値）
                    if obj.params.classifier.svm.threshold.useOptimal && ...
                            ~isempty(obj.params.classifier.svm.threshold.optimal)
                        threshold = obj.params.classifier.svm.threshold.optimal;
                    else
                        threshold = obj.params.classifier.svm.threshold.rest;
                    end
                    
                    % 分類の実行
                    [label, score] = obj.classifier.predictOnline(...
                        currentFeatures, obj.svmclassifier, threshold);
                    
                    % 結果の保存
                    currentTime = toc(obj.processingTimer)*1000;
                    newPredictResult = struct(...
                        'label', label, ...
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
                % udpData = emotionCoords;    % 感情分類
                % udpData = label;
                udpData = faaValue;
                obj.udpManager.sendTrigger(udpData);
                
            catch ME
                warning(ME.identifier, 'Error in processOnline: %s\n', ME.message);
                fprintf('Error stack:\n');
                for k = 1:length(ME.stack)
                    fprintf('File: %s, Line: %d, Function: %s\n', ...
                        ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
                end
            end
        end
        
        function processGUI(obj)
            try
                % 15秒分のデータに対してフィルタリング処理
                preprocessedBuffer = obj.signalProcessor.preprocess(obj.dataBuffer);
                
                % 最新の5秒分のデータを抽出
                displaySeconds = obj.params.gui.display.visualization.scale.displaySeconds;
                displaySamples = round(displaySeconds * obj.params.device.sampleRate);
                
                % 表示用データの準備
                displayData = struct();
                
                % 生データの最新5秒分を抽出
                displayData.rawData = obj.dataBuffer(:, end-displaySamples+1:end);
                
                % フィルタリング済みデータの最新5秒分を抽出
                displayData.processedData = preprocessedBuffer(:, end-displaySamples+1:end);
                
                % パワースペクトル計算
                if obj.params.gui.display.visualization.enable.spectrum
                    [pxx, f] = obj.powerExtractor.calculateSpectrum(displayData.processedData);
                    displayData.spectrum = struct('pxx', pxx, 'f', f);
                end
                
                % ERSP計算
                if obj.params.gui.display.visualization.enable.ersp
                    [ersp, times, freqs] = obj.powerExtractor.calculateERSP(displayData.processedData);
                    displayData.ersp = struct('ersp', ersp, 'times', times, 'freqs', freqs);
                end
                
                % GUI更新
                obj.guiController.updateResults(displayData);
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
                fprintf('Error in GUI processing: %s\n', ME.message);
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
                        obj.params.classifier.svm.threshold.rest = value;
                        obj.optimalThreshold = value;  % オンライン処理用の閾値も更新
                        fprintf('Threshold updated: %.3f\n', value);
                end
                
                % 必要に応じて処理パイプラインの再設定
                if obj.isOnlineMode
                    obj.reconfigureForMode();
                end
            catch ME
                obj.guiController.showError('Parameter Update Error', ME.message);
            end
        end
        
        function coords = getEmotionCoordinates(~, emotionState)
            % 感情状態を4次元座標に変換するメソッド
            % 座標は [快活性, 快不活性, 不快不活性, 不快活性] を表す
            emotionMap = containers.Map(...
                {'安静', '興奮', '喜び', '快適', 'リラックス', '眠気', '憂鬱', '不快', '緊張'}, ...
                {[0 0 0 0], [100 0 0 100], [100 0 0 0], [100 100 0 0], ...
                [0 100 0 0], [0 100 100 0], [0 0 100 0], [0 0 100 100], [0 0 0 100]});
            
            if emotionMap.isKey(emotionState)
                coords = emotionMap(emotionState);
            else
                coords = [0 0 0 0];  % デフォルト値（安静状態）
                warning('Emotion state "%s" not found. Using default coordinates.', emotionState);
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