classdef EEGAcquisitionManager < handle
    properties (Access = private)
        % 各種マネージャーインスタンス
        lslManager
        udpManager
        guiController
        dataManager
        
        % 前処理コンポーネント
        artifactRemover
        baselineCorrector
        downSampler
        firFilter
        notchFilter
        normalizer
        
        % 特徴抽出コンポーネント
        powerExtractor
        faaExtractor
        abRatioExtractor
        cspExtractor
        emotionExtractor
        
        % 分類器コンポーネント
        svmClassifier
        ecocClassifier
        cnnClassifier
        
        % 設定とデータ管理
        params
        rawData
        labels
        processedData
        processedLabel
        processingInfo
        classifiers      % 分類器モデルを保持する構造体
        results          % 解析結果を保持する構造体
        normParams
        optimalThreshold
       
        % 状態管理
        isRunning
        isPaused
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
        latestResults           % UDP送信用の最新結果保持用
        
        % サンプル数管理用の変数を追加
        totalSampleCount    % 累積サンプル数
        lastResetSampleCount % 最後のリセット時のサンプル数
    end
    
    methods (Access = public)
        function obj = EEGAcquisitionManager(params)
            obj.params = params;
            obj.isRunning = false;
            obj.isPaused = false;
            
            % 初期化
            obj.initializeDataBuffers();
            obj.initializeResults();
            obj.initializeManagers();
            obj.setupTimers();

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
                
                obj.mergeAndSaveData();
                
                % タイマーの停止
                stop(obj.acquisitionTimer);
                obj.guiController.updateStatus('Stopped');
                
                % 全てのウィンドウを閉じる
                if ~isempty(obj.guiController)
                    obj.guiController.closeAllWindows();
                end
                
                % 全てのリソースを解放
                delete(obj);
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
                obj.handleError(ME);
            end
        end
    end
    
    methods (Access = private)
        function initializeManagers(obj)
            % 各マネージャーの初期化
            try
                obj.lslManager = LSLManager(obj.params);
                obj.udpManager = UDPManager(obj.params);
                obj.dataManager = DataManager(obj.params);
                obj.guiController = GUIControllerManager(obj.params);
                
                % 前処理コンポーネントの初期化
                obj.artifactRemover = ArtifactRemover(obj.params);
                obj.baselineCorrector = BaselineCorrector(obj.params);
                obj.downSampler = DownSampler(obj.params);
                obj.firFilter = FIRFilterDesigner(obj.params);
                obj.notchFilter = NotchFilterDesigner(obj.params);
                obj.normalizer = EEGNormalizer(obj.params);
                
                % 特徴抽出コンポーネントの初期化
                obj.powerExtractor = PowerExtractor(obj.params);
                obj.faaExtractor = FAAExtractor(obj.params);
                obj.abRatioExtractor = ABRatioExtractor(obj.params);
                obj.cspExtractor = CSPExtractor(obj.params);
                obj.emotionExtractor = EmotionExtractor(obj.params);
                
                % 分類器コンポーネントの初期化
                obj.svmClassifier = SVMClassifier(obj.params);
                obj.ecocClassifier = ECOCClassifier(obj.params);
                obj.cnnClassifier = CNNClassifier(obj.params);
                
                % classifiersストラクチャの初期化
                obj.classifiers = struct(...
                    'svm', [], ...
                    'ecoc', [], ...
                    'cnn', [] ...
                );
                
                % GUIコールバック
                obj.setupGUICallbacks();
                
                % オンラインモードならオンライン初期化
                if strcmpi(obj.params.acquisition.mode, 'online')
                    obj.initializeOnline();
                end
            catch ME
                error('Failed to initialize managers: %s', ME.message);
            end
        end
        
        function initializeResults(obj)
            obj.results = struct(...
                'power', [], ...
                'faa', [], ...
                'abRatio', [], ...
                'emotion', [], ...
                'csp', struct(...
                    'filters', [], ...
                    'features', [], ...
                    'parameters', struct(...
                        'numFilters', obj.params.feature.csp.patterns, ...
                        'regularization', obj.params.feature.csp.regularization, ...
                        'method', 'standard' ...
                    ) ...
                ), ...
                'predict', [] ...
            );
        end
        
        function setupGUICallbacks(obj)
            % 既存のコールバック設定に追加
            callbacks = struct(...
                'onStart', @() obj.start(), ...
                'onStop', @() obj.stop(), ...
                'onPause', @() obj.pause(), ...
                'onResume', @() obj.resume(), ...
                'onLabel', @(value) obj.handleLabel(value), ...
                'onParamChange', @(param, value) obj.updateParameter(param, value));
            
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
        
        function mergeAndSaveData(obj)
            try
                % データの統合
                [mergedData, mergedLabels] = obj.mergeTempFiles();

                % 保存データの構造体を作成
                saveData = struct();
                saveData.params = obj.params;
                saveData.rawData = mergedData;
                saveData.labels = mergedLabels;

                % 解析結果の保存
                if ~isempty(obj.results)
                    saveData.results = obj.results;
                end

                % CSPフィルタと分類器の保存（オンラインモード用）
                if ~isempty(obj.classifiers) && isfield(obj.classifiers, obj.params.classifier.activeClassifier)
                    saveData.([obj.params.classifier.activeClassifier 'classifier']) = obj.classifiers.(obj.params.classifier.activeClassifier);
                end
                if isfield(obj.results, 'csp') && ~isempty(obj.results.csp.filters)
                    saveData.cspFilters = obj.results.csp.filters;
                end

                % データの保存とパスの取得
                obj.lastSavedFilePath = obj.dataManager.saveDataset(saveData);
                fprintf('Data saved to: %s\n', obj.lastSavedFilePath);

            catch ME
                obj.logError(ME);
                rethrow(ME);
            end
        end
        
        function initializeOnline(obj)
            try
                % オンライン処理用の学習済みモデル読み込み
                loadedData = DataLoader.loadDataBrowserWithPrompt('オンライン処理');
                
                % 正規化パラメータの読み込みと検証
                if obj.params.signal.preprocessing.normalize.enable
                    if isfield(loadedData, 'processingInfo') && ...
                       isfield(loadedData.processingInfo, 'normalize')
                        obj.normParams = loadedData.processingInfo.normalize;
                        obj.validateNormalizationParams(obj.normParams);
                        obj.displayNormalizationParams(obj.normParams);
                    else
                        error('Normalization parameters not found in loaded data (processingInfo.normalize)');
                    end
                end

                % アクティブな分類器の確認
                if ~isfield(loadedData.classifier, obj.params.classifier.activeClassifier)
                    error('Selected classifier "%s" not found in loaded data', obj.params.classifier.activeClassifier);
                end

                % 分類器モデルの設定
                if ~isfield(loadedData.classifier.(obj.params.classifier.activeClassifier), 'model')
                    error('%s model not found in loaded data', upper(obj.params.classifier.activeClassifier));
                end
                obj.classifiers = loadedData.classifier;

                % SVMの場合のみ最適閾値を設定
                if strcmp(obj.params.classifier.activeClassifier, 'svm')
                    if obj.params.classifier.svm.probability
                        % 分類器の性能情報から最適閾値を取得
                        if isfield(loadedData.classifier.svm, 'performance') && ...
                           isfield(loadedData.classifier.svm.performance, 'optimalThreshold') && ...
                           ~isempty(loadedData.classifier.svm.performance.optimalThreshold)
                            obj.optimalThreshold = loadedData.classifier.svm.performance.optimalThreshold;
                        end
                    end
                    % デフォルトの閾値を設定
                    obj.optimalThreshold = obj.params.classifier.svm.threshold.rest;
                end

                % CSPフィルタの設定
                if ~isfield(loadedData, 'results') || ~isfield(loadedData.results, 'csp')
                    error('CSP filters not found in loaded data');
                end
                obj.results.csp.filters = loadedData.results.csp.filters;

                fprintf('Successfully initialized online processing with %s classifier\n', ...
                    upper(obj.params.classifier.activeClassifier));

            catch ME
                obj.guiController.showError('Online processing initialization failed', ME.message);
                obj.stop();
                rethrow(ME);
            end
        end
        
        function validateNormalizationParams(obj, normParams)
            % 正規化パラメータの妥当性チェック
            switch obj.normParams.method
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
            
            switch obj.normParams.method
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
                    % オンライン処理の更新
                    if strcmpi(obj.params.acquisition.mode, 'online')
                        obj.processOnline();
                    end

                    % GUI処理の更新
                    if any([obj.params.gui.display.visualization.enable.rawData, ...
                            obj.params.gui.display.visualization.enable.processedData, ...
                            obj.params.gui.display.visualization.enable.spectrum, ...
                            obj.params.gui.display.visualization.enable.ersp])
                        obj.processGUI();
                    end

                    % スライダー処理
                    if ~obj.isPaused
                        obj.updateSliderValue();
                    end

                    % バッファのシフト
                    obj.dataBuffer = obj.dataBuffer(:, obj.slidingStep:end);
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function processOnline(obj)
            try
                obj.currentTotalSamples = obj.totalSampleCount + size(obj.rawData, 2);

                % 前処理
                preprocessedSegment = obj.preprocessSignal();
                if isempty(preprocessedSegment)
                    return;
                end

                % 最新の解析ウィンドウを抽出
                windowSamples = round(obj.params.signal.window.analysis * obj.params.device.sampleRate);
                if size(preprocessedSegment, 2) > windowSamples
                    % 最新のwindowSamplesのデータを抽出
                    analysisSegment = preprocessedSegment(:, end-windowSamples+1:end);
                else
                    % データが解析ウィンドウより小さい場合はそのまま使用
                    analysisSegment = preprocessedSegment;
                end

                % 各特徴量の処理（最新の解析ウィンドウのみを使用）
                obj.processPowerFeatures(analysisSegment);
                obj.processFAAFeatures(analysisSegment);
                obj.processABRatioFeatures(analysisSegment);
                obj.processEmotionFeatures(analysisSegment);

                % CSP特徴量の抽出と分類
                if obj.params.feature.csp.enable
                    currentFeatures = obj.processCSPFeatures(analysisSegment);
                end
                
                switch obj.params.classifier.activeClassifier
                    case 'svm'
                         [label, score] = obj.processClassification(currentFeatures);
                         
                    case 'ecoc'
                         [label, score] = obj.processClassification(currentFeatures);

                    case 'cnn'
                         [label, score] = obj.processClassification(analysisSegment);
                end

                % 最新の結果を構造体として保存
                obj.latestResults = struct(...
                    'prediction', struct(...
                        'label', label, ...
                        'score', score ...
                    ), ...
                    'features', struct(...
                        'faa', obj.getLatestFeature(obj.results.faa), ...
                        'abRatio', obj.getLatestFeature(obj.results.abRatio), ...
                        'emotion', obj.getLatestFeature(obj.results.emotion) ...
                    ) ...
                );

                % UDP送信
                jsonStr = jsonencode(obj.latestResults);
                fprintf('JSON data size before sending: %d bytes\n', strlength(jsonStr));
                obj.sendResults(obj.latestResults);

            catch ME
                obj.handleError(ME);
            end
        end
        
        function preprocessedSegment = preprocessSignal(obj)
            try
                preprocessedSegment = [];
                if obj.params.signal.enable && ~isempty(obj.dataBuffer)
                    data = obj.dataBuffer;

                    % アーティファクト除去
                    if obj.params.signal.preprocessing.artifact.enable
                        [data, ~] = obj.artifactRemover.removeArtifacts(data, 'all');
                    end

                    % ベースライン補正
                    if obj.params.signal.preprocessing.baseline.enable
                        [data, ~] = obj.baselineCorrector.correctBaseline(...
                            data, obj.params.signal.preprocessing.baseline.method);
                    end

                    % ダウンサンプリング
                    if obj.params.signal.preprocessing.downsample.enable
                        [data, ~] = obj.downSampler.downsample(...
                            data, obj.params.signal.preprocessing.downsample.targetRate);
                    end

                    % フィルタリング
                    if obj.params.signal.preprocessing.filter.notch.enable
                        [data, ~] = obj.notchFilter.designAndApplyFilter(data);
                    end
                    if obj.params.signal.preprocessing.filter.fir.enable
                        [data, ~] = obj.firFilter.designAndApplyFilter(data);
                    end

                    % 正規化
                    if obj.params.signal.preprocessing.normalize.enable
                        if strcmpi(obj.params.acquisition.mode, 'online')
                            data = obj.normalizer.normalizeOnline(data, obj.normParams);
                        else
                            [data, ~] = obj.normalizer.normalize(data);
                        end
                    end
                    
                    preprocessedSegment = data;
                    
                end
            catch ME
                error('Preprocessing failed: %s', ME.message);
            end
        end
        
        function processPowerFeatures(obj, preprocessedSegment)
            if ~obj.params.feature.power.enable || isempty(preprocessedSegment)
                return;
            end
        
            % PowerExtractorを使用してパワー値を計算
            bandNames = obj.params.feature.power.bands.names;
            if iscell(bandNames{1})
                bandNames = bandNames{1};
            end
        
            bandPowers = struct();
            for i = 1:length(bandNames)
                bandName = bandNames{i};
                freqRange = obj.params.feature.power.bands.(bandName);
                bandPowers.(bandName) = obj.powerExtractor.calculatePower(preprocessedSegment, freqRange);
            end
            
            currentTime = toc(obj.processingTimer)*1000;
            newPowerResult = struct(...
                'bands', bandPowers, ...
                'time', currentTime, ...
                'sample', obj.currentTotalSamples);
        
            if isempty(obj.results.power)
                obj.results.power = newPowerResult;
            else
                obj.results.power(end+1) = newPowerResult;
            end
        end
        
        function processFAAFeatures(obj, preprocessedSegment)
            if ~obj.params.feature.faa.enable || isempty(preprocessedSegment)
                return;
            end
        
            % FAAExtractorを使用してFAA値を計算
            faaResults = obj.faaExtractor.calculateFAA(preprocessedSegment);
            
            if ~isempty(faaResults)
                if iscell(faaResults)
                    faaResult = faaResults{1};
                else
                    faaResult = faaResults;
                end
                
                currentTime = toc(obj.processingTimer)*1000;
                newFAAResult = struct(...
                    'faa', faaResult.faa, ...
                    'arousal', faaResult.pleasureState, ...
                    'time', currentTime, ...
                    'sample', obj.currentTotalSamples);
        
                if isempty(obj.results.faa)
                    obj.results.faa = newFAAResult;
                else
                    obj.results.faa(end+1) = newFAAResult;
                end
            end
        end
        
        function processABRatioFeatures(obj, preprocessedSegment)
            if ~obj.params.feature.abRatio.enable || isempty(preprocessedSegment)
                return;
            end

            % α/β比の計算
            [abRatio, arousalState] = obj.abRatioExtractor.calculateABRatio(preprocessedSegment);
            
            currentTime = toc(obj.processingTimer)*1000;
            newABRatioResult = struct(...
                'ratio', abRatio, ...
                'state', arousalState, ...
                'time', currentTime, ...
                'sample', obj.currentTotalSamples);

            if isempty(obj.results.abRatio)
                obj.results.abRatio = newABRatioResult;
            else
                obj.results.abRatio(end+1) = newABRatioResult;
            end
        end
        
        function processEmotionFeatures(obj, preprocessedSegment)
            if ~obj.params.feature.emotion.enable || isempty(preprocessedSegment)
                return;
            end

            emotionResults = obj.emotionExtractor.classifyEmotion(preprocessedSegment);
            if iscell(emotionResults)
                emotionResult = emotionResults{1};
            else
                emotionResult = emotionResults;
            end
            
            currentTime = toc(obj.processingTimer)*1000;
            newEmotionResult = struct(...
                'state', emotionResult.state, ...
                'coordinates', emotionResult.coordinates, ...
                'emotionCoords', emotionResult.emotionCoords, ...
                'time', currentTime, ...
                'sample', obj.currentTotalSamples);

            if isempty(obj.results.emotion)
                obj.results.emotion = newEmotionResult;
            else
                obj.results.emotion(end+1) = newEmotionResult;
            end
        end
        
        function currentFeatures = processCSPFeatures(obj, preprocessedSegment)
            if ~obj.params.feature.csp.enable || isempty(preprocessedSegment)
                currentFeatures = [];
                return;
            end

            try
                % データとフィルタのサイズを確認
                [n_channels, n_samples] = size(preprocessedSegment);
                [filter_rows, filter_cols] = size(obj.results.csp.filters);

                % CSP特徴量の抽出
                currentFeatures = obj.cspExtractor.extractFeatures(...
                    preprocessedSegment, obj.results.csp.filters);
                
                currentTime = toc(obj.processingTimer)*1000;
                if ~isempty(currentFeatures)
                    % CSP特徴量を結果に保存
                    newCSPResult = struct(...
                        'features', currentFeatures, ...
                        'time', currentTime, ...
                        'sample', obj.currentTotalSamples);

                    if isempty(obj.results.csp.features)
                        obj.results.csp.features = newCSPResult;
                    else
                        obj.results.csp.features(end+1) = newCSPResult;
                    end
                end
            catch ME
                % より詳細なエラー情報を提供
                warning('CSPFeatures:Error', 'Error in processCSPFeatures: %s\nData size: %dx%d, Filter size: %dx%d', ...
                    ME.message, n_channels, n_samples, filter_rows, filter_cols);
                currentFeatures = [];
            end
        end
        
        function [label, score] = processClassification(obj, currentFeatures)
            label = [];
            score = [];

            try
                switch obj.params.classifier.activeClassifier
                    case 'svm'
                        % SVMモデルの存在確認
                        if ~isfield(obj.classifiers, 'svm') || isempty(obj.classifiers.svm.model)
                            fprintf('Warning: SVM model not found\n');
                            return;
                        end

                        [label, score] = obj.svmClassifier.predictOnline(...
                            currentFeatures, obj.classifiers.svm.model, obj.optimalThreshold);

                    case 'ecoc'
                        [label, score] = obj.ecocClassifier.predictOnline(...
                            currentFeatures, obj.classifiers.ecoc.model);

                    case 'cnn'
                        [label, score] = obj.cnnClassifier.predictOnline(...
                            currentFeatures, obj.classifiers.cnn.model);
                end

                % 予測結果の保存
                if ~isempty(label)
                    currentTime = toc(obj.processingTimer)*1000;
                    newPredictResult = struct(...
                        'label', label, ...
                        'score', score, ...
                        'time', currentTime, ...
                        'classifier', obj.params.classifier.activeClassifier);

                    % SVMの場合のみ閾値情報を追加
                    if strcmp(obj.params.classifier.activeClassifier, 'svm') && obj.params.classifier.svm.probability
                        newPredictResult.threshold = obj.optimalThreshold;
                    end

                    if isempty(obj.results.predict)
                        obj.results.predict = newPredictResult;
                    else
                        obj.results.predict(end+1) = newPredictResult;
                    end
                end
            catch ME
                fprintf('Error in processClassification: %s\n', ME.message);
                warning(ME.identifier, '%s', ME.message);
            end
        end

        function sendResults(obj, udpData)
            if ~isempty(udpData)
                obj.udpManager.sendTrigger(udpData);
            end
        end
        
        function handleError(~, ME)
            warning(ME.identifier, 'Error in processOnline: %s\n', ME.message);
            fprintf('Error stack:\n');
            for k = 1:length(ME.stack)
                fprintf('File: %s, Line: %d, Function: %s\n', ...
                    ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
            end
        end
        
        function processGUI(obj)
            try
                displayData = struct();
                displaySeconds = obj.params.gui.display.visualization.scale.displaySeconds;
                displaySamples = round(displaySeconds * obj.params.device.sampleRate);

                % 生データの保存
                if obj.params.gui.display.visualization.enable.rawData
                    displayData.rawData = obj.dataBuffer(:, end-displaySamples+1:end);
                end

                % 前処理パイプラインの実行
                if obj.params.gui.display.visualization.enable.processedData || ...
                   obj.params.gui.display.visualization.enable.spectrum || ...
                   obj.params.gui.display.visualization.enable.ersp

                    preprocessedSegment = obj.preprocessSignal();
                    
                    % 処理ウィンドウの抽出
                    endIdx = size(preprocessedSegment, 2);
                    startIdx = endIdx - displaySamples + 1;
                    if startIdx > 0
                        processedBuffer = preprocessedSegment(:, startIdx:endIdx);
                    end

                    if ~isempty(processedBuffer)
                        % 処理済みデータの保存
                        if obj.params.gui.display.visualization.enable.processedData
                            displayData.processedData = preprocessedSegment;
                        end

                        % パワースペクトル計算
                        if obj.params.gui.display.visualization.enable.spectrum
                            [pxx, f] = obj.powerExtractor.calculateSpectrum(processedBuffer);
                            displayData.spectrum = struct('pxx', pxx, 'f', f);
                        end

                        % ERSP計算
                        if obj.params.gui.display.visualization.enable.ersp
                            [ersp, times, freqs] = obj.powerExtractor.calculateERSP(processedBuffer);
                            displayData.ersp = struct('ersp', ersp, 'times', times, 'freqs', freqs);
                        end

                    end
                end

                % GUI更新
                obj.guiController.updateDisplayData(displayData);

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function updateSliderValue(obj)
            sliderValue = obj.guiController.getSliderValue();
            if ~isempty(sliderValue)
                currentTime = toc(obj.processingTimer)*1000;
                obj.currentTotalSamples = obj.totalSampleCount + size(obj.rawData, 2);

                trigger = struct(...
                    'value', sliderValue, ...
                    'time', uint64(currentTime), ...
                    'sample', obj.currentTotalSamples);
                obj.labels = [obj.labels; trigger];
            end
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
        
        % ヘルパー関数：最新の結果を取得
        function latest = getLatestFeature(~, data)
            if isempty(data)
                latest = [];
                return;
            end

            if isstruct(data)
                % 構造体配列の場合、最後の要素を取得
                latest = data(end);

                % CSP特徴量の特別処理
                if isfield(latest, 'features')
                    latest = latest.features;
                end
            elseif isnumeric(data)
                % 数値配列の場合、最後の行を取得
                latest = data(end,:);
            else
                % その他のデータ型の場合は空を返す
                latest = [];
            end
        end
        
        % 配列の各要素に対して小数点以下2桁に制限するヘルパー関数
        function formattedArray = formatArray(data, precision)
            if nargin < 2
                precision = 2;  % デフォルトは小数点以下2桁
            end

            % 配列の各要素に対してフォーマットを適用
            formattedArray = arrayfun(@(x) str2double(sprintf('%.*f', precision, x)), data);
        end
    end
end