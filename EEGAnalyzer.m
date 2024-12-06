classdef EEGAnalyzer < handle
    % 使用例：params = getConfig('epocx');  analysis = EEGAnalyzer(params);
    % UDP系エラー初期化：clear all; instrreset;
    
    properties (Access = private)
        params              % 設定パラメータ
        signalProcessor     % 信号処理器
        featureExtractor   % 特徴抽出器
        classifier         % 分類器
        dataManager        % データ管理
        
        % データ保持用プロパティ
        rawData           % 生データ
        labels            % ラベルデータ
        processedData     % 処理済みデータ
        processedLabel    % 処理済みラベル
        cspFilters       % CSPフィルタ
        cspFeatures      % CSP特徴量
        processingInfo    % 処理情報
    end
    
    methods (Access = public)
        function obj = EEGAnalyzer(params)
            obj.params = params;
            obj.initializeComponents();
            
        end
        
        function executeAnalysis(obj)
            try
                % データの読み込み
                loadedData = loadDataBrowser(obj);
                
                % 読み込んだデータを設定
                if isfield(loadedData, 'rawData')
                    obj.rawData = loadedData.rawData;
                end
                if isfield(loadedData, 'labels')
                    obj.labels = loadedData.labels;
                end
                
                % 信号処理の実行
                if obj.params.signal.enable
                    [obj.processedData, obj.processedLabel, obj.processingInfo] = ...
                        obj.signalProcessor.process(obj.rawData, obj.labels);
                    if ~isempty(obj.processedData)
                        obj.mergeAndSaveData();
                        fprintf('Processed data and previous data saved\n');
                    end
                end

                % 特徴抽出の実行
                if ~isempty(obj.labels) && ...
                   isfield(obj.params.feature, 'csp') && ...
                   isfield(obj.params.feature.csp, 'enable') && ...
                   obj.params.feature.csp.enable
                    % CSPフィルタの学習
                    obj.cspFilters = obj.featureExtractor.trainCSP(obj.processedData, obj.processedLabel);
                    % 学習したフィルタを用いて特徴抽出
                    obj.cspFeatures = obj.featureExtractor.extractFeatures(obj.processedData);
                    if ~isempty(obj.cspFeatures)
                        obj.mergeAndSaveData();
                        fprintf('CSP Features and previous data saved\n');
                    end
                end

                % 分類処理の実行
                if ~isempty(obj.cspFeatures) && ~isempty(obj.processedLabel) && ...
                   isfield(obj.params.classifier, 'svm') && ...
                   isfield(obj.params.classifier.svm, 'enable') && ...
                   obj.params.classifier.svm.enable

               assert(isnumeric(obj.processedLabel) && ...
                        size(obj.processedLabel, 2) == 1, ...
                        'ProcessedLabel must be a column vector of type double');
                    % 分類器の学習を実行
                    obj.classifier.trainOffline(obj.cspFeatures, obj.processedLabel);
                    obj.mergeAndSaveData();
                    fprintf('Classifier training completed and data saved\n');
                end

            catch ME
                error('Analysis failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function initializeComponents(obj)
            obj.signalProcessor = SignalProcessor(obj.params);
            obj.featureExtractor = CSPFeatureExtractor(obj.params);
            obj.classifier = SVMClassifier(obj.params);
            obj.dataManager = DataManager(obj.params);
            
            obj.executeAnalysis()
        end
        
        function mergeAndSaveData(obj)
            try
                saveData = struct();
                
                saveData.params = obj.params;
                saveData.rawData = obj.rawData;
                saveData.labels = obj.labels;     
                saveData.metadata = struct(...
                    'samplingRate', obj.params.device.sampleRate, ...
                    'channelCount', obj.params.device.channelCount, ...
                    'channelLabels', {obj.params.device.channels}, ...
                    'recordingTime', datetime('now'), ...
                    'deviceType', obj.params.device.name ...
                );
            
                if ~isempty(obj.processedData)
                    saveData.processedData = obj.processedData;
                    saveData.processedLabel = obj.processedLabel;
                    saveData.processingInfo = obj.processingInfo;
                end
                if ~isempty(obj.cspFeatures)
                    saveData.cspFeatures = obj.cspFeatures;
                    saveData.cspFilters = obj.cspFilters;
                end
                if ~isempty(obj.classifier) && ~isempty(obj.classifier.svmModel)
                    saveData.svmClassifier = obj.classifier.svmModel;
                    if ~isempty(obj.classifier.performance)
                        saveData.results = obj.classifier.performance;
                    end
                end

                % 保存したデータの概要を表示
                obj.displaySavedDataSummary(saveData);
                
                obj.dataManager.saveDataset(saveData);
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
                fprintf('Save operation failed: %s\n', ME.message);
            end
        end

        function loadedData = loadDataBrowser(obj)
            try
                [filename, pathname] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, ...
                    'Select EEG data file', obj.params.acquisition.save.path);
                
                if filename ~= 0
                    fullpath = fullfile(pathname, filename);
                    loadedData = obj.dataManager.loadDataset(fullpath);
                    fprintf('Successfully loaded data from: %s\n', fullpath);
                    fprintf('Raw data size: [%s], Labels: %d\n', ...
                        num2str(size(obj.rawData)), length(obj.labels));
                end
            catch ME
                error('Failed to load data: %s', ME.message);
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
    end
end