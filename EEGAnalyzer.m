classdef EEGAnalyzer < handle
    properties (Access = private)
        params              % パラメータ設定
        signalProcessor     % 信号処理
        powerExtractor     % パワー特徴抽出
        featureExtractor   % CSP特徴抽出
        classifier         % SVM分類器
        dataManager        % データ管理
        
        % データ保持用
        rawData           % 生データ
        labels            % ラベル
        processedData     % 処理済みデータ
        processedLabel    % 処理済みラベル
        processingInfo    % 処理情報
        baselineData      % ベースラインデータ
        cspfilters       % CSPフィルタ
        cspfeatures      % CSP特徴量
        svmclassifier    % SVM分類器
        results          % 解析結果
    end
    
    methods (Access = public)
        function obj = EEGAnalyzer(params)
            % コンストラクタ
            obj.params = params;
            
            % 各プロセッサの初期化
            obj.signalProcessor = SignalProcessor(params);
            obj.powerExtractor = PowerFeatureExtractor(params);
            obj.featureExtractor = CSPFeatureExtractor(params);
            obj.classifier = SVMClassifier(params);
            obj.dataManager = DataManager(params);
            
            % 結果構造体の初期化
            obj.results = struct('predict', [], 'power', [], 'faa', [], ...
                'erd', [], 'emotion', []);
        end
        
        function analyze(obj)
            try
                % 解析用データの読み込み
                fprintf('解析対象のデータファイルを選択してください．\n');
                loadedData = DataLoading.loadDataBrowserWithPrompt('classifier');
                
                % 読み込んだデータの確認と設定
                if isfield(loadedData, 'rawData')
                    obj.rawData = loadedData.rawData;
                else
                    error('Raw data not found in the loaded file');
                end
                
                if isfield(loadedData, 'labels')
                    obj.labels = loadedData.labels;
                else
                    error('Labels not found in the loaded file');
                end
                
                % ベースラインデータの読み込み（ERD計算が有効な場合のみ）
                if obj.params.feature.erd.enable
                    fprintf('ベースラインデータファイルを選択してください．\n');
                    baselineData = DataLoading.loadDataBrowserWithPrompt('baseline');
                    if isfield(baselineData, 'processedData')
                        obj.baselineData = baselineData.processedData;
                        % ベースラインパワーの計算
                        obj.powerExtractor.calculateBaseline(obj.baselineData);
                    else
                        error('Processed data for baseline not found in the loaded file');
                    end
                end
                
                % 信号処理の実行
                if obj.params.signal.enable && ~isempty(obj.rawData)
                    [obj.processedData, obj.processedLabel, obj.processingInfo] = ...
                        obj.signalProcessor.process(obj.rawData, obj.labels);
                    
                    % パワー値の計算
                    if obj.params.feature.power.enable && ~isempty(obj.processedData)
                        obj.results.power = obj.powerExtractor.calculatePowerForAllBands(obj.processedData);
                        
                        % 結果の表示
                        for epoch = 1:length(obj.results.power)
                            fprintf('Epoch %d Power Results:\n', epoch);
                            bandNames = fieldnames(obj.results.power{epoch});
                            for i = 1:length(bandNames)
                                fprintf('  %s: %.4f\n', bandNames{i}, ...
                                    mean(obj.results.power{epoch}.(bandNames{i})));
                            end
                        end
                    end
                    
                    % FAA計算
                    if obj.params.feature.faa.enable && ~isempty(obj.processedData)
                        obj.results.faa = obj.powerExtractor.calculateFAA(obj.processedData);
                        
                        % 結果の表示
                        for epoch = 1:length(obj.results.faa)
                            fprintf('Epoch %d - FAA Value: %.4f, Arousal State: %s\n', ...
                                epoch, obj.results.faa{epoch}.faa, obj.results.faa{epoch}.arousal);
                        end
                    end
                    
                    % ERD計算
                    if obj.params.feature.erd.enable && ~isempty(obj.processedData)
                        obj.results.erd = obj.powerExtractor.calculateERD(obj.processedData);
                        
                        % 結果の表示
                        for epoch = 1:length(obj.results.erd)
                            fprintf('Epoch %d ERD Results:\n', epoch);
                            bandNames = fieldnames(obj.results.erd{epoch}.values);
                            for i = 1:length(bandNames)
                                fprintf('  %s band - ERD: %.2f%% (Value: %.4f)\n', ...
                                    bandNames{i}, ...
                                    mean(obj.results.erd{epoch}.percent.(bandNames{i})), ...
                                    mean(obj.results.erd{epoch}.values.(bandNames{i})));
                            end
                        end
                    end
                    
                    % 感情状態の分類
                    if obj.params.feature.emotion.enable && ~isempty(obj.processedData)
                        obj.results.emotion = obj.powerExtractor.classifyEmotion(obj.processedData);
                        
                        % 結果の表示
                        for epoch = 1:length(obj.results.emotion)
                            fprintf('Epoch %d - Emotion State: %s\n', epoch, ...
                                obj.results.emotion{epoch}.state);
                            fprintf('  Valence: %.2f, Arousal: %.2f\n', ...
                                obj.results.emotion{epoch}.coordinates.valence, ...
                                obj.results.emotion{epoch}.coordinates.arousal);
                        end
                    end
                    
                    % CSP特徴抽出
                    if obj.params.feature.csp.enable && ~isempty(obj.processedData)
                        % processedDataとprocessedLabelの形式を確認
                        fprintf('Data format: %s\n', class(obj.processedData));
                        if iscell(obj.processedData)
                            fprintf('Cell array size: %d\n', length(obj.processedData));
                        else
                            fprintf('Array dimensions: %dx%dx%d\n', size(obj.processedData));
                        end

                        % CSPフィルタの学習
                        obj.cspfilters = obj.featureExtractor.trainCSP(...
                            obj.processedData, obj.processedLabel);

                        if ~isempty(obj.cspfilters)
                            % 特徴量の抽出
                            obj.cspfeatures = obj.featureExtractor.extractFeatures(...
                                obj.processedData, obj.cspfilters);
                        end
                    end
                    
                    % 分類器の学習
                    if obj.params.classifier.svm.enable && ~isempty(obj.cspfeatures)
                        classifierResults = obj.classifier.trainOffline(...
                            obj.cspfeatures, obj.processedLabel);
                        obj.results.performance = classifierResults.performance;
                        obj.svmclassifier = classifierResults.classifier;
                    end
                    
                    % 結果の保存
                    obj.saveResults();
                end
                
            catch ME
                error('Analysis failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function calculatePowerFeatures(obj)
            % パワー値の計算と保存
            bandNames = obj.params.feature.power.bands.names;
            if iscell(bandNames{1})
                bandNames = bandNames{1};
            end
            
            bandPowers = struct();
            for i = 1:length(bandNames)
                bandName = bandNames{i};
                freqRange = obj.params.feature.power.bands.(bandName);
                bandPower = obj.powerExtractor.calculatePower(obj.processedData, freqRange);
                bandPowers.(bandName) = bandPower;
            end
            
            obj.results.power = struct('bands', bandPowers);
        end
        
        function calculateFAA(obj)
            % FAA値の計算と保存
            [faaValue, arousalState] = obj.powerExtractor.calculateFAA(obj.processedData);
            obj.results.faa = struct('faa', faaValue, 'arousal', arousalState);
            fprintf('FAA Value: %.4f, Arousal State: %s\n', faaValue, arousalState);
        end
        
        function calculateERD(obj)
            % ERDの計算と保存
            [erdValues, erdPercent] = obj.powerExtractor.calculateERD(obj.processedData);
            obj.results.erd = struct('values', erdValues, 'percent', erdPercent);
            
            % ERD値の結果表示
            bandNames = fieldnames(erdValues);
            for i = 1:length(bandNames)
                bandName = bandNames{i};
                fprintf('%s band - ERD: %.2f%% (Value: %.4f)\n', ...
                    bandName, ...
                    mean(erdPercent.(bandName)), ...
                    mean(erdValues.(bandName)));
            end
        end
        
        function classifyEmotion(obj)
            % 感情状態の分類と保存
            [emotionState, coordinates] = obj.powerExtractor.classifyEmotion(obj.processedData);
            obj.results.emotion = struct('state', emotionState, 'coordinates', coordinates);
        end
        
        function saveResults(obj)
            % 保存データの構築
            saveData = struct();
            
            % 基本データの保存
            saveData.params = obj.params;
            saveData.rawData = obj.rawData;
            saveData.labels = obj.labels;
            
            % 処理済みデータの保存
            if ~isempty(obj.processedData)
                saveData.processedData = obj.processedData;
                saveData.processedLabel = obj.processedLabel;
                saveData.processingInfo = obj.processingInfo;
            end
            
            % ベースラインデータの保存
            if ~isempty(obj.baselineData)
                saveData.baselineData = obj.baselineData;
            end
            
            % CSP関連データの保存
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
                saveData.results = obj.results;
            end
            
            % DataManagerを使用してデータを保存
            obj.dataManager.saveDataset(saveData);
        end
    end
end