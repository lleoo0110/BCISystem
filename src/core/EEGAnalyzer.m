classdef EEGAnalyzer < handle
    properties (Access = private)
        % データ保持用
        params              % パラメータ設定
        rawData           % 生データ
        labels            % ラベル
        processedData     % 処理済みデータ
        processedLabel    % 処理済みラベル
        processingInfo    % 処理情報
        baselineData      % ベースラインデータ
        svm              % SVM分類器の結果
        ecoc             % ECOC分類器の結果
        cnn              % CNN分類器の結果
        results          % 解析結果
        
        % 前処理コンポーネント
        artifactRemover     % アーティファクト除去
        baselineCorrector   % ベースライン補正
        dataAugmenter        % データ拡張
        downSampler        % ダウンサンプリング
        firFilter          % FIRフィルタ
        notchFilter        % ノッチフィルタ
        normalizer         % 正規化
        epoching          % エポック化コンポーネント
        
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
        
        %  データ管理コンポーネント
        dataManager
    end
    
    methods (Access = public)
        function obj = EEGAnalyzer(params)
            % コンストラクタ
            obj.params = params;
            
            % 前処理コンポーネントの初期化
            obj.initializePreprocessors();
            
            % 特徴抽出器の初期化
            obj.initializeExtractors();
            
            % 特徴分類器の初期化
            obj.initializeClassifiers();
            
            % 結果構造体の初期化
            obj.initializeResults();
            
            % データ管理の初期化
            obj.dataManager = DataManager(params);
        end
        
        function analyze(obj)
            try
                fprintf('解析対象のデータファイルを選択してください．\n');
                loadedData = DataLoader.loadDataBrowserWithPrompt('オフライン解析');
                obj.setData(loadedData);
                obj.executePreprocessingPipeline();
                
                if obj.params.signal.enable && ~isempty(obj.processedData)
                    obj.extractFeatures();
                    obj.performClassification();
                end

                obj.saveResults();
                
                % 全てのウィンドウを閉じる
                close all;
            catch ME
                error('Analysis failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function initializePreprocessors(obj)
            % 前処理コンポーネントの初期化
            obj.artifactRemover = ArtifactRemover(obj.params);
            obj.baselineCorrector = BaselineCorrector(obj.params);
            obj.dataAugmenter = DataAugmenter(obj.params);
            obj.downSampler = DownSampler(obj.params);
            obj.firFilter = FIRFilterDesigner(obj.params);
            obj.notchFilter = NotchFilterDesigner(obj.params);
            obj.normalizer = EEGNormalizer(obj.params);
            obj.epoching = Epoching(obj.params);
        end
        
        function initializeExtractors(obj)
            % 特徴抽出器の初期化
            obj.powerExtractor = PowerExtractor(obj.params);
            obj.faaExtractor = FAAExtractor(obj.params);
            obj.abRatioExtractor = ABRatioExtractor(obj.params);
            obj.cspExtractor = CSPExtractor(obj.params);
            obj.emotionExtractor = EmotionExtractor(obj.params);
        end
        
        function initializeClassifiers(obj)
            obj.svmClassifier = SVMClassifier(obj.params);
            obj.ecocClassifier = ECOCClassifier(obj.params);
            obj.cnnClassifier = CNNClassifier(obj.params);
        end
        
        function initializeResults(obj)
            % 結果構造体の初期化
            obj.results = struct(...
                'power', [], ...     % パワー解析結果
                'faa', [], ...      % FAA解析結果
                'abRatio', [], ...  % α/β比解析結果
                'emotion', [], ...  % 感情分析結果
                'csp', struct(...   % CSP関連の結果
                    'filters', [], ...
                    'features', [], ...
                    'parameters', struct(...
                        'numFilters', obj.params.feature.csp.patterns, ...
                        'regularization', obj.params.feature.csp.regularization, ...
                        'method', 'standard' ...
                    ) ...
                ) ...
            );

            % 分類器結果の初期化
            obj.initializeClassifierResults();
        end
        
        function initializeClassifierResults(obj)
            % 分類器結果構造体の基本構造
            classifierStruct = struct(...
                'model', [], ...
                'performance', struct(...
                    'overallAccuracy', [], ...
                    'crossValidation', struct(...
                        'accuracy', [], ...
                        'std', [] ...
                    ), ...
                    'precision', [], ...
                    'recall', [], ...
                    'f1score', [], ...
                    'auc', [], ...
                    'confusionMatrix', [] ...
                ), ...
                'trainingInfo', [], ...
                'crossValidation', [], ...
                'overfitting', [] ...
            );

            % 各分類器の結果を初期化
            obj.svm = classifierStruct;
            obj.ecoc = classifierStruct;
            obj.cnn = classifierStruct;
        end
        
        function setData(obj, loadedData)
            % 必須フィールドの確認
            requiredFields = {'rawData', 'labels'};
            for i = 1:length(requiredFields)
                if ~isfield(loadedData, requiredFields{i})
                    error('Required field %s not found in loaded data', requiredFields{i});
                end
            end

            % データの設定
            obj.rawData = loadedData.rawData;
            obj.labels = loadedData.labels;           
        end
        
        function executePreprocessingPipeline(obj)
            try
                data = obj.rawData;
                obj.processingInfo = struct('startTime', datestr(now), 'dataSize', size(data));

                % ダウンサンプリング
                if obj.params.signal.preprocessing.downsample.enable
                    [data, info] = obj.downSampler.downsample(data, obj.params.signal.preprocessing.downsample.targetRate);
                    obj.processingInfo.downsample = info;
                end

                % アーティファクト除去
                if obj.params.signal.preprocessing.artifact.enable
                    [data, info] = obj.artifactRemover.removeArtifacts(data, 'all');
                    obj.processingInfo.artifact = info;
                end

                % ベースライン補正
                if obj.params.signal.preprocessing.baseline.enable
                    [data, info] = obj.baselineCorrector.correctBaseline(data, obj.params.signal.preprocessing.baseline.method);
                    obj.processingInfo.baseline = info;
                end

                % フィルタリング
                if obj.params.signal.preprocessing.filter.notch.enable
                    [data, info] = obj.notchFilter.designAndApplyFilter(data);
                    obj.processingInfo.notchFilter = info;
                end

                if obj.params.signal.preprocessing.filter.fir.enable
                    [data, info] = obj.firFilter.designAndApplyFilter(data);
                    obj.processingInfo.firFilter = info;
                end

                % 正規化
                if obj.params.signal.preprocessing.normalize.enable
                    [data, info] = obj.normalizer.normalize(data);
                    obj.processingInfo.normalize = info;
                end

                % エポック化
                [epochs, epochLabels, info] = obj.epoching.epoching(data, obj.labels);
                obj.processingInfo.epoch = info;
                
                % データ拡張
                if obj.params.signal.preprocessing.augmentation.enable
                    [augData, augLabels, info] = obj.dataAugmenter.augmentData(epochs, epochLabels);
                    obj.processingInfo.augmentation = info;
                    obj.processedLabel = augLabels;
                    epochs = augData;
                else
                    obj.processedLabel = epochLabels;
                end
                
                obj.processedData = epochs;

            catch ME
                error('Preprocessing pipeline failed: %s', ME.message);
            end
        end
        
        function extractFeatures(obj)
            try
                % CSP特徴抽出
                if obj.params.feature.csp.enable && ~isempty(obj.processedData)
                    % CSPフィルタの学習
                    [filters, parameters] = obj.cspExtractor.trainCSP(...
                        obj.processedData, obj.processedLabel);

                    % 特徴量の抽出
                    if ~isempty(filters)
                        features = obj.cspExtractor.extractFeatures(...
                            obj.processedData, filters);

                        % 結果の保存
                        obj.results.csp.filters = filters;
                        obj.results.csp.features = features;
                        obj.results.csp.parameters = parameters;
                    end
                end

                % パワー特徴量の抽出
                if obj.params.feature.power.enable
                    obj.extractPowerFeatures();
                end

                % FAA特徴量の抽出
                if obj.params.feature.faa.enable
                    obj.extractFAAFeatures();
                end
                
                % α/β特徴量の抽出
                if obj.params.feature.abRatio.enable
                    obj.extractABRatioFeatures();
                end

                % 感情特徴量の抽出
                if obj.params.feature.emotion.enable
                    obj.extractEmotionFeatures();
                end

            catch ME
                error('Feature extraction failed: %s', ME.message);
            end
        end
        
        function performClassification(obj)
            try
                % CSP特徴抽出→SVM特徴分類
                if ~isempty(obj.results.csp.features)
                    if obj.params.classifier.svm.enable
                        obj.svm = obj.svmClassifier.trainSVM(...
                            obj.results.csp.features, obj.processedLabel);
                    end
                    if obj.params.classifier.ecoc.enable
                        obj.ecoc = obj.ecocClassifier.trainECOC(...
                            obj.results.csp.features, obj.processedLabel);
                    end
                end
                
                % CNN特徴分類
                if obj.params.classifier.cnn.enable
                    obj.cnn = obj.cnnClassifier.trainCNN(...
                        obj.processedData, obj.processedLabel);
                end
            catch ME
                error('Classification failed: %s', ME.message);
            end
        end

        function extractPowerFeatures(obj)
            try
                if isempty(obj.processedData) || isempty(obj.labels)
                    warning('PowerExtractor:NoData', 'データまたはラベルが空です');
                    return;
                end

                % データの形状を確認
                numEpochs = size(obj.processedData, 3);

                % タイミング情報の取得
                timings = zeros(numEpochs, 1);
                samples = zeros(numEpochs, 1);

                for i = 1:length(obj.labels)
                    if i <= numEpochs
                        timings(i) = obj.labels(i).time;
                        samples(i) = obj.labels(i).sample;
                    end
                end

                % エポックごとの処理
                for epoch = 1:numEpochs
                    % 周波数帯域ごとのパワーを計算
                    bandNames = obj.params.feature.power.bands.names;
                    if iscell(bandNames{1})
                        bandNames = bandNames{1};
                    end

                    % 各周波数帯域のパワーを計算
                    bandPowers = struct();
                    for i = 1:length(bandNames)
                        bandName = bandNames{i};
                        freqRange = obj.params.feature.power.bands.(bandName);
                        bandPowers.(bandName) = obj.powerExtractor.calculatePower(...
                            obj.processedData(:,:,epoch), freqRange);
                    end

                    % 新しい結果の構築
                    newResult = struct(...
                        'powers', bandPowers, ...
                        'bands', {bandNames}, ...
                        'time', timings(epoch), ...
                        'sample', samples(epoch));

                    % 結果の追加
                    if isempty(obj.results.power)
                        obj.results.power = newResult;
                    else
                        obj.results.power(end+1) = newResult;
                    end
                end

                fprintf('パワー特徴量の抽出が完了しました（%d エポック）\n', numEpochs);

            catch ME
                error('PowerExtractor:ExtractionFailed', 'パワー特徴量の抽出に失敗しました: %s', ME.message);
            end
        end
        
        function extractFAAFeatures(obj)
            try
                if isempty(obj.processedData) || isempty(obj.labels)
                    warning('FAAExtractor:NoData', 'データまたはラベルが空です');
                    return;
                end

                numEpochs = size(obj.processedData, 3);
                timings = zeros(numEpochs, 1);
                samples = zeros(numEpochs, 1);

                % タイミング情報の取得
                for i = 1:length(obj.labels)
                    if i <= numEpochs
                        timings(i) = obj.labels(i).time;
                        samples(i) = obj.labels(i).sample;
                    end
                end

                % エポックごとのFAA値を計算
                for epoch = 1:numEpochs
                    faaResults = obj.faaExtractor.calculateFAA(obj.processedData(:,:,epoch));

                    % 新しい結果の構築
                    newResult = struct(...
                        'faa', faaResults{1}.faa, ...
                        'arousal', faaResults{1}.arousal, ...
                        'time', timings(epoch), ...
                        'sample', samples(epoch));

                    % 結果の追加
                    if isempty(obj.results.faa)
                        obj.results.faa = newResult;
                    else
                        obj.results.faa(end+1) = newResult;
                    end
                end

                fprintf('FAA特徴量の抽出が完了しました（%d エポック）\n', numEpochs);

            catch ME
                error('FAAExtractor:ExtractionFailed', 'FAA特徴量の抽出に失敗しました: %s', ME.message);
            end
        end

        function extractABRatioFeatures(obj)
            try
                if isempty(obj.processedData) || isempty(obj.labels)
                    warning('ABRatioExtractor:NoData', 'データまたはラベルが空です');
                    return;
                end

                % データの形状を確認
                numEpochs = size(obj.processedData, 3);

                % ラベルからタイムスタンプとサンプル情報を取得
                timings = zeros(numEpochs, 1);
                samples = zeros(numEpochs, 1);

                for i = 1:length(obj.labels)
                    if i <= numEpochs
                        timings(i) = obj.labels(i).time;
                        samples(i) = obj.labels(i).sample;
                    end
                end

                % エポックごとの処理
                for epoch = 1:numEpochs
                    % α/β比の計算
                    [abRatio, arousalState] = obj.abRatioExtractor.calculateABRatio(...
                        obj.processedData(:,:,epoch));

                    % 新しい結果の構築
                    newResult = struct(...
                        'ratio', abRatio, ...
                        'state', arousalState, ...
                        'time', timings(epoch), ...     % ラベルから取得したタイムスタンプ
                        'sample', samples(epoch));      % ラベルから取得したサンプル番号

                    % 結果の追加
                    if isempty(obj.results.abRatio)
                        obj.results.abRatio = newResult;
                    else
                        obj.results.abRatio(end+1) = newResult;
                    end
                end

                % 処理結果のサマリーを表示
                fprintf('α/β比の特徴抽出が完了しました（%d エポック）\n', numEpochs);

            catch ME
                error('ABRatioExtractor:ExtractionFailed', ME.message);
            end
        end

        function extractEmotionFeatures(obj)
            try
                if isempty(obj.processedData) || isempty(obj.labels)
                    warning('EmotionExtractor:NoData', 'データまたはラベルが空です');
                    return;
                end

                numEpochs = size(obj.processedData, 3);
                timings = zeros(numEpochs, 1);
                samples = zeros(numEpochs, 1);

                % タイミング情報の取得
                for i = 1:length(obj.labels)
                    if i <= numEpochs
                        timings(i) = obj.labels(i).time;
                        samples(i) = obj.labels(i).sample;
                    end
                end

                % エポックごとの感情状態を分類
                for epoch = 1:numEpochs
                    emotionResult = obj.emotionExtractor.classifyEmotion(obj.processedData(:,:,epoch));

                    % 新しい結果の構築
                    newResult = struct(...
                        'state', emotionResult{1}.state, ...
                        'coordinates', emotionResult{1}.coordinates, ...
                        'emotionCoords', emotionResult{1}.emotionCoords, ...
                        'time', timings(epoch), ...
                        'sample', samples(epoch));

                    % 結果の追加
                    if isempty(obj.results.emotion)
                        obj.results.emotion = newResult;
                    else
                        obj.results.emotion(end+1) = newResult;
                    end
                end

                fprintf('感情特徴量の抽出が完了しました（%d エポック）\n', numEpochs);

            catch ME
                error('EmotionExtractor:ExtractionFailed', '感情特徴量の抽出に失敗しました: %s', ME.message);
            end
        end

        function saveResults(obj)
            try
                saveData = struct();

                % 基本データの保存
                saveData.params = obj.params;
                saveData.rawData = obj.rawData;
                saveData.labels = obj.labels;
                saveData.processedData = obj.processedData;
                saveData.processedLabel = obj.processedLabel;
                saveData.processingInfo = obj.processingInfo;

                % 特徴抽出結果
                if ~isempty(obj.results)
                    saveData.results = obj.results;
                end

                % 分類器結果の保存
                saveData.classifier = struct();

                % SVM結果
                if obj.params.classifier.svm.enable && ~isempty(obj.svm)
                    saveData.classifier.svm = struct(...
                        'model', obj.svm.model, ...
                        'performance', obj.svm.performance);
                end

                % ECOC結果
                if obj.params.classifier.ecoc.enable && ~isempty(obj.ecoc)
                    saveData.classifier.ecoc = struct(...
                        'model', obj.ecoc.model, ...
                        'performance', obj.ecoc.performance);
                end

                % CNN結果
                if obj.params.classifier.cnn.enable && ~isempty(obj.cnn)
                    saveData.classifier.cnn = struct(...
                        'model', obj.cnn.model, ...
                        'performance', obj.cnn.performance, ...
                        'trainingInfo', obj.cnn.trainingInfo, ...
                        'crossValidation', obj.cnn.crossValidation, ...
                        'overfitting', obj.cnn.overfitting ...
                    );
                end

                savedFile = obj.dataManager.saveDataset(saveData);
                fprintf('Results saved to: %s\n', savedFile);

            catch ME
                error('Failed to save results: %s', ME.message);
            end
        end
    end
end