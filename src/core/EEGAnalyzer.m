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
                fprintf('\n=== 解析処理開始 ===\n');
                fprintf('解析対象のデータファイルを選択してください．\n');

                % DataLoaderを使用してデータを読み込み
                [loadedData, fileInfo] = DataLoader.loadDataBrowserWithPrompt('analysis');

                if isempty(loadedData)
                    fprintf('データが選択されませんでした。\n');
                    return;
                end

                fprintf('データ読み込み完了\n');
                fprintf('読み込んだファイル:\n');
                for i = 1:length(fileInfo.filenames)
                    fprintf('%d: %s\n', i, fileInfo.filenames{i});
                end

                % 複数ファイルの処理
                if length(loadedData) > 1
                    fprintf('複数ファイル処理を開始します（%d ファイル）\n', length(loadedData));

                    % 保存モードの選択
                    saveMode = questdlg('保存モードを選択してください:', ...
                        '保存モードの選択', '一括保存', '個別保存', 'キャンセル', '個別保存');

                    if strcmp(saveMode, 'キャンセル')
                        fprintf('処理がキャンセルされました。\n');
                        return;
                    end

                    fprintf('選択された保存モード: %s\n', saveMode);
                    fprintf('選択された読み込みモード: %s\n', fileInfo.loadMode);

                    % 保存先の設定
                    savePaths = cell(1, length(loadedData));

                    % 一括保存の場合のディレクトリ選択
                    if strcmp(saveMode, '一括保存')
                        saveDir = uigetdir('', '保存先のフォルダを選択してください');
                        if saveDir == 0
                            fprintf('保存先フォルダの選択がキャンセルされました。\n');
                            return;
                        end
                        fprintf('保存先フォルダ: %s\n', saveDir);
                    end

                    % 各ファイルの保存パスを設定
                    for i = 1:length(loadedData)
                        if ~isempty(loadedData{i})
                            % 元のファイル名から .mat を除去してタイムスタンプを追加
                            [~, baseFileName, ~] = fileparts(fileInfo.filenames{i});
                            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                            analysisFileName = [baseFileName '_analysis_' timestamp '.mat'];

                            if strcmp(saveMode, '個別保存')
                                [saveName, tempDir] = uiputfile(analysisFileName, ...
                                    sprintf('データセット %d/%d の保存先を選択', i, length(loadedData)));

                                if saveName == 0
                                    fprintf('ファイル %d の保存がキャンセルされました。\n', i);
                                    continue;
                                end
                                savePaths{i} = fullfile(tempDir, saveName);
                            else
                                savePaths{i} = fullfile(saveDir, analysisFileName);
                            end
                            fprintf('保存パス %d: %s\n', i, savePaths{i});
                        end
                    end

                    % データの処理
                    if strcmp(fileInfo.loadMode, 'batch')
                        % 統合データの処理
                        fprintf('\n=== 統合データの処理開始 ===\n');
                        obj.setData(loadedData{1}); % 統合データは最初の要素に格納されている
                        obj.executePreprocessingPipeline();
                        if obj.params.signal.enable && ~isempty(obj.processedData)
                            obj.extractFeatures();
                            obj.performClassification();
                        end
                        obj.saveResults(savePaths{1});
                    else
                        % 個別データの処理
                        for i = 1:length(loadedData)
                            if ~isempty(loadedData{i}) && ~isempty(savePaths{i})
                                fprintf('\n=== データセット %d/%d (%s) の処理開始 ===\n', ...
                                    i, length(loadedData), fileInfo.filenames{i});
                                obj.setData(loadedData{i});
                                obj.executePreprocessingPipeline();
                                if obj.params.signal.enable && ~isempty(obj.processedData)
                                    obj.extractFeatures();
                                    obj.performClassification();
                                end
                                obj.saveResults(savePaths{i});
                            end
                        end
                    end

                else
                    % 単一ファイルの処理
                    fprintf('\n=== 単一ファイル処理開始 ===\n');
                    fprintf('処理対象: %s\n', fileInfo.filenames{1});

                    [~, originalName, ~] = fileparts(fileInfo.filenames{1});
                    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                    defaultFileName = sprintf('%s_analysis_%s.mat', originalName, timestamp);

                    [saveName, saveDir] = uiputfile('*.mat', '保存先を選択してください', defaultFileName);
                    if saveName == 0
                        fprintf('保存がキャンセルされました。\n');
                        return;
                    end
                    savePath = fullfile(saveDir, saveName);

                    obj.setData(loadedData{1});
                    obj.executePreprocessingPipeline();
                    if obj.params.signal.enable && ~isempty(obj.processedData)
                        obj.extractFeatures();
                        obj.performClassification();
                    end
                    obj.saveResults(savePath);
                end

                close all;
                fprintf('\n=== 解析処理完了 ===\n');

            catch ME
                fprintf('\n=== エラー発生 ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラー発生場所:\n');
                for i = 1:length(ME.stack)
                    fprintf('  File: %s\n  Line: %d\n  Function: %s\n\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end
                rethrow(ME);
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
                % データ形式のチェックと必要な変換
                if isempty(obj.processedData)
                    error('処理済みデータが空です');
                end

                % データ形式の判定
                isEpochCell = iscell(obj.processedData);

                % CSP特徴抽出
                if obj.params.feature.csp.enable
                    % cell形式の場合は3D配列に変換
                    if isEpochCell
                        numEpochs = length(obj.processedData);
                        [channels, samples] = size(obj.processedData{1});
                        tempData = zeros(channels, samples, numEpochs);
                        for i = 1:numEpochs
                            tempData(:,:,i) = obj.processedData{i};
                        end

                        % CSPフィルタの学習
                        [filters, parameters] = obj.cspExtractor.trainCSP(tempData, obj.processedLabel);

                        % 特徴量の抽出
                        if ~isempty(filters)
                            features = obj.cspExtractor.extractFeatures(tempData, filters);

                            % 結果の保存
                            obj.results.csp.filters = filters;
                            obj.results.csp.features = features;
                            obj.results.csp.parameters = parameters;
                        end
                    else
                        % 従来の3D配列処理
                        [filters, parameters] = obj.cspExtractor.trainCSP(...
                            obj.processedData, obj.processedLabel);

                        if ~isempty(filters)
                            features = obj.cspExtractor.extractFeatures(...
                                obj.processedData, filters);

                            obj.results.csp.filters = filters;
                            obj.results.csp.features = features;
                            obj.results.csp.parameters = parameters;
                        end
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
                error('特徴抽出に失敗しました: %s', ME.message);
            end
        end
        
        function performClassification(obj)
            try
                if ~isempty(obj.results.csp.features)
                    % SVM分類
                    if obj.params.classifier.svm.enable
                        obj.svm = obj.svmClassifier.trainSVM(...
                            obj.results.csp.features, obj.processedLabel);
                    end

                    % ECOC分類
                    if obj.params.classifier.ecoc.enable
                        obj.ecoc = obj.ecocClassifier.trainECOC(...
                            obj.results.csp.features, obj.processedLabel);
                    end
                end

                % CNN分類
                if obj.params.classifier.cnn.enable
                    optimizer = CNNOptimizer(obj.params);
                    [optimizedParams, ~, ~] = optimizer.optimize(obj.processedData, obj.processedLabel);

                    if obj.params.classifier.cnn.optimize && ~isempty(optimizedParams)
                        % パラメータの更新
                        updatedParams = obj.params;
                        updatedParams.classifier.cnn = obj.updateCNNParams(obj.params.classifier.cnn, optimizedParams);

                        % 更新したパラメータでCNNClassifierを再初期化
                        updatedCnnClassifier = CNNClassifier(updatedParams);
                        obj.cnn = updatedCnnClassifier.trainCNN(obj.processedData, obj.processedLabel);
                    else
                        obj.cnn = obj.cnnClassifier.trainCNN(obj.processedData, obj.processedLabel);
                    end
                end
            catch ME
                error('Classification failed: %s', ME.message);
            end
        end
        
        function params = updateCNNParams(~, baseParams, optimizedParams)
            % CNNパラメータの更新
            params = baseParams;
            params.training.optimizer.learningRate = optimizedParams(1);
            params.training.miniBatchSize = optimizedParams(2);

            % アーキテクチャの更新
            params.architecture.convLayers.conv1.size = optimizedParams(3);
            params.architecture.convLayers.conv1.filters = optimizedParams(4);
            params.architecture.dropoutLayers.dropout1 = optimizedParams(5);
            params.architecture.fullyConnected = [optimizedParams(6)];
        end

        function extractPowerFeatures(obj)
            try
                if isempty(obj.processedData) || isempty(obj.labels)
                    warning('PowerExtractor:NoData', 'データまたはラベルが空です');
                    return;
                end

                % データ形式の判定とエポック数の取得
                if iscell(obj.processedData)
                    numEpochs = length(obj.processedData);
                else
                    numEpochs = size(obj.processedData, 3);
                end

                % エポックごとの処理
                for epoch = 1:numEpochs
                    % データの取得
                    if iscell(obj.processedData)
                        epochData = obj.processedData{epoch};
                    else
                        epochData = obj.processedData(:,:,epoch);
                    end

                    % 周波数帯域ごとのパワーを計算
                    bandNames = obj.params.feature.power.bands.names;
                    if iscell(bandNames{1})
                        bandNames = bandNames{1};
                    end

                    bandPowers = struct();
                    for i = 1:length(bandNames)
                        bandName = bandNames{i};
                        freqRange = obj.params.feature.power.bands.(bandName);
                        bandPowers.(bandName) = obj.powerExtractor.calculatePower(epochData, freqRange);
                    end

                    % 新しい結果の構築
                    newResult = struct(...
                        'labels', obj.labels(epoch).value, ...
                        'powers', bandPowers, ...
                        'bands', {bandNames} ...
                    );

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

                % データ形式の判定とエポック数の取得
                if iscell(obj.processedData)
                    numEpochs = length(obj.processedData);
                else
                    numEpochs = size(obj.processedData, 3);
                end

                % エポックごとの処理
                for epoch = 1:numEpochs
                    % データの取得
                    if iscell(obj.processedData)
                        epochData = obj.processedData{epoch};
                    else
                        epochData = obj.processedData(:,:,epoch);
                    end

                    % FAA値の計算
                    faaResults = obj.faaExtractor.calculateFAA(epochData);

                    if iscell(faaResults)
                        faaResult = faaResults{1};
                    else
                        faaResult = faaResults;
                    end

                    % 新しい結果の構築
                    newResult = struct(...
                        'labels', obj.labels(epoch).value, ...
                        'faa', faaResult.faa, ...
                        'pleasureState', faaResult.pleasureState ...
                    );

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

                % データ形式の判定とエポック数の取得
                if iscell(obj.processedData)
                    numEpochs = length(obj.processedData);
                else
                    numEpochs = size(obj.processedData, 3);
                end

                % エポックごとの処理
                for epoch = 1:numEpochs
                    % データの取得
                    if iscell(obj.processedData)
                        epochData = obj.processedData{epoch};
                    else
                        epochData = obj.processedData(:,:,epoch);
                    end

                    % α/β比の計算
                    [abRatio, arousalState] = obj.abRatioExtractor.calculateABRatio(epochData);

                    % 新しい結果の構築
                    newResult = struct(...
                        'labels', obj.labels(epoch).value, ...
                        'ratio', abRatio, ...
                        'arousalState', arousalState ...
                    );

                    % 結果の追加
                    if isempty(obj.results.abRatio)
                        obj.results.abRatio = newResult;
                    else
                        obj.results.abRatio(end+1) = newResult;
                    end
                end

                fprintf('α/β比の特徴抽出が完了しました（%d エポック）\n', numEpochs);

            catch ME
                error('ABRatioExtractor:ExtractionFailed', 'α/β比の特徴抽出に失敗しました: %s', ME.message);
            end
        end
        
        function extractEmotionFeatures(obj)
            try
                if isempty(obj.processedData) || isempty(obj.labels)
                    warning('EmotionExtractor:NoData', 'データまたはラベルが空です');
                    return;
                end

                % データ形式の判定とエポック数の取得
                if iscell(obj.processedData)
                    numEpochs = length(obj.processedData);
                else
                    numEpochs = size(obj.processedData, 3);
                end

                % エポックごとの処理
                for epoch = 1:numEpochs
                    % データの取得
                    if iscell(obj.processedData)
                        epochData = obj.processedData{epoch};
                    else
                        epochData = obj.processedData(:,:,epoch);
                    end

                    % 感情特徴量の抽出
                    emotionResult = obj.emotionExtractor.classifyEmotion(epochData);

                    if iscell(emotionResult)
                        currentResult = emotionResult{1};
                    else
                        currentResult = emotionResult;
                    end

                    % 新しい結果の構築
                    newResult = struct(...
                        'labels', obj.labels(epoch).value, ...
                        'state', currentResult.state, ...
                        'coordinates', currentResult.coordinates, ...
                        'emotionCoords', currentResult.emotionCoords ...
                    );

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
        
        function saveResults(obj, savePath)
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
                        'trainInfo', obj.cnn.trainInfo, ...
                        'crossValidation', obj.cnn.crossValidation, ...
                        'overfitting', obj.cnn.overfitting ...
                    );
                end

                % DataManagerを使用して保存
                obj.dataManager.saveDataset(saveData, savePath);
                fprintf('Results saved to: %s\n', savePath);

            catch ME
                error('Failed to save results: %s', ME.message);
            end
        end
    end
end