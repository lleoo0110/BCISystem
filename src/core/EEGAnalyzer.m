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
        lstm             % LSTM分類器の結果
        hybrid          % Hybrid分類器の結果
        results          % 解析結果
        
        % 前処理コンポーネント
        artifactRemover     % アーティファクト除去
        baselineCorrector   % ベースライン補正
        dataAugmenter      % データ拡張
        downSampler        % ダウンサンプリング
        firFilter          % FIRフィルタ
        iirFilter          % IIRフィルタ
        notchFilter        % ノッチフィルタ
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
        lstmClassifier 
        hybridClassifier
        
        % データ管理コンポーネント
        dataManager
        dataLoader
    end
    
    methods (Access = public)
        function obj = EEGAnalyzer(params)
            obj.params = params;
            obj.initializePreprocessors();
            obj.initializeExtractors();
            obj.initializeClassifiers();
            obj.initializeResults();
            obj.dataManager = DataManager(params);
            obj.dataLoader = DataLoader(params);
        end
        
        function analyze(obj)
            try
                fprintf('\n=== 解析処理開始 ===\n');
                fprintf('解析対象のデータファイルを選択してください．\n');
        
                % 既に初期化されているDataLoaderを使用
                [loadedData, fileInfo] = obj.dataLoader.loadDataBrowser();
        
                if isempty(loadedData)
                    fprintf('データが選択されませんでした。\n');
                    return;
                end
        
                fprintf('データ読み込み完了\n');
                fprintf('読み込んだファイル:\n');
                for i = 1:length(fileInfo.filenames)
                    fprintf('%d: %s\n', i, fileInfo.filenames{i});
                end
        
                % 保存先の設定
                savePaths = cell(length(loadedData), 1);
                batchSave = obj.params.acquisition.save.batchSave && length(loadedData) > 1;
                
                if batchSave
                    % 一括保存モード: 共通の保存先フォルダを選択
                    batchSavePath = uigetdir(fileInfo.filepath, '一括保存用のフォルダを選択してください');
                    if isequal(batchSavePath, 0)
                        fprintf('保存先の選択がキャンセルされました。処理を中止します。\n');
                        return;
                    end
                    
                    fprintf('一括保存モード: 処理結果を %s に保存します。\n', batchSavePath);
                    
                    % 各ファイルの保存先パスを生成
                    for i = 1:length(loadedData)
                        if ~isempty(loadedData{i})
                            [~, originalName, ~] = fileparts(fileInfo.filenames{i});
                            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                            defaultFileName = sprintf('%s_analysis_%s.mat', originalName, timestamp);
                            savePaths{i} = fullfile(batchSavePath, defaultFileName);
                        end
                    end
                else
                    % 個別保存モード: 各ファイルの保存先を事前に選択
                    fprintf('\n各ファイルの保存先を選択してください。\n');
                    for i = 1:length(loadedData)
                        if ~isempty(loadedData{i})
                            [~, originalName, ~] = fileparts(fileInfo.filenames{i});
                            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                            defaultFileName = sprintf('%s_analysis_%s.mat', originalName, timestamp);
                            
                            [saveName, saveDir] = uiputfile('*.mat', ...
                                sprintf('ファイル %s の保存先を選択してください (%d/%d)', ...
                                fileInfo.filenames{i}, i, length(loadedData)), ...
                                defaultFileName);
                            
                            if saveName == 0
                                fprintf('ファイル %s の保存先選択がキャンセルされました。このファイルはスキップします。\n', ...
                                    fileInfo.filenames{i});
                                savePaths{i} = '';
                            else
                                savePaths{i} = fullfile(saveDir, saveName);
                                fprintf('ファイル %s の保存先: %s\n', fileInfo.filenames{i}, savePaths{i});
                            end
                        end
                    end
                end
        
                % 複数ファイルの処理
                for i = 1:length(loadedData)
                    if ~isempty(loadedData{i}) && ~isempty(savePaths{i})
                        fprintf('\n=== データセット %d/%d (%s) の処理開始 ===\n', ...
                            i, length(loadedData), fileInfo.filenames{i});
                        
                        % データの処理
                        obj.setData(loadedData{i});
                        obj.executePreprocessingPipeline();
                        obj.extractFeatures();
                        obj.performClassification();
                        
                        % 結果の保存
                        obj.saveResults(savePaths{i});
                        fprintf('\n解析結果を保存しました: %s\n', savePaths{i});
                    elseif ~isempty(loadedData{i})
                        fprintf('\n=== データセット %d/%d (%s) はスキップされました ===\n', ...
                            i, length(loadedData), fileInfo.filenames{i});
                    end
                end
        
                close all;
                fprintf('\n=== 解析処理完了 ===\n');
<<<<<<< HEAD

                if obj.params.signal.createEEGLABset
                    fprintf('\n=== EEGLAB起動 ===\n');
                    create_eeglabset(obj.rawData,obj.labels,obj.params,saveDir);
                end

=======
        
>>>>>>> 77be08a1646b3e6dca6b51b459a238fdd6ad0b8a
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
            obj.iirFilter = IIRFilterDesigner(obj.params);
            obj.notchFilter = NotchFilterDesigner(obj.params);
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
            % 分類器コンポーネントの初期化
            obj.svmClassifier = SVMClassifier(obj.params);
            obj.ecocClassifier = ECOCClassifier(obj.params);
            obj.cnnClassifier = CNNClassifier(obj.params);
            obj.lstmClassifier = LSTMClassifier(obj.params);
            obj.hybridClassifier = HybridClassifier(obj.params);
        end
        
        function initializeResults(obj)
            % 結果構造体の初期化
            obj.results = struct(...
                'power', [], ...     % パワー解析結果
                'faa', [], ...      % FAA解析結果
                'abRatio', [], ...  % α/β比解析結果
                'emotion', [] ...  % 感情分析結果
            );

            % 分類器結果の初期化
            obj.initializeClassifierResults();
        end
        
        function initializeClassifierResults(obj)
            % 基本の分類器結果構造体
            classifierStruct = struct(...
                'model', [], ...
                'performance', [], ...
                'trainingInfo', [], ...
                'crossValidation', [], ...
                'overfitting', [], ...
                'normParams', [] ...
            );

            % 各分類器の結果を初期化
            obj.svm = classifierStruct;
            obj.ecoc = classifierStruct;
            obj.cnn = classifierStruct;
            obj.lstm = classifierStruct;
            obj.hybrid = classifierStruct;
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

                % FIRフィルタ
                if obj.params.signal.preprocessing.filter.fir.enable
                    [data, info] = obj.firFilter.designAndApplyFilter(data);
                    obj.processingInfo.firFilter = info;
                end

                % IIRフィルタ
                if obj.params.signal.preprocessing.filter.iir.enable
                    [data, info] = obj.iirFilter.designAndApplyFilter(data);
                    obj.processingInfo.iirFilter = info;
                end

                % エポック化
                [epochs, epochLabels, info] = obj.epoching.epoching(data, obj.labels);
                obj.processedData = epochs;
                obj.processedLabel = epochLabels;
                obj.processingInfo.epoch = info;

            catch ME
                error('Preprocessing pipeline failed: %s', ME.message);
            end
        end
        
        function extractFeatures(obj)
            try
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
                % SVM分類
                if obj.params.classifier.svm.enable
                    obj.svm = obj.svmClassifier.trainSVM(obj.processedData, obj.processedLabel);
                end
        
                % ECOC分類
                if obj.params.classifier.ecoc.enable
                    obj.ecoc = obj.ecocClassifier.trainECOC(obj.processedData, obj.processedLabel);
                end
        
                % CNN分類
                if obj.params.classifier.cnn.enable
                    if obj.params.classifier.cnn.optimize
                        cnnOptimizer = CNNOptimizer(obj.params);
                        obj.cnn = cnnOptimizer.optimize(obj.processedData, obj.processedLabel);
                    else
                        obj.cnn = obj.cnnClassifier.trainCNN(obj.processedData, obj.processedLabel);
                    end
                end
        
                % LSTM分類
                if obj.params.classifier.lstm.enable
                    if obj.params.classifier.lstm.optimize
                        lstmOptimizer = LSTMOptimizer(obj.params);
                        obj.lstm = lstmOptimizer.optimize(obj.processedData, obj.processedLabel);
                    else
                        obj.lstm = obj.lstmClassifier.trainLSTM(obj.processedData, obj.processedLabel);
                    end
                end
        
                % Hybrid分類
                if obj.params.classifier.hybrid.enable
                    try
                        fprintf('\n=== Hybridモデルの学習開始 ===\n');
                        
                        % ハイブリッド分類器の学習
                        if obj.params.classifier.hybrid.optimize
                            hybridOptimizer = HybridOptimizer(obj.params);
                            obj.hybrid = hybridOptimizer.optimize(obj.processedData, obj.processedLabel);
                        else
                            % ハイブリッド分類器インスタンスの作成と学習
                            hybridClassifier = HybridClassifier(obj.params);
                            obj.hybrid = hybridClassifier.trainHybrid(obj.processedData, obj.processedLabel);
                            
                            % 結果構造体の正常性を確認
                            if ~isstruct(obj.hybrid) || ~isfield(obj.hybrid, 'model') || ~isfield(obj.hybrid, 'performance')
                                error('Hybridモデルから無効な結果が返されました');
                            end
                            
                            % モデル構造の確認とデバッグ情報の出力
                            fprintf('Hybridモデル構造を確認中...\n');
                            fprintf('  - model フィールド: %s\n', mat2str(isfield(obj.hybrid, 'model')));
                            if isfield(obj.hybrid, 'model')
                                fprintf('  - featureExtractor: %s\n', mat2str(isfield(obj.hybrid.model, 'featureExtractor')));
                                fprintf('  - adaBoostModel: %s\n', mat2str(isfield(obj.hybrid.model, 'adaBoostModel')));
                            end
                        end
                        
                        fprintf('Hybridモデルの学習完了\n');
                        
                    catch ME
                        fprintf('\n=== Hybridモデルの学習でエラーが発生 ===\n');
                        fprintf('エラー詳細: %s\n', ME.message);
                        fprintf('スタックトレース:\n');
                        disp(getReport(ME, 'extended'));
                        
                        % エラー発生時でも処理を継続
                        fprintf('分類はスキップして処理を継続します。\n');
                        obj.hybrid = struct('model', [], 'performance', struct());
                    end
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
                    try
                        saveData.classifier.svm = struct(...
                            'model', obj.svm.model, ...
                            'performance', obj.svm.performance);
                        
                        % 追加フィールドの安全な追加
                        if isfield(obj.svm, 'normParams')
                            saveData.classifier.svm.normParams = obj.svm.normParams;
                        end
                        if isfield(obj.svm, 'cspFilters')
                            saveData.classifier.svm.cspFilters = obj.svm.cspFilters;
                        end
                    catch ME
                        error('SVM結果の保存中にエラーが発生: %s', ME.message);
                    end
                end
        
                % ECOC結果
                if obj.params.classifier.ecoc.enable && ~isempty(obj.ecoc)
                    try
                        saveData.classifier.ecoc = struct(...
                            'model', obj.ecoc.model, ...
                            'performance', obj.ecoc.performance);
                        
                        % 追加フィールドの安全な追加
                        if isfield(obj.ecoc, 'normParams')
                            saveData.classifier.ecoc.normParams = obj.ecoc.normParams;
                        end
                        if isfield(obj.ecoc, 'cspFilters')
                            saveData.classifier.ecoc.cspFilters = obj.ecoc.cspFilters;
                        end
                    catch ME
                        error('ECOC結果の保存中にエラーが発生: %s', ME.message);
                    end
                end
        
                % CNN結果
                if obj.params.classifier.cnn.enable && ~isempty(obj.cnn)
                    try
                        saveData.classifier.cnn = struct(...
                            'model', obj.cnn.model);
                        
                        % フィールドの安全な追加
                        if isfield(obj.cnn, 'performance')
                            saveData.classifier.cnn.performance = obj.cnn.performance;
                        end
                        if isfield(obj.cnn, 'trainInfo')
                            saveData.classifier.cnn.trainInfo = obj.cnn.trainInfo;
                        end
                        if isfield(obj.cnn, 'overfitting')
                            saveData.classifier.cnn.overfitting = obj.cnn.overfitting;
                        end
                        if isfield(obj.cnn, 'normParams')
                            saveData.classifier.cnn.normParams = obj.cnn.normParams;
                        end
                    catch ME
                        error('CNN結果の保存中にエラーが発生: %s', ME.message);
                    end
                end
        
                % LSTM結果
                if obj.params.classifier.lstm.enable && ~isempty(obj.lstm)
                    try
                        saveData.classifier.lstm = struct('model', obj.lstm.model);
                        
                        % フィールドの安全な追加
                        if isfield(obj.lstm, 'performance')
                            saveData.classifier.lstm.performance = obj.lstm.performance;
                        end
                        if isfield(obj.lstm, 'trainInfo')
                            saveData.classifier.lstm.trainInfo = obj.lstm.trainInfo;
                        end
                        if isfield(obj.lstm, 'overfitting')
                            saveData.classifier.lstm.overfitting = obj.lstm.overfitting;
                        end
                        if isfield(obj.lstm, 'normParams')
                            saveData.classifier.lstm.normParams = obj.lstm.normParams;
                        end
                    catch ME
                        error('LSTM結果の保存中にエラーが発生: %s', ME.message);
                    end
                end
        
                % Hybrid結果
                if obj.params.classifier.hybrid.enable && ~isempty(obj.hybrid)
                    try
                        % 基本構造を初期化
                        saveData.classifier.hybrid = struct();
                        
                        % フィールドの安全な追加
                        if isfield(obj.hybrid, 'model')
                            saveData.classifier.hybrid.model = obj.hybrid.model;
                        end
                        if isfield(obj.hybrid, 'performance')
                            saveData.classifier.hybrid.performance = obj.hybrid.performance;
                        end
                        if isfield(obj.hybrid, 'trainInfo')
                            saveData.classifier.hybrid.trainInfo = obj.hybrid.trainInfo;
                        end
                        if isfield(obj.hybrid, 'overfitting')
                            saveData.classifier.hybrid.overfitting = obj.hybrid.overfitting;
                        end
                        if isfield(obj.hybrid, 'normParams')
                            saveData.classifier.hybrid.normParams = obj.hybrid.normParams;
                        end
                        % エラーがあれば保存
                        if isfield(obj.hybrid, 'error')
                            saveData.classifier.hybrid.error = obj.hybrid.error;
                        end
                    catch ME
                        error('Hybrid結果の保存中にエラーが発生: %s', ME.message);
                    end
                end
        
                % 保存パスが指定されていない場合は保存ダイアログを表示
                if nargin < 2 || isempty(savePath)
                    % ファイル名の生成
                    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                    defaultFileName = sprintf('eeg_analysis_%s.mat', timestamp);
                    
                    % 保存先の選択
                    [saveName, saveDir] = uiputfile('*.mat', '解析結果の保存先を選択してください', defaultFileName);
                    if saveName == 0
                        fprintf('保存がキャンセルされました。\n');
                        return;
                    end
                    savePath = fullfile(saveDir, saveName);
                end
        
                % DataManagerを使用して保存
                obj.dataManager.saveDataset(saveData, savePath);
                fprintf('解析結果を保存しました: %s\n', savePath);
        
            catch ME
                error('解析結果の保存に失敗: %s\n詳細: %s', ME.message, getReport(ME, 'extended'));
            end
        end
    end
end