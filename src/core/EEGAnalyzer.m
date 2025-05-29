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
        cnnOptimizer
        lstmClassifier 
        lstmOptimizer
        hybridClassifier
        hybridOptimizer
        
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
            obj.cnnOptimizer = CNNOptimizer(obj.params);
            obj.lstmClassifier = LSTMClassifier(obj.params);
            obj.lstmOptimizer = LSTMOptimizer(obj.params);
            obj.hybridClassifier = HybridClassifier(obj.params);
            obj.hybridOptimizer = HybridOptimizer(obj.params);
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
                fprintf('\n=== 分類処理を開始 ===\n');
                
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
                    if obj.params.classifier.cnn.training.validation.enable

                        % モンテカルロ交差検証を実行
                        numRepetitions = obj.params.classifier.cnn.training.validation.repetitions;  % 反復回数
                        fprintf('\n=== CNN: モンテカルロ交差検証 (%d回反復) ===\n', numRepetitions);
                        
                        % 結果保存
                        cnnModels = cell(1, numRepetitions);
                        cnnAccuracies = zeros(1, numRepetitions);
                        
                        for rep = 1:numRepetitions
                            try
                                fprintf('\n--- CNN: 反復 %d/%d ---\n', rep, numRepetitions);
                                if obj.params.classifier.cnn.optimize
                                    % 最適化実行
                                    result = obj.cnnOptimizer.optimize(obj.processedData, obj.processedLabel);
                                else
                                    % 通常のCNN学習
                                    result = obj.cnnClassifier.trainCNN(obj.processedData, obj.processedLabel, baseSeed + rep);
                                end
                                
                                % 結果保存
                                cnnModels{rep} = result;
                                
                                % 精度記録
                                if isfield(result, 'performance') && isfield(result.performance, 'accuracy')
                                    cnnAccuracies(rep) = result.performance.accuracy;
                                    fprintf('反復 %d の精度: %.2f%%\n', rep, cnnAccuracies(rep) * 100);
                                end
                                
                                % GPUメモリの解放
                                if obj.params.classifier.cnn.gpu
                                    gpuDevice([]);
                                end
                                
                            catch ME
                                fprintf('反復 %d でエラー発生: %s\n', rep, ME.message);
                                disp(getReport(ME, 'extended'));
                            end
                        end
                        
                        % 有効結果の統計
                        validAccuracies = cnnAccuracies(cnnAccuracies > 0);
                        
                        if ~isempty(validAccuracies)
                            % 統計情報の集計
                            cvSummary = struct(...
                                'iterations', numRepetitions, ...
                                'cvAccuracy', validAccuracies, ...
                                'meanAccuracy', mean(validAccuracies), ...
                                'stdAccuracy', std(validAccuracies), ...
                                'minAccuracy', min(validAccuracies), ...
                                'maxAccuracy', max(validAccuracies), ...
                                'validCount', length(validAccuracies) ...
                            );
                            
                            fprintf('\n=== CNN交差検証結果サマリー ===\n');
                            fprintf('平均精度: %.2f%% (±%.2f%%)\n', cvSummary.meanAccuracy * 100, cvSummary.stdAccuracy * 100);
                            fprintf('精度範囲: %.2f%% - %.2f%%\n', cvSummary.minAccuracy * 100, cvSummary.maxAccuracy * 100);
                            fprintf('有効反復数: %d/%d\n', cvSummary.validCount, numRepetitions);
                            
                            % 最良モデルを選択
                            [~, bestIdx] = max(cnnAccuracies);
                            obj.cnn = cnnModels{bestIdx};
                            
                            % 交差検証情報を追加
                            obj.cnn.crossValidation = cvSummary;
                            
                            fprintf('最良モデル（反復 %d）を選択しました。精度: %.2f%%\n', bestIdx, cnnAccuracies(bestIdx) * 100);
                        else
                            fprintf('\n警告: 有効な結果が得られませんでした\n');
                        end
                    else
                        % 通常の学習
                        fprintf('\n=== CNN: 標準モードを使用 ===\n');
                        if obj.params.classifier.cnn.optimize
                            obj.cnn = obj.cnnOptimizer.optimize(obj.processedData, obj.processedLabel);
                        else
                            obj.cnn = obj.cnnClassifier.trainCNN(obj.processedData, obj.processedLabel);
                        end
                    end
                end
        
                % LSTM分類
                if obj.params.classifier.lstm.enable
                    if  obj.params.classifier.lstm.training.validation.enable
                        
                        % モンテカルロ交差検証を実行
                        numRepetitions = obj.params.classifier.lstm.training.validation.repetitions;  % 反復回数
                        fprintf('\n=== LSTM: モンテカルロ交差検証 (%d回反復) ===\n', numRepetitions);
                        
                        % 結果保存
                        lstmModels = cell(1, numRepetitions);
                        lstmAccuracies = zeros(1, numRepetitions);
                        
                        for rep = 1:numRepetitions
                            try
                                fprintf('\n--- LSTM: 反復 %d/%d ---\n', rep, numRepetitions);
                                
                                if obj.params.classifier.lstm.optimize
                                    % 最適化実行
                                    result = obj.lstmOptimizer.optimize(obj.processedData, obj.processedLabel);
                                else
                                    % 通常のLSTM学習
                                    result = obj.lstmClassifier.trainLSTM(obj.processedData, obj.processedLabel, baseSeed + rep);
                                end
                                
                                % 結果保存
                                lstmModels{rep} = result;
                                
                                % 精度記録
                                if isfield(result, 'performance') && isfield(result.performance, 'accuracy')
                                    lstmAccuracies(rep) = result.performance.accuracy;
                                    fprintf('反復 %d の精度: %.2f%%\n', rep, lstmAccuracies(rep) * 100);
                                end
                                
                                % GPUメモリの解放
                                if obj.params.classifier.lstm.gpu
                                    gpuDevice([]);
                                end
                                
                            catch ME
                                fprintf('反復 %d でエラー発生: %s\n', rep, ME.message);
                                disp(getReport(ME, 'extended'));
                            end
                        end
                        
                        % 有効結果の統計
                        validAccuracies = lstmAccuracies(lstmAccuracies > 0);
                        
                        if ~isempty(validAccuracies)
                            % 統計情報の集計
                            cvSummary = struct(...
                                'iterations', numRepetitions, ...
                                'cvAccuracy', validAccuracies, ...
                                'meanAccuracy', mean(validAccuracies), ...
                                'stdAccuracy', std(validAccuracies), ...
                                'minAccuracy', min(validAccuracies), ...
                                'maxAccuracy', max(validAccuracies), ...
                                'validCount', length(validAccuracies) ...
                            );
                            
                            fprintf('\n=== LSTM交差検証結果サマリー ===\n');
                            fprintf('平均精度: %.2f%% (±%.2f%%)\n', cvSummary.meanAccuracy * 100, cvSummary.stdAccuracy * 100);
                            fprintf('精度範囲: %.2f%% - %.2f%%\n', cvSummary.minAccuracy * 100, cvSummary.maxAccuracy * 100);
                            fprintf('有効反復数: %d/%d\n', cvSummary.validCount, numRepetitions);
                            
                            % 最良モデルを選択
                            [~, bestIdx] = max(lstmAccuracies);
                            obj.lstm = lstmModels{bestIdx};
                            
                            % 交差検証情報を追加
                            obj.lstm.crossValidation = cvSummary;
                            
                            fprintf('最良モデル（反復 %d）を選択しました。精度: %.2f%%\n', bestIdx, lstmAccuracies(bestIdx) * 100);
                        else
                            fprintf('\n警告: 有効な結果が得られませんでした\n');
                        end
                    else
                        % 通常の学習
                        fprintf('\n=== LSTM: 標準モードを使用 ===\n');
                        if obj.params.classifier.lstm.optimize
                            obj.lstm = obj.lstmOptimizer.optimize(obj.processedData, obj.processedLabel);
                        else
                            obj.lstm = obj.lstmClassifier.trainLSTM(obj.processedData, obj.processedLabel);
                        end
                    end
                end
        
                % Hybrid分類
                if obj.params.classifier.hybrid.enable
                    if  obj.params.classifier.hybrid.training.validation.enable
                        
                        % モンテカルロ交差検証を実行
                        numRepetitions = obj.params.classifier.hybrid.training.validation.repetitions;  % 反復回数
                        fprintf('\n=== Hybrid: モンテカルロ交差検証 (%d回反復) ===\n', numRepetitions);
                        
                        % 結果保存
                        hybridModels = cell(1, numRepetitions);
                        hybridAccuracies = zeros(1, numRepetitions);
                        
                        for rep = 1:numRepetitions
                            try
                                fprintf('\n--- Hybrid: 反復 %d/%d ---\n', rep, numRepetitions);

                                if obj.params.classifier.hybrid.optimize                                    
                                    % 最適化実行
                                    result = obj.hybridOptimizer.optimize(obj.processedData, obj.processedLabel);
                                else
                                    % 通常のHybrid学習
                                    result = obj.hybridClassifier.trainHybrid(obj.processedData, obj.processedLabel, baseSeed + rep);
                                end
                                
                                % 結果保存
                                hybridModels{rep} = result;
                                
                                % 精度記録
                                if isfield(result, 'performance') && isfield(result.performance, 'accuracy')
                                    hybridAccuracies(rep) = result.performance.accuracy;
                                    fprintf('反復 %d の精度: %.2f%%\n', rep, hybridAccuracies(rep) * 100);
                                end
                                
                                % GPUメモリの解放
                                if obj.params.classifier.hybrid.gpu
                                    gpuDevice([]);
                                end
                                
                            catch ME
                                fprintf('反復 %d でエラー発生: %s\n', rep, ME.message);
                                disp(getReport(ME, 'extended'));
                            end
                        end
                        
                        % 有効結果の統計
                        validAccuracies = hybridAccuracies(hybridAccuracies > 0);
                        
                        if ~isempty(validAccuracies)
                            % 統計情報の集計
                            cvSummary = struct(...
                                'iterations', numRepetitions, ...
                                'cvAccuracy', validAccuracies, ...
                                'meanAccuracy', mean(validAccuracies), ...
                                'stdAccuracy', std(validAccuracies), ...
                                'minAccuracy', min(validAccuracies), ...
                                'maxAccuracy', max(validAccuracies), ...
                                'validCount', length(validAccuracies) ...
                            );
                            
                            fprintf('\n=== Hybrid交差検証結果サマリー ===\n');
                            fprintf('平均精度: %.2f%% (±%.2f%%)\n', cvSummary.meanAccuracy * 100, cvSummary.stdAccuracy * 100);
                            fprintf('精度範囲: %.2f%% - %.2f%%\n', cvSummary.minAccuracy * 100, cvSummary.maxAccuracy * 100);
                            fprintf('有効反復数: %d/%d\n', cvSummary.validCount, numRepetitions);
                            
                            % 最良モデルを選択
                            [~, bestIdx] = max(hybridAccuracies);
                            obj.hybrid = hybridModels{bestIdx};
                            
                            % 交差検証情報を追加
                            obj.hybrid.crossValidation = cvSummary;
                            
                            fprintf('最良モデル（反復 %d）を選択しました。精度: %.2f%%\n', bestIdx, hybridAccuracies(bestIdx) * 100);
                        else
                            fprintf('\n警告: 有効な結果が得られませんでした\n');
                        end
                        
                    else
                        % 通常の学習（既存コード）
                        try
                            fprintf('\n=== Hybridモデルの学習開始 ===\n');
                            
                            if obj.params.classifier.hybrid.optimize
                                obj.hybrid = obj.hybridOptimizer.optimize(obj.processedData, obj.processedLabel);
                            else
                                obj.hybrid = obj.hybridClassifier.trainHybrid(obj.processedData, obj.processedLabel);
                                
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
        
                % 分類器結果の保存（結果構造体をそのまま保存）
                saveData.classifier = struct();
                
                % SVM結果（構造体全体をコピー）
                if obj.params.classifier.svm.enable && ~isempty(obj.svm)
                    try
                        saveData.classifier.svm = obj.svm;
                        fprintf('SVM結果を保存しました\n');
                    catch ME
                        fprintf('警告: SVM結果の保存中にエラーが発生: %s\n', ME.message);
                        saveData.classifier.svm = struct('error', ME.message);
                    end
                end
        
                % ECOC結果（構造体全体をコピー）
                if obj.params.classifier.ecoc.enable && ~isempty(obj.ecoc)
                    try
                        saveData.classifier.ecoc = obj.ecoc;
                        fprintf('ECOC結果を保存しました\n');
                        
                        % 交差検証結果の確認
                        if isfield(obj.ecoc, 'crossValidation') && ~isempty(obj.ecoc.crossValidation)
                            if isfield(obj.ecoc.crossValidation, 'completed') && obj.ecoc.crossValidation.completed
                                fprintf('  - 交差検証結果も保存されました (平均精度: %.2f%%)\n', ...
                                    obj.ecoc.crossValidation.meanAccuracy * 100);
                            else
                                fprintf('  - 交差検証は実行されませんでした\n');
                            end
                        end
                    catch ME
                        fprintf('警告: ECOC結果の保存中にエラーが発生: %s\n', ME.message);
                        saveData.classifier.ecoc = struct('error', ME.message);
                    end
                end
        
                % CNN結果（構造体全体をコピー）
                if obj.params.classifier.cnn.enable && ~isempty(obj.cnn)
                    try
                        saveData.classifier.cnn = obj.cnn;
                        fprintf('CNN結果を保存しました\n');
                        
                        % 交差検証結果の確認
                        if isfield(obj.cnn, 'crossValidation') && ~isempty(obj.cnn.crossValidation)
                            if isfield(obj.cnn.crossValidation, 'meanAccuracy')
                                fprintf('  - モンテカルロ交差検証結果も保存されました (平均精度: %.2f%%)\n', ...
                                    obj.cnn.crossValidation.meanAccuracy * 100);
                            end
                        end
                    catch ME
                        fprintf('警告: CNN結果の保存中にエラーが発生: %s\n', ME.message);
                        saveData.classifier.cnn = struct('error', ME.message);
                    end
                end
        
                % LSTM結果（構造体全体をコピー）
                if obj.params.classifier.lstm.enable && ~isempty(obj.lstm)
                    try
                        saveData.classifier.lstm = obj.lstm;
                        fprintf('LSTM結果を保存しました\n');
                        
                        % 交差検証結果の確認
                        if isfield(obj.lstm, 'crossValidation') && ~isempty(obj.lstm.crossValidation)
                            if isfield(obj.lstm.crossValidation, 'meanAccuracy')
                                fprintf('  - モンテカルロ交差検証結果も保存されました (平均精度: %.2f%%)\n', ...
                                    obj.lstm.crossValidation.meanAccuracy * 100);
                            end
                        end
                    catch ME
                        fprintf('警告: LSTM結果の保存中にエラーが発生: %s\n', ME.message);
                        saveData.classifier.lstm = struct('error', ME.message);
                    end
                end
        
                % Hybrid結果（構造体全体をコピー）
                if obj.params.classifier.hybrid.enable && ~isempty(obj.hybrid)
                    try
                        saveData.classifier.hybrid = obj.hybrid;
                        fprintf('Hybrid結果を保存しました\n');
                        
                        % 交差検証結果の確認
                        if isfield(obj.hybrid, 'crossValidation') && ~isempty(obj.hybrid.crossValidation)
                            if isfield(obj.hybrid.crossValidation, 'meanAccuracy')
                                fprintf('  - モンテカルロ交差検証結果も保存されました (平均精度: %.2f%%)\n', ...
                                    obj.hybrid.crossValidation.meanAccuracy * 100);
                            end
                        end
                    catch ME
                        fprintf('警告: Hybrid結果の保存中にエラーが発生: %s\n', ME.message);
                        saveData.classifier.hybrid = struct('error', ME.message);
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
                
                % 保存内容のサマリー表示
                fprintf('\n=== 保存完了サマリー ===\n');
                fprintf('保存先: %s\n', savePath);
                fprintf('保存された分類器:\n');
                
                if isfield(saveData.classifier, 'svm') && ~isempty(saveData.classifier.svm)
                    fprintf('  - SVM: ✓\n');
                end
                if isfield(saveData.classifier, 'ecoc') && ~isempty(saveData.classifier.ecoc)
                    fprintf('  - ECOC: ✓\n');
                end
                if isfield(saveData.classifier, 'cnn') && ~isempty(saveData.classifier.cnn)
                    fprintf('  - CNN: ✓\n');
                end
                if isfield(saveData.classifier, 'lstm') && ~isempty(saveData.classifier.lstm)
                    fprintf('  - LSTM: ✓\n');
                end
                if isfield(saveData.classifier, 'hybrid') && ~isempty(saveData.classifier.hybrid)
                    fprintf('  - Hybrid: ✓\n');
                end
                
                fprintf('解析結果の保存が完了しました\n');
        
            catch ME
                error('解析結果の保存に失敗: %s\n詳細: %s', ME.message, getReport(ME, 'extended'));
            end
        end
    end
end