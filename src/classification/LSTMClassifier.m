classdef LSTMClassifier < handle
    %% LSTMClassifier - EEGデータ用長短期記憶(LSTM)ニューラルネットワーク分類器
    %
    % このクラスはEEGデータに対してLSTMニューラルネットワークベースの
    % 分類を行います。脳波特有の時系列データ構造に適したアーキテクチャを使用し、
    % データ前処理、モデル学習、評価、予測の機能を提供します。
    %
    % 主な機能:
    %   - EEGデータの前処理と次元変換
    %   - データ拡張と正規化
    %   - EEG特化型LSTMアーキテクチャの構築
    %   - モデルのトレーニングと評価
    %   - 過学習の検出と詳細分析
    %   - 交差検証
    %   - オンライン予測
    %
    % 使用例:
    %   params = getConfig('epocx');
    %   lstm = LSTMClassifier(params);
    %   results = lstm.trainLSTM(processedData, processedLabel);
    %   [label, score] = lstm.predictOnline(newData, results.model);
    
    properties (Access = private)
        params              % システム設定パラメータ
        net                 % 学習済みLSTMネットワーク
        isInitialized       % 初期化完了フラグ
        useGPU              % GPU使用フラグ
        
        % 学習進捗の追跡用
        trainingHistory     % 学習履歴データ
        validationHistory   % 検証履歴データ
        bestValAccuracy     % 最良の検証精度
        patienceCounter     % Early Stopping用カウンター
        currentEpoch        % 現在のエポック
        
        % 過学習監視用
        overfitMetrics      % 過学習メトリクス
        
        % 前処理コンポーネント
        dataAugmenter       % データ拡張コンポーネント
        normalizer          % 正規化コンポーネント
        
        % パフォーマンス監視
        gpuMemory           % GPU使用メモリ監視
    end
    
    properties (Access = public)
        performance         % 評価メトリクス (精度・確率など)
    end
    
    methods (Access = public)
        %% コンストラクタ - 初期化処理
        function obj = LSTMClassifier(params)
            % LSTMClassifierのインスタンスを初期化
            %
            % 入力:
            %   params - 設定パラメータ（getConfig関数から取得）
            
            % 基本パラメータの設定
            obj.params = params;
            obj.isInitialized = false;
            obj.useGPU = params.classifier.lstm.gpu;
            
            % プロパティの初期化
            obj.initializeProperties();
            
            % コンポーネントの初期化
            obj.dataAugmenter = DataAugmenter(params);
            obj.normalizer = EEGNormalizer(params);
        end
        
        %% LSTM学習メソッド - モデルの学習と評価を実行
        function results = trainLSTM(obj, processedData, processedLabel)
            % EEGデータを使用してLSTMモデルを学習
            %
            % 入力:
            %   processedData - 前処理済みEEGデータ [チャンネル x サンプル x エポック]
            %   processedLabel - クラスラベル [エポック x 1]
            %
            % 出力:
            %   results - 学習結果を含む構造体（モデル、性能評価、正規化パラメータなど）
            try                
                fprintf('\n=== LSTM学習処理を開始 ===\n');
                
                % データの次元を確認し、必要に応じて調整
                [processedData, processInfo] = obj.validateAndPrepareData(processedData);
                fprintf('データ検証: %s\n', processInfo);
                
                % データを学習・検証・テストセットに分割
                [trainData, trainLabels, valData, valLabels, testData, testLabels] = obj.splitDataset(processedData, processedLabel);

                % 学習データの拡張処理
                [trainData, trainLabels, normParams] = obj.preprocessTrainingData(trainData, trainLabels);
                
                % 検証・テストデータも同じ正規化パラメータで処理
                valData = obj.normalizer.normalizeOnline(valData, normParams);
                testData = obj.normalizer.normalizeOnline(testData, normParams);
                
                % LSTM用のデータ形式への変換
                prepTrainData = obj.prepareDataForLSTM(trainData);
                prepValData = obj.prepareDataForLSTM(valData);
                prepTestData = obj.prepareDataForLSTM(testData);
                
                % モデルの学習
                [lstmModel, trainInfo] = obj.trainLSTMModel(prepTrainData, trainLabels, prepValData, valLabels);
                
                % テストデータでの最終評価
                testMetrics = obj.evaluateModel(lstmModel, prepTestData, testLabels);
                
                % 過学習の分析
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);
                
                if isOverfit
                    fprintf('\n警告: モデルに過学習の兆候が検出されました (%s)\n', obj.overfitMetrics.severity);
                end

                % 性能指標の更新
                obj.updatePerformanceMetrics(testMetrics);
                
                % 結果構造体の構築
                results = obj.buildResultsStruct(lstmModel, testMetrics, trainInfo, normParams);

                % 結果のサマリー表示
                obj.displayResults();
                
                % 使用リソースのクリーンアップ
                obj.resetGPUMemory();
                fprintf('\n=== LSTM学習処理が完了しました ===\n');

            catch ME
                % エラー発生時の詳細情報出力
                fprintf('\n=== LSTM学習中にエラーが発生しました ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラー発生場所:\n');
                for i = 1:length(ME.stack)
                    fprintf('  ファイル: %s\n  行: %d\n  関数: %s\n\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end

                % クリーンアップ処理
                obj.resetGPUMemory();
                rethrow(ME);
            end
        end

        %% 過学習検証メソッド - モデルの過学習程度を評価
        function [isOverfit, metrics] = validateOverfitting(obj, trainInfo, testMetrics)
            % トレーニング結果とテスト結果から過学習を分析
            %
            % 入力:
            %   trainInfo - トレーニング情報（学習曲線を含む）
            %   testMetrics - テストデータでの評価結果
            %
            % 出力:
            %   isOverfit - 過学習の有無（論理値）
            %   metrics - 詳細な過学習メトリクス
            
            fprintf('\n=== 過学習検証の実行 ===\n');
            
            % 初期化
            isOverfit = false;
            metrics = struct();
            
            try
                % trainInfoの構造を検証
                obj.validateTrainInfo(trainInfo);
                
                % 精度データの取得
                trainAcc = trainInfo.History.TrainingAccuracy;
                valAcc = trainInfo.History.ValidationAccuracy;
                
                % テスト精度を取得
                testAcc = testMetrics.accuracy;
                
                % 検証-テスト精度ギャップ分析を実行
                [gapOverfit, gapMetrics] = obj.validateTestValidationGap(valAcc, testAcc, size(testMetrics.confusionMat, 1));
                
                % 学習曲線の詳細分析
                [trainTrend, valTrend] = obj.analyzeLearningCurves(trainAcc, valAcc);
                
                % バイアスの検出（特定クラスへの偏り）
                isCompletelyBiased = obj.detectClassificationBias(testMetrics);
                
                % 学習進行の評価
                isLearningProgressing = std(diff(trainAcc)) > 0.01;
                
                % 最適エポックの分析
                [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(valAcc);
                fprintf('最適エポック: %d/%d (%.1f%%)\n', optimalEpoch, totalEpochs, ... 
                    (optimalEpoch/totalEpochs)*100);
                
                % Early Stoppingの効果分析
                earlyStoppingEffect = obj.analyzeEarlyStoppingEffect(trainInfo);

                % 複合判定 - バイアス検出を優先
                if isCompletelyBiased
                    severity = 'critical';  % 完全な偏りがある場合は最重度の過学習と判定
                    fprintf('完全な分類バイアスが検出されたため、過学習を「%s」と判定\n', severity);
                else
                    % 通常のギャップベース判定を使用
                    severity = gapMetrics.severity;
                end
                
                % メトリクスの構築
                metrics = struct(...
                    'gapMetrics', gapMetrics, ...
                    'performanceGap', gapMetrics.rawGap, ...
                    'isCompletelyBiased', isCompletelyBiased, ...
                    'isLearningProgressing', isLearningProgressing, ...
                    'validationTrend', valTrend, ...
                    'trainingTrend', trainTrend, ...
                    'severity', severity, ...
                    'optimalEpoch', optimalEpoch, ...
                    'totalEpochs', totalEpochs, ...
                    'earlyStoppingEffect', earlyStoppingEffect ...
                );
                
                % 過学習判定
                isOverfit = gapOverfit || isCompletelyBiased || (~isLearningProgressing && severity ~= 'none');
                fprintf('過学習判定: %s (重大度: %s)\n', mat2str(isOverfit), severity);
                
            catch ME
                fprintf('過学習検証でエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                
                % エラー時のフォールバック値を設定
                metrics = obj.createFallbackOverfitMetrics();
                isOverfit = true;
            end
        end

        %% オンライン予測メソッド - 新しいデータの分類を実行
        function [label, score] = predictOnline(obj, data, lstm)
            % 学習済みモデルを使用して新しいEEGデータを分類
            %
            % 入力:
            %   data - 分類するEEGデータ [チャンネル x サンプル]
            %   lstmModel - 学習済みLSTMモデル
            %
            % 出力:
            %   label - 予測クラスラベル
            %   score - 予測確率スコア        
            try
                % 入力データの検証
                if isempty(data)
                    error('データが空です');
                end
        
                % モデルの存在確認
                if isempty(lstm) || ~isfield(lstm, 'model') || isempty(lstm.model)
                    error('LSTMモデルが利用できません');
                end

                % 正規化パラメータを取得して正規化を実行
                normalizedData = data;
                if isfield(lstm, 'normParams')
                    normalizedData = obj.normalizer.normalizeOnline(data, lstm.normParams);
                end

                % データの形状変換
                prepData = obj.prepareDataForLSTM(normalizedData);
                
                % 予測の実行
                [label, score] = classify(lstm.model, prepData);
        
            catch ME
                fprintf('LSTM予測中にエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                label = [];
                score = [];
                rethrow(ME);
            end
        end
    end
    
    methods (Access = private)
        %% プロパティ初期化メソッド
        function initializeProperties(obj)
            % クラスプロパティの初期化
            obj.trainingHistory = struct('loss', [], 'accuracy', []);
            obj.validationHistory = struct('loss', [], 'accuracy', []);
            obj.bestValAccuracy = 0;
            obj.patienceCounter = 0;
            obj.currentEpoch = 0;
            obj.overfitMetrics = struct();
            obj.gpuMemory = struct('total', 0, 'used', 0, 'peak', 0);
        end
        
        %% データ検証と準備
        function [validatedData, infoMsg] = validateAndPrepareData(~, data)
            % 入力データの検証と適切な形式への変換
            
            % データの次元と形状を確認
            dataSize = size(data);
            dimCount = ndims(data);
            
            if dimCount > 3
                error('対応していないデータ次元数: %d (最大3次元まで対応)', dimCount);
            end
            
            validatedData = data;
            
            % 必要に応じてデータ次元を調整
            if dimCount == 2
                [channels, samples] = size(data);
                validatedData = reshape(data, [channels, samples, 1]);
                infoMsg = sprintf('2次元データを3次元に変換 [%d×%d] → [%d×%d×1]', ...
                    channels, samples, channels, samples);
            else
                [channels, samples, epochs] = size(data);
                infoMsg = sprintf('3次元データを検証 [%d×%d×%d]', channels, samples, epochs);
            end
            
            % データの妥当性検証
            validateattributes(validatedData, {'numeric'}, {'finite', 'nonnan'}, ...
                'validateAndPrepareData', 'data');
            
            return;
        end
        
        %% 学習データの前処理
        function [procTrainData, procTrainLabels, normParams] = preprocessTrainingData(obj, trainData, trainLabels)
            % 学習データの拡張と正規化を実行
            
            procTrainData = trainData;
            procTrainLabels = trainLabels;
            normParams = [];
            
            % データ拡張処理
            if obj.params.classifier.augmentation.enable
                fprintf('\nデータ拡張を実行...\n');
                [procTrainData, procTrainLabels, ~] = obj.dataAugmenter.augmentData(trainData, trainLabels);
                fprintf('  - 拡張前: %d サンプル\n', length(trainLabels));
                fprintf('  - 拡張後: %d サンプル (%.1f倍)\n', length(procTrainLabels), ... 
                    length(procTrainLabels)/length(trainLabels));
            end
            
            % 正規化処理
            if obj.params.classifier.normalize.enable
                [procTrainData, normParams] = obj.normalizer.normalize(procTrainData);
                fprintf('データを正規化しました（%s法）\n', normParams.method);
                
                % 正規化パラメータの検証
                obj.validateNormalizationParams(normParams);
            end
            
            return;
        end
        
        %% 正規化パラメータの検証
        function validateNormalizationParams(~, params)
            % 正規化パラメータの妥当性を検証
            
            if ~isstruct(params)
                error('正規化パラメータが構造体ではありません');
            end
            
            % 正規化方法ごとの必須パラメータを確認
            switch params.method
                case 'zscore'
                    % Z-score正規化パラメータの検証
                    if ~isfield(params, 'mean') || ~isfield(params, 'std')
                        error('Z-score正規化に必要なパラメータ(mean, std)が不足しています');
                    end
                    
                    if any(params.std == 0)
                        warning('標準偏差が0のチャンネルがあります。正規化で問題が発生する可能性があります');
                    end
                    
                case 'minmax'
                    % MinMax正規化パラメータの検証
                    if ~isfield(params, 'min') || ~isfield(params, 'max')
                        error('MinMax正規化に必要なパラメータ(min, max)が不足しています');
                    end
                    
                    if any(params.max - params.min < eps)
                        warning('最大値と最小値がほぼ同じチャンネルがあります。正規化で問題が発生する可能性があります');
                    end
                    
                case 'robust'
                    % Robust正規化パラメータの検証
                    if ~isfield(params, 'median') || ~isfield(params, 'mad')
                        error('Robust正規化に必要なパラメータ(median, mad)が不足しています');
                    end
                    
                    if any(params.mad < eps)
                        warning('MADが極めて小さいチャンネルがあります。正規化で問題が発生する可能性があります');
                    end
                    
                otherwise
                    warning('未知の正規化方法: %s', params.method);
            end
        end
        
        %% データセット分割メソッド
        function [trainData, trainLabels, valData, valLabels, testData, testLabels] = splitDataset(obj, data, labels)
            % データを学習・検証・テストセットに分割（クラスバランスを維持）
            %
            % 入力:
            %   data - 前処理済みEEGデータ [チャンネル x サンプル x エポック]
            %   labels - クラスラベル [エポック x 1]
            %
            % 出力:
            %   trainData - 学習データ
            %   trainLabels - 学習ラベル
            %   valData - 検証データ
            %   valLabels - 検証ラベル
            %   testData - テストデータ
            %   testLabels - テストラベル
            
            try
                % 分割数の取得
                k = obj.params.classifier.cnn.training.validation.kfold;

                % データサイズの取得
                [~, ~, numEpochs] = size(data);
                fprintf('\nデータセット分割:\n');
                fprintf('  - 総エポック数: %d\n', numEpochs);

                % 分割比率の計算
                trainRatio = (k-1)/k;  % (k-1)/k
                valRatio = 1/(2*k);    % 0.5/k
                testRatio = 1/(2*k);   % 0.5/k
        
                % クラスバランスを維持するため、層別化サンプリングを使用
                uniqueLabels = unique(labels);
                numClasses = length(uniqueLabels);
                
                fprintf('  - クラス数: %d\n', numClasses);
                
                % クラスごとのインデックスを取得
                classIndices = cell(numClasses, 1);
                for i = 1:numClasses
                    classIndices{i} = find(labels == uniqueLabels(i));
                    fprintf('  - クラス %d: %d サンプル\n', uniqueLabels(i), length(classIndices{i}));
                end
                
                % 各クラスごとに分割
                trainIdx = [];
                valIdx = [];
                testIdx = [];
                
                for i = 1:numClasses
                    currentIndices = classIndices{i};
                    
                    % インデックスをランダムに並べ替え（毎回異なる結果になる）
                    randomOrder = randperm(length(currentIndices));
                    shuffledIndices = currentIndices(randomOrder);
                    
                    % 分割数の計算
                    numTrain = round(length(shuffledIndices) * trainRatio);
                    numVal = round(length(shuffledIndices) * valRatio);
                    
                    % インデックスの分割
                    trainIdx = [trainIdx; shuffledIndices(1:numTrain)];
                    valIdx = [valIdx; shuffledIndices(numTrain+1:numTrain+numVal)];
                    testIdx = [testIdx; shuffledIndices(numTrain+numVal+1:end)];
                end
        
                % データの分割
                trainData = data(:,:,trainIdx);
                trainLabels = labels(trainIdx);
                
                valData = data(:,:,valIdx);
                valLabels = labels(valIdx);
                
                testData = data(:,:,testIdx);
                testLabels = labels(testIdx);
        
                % 分割結果のサマリー表示
                fprintf('  - 学習データ: %d サンプル (%.1f%%)\n', ...
                    length(trainIdx), (length(trainIdx)/numEpochs)*100);
                fprintf('  - 検証データ: %d サンプル (%.1f%%)\n', ...
                    length(valIdx), (length(valIdx)/numEpochs)*100);
                fprintf('  - テストデータ: %d サンプル (%.1f%%)\n', ...
                    length(testIdx), (length(testIdx)/numEpochs)*100);
        
                % データの検証
                if isempty(trainData) || isempty(valData) || isempty(testData)
                    error('分割後に空のデータセットが存在します');
                end
        
                % 分割後のクラス分布確認
                obj.checkClassDistribution('学習', trainLabels);
                obj.checkClassDistribution('検証', valLabels);
                obj.checkClassDistribution('テスト', testLabels);
        
            catch ME
                error('データ分割でエラーが発生しました: %s', ME.message);
            end
        end
        
        %% クラス分布確認メソッド
        function checkClassDistribution(~, setName, labels)
            % データセット内のクラス分布を解析して表示
            
            uniqueLabels = unique(labels);
            fprintf('\n%sデータのクラス分布:\n', setName);
            
            for i = 1:length(uniqueLabels)
                count = sum(labels == uniqueLabels(i));
                fprintf('  - クラス %d: %d サンプル (%.1f%%)\n', ...
                    uniqueLabels(i), count, (count/length(labels))*100);
            end
            
            % クラス不均衡の評価
            maxCount = max(histcounts(labels));
            minCount = min(histcounts(labels));
            imbalanceRatio = maxCount / max(minCount, 1);
            
            if imbalanceRatio > 3
                warning('%sデータセットのクラス不均衡が大きいです (比率: %.1f:1)', ...
                    setName, imbalanceRatio);
            end
        end

        %% LSTM用のデータ前処理（セル配列へ変換）
        function preparedData = prepareDataForLSTM(obj, data)
            % データをLSTMに適した形式に変換
            % MATLABのLSTMでは、各シーケンスのデータは [特徴量 × 時間ステップ] の形式である必要があります
            % (特徴量を行、時間ステップを列とする)
            %
            % 入力:
            %   data - 入力データ (3次元数値配列、2次元配列、またはセル配列)
            %
            % 出力:
            %   preparedData - LSTM用に整形されたデータ (セル配列)
            
            try
                if iscell(data)
                    % 入力が既にセル配列の場合
                    trials = numel(data);
                    preparedData = cell(trials, 1);
                    
                    for i = 1:trials
                        currentData = data{i};
                        if ~isa(currentData, 'double')
                            currentData = double(currentData);
                        end
                        
                        % 重要: MATLABのLSTMでは [特徴量 × 時間ステップ] の形式が必要
                        [dim1, dim2] = size(currentData);
                        % 時間ステップの方が特徴量より多いことが一般的
                        if dim1 > dim2
                            % データは [時間ステップ × 特徴量] の形式と思われるので、転置して
                            % [特徴量 × 時間ステップ] の形式に変換
                            currentData = currentData';
                            fprintf('  データ形状を変換: [%d×%d] → [%d×%d] (特徴量×時間ステップ)\n', dim1, dim2, dim2, dim1);
                        end
                        
                        % NaN/Inf値の検出と補間処理
                        currentData = obj.interpolateInvalidValues(currentData, i);
                        
                        preparedData{i} = currentData;
                    end
                    
                elseif ndims(data) == 3
                    % 3次元数値配列の処理 [チャンネル × 時間ポイント × 試行]
                    [channels, timepoints, trials] = size(data);
                    preparedData = cell(trials, 1);
                    
                    for i = 1:trials
                        currentData = data(:, :, i);
                        if ~isa(currentData, 'double')
                            currentData = double(currentData);
                        end
                        
                        % 既に [チャンネル × 時間ポイント] の形式になっているので、そのまま使用
                        % これはMATLABのLSTMが期待する [特徴量 × 時間ステップ] の形式と一致
                        
                        % NaN/Inf値の検出と補間処理
                        currentData = obj.interpolateInvalidValues(currentData, i);
                        
                        preparedData{i} = currentData;
                    end
                    
                elseif ismatrix(data)
                    % 2次元データ（単一試行）の処理
                    [dim1, dim2] = size(data);
                    
                    currentData = data;
                    if ~isa(currentData, 'double')
                        currentData = double(currentData);
                    end
                    
                    % チャンネル数が時点数より少ないことが一般的
                    if dim1 > dim2
                        % データは [時間ステップ × 特徴量] の形式と思われるので、転置して
                        % [特徴量 × 時間ステップ] の形式に変換
                        currentData = currentData';
                        fprintf('  データ形状を変換: [%d×%d] → [%d×%d] (特徴量×時間ステップ)\n', dim1, dim2, dim2, dim1);
                    end
                    
                    % NaN/Inf値の検出と補間処理
                    currentData = obj.interpolateInvalidValues(currentData, i);
                    
                    preparedData = {currentData};
                else
                    error('対応していないデータ次元数: %d (最大3次元まで対応)', ndims(data));
                end
                
                % 結果検証
                if isempty(preparedData)
                    error('変換後のデータが空です');
                end
                
            catch ME
                fprintf('LSTM用データ準備でエラーが発生: %s\n', ME.message);
                fprintf('入力データサイズ: [%s]\n', num2str(size(data)));
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end
        
        function processedData = interpolateInvalidValues(~, data, trialIndex)
            % NaN/Infなどの無効値を線形補間で処理
            % MATLABのLSTMでは、データは [特徴量 × 時間ステップ] の形式
            %
            % 入力:
            %   data - 処理するデータ [特徴量 × 時間ステップ]
            %   trialIndex - 試行インデックス（デバッグ情報用）
            %
            % 出力:
            %   processedData - 補間処理済みデータ
            
            processedData = data;
            [features, timepoints] = size(data);
            
            % 無効値の検出
            hasInvalidData = false;
            invalidCount = 0;
            
            for f = 1:features
                featureData = data(f, :);
                invalidIndices = isnan(featureData) | isinf(featureData);
                invalidCount = invalidCount + sum(invalidIndices);
                
                if any(invalidIndices)
                    hasInvalidData = true;
                    validIndices = ~invalidIndices;
                    
                    % 有効なデータポイントが十分ある場合は線形補間を使用
                    if sum(validIndices) > 1
                        % 補間のための準備
                        validTimePoints = find(validIndices);
                        validValues = featureData(validIndices);
                        invalidTimePoints = find(invalidIndices);
                        
                        % 線形補間を適用
                        interpolatedValues = interp1(validTimePoints, validValues, invalidTimePoints, 'linear', 'extrap');
                        featureData(invalidIndices) = interpolatedValues;
                    else
                        % 有効データが不足している場合は特徴量の平均値または0で置換
                        if sum(validIndices) == 1
                            % 1点のみ有効な場合はその値を使用
                            replacementValue = featureData(validIndices);
                        else
                            % 全て無効な場合は0を使用
                            replacementValue = 0;
                            fprintf('警告: 試行 %d, 特徴量 %d の全データポイントが無効です。0で置換します。\n', ...
                                trialIndex, f);
                        end
                        featureData(invalidIndices) = replacementValue;
                    end
                    
                    processedData(f, :) = featureData;
                end
            end
            
            % 無効値があった場合に情報を表示
            if hasInvalidData
                fprintf('試行 %d: %d個の無効値を検出し補間処理しました (%.1f%%)\n', ...
                    trialIndex, invalidCount, (invalidCount/(features*timepoints))*100);
            end
        end

        %% LSTMモデル学習メソッド
        function [lstmModel, trainInfo] = trainLSTMModel(obj, trainData, trainLabels, valData, valLabels)
            % LSTMモデルの構築と学習を実行
            %
            % 入力:
            %   trainData - 学習データ
            %   trainLabels - 学習ラベル
            %   valData - 検証データ
            %   valLabels - 検証ラベル
            %
            % 出力:
            %   lstmModel - 学習済みLSTMモデル
            %   trainInfo - 学習に関する情報
            
           try        
               fprintf('\n=== LSTMモデルの学習開始 ===\n');
               
               % GPUメモリの確認
               obj.checkGPUMemory();
               
               % データの構造確認と検証
               if ~isempty(trainData) && iscell(trainData) && ~isempty(trainData{1})
                   sampleData = trainData{1};
                   [numFeatures, numTimeSteps] = size(sampleData);
               else
                   error('不正な学習データ形式');
               end
        
               % ラベルのカテゴリカル変換
               uniqueLabels = unique(trainLabels);
               trainLabels = categorical(trainLabels, uniqueLabels);
        
               % 検証データの処理
               valDS = {};
               if ~isempty(valData)
                   valLabels = categorical(valLabels, uniqueLabels);
                   valDS = {valData, valLabels};
               end
        
               % トレーニング情報の初期化
               trainInfo = struct(...
                   'History', struct(...
                       'TrainingLoss', [], ...
                       'ValidationLoss', [], ...
                       'TrainingAccuracy', [], ...
                       'ValidationAccuracy', [] ...
                   ), ...
                   'FinalEpoch', 0 ...
               );
        
               % 実行環境の選択
               executionEnvironment = 'cpu';
               if obj.useGPU
                   executionEnvironment = 'gpu';
               end
        
               % レイヤーの構築 - 入力サイズを正しく渡す
               fprintf('LSTMアーキテクチャを構築中...\n');
               layers = obj.buildLSTMLayers(numFeatures);  % ここで正しい特徴量次元を渡す
               
               % トレーニングオプションの設定
               options = trainingOptions(obj.params.classifier.lstm.training.optimizer.type, ...
                   'InitialLearnRate', obj.params.classifier.lstm.training.optimizer.learningRate, ...
                   'MaxEpochs', obj.params.classifier.lstm.training.maxEpochs, ...
                   'MiniBatchSize', obj.params.classifier.lstm.training.miniBatchSize, ...
                   'Plots', 'none', ...
                   'Shuffle', obj.params.classifier.lstm.training.shuffle, ...
                   'ExecutionEnvironment', executionEnvironment, ...
                   'OutputNetwork', 'best-validation', ...
                   'Verbose', true, ...
                   'ValidationData', valDS, ...
                   'ValidationFrequency', obj.params.classifier.lstm.training.frequency, ...
                   'ValidationPatience', obj.params.classifier.lstm.training.patience, ...
                   'GradientThreshold', obj.params.classifier.lstm.training.optimizer.gradientThreshold);
               
               % モデルの学習
               fprintf('LSTMモデルの学習を開始...\n');
               [lstmModel, trainHistory] = trainNetwork(trainData, trainLabels, layers, options);
        
               % GPU使用時は、データをCPUに明示的に移動
               if obj.useGPU
                   try
                       trainHistory.TrainingLoss = gather(trainHistory.TrainingLoss);
                       trainHistory.ValidationLoss = gather(trainHistory.ValidationLoss);
                       trainHistory.TrainingAccuracy = gather(trainHistory.TrainingAccuracy);
                       trainHistory.ValidationAccuracy = gather(trainHistory.ValidationAccuracy);
                   catch ME
                       warning('GPU上のデータをCPUに移動中にエラーが発生しました');
                   end
               end
               
               % 学習履歴の保存
               trainInfo.History = trainHistory;
               trainInfo.FinalEpoch = length(trainHistory.TrainingLoss);
               
               % 学習情報の詳細ログ出力（デバッグ用）
               obj.logTrainInfo(trainInfo);
        
               fprintf('学習完了: %d エポック\n', trainInfo.FinalEpoch);
               fprintf('  - 最終学習損失: %.4f\n', trainHistory.TrainingLoss(end));
               fprintf('  - 最終検証損失: %.4f\n', trainHistory.ValidationLoss(end));
               fprintf('  - 最終学習精度: %.2f%%\n', trainHistory.TrainingAccuracy(end));
               fprintf('  - 最終検証精度: %.2f%%\n', trainHistory.ValidationAccuracy(end));
               
               % 最良の検証精度
               [bestValAcc, bestEpoch] = max(trainHistory.ValidationAccuracy);
               fprintf('  - 最良検証精度: %.2f%% (エポック %d)\n', bestValAcc, bestEpoch);
        
               % GPUメモリ解放
               obj.resetGPUMemory();
        
           catch ME
               fprintf('LSTMモデル学習でエラーが発生: %s\n', ME.message);
               obj.resetGPUMemory();
               rethrow(ME);
           end
        end

        %% LSTM層構築メソッド
        function layers = buildLSTMLayers(obj, inputSize)
            % 入力データに適したLSTMアーキテクチャを構築
            %
            % 入力:
            %   inputSize - 入力特徴量次元
            %
            % 出力:
            %   layers - LSTMレイヤー構造
            
            try
                % アーキテクチャパラメータの取得
                arch = obj.params.classifier.lstm.architecture;
                numClasses = arch.numClasses;
                
                fprintf('LSTMレイヤー構築: 入力特徴量=%d, クラス数=%d\n', inputSize, numClasses);
                
                % レイヤー数の計算
                numLSTMLayers = length(fieldnames(arch.lstmLayers));
                numFCLayers = length(arch.fullyConnected);
                
                % セル配列でレイヤーを構築
                layers = cell(1, numLSTMLayers * 3 + numFCLayers * 2 + 3);
                layerIdx = 1;
                
                % 入力層 - ここで正しい入力サイズを指定することが重要
                layers{layerIdx} = sequenceInputLayer(inputSize, ...
                    'Normalization', 'none', ...
                    'Name', 'input');
                layerIdx = layerIdx + 1;
                
                % LSTM層、バッチ正規化、ドロップアウト層の追加
                lstmLayerNames = fieldnames(arch.lstmLayers);
                for i = 1:length(lstmLayerNames)
                    lstmParams = arch.lstmLayers.(lstmLayerNames{i});
                    
                    % ドロップアウト率の取得
                    if isfield(arch.dropoutLayers, ['dropout' num2str(i)])
                        dropoutRate = arch.dropoutLayers.(['dropout' num2str(i)]);
                    else
                        dropoutRate = 0.5;  % デフォルト値
                    end
                    
                    % LSTM層の追加
                    layers{layerIdx} = lstmLayer(lstmParams.numHiddenUnits, ...
                        'OutputMode', lstmParams.OutputMode, ...
                        'Name', ['lstm' num2str(i)]);
                    layerIdx = layerIdx + 1;
                    
                    % バッチ正規化層（オプション）
                    if arch.batchNorm
                        layers{layerIdx} = batchNormalizationLayer('Name', ['bn' num2str(i)]);
                        layerIdx = layerIdx + 1;
                    end
                    
                    % ドロップアウト層
                    layers{layerIdx} = dropoutLayer(dropoutRate, 'Name', ['dropout' num2str(i)]);
                    layerIdx = layerIdx + 1;
                end
                
                % 全結合層とReLU層の追加
                for i = 1:length(arch.fullyConnected)
                    layers{layerIdx} = fullyConnectedLayer(arch.fullyConnected(i), ...
                        'Name', ['fc' num2str(i)]);
                    layerIdx = layerIdx + 1;
                    
                    layers{layerIdx} = reluLayer('Name', ['relu_fc' num2str(i)]);
                    layerIdx = layerIdx + 1;
                end
                
                % 出力層
                layers{layerIdx} = fullyConnectedLayer(numClasses, 'Name', 'fc_output');
                layerIdx = layerIdx + 1;
                
                layers{layerIdx} = softmaxLayer('Name', 'softmax');
                layerIdx = layerIdx + 1;
                
                layers{layerIdx} = classificationLayer('Name', 'output');
                
                % 不要な要素を削除
                layers = layers(1:layerIdx);
                
                % レイヤー配列に変換（セル配列を結合して配列に変換）
                layers = [layers{:}];
                
                fprintf('LSTMアーキテクチャ構築完了: %dレイヤー\n', layerIdx);
                
            catch ME
                fprintf('LSTMレイヤー構築でエラーが発生: %s\n', ME.message);
                rethrow(ME);
            end
        end

        %% モデル評価メソッド
        function metrics = evaluateModel(~, model, testData, testLabels)
            % 学習済みモデルの性能を評価
            %
            % 入力:
            %   model - 学習済みLSTMモデル
            %   testData - テストデータ
            %   testLabels - テストラベル
            %
            % 出力:
            %   metrics - 詳細な評価メトリクス
            
            fprintf('\n=== モデル評価を実行 ===\n');
            metrics = struct(...
                'accuracy', [], ...
                'score', [], ...
                'confusionMat', [], ...
                'classwise', [], ...
                'roc', [], ...
                'auc', [] ...
            );
            
           % テストラベルをカテゴリカル型に変換
           uniqueLabels = unique(testLabels);
           testLabels = categorical(testLabels, uniqueLabels);

           % モデルの評価
           [pred, score] = classify(model, testData);
           metrics.score = score;

           % 基本的な指標の計算
           metrics.accuracy = mean(pred == testLabels);
           metrics.confusionMat = confusionmat(testLabels, pred);
           
           fprintf('テスト精度: %.2f%%\n', metrics.accuracy * 100);

           % クラスごとの性能評価
           classes = unique(testLabels);
           metrics.classwise = struct('precision', zeros(1,length(classes)), ...
                                    'recall', zeros(1,length(classes)), ...
                                    'f1score', zeros(1,length(classes)));

           fprintf('\nクラスごとの評価:\n');
           for i = 1:length(classes)
               className = classes(i);
               classIdx = (testLabels == className);

               % 各クラスの指標計算
               TP = sum(pred(classIdx) == className);
               FP = sum(pred == className) - TP;
               FN = sum(classIdx) - TP;

               % 0による除算を回避
               if (TP + FP) > 0
                   precision = TP / (TP + FP);
               else
                   precision = 0;
               end
               
               if (TP + FN) > 0
                   recall = TP / (TP + FN);
               else
                   recall = 0;
               end
               
               if (precision + recall) > 0
                   f1 = 2 * (precision * recall) / (precision + recall);
               else
                   f1 = 0;
               end

               metrics.classwise(i).precision = precision;
               metrics.classwise(i).recall = recall;
               metrics.classwise(i).f1score = f1;
               
               fprintf('  - クラス %d:\n', i);
               fprintf('    - 精度 (Precision): %.2f%%\n', precision * 100);
               fprintf('    - 再現率 (Recall): %.2f%%\n', recall * 100);
               fprintf('    - F1スコア: %.2f\n', f1);
           end

           % ROC曲線とAUC（2クラス分類の場合）
           if length(classes) == 2
               [X, Y, T, AUC] = perfcurve(testLabels, score(:,2), classes(2));
               metrics.roc = struct('X', X, 'Y', Y, 'T', T);
               metrics.auc = AUC;
               fprintf('\nAUC: %.3f\n', AUC);
           end
           
           % 混同行列の表示
           fprintf('\n混同行列:\n');
           disp(metrics.confusionMat);
        end

        %% トレーニング情報検証メソッド
        function validateTrainInfo(~, trainInfo)
            % トレーニング情報の構造を検証
            
            if ~isstruct(trainInfo) 
                error('trainInfoが構造体ではありません');
            end
            
            if ~isfield(trainInfo, 'History')
                error('trainInfoにHistoryフィールドがありません');
            end
            
            if ~isfield(trainInfo.History, 'TrainingAccuracy') || ...
               ~isfield(trainInfo.History, 'ValidationAccuracy')
                error('Historyに必要な精度フィールドがありません');
            end
        end

        %% 学習曲線分析メソッド
        function [trainTrend, valTrend] = analyzeLearningCurves(~, trainAcc, valAcc)
            % 学習曲線を分析し、傾向と特性を抽出
            %
            % 入力:
            %   trainAcc - 学習精度履歴
            %   valAcc - 検証精度履歴
            %
            % 出力:
            %   trainTrend - 学習精度の傾向分析結果
            %   valTrend - 検証精度の傾向分析結果
            
            fprintf('\n=== 学習曲線の分析 ===\n');
            
            try
                % データの検証
                if isempty(trainAcc) || isempty(valAcc)
                    error('学習曲線データが空です');
                end
                
                % NaNチェックと除去
                trainAcc = trainAcc(~isnan(trainAcc));
                valAcc = valAcc(~isnan(valAcc));
                
                if isempty(trainAcc) || isempty(valAcc)
                    error('NaN除去後に有効なデータがありません');
                end
                
                % エポック数の取得
                numEpochs = length(trainAcc);
                
                % 移動平均の計算（適応的なウィンドウサイズ）
                windowSize = min(5, floor(numEpochs/3));
                windowSize = max(windowSize, 1);  % 最小値を保証
                
                trainSmooth = movmean(trainAcc, windowSize);
                valSmooth = movmean(valAcc, windowSize);
                
                % 変化率の計算
                if length(trainSmooth) > 1
                    trainDiff = diff(trainSmooth);
                    valDiff = diff(valSmooth);
                else
                    trainDiff = 0;
                    valDiff = 0;
                end
                
                % トレンド指標の計算
                trainTrend = struct(...
                    'mean_change', mean(trainDiff), ...
                    'volatility', std(trainDiff), ...
                    'increasing_ratio', sum(trainDiff > 0) / max(length(trainDiff), 1), ...
                    'plateau_detected', detectPlateau(trainSmooth), ...
                    'oscillation_strength', calculateOscillation(trainDiff), ...
                    'convergence_epoch', estimateConvergenceEpoch(trainSmooth) ...
                );
                
                valTrend = struct(...
                    'mean_change', mean(valDiff), ...
                    'volatility', std(valDiff), ...
                    'increasing_ratio', sum(valDiff > 0) / max(length(valDiff), 1), ...
                    'plateau_detected', detectPlateau(valSmooth), ...
                    'oscillation_strength', calculateOscillation(valDiff), ...
                    'convergence_epoch', estimateConvergenceEpoch(valSmooth) ...
                );
                
                % 詳細な分析結果の表示
                fprintf('学習精度の傾向:\n');
                fprintf('  - 平均変化率: %.4f/エポック\n', trainTrend.mean_change);
                fprintf('  - 変動性: %.4f\n', trainTrend.volatility);
                fprintf('  - 上昇率: %.1f%%\n', trainTrend.increasing_ratio*100);
                fprintf('  - プラトー検出: %s\n', string(trainTrend.plateau_detected));
                fprintf('  - 収束エポック: %d/%d\n', trainTrend.convergence_epoch, numEpochs);
                
                fprintf('検証精度の傾向:\n');
                fprintf('  - 平均変化率: %.4f/エポック\n', valTrend.mean_change);
                fprintf('  - 変動性: %.4f\n', valTrend.volatility);
                fprintf('  - 上昇率: %.1f%%\n', valTrend.increasing_ratio*100);
                fprintf('  - プラトー検出: %s\n', string(valTrend.plateau_detected));
                fprintf('  - 収束エポック: %d/%d\n', valTrend.convergence_epoch, numEpochs);
                
            catch ME
                fprintf('学習曲線分析でエラーが発生: %s\n', ME.message);
                
                % エラー時のフォールバック値を設定
                trainTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, ...
                    'plateau_detected', false, 'oscillation_strength', 0, 'convergence_epoch', 0);
                valTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, ...
                    'plateau_detected', false, 'oscillation_strength', 0, 'convergence_epoch', 0);
            end

            % 補助関数: プラトー検出
            function isPlateau = detectPlateau(smoothedCurve)
                % 学習曲線でのプラトー（停滞）状態を検出
                
                % 最小必要長のチェック
                if length(smoothedCurve) < 5
                    isPlateau = false;
                    return;
                end
                
                % 後半部分の変化率を分析
                halfLength = max(3, floor(length(smoothedCurve)/2));
                lastSegment = smoothedCurve(end-halfLength+1:end);
                segmentDiff = diff(lastSegment);
                
                % 変化が非常に小さい場合はプラトーと見なす
                isPlateau = mean(abs(segmentDiff)) < 0.001;
            end

            % 補助関数: 振動強度計算
            function oscillation = calculateOscillation(diffValues)
                % 学習曲線の振動強度を計算
                
                % 最小必要長のチェック
                if length(diffValues) < 2
                    oscillation = 0;
                    return;
                end
                
                % 符号変化の数をカウント（振動の指標）
                signChanges = sum(diff(sign(diffValues)) ~= 0);
                oscillation = signChanges / (length(diffValues) - 1);  % 正規化
            end

            % 補助関数: 収束エポック推定
            function convergenceEpoch = estimateConvergenceEpoch(smoothedCurve)
                % 学習が収束したエポックを推定
                
                % 最小必要長のチェック
                if length(smoothedCurve) < 5
                    convergenceEpoch = length(smoothedCurve);
                    return;
                end
                
                % 曲線の変化率を計算
                diffValues = abs(diff(smoothedCurve));
                threshold = 0.001;  % 収束判定の閾値
                
                % 連続する数ポイントで閾値以下を検出
                convergedIdx = find(diffValues < threshold, 3, 'first');
                
                if length(convergedIdx) >= 3 && (convergedIdx(3) - convergedIdx(1)) == 2
                    % 3ポイント連続で変化が小さい場合
                    convergenceEpoch = convergedIdx(1) + 1;  % +1は差分からインデックスへの調整
                else
                    % 収束点が見つからない場合は終点を返す
                    convergenceEpoch = length(smoothedCurve);
                end
            end
        end

        %% 最適エポック検出メソッド
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(~, valAcc)
            % 検証精度に基づいて最適なエポックを特定
            %
            % 入力:
            %   valAcc - 検証精度履歴
            %
            % 出力:
            %   optimalEpoch - 最適エポック
            %   totalEpochs - 総エポック数
            
            try
                totalEpochs = length(valAcc);
                [~, optimalEpoch] = max(valAcc);
                
                % 最適エポックが最後のエポックの場合、改善の余地がある可能性
                if optimalEpoch == totalEpochs
                    fprintf('警告: 最適エポックが最終エポックと一致。より長い学習が有益かもしれません。\n');
                end
        
            catch ME
                fprintf('最適エポック検出でエラーが発生: %s\n', ME.message);
                optimalEpoch = 0;
                totalEpochs = 0;
            end
        end

        %% Early Stopping効果分析メソッド
        function effect = analyzeEarlyStoppingEffect(~, trainInfo)
            % Early Stoppingの効果を分析
            %
            % 入力:
            %   trainInfo - トレーニング情報
            %
            % 出力:
            %   effect - Early Stopping効果の分析結果
            
            % バリデーション損失を取得
            valLoss = trainInfo.History.ValidationLoss;
            
            % 最小損失エポックを特定
            [~, minLossEpoch] = min(valLoss);
            totalEpochs = length(valLoss);
            
            % 効果分析の構造体を作成
            effect = struct(...
                'optimal_epoch', minLossEpoch, ...
                'total_epochs', totalEpochs, ...
                'stopping_efficiency', minLossEpoch / totalEpochs, ...
                'potential_savings', totalEpochs - minLossEpoch ...
            );
            
            fprintf('Early Stopping分析:\n');
            fprintf('  - 最適エポック: %d/%d (%.1f%%)\n', ...
                effect.optimal_epoch, effect.total_epochs, effect.stopping_efficiency*100);
            
            if effect.potential_savings > 0
                fprintf('  - 潜在的節約: %dエポック\n', effect.potential_savings);
            end
        end

        %% 検証-テスト精度ギャップ分析メソッド
        function [isOverfit, metrics] = validateTestValidationGap(~, valAccHistory, testAcc, dataSize)
            % 検証精度とテスト精度の差を統計的に評価
            %
            % 入力:
            %   valAccHistory - 検証精度の履歴
            %   testAcc - テスト精度
            %   dataSize - データサイズ（サンプル数）
            %
            % 出力:
            %   isOverfit - 過学習の有無（論理値）
            %   metrics - 詳細な評価メトリクス

            % NaNチェックを追加
            if isempty(valAccHistory) || all(isnan(valAccHistory))
                fprintf('警告: 有効な検証精度履歴がありません\n');
                metrics = struct(...
                    'rawGap', NaN, ...
                    'normalizedGap', NaN, ...
                    'adjustedGap', NaN, ...
                    'meanValAcc', NaN, ...
                    'stdValAcc', NaN, ...
                    'testAcc', testAcc*100, ...
                    'severity', 'unknown' ...
                );
                isOverfit = true;
                return;
            end
            
            % NaNを除去
            valAccHistory = valAccHistory(~isnan(valAccHistory));
            
            if isempty(valAccHistory)
                fprintf('警告: NaN除去後に有効なデータがありません\n');
                metrics = struct(...
                    'rawGap', NaN, ...
                    'normalizedGap', NaN, ...
                    'adjustedGap', NaN, ...
                    'meanValAcc', NaN, ...
                    'stdValAcc', NaN, ...
                    'testAcc', testAcc*100, ...
                    'severity', 'unknown' ...
                );
                isOverfit = true;
                return;
            end

            % 安定区間の計算（最低5ポイント、または全データの30%のいずれか大きい方）
            minStablePoints = min(5, length(valAccHistory));
            stableIdx = max(1, length(valAccHistory) - minStablePoints + 1):length(valAccHistory);
            stableValAcc = valAccHistory(stableIdx);
            
            % 統計量の計算
            meanValAcc = mean(stableValAcc);
            stdValAcc = std(stableValAcc);
            
            % ゼロ除算防止
            if stdValAcc < 0.001
                stdValAcc = 0.001; 
            end
            
            % スケールされたギャップ計算（データサイズで調整）
            % 小さいデータセットでは大きなギャップが生じやすい
            scaleFactor = min(1, sqrt(dataSize / 1000));  % データサイズに基づく調整
            
            % 正規化された差（z-スコア的アプローチ）
            normalizedGap = abs(meanValAcc - testAcc*100) / max(stdValAcc, 1);
            
            % スケール調整されたギャップ
            adjustedGap = normalizedGap * scaleFactor;
            
            % 過学習の判定（調整されたギャップに基づく）
            if adjustedGap > 3
                severity = 'critical';      % 3標準偏差以上
            elseif adjustedGap > 2
                severity = 'severe';        % 2標準偏差以上
            elseif adjustedGap > 1.5
                severity = 'moderate';      % 1.5標準偏差以上
            elseif adjustedGap > 1
                severity = 'mild';          % 1標準偏差以上
            else
                severity = 'none';
            end
            
            % 結果の格納
            metrics = struct(...
                'rawGap', abs(meanValAcc - testAcc*100), ...
                'normalizedGap', normalizedGap, ...
                'adjustedGap', adjustedGap, ...
                'meanValAcc', meanValAcc, ...
                'stdValAcc', stdValAcc, ...
                'testAcc', testAcc*100, ...
                'severity', severity ...
            );
            
            isOverfit = ~strcmp(severity, 'none');
            
            % 結果の表示
            fprintf('\n=== 検証-テスト精度ギャップ分析 ===\n');
            fprintf('  平均検証精度: %.2f%% (±%.2f%%)\n', meanValAcc, stdValAcc);
            fprintf('  テスト精度: %.2f%%\n', testAcc*100);
            fprintf('  基本ギャップ: %.2f%%\n', metrics.rawGap);
            fprintf('  正規化ギャップ: %.2f (スケーリング後: %.2f)\n', normalizedGap, adjustedGap);
            fprintf('  判定結果: %s\n', severity);
        end

        %% 分類バイアス検出メソッド
        function isCompletelyBiased = detectClassificationBias(~, testMetrics)
            % 混同行列から分類バイアスを検出
            %
            % 入力:
            %   testMetrics - 評価メトリクス
            %
            % 出力:
            %   isCompletelyBiased - 完全なバイアスの有無
            
            isCompletelyBiased = false;
    
            if isfield(testMetrics, 'confusionMat')
                cm = testMetrics.confusionMat;
                
                % 各実際のクラス（行）のサンプル数を確認
                rowSums = sum(cm, 2);
                missingActual = any(rowSums == 0);
                
                % 各予測クラス（列）の予測件数を確認
                colSums = sum(cm, 1);
                missingPredicted = any(colSums == 0);
                
                % すべての予測が1クラスに集中しているかを検出
                predictedClassCount = sum(colSums > 0);
                
                % いずれかが true ならば、全く現れないクラスがあると判断
                isCompletelyBiased = missingActual || missingPredicted || predictedClassCount <= 1;
                
                if isCompletelyBiased
                    fprintf('\n警告: 分類に完全な偏りが検出されました\n');
                    fprintf('  - 分類された実際のクラス数: %d / %d\n', sum(rowSums > 0), size(cm, 1));
                    fprintf('  - 予測されたクラス数: %d / %d\n', predictedClassCount, size(cm, 2));
                    
                    % 混同行列の出力
                    fprintf('  混同行列:\n');
                    disp(cm);
                end
            end
        end
        
        %% フォールバック過学習メトリクス作成メソッド
        function metrics = createFallbackOverfitMetrics(~)
            % エラー発生時のフォールバックメトリクスを作成
            
            metrics = struct(...
                'performanceGap', Inf, ...
                'isCompletelyBiased', true, ...
                'isLearningProgressing', false, ...
                'validationTrend', struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0), ...
                'trainingTrend', struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0), ...
                'severity', 'error', ...
                'optimalEpoch', 0, ...
                'totalEpochs', 0);
        end

        %% 学習情報ログ出力メソッド
        function logTrainInfo(~, trainInfo)
            % trainInfoの内容を詳細にログ出力（デバッグ用）
            
            fprintf('\n=== 学習情報の詳細 ===\n');
            
            if ~isstruct(trainInfo)
                fprintf('trainInfoが構造体ではありません\n');
                return;
            end
            
            % フィールド一覧の表示
            fprintf('trainInfoのフィールド: %s\n', strjoin(fieldnames(trainInfo), ', '));
            
            % Historyフィールドの詳細
            if isfield(trainInfo, 'History')
                fprintf('Historyフィールドのサブフィールド: %s\n', strjoin(fieldnames(trainInfo.History), ', '));
                
                % 各履歴データのサイズと値
                if isfield(trainInfo.History, 'TrainingLoss')
                    loss = trainInfo.History.TrainingLoss;
                    fprintf('TrainingLoss: %d要素, 範囲[%.4f, %.4f]\n', ...
                        length(loss), min(loss), max(loss));
                end
                
                if isfield(trainInfo.History, 'ValidationLoss')
                    loss = trainInfo.History.ValidationLoss;
                    fprintf('ValidationLoss: %d要素, 範囲[%.4f, %.4f]\n', ...
                        length(loss), min(loss), max(loss));
                end
                
                if isfield(trainInfo.History, 'TrainingAccuracy')
                    acc = trainInfo.History.TrainingAccuracy;
                    fprintf('TrainingAccuracy: %d要素, 範囲[%.2f%%, %.2f%%]\n', ...
                        length(acc), min(acc), max(acc));
                end
                
                if isfield(trainInfo.History, 'ValidationAccuracy')
                    acc = trainInfo.History.ValidationAccuracy;
                    fprintf('ValidationAccuracy: %d要素, 範囲[%.2f%%, %.2f%%]\n', ...
                        length(acc), min(acc), max(acc));
                end
            else
                fprintf('Historyフィールドがありません\n');
            end
            
            % その他のフィールド
            if isfield(trainInfo, 'FinalEpoch')
                fprintf('FinalEpoch: %d\n', trainInfo.FinalEpoch);
            else
                fprintf('FinalEpochフィールドがありません\n');
            end
            
            fprintf('=== 学習情報の詳細終了 ===\n\n');
        end

        %% 性能指標の更新
        function updatePerformanceMetrics(obj, testMetrics)
            % 評価結果から性能メトリクスを更新
            obj.performance = testMetrics;
        end
        
        %% GPU使用状況確認メソッド
        function checkGPUMemory(obj)
            % GPU使用状況の確認とメモリ使用率の監視
            
            if obj.useGPU
                try
                    device = gpuDevice();
                    totalMem = device.TotalMemory / 1e9;  % GB
                    availMem = device.AvailableMemory / 1e9;  % GB
                    usedMem = totalMem - availMem;
                    
                    % メモリ使用情報の更新
                    obj.gpuMemory.total = totalMem;
                    obj.gpuMemory.used = usedMem;
                    obj.gpuMemory.peak = max(obj.gpuMemory.peak, usedMem);
                    
                    fprintf('GPU使用状況: %.2f/%.2f GB (%.1f%%)\n', ...
                        usedMem, totalMem, (usedMem/totalMem)*100);
                    
                    % メモリ使用率が高い場合は警告と対応
                    if usedMem/totalMem > 0.8
                        warning('GPU使用率が80%%を超えています。パフォーマンスに影響する可能性があります。');
                        
                        % 適応的なバッチサイズ調整
                        if obj.params.classifier.lstm.training.miniBatchSize > 32
                            newBatchSize = max(16, floor(obj.params.classifier.lstm.training.miniBatchSize / 2));
                            fprintf('高メモリ使用率のため、バッチサイズを%dから%dに削減します\n', ...
                                obj.params.classifier.lstm.training.miniBatchSize, newBatchSize);
                            obj.params.classifier.lstm.training.miniBatchSize = newBatchSize;
                        end
                    end
                catch ME
                    fprintf('GPUメモリチェックでエラー: %s', ME.message);
                end
            end
        end
        
        %% GPUメモリ解放メソッド
        function resetGPUMemory(obj)
            % GPUメモリのリセットと解放
            
            if obj.useGPU
                try
                    % GPUデバイスのリセット
                    currentDevice = gpuDevice();
                    reset(currentDevice);
                    
                    % リセット後のメモリ状況
                    availMem = currentDevice.AvailableMemory / 1e9;  % GB
                    totalMem = currentDevice.TotalMemory / 1e9;  % GB
                    fprintf('GPUメモリをリセットしました (利用可能: %.2f/%.2f GB)\n', ...
                        availMem, totalMem);
                        
                    % 使用状況の更新
                    obj.gpuMemory.used = totalMem - availMem;
                    
                catch ME
                    fprintf('GPUメモリのリセットに失敗: %s', ME.message);
                end
            end
        end

        %% 結果構造体構築メソッド
        function results = buildResultsStruct(obj, lstmModel, metrics, trainInfo, normParams)
            % 結果構造体の構築
            
            % trainInfoの検証と安全なディープコピー
            obj.validateTrainInfo(trainInfo);
            trainInfoCopy = obj.createDeepCopy(trainInfo);
            
            % 結果構造体の構築
            results = struct(...
                'model', lstmModel, ...
                'performance', metrics, ...
                'trainInfo', trainInfoCopy, ...
                'overfitting', obj.overfitMetrics, ...
                'normParams', normParams, ...
                'params', obj.extractParams() ...
            );
            
            % 出力前に結果の検証
            obj.validateResults(results);
        end
        
        %% パラメータ抽出メソッド
        function params = extractParams(obj)
            % LSTMのパラメータを抽出して配列として返す
            
            % デフォルト値で初期化
            params = [0, 0, 0, 0, 0, 0];
            
            try
                % 1. 学習率
                if isfield(obj.params.classifier.lstm.training.optimizer, 'learningRate')
                    params(1) = obj.params.classifier.lstm.training.optimizer.learningRate;
                end
                
                % 2. バッチサイズ
                if isfield(obj.params.classifier.lstm.training, 'miniBatchSize')
                    params(2) = obj.params.classifier.lstm.training.miniBatchSize;
                end
                
                % 3. LSTMユニット数
                if isfield(obj.params.classifier.lstm.architecture, 'lstmLayers')
                    lstmFields = fieldnames(obj.params.classifier.lstm.architecture.lstmLayers);
                    if ~isempty(lstmFields)
                        firstLSTM = obj.params.classifier.lstm.architecture.lstmLayers.(lstmFields{1});
                        if isfield(firstLSTM, 'numHiddenUnits')
                            params(3) = firstLSTM.numHiddenUnits;
                        end
                    end
                end
                
                % 4. LSTM層数
                if isfield(obj.params.classifier.lstm.architecture, 'lstmLayers')
                    params(4) = length(fieldnames(obj.params.classifier.lstm.architecture.lstmLayers));
                end
                
                % 5. ドロップアウト率
                if isfield(obj.params.classifier.lstm.architecture, 'dropoutLayers')
                    dropout = obj.params.classifier.lstm.architecture.dropoutLayers;
                    dropoutFields = fieldnames(dropout);
                    if ~isempty(dropoutFields)
                        params(5) = dropout.(dropoutFields{1});
                    end
                end
                
                % 6. 全結合層ユニット数
                if isfield(obj.params.classifier.lstm.architecture, 'fullyConnected')
                    fc = obj.params.classifier.lstm.architecture.fullyConnected;
                    if ~isempty(fc)
                        params(6) = fc(1);
                    end
                end
            catch ME
                warning('パラメータ抽出中にエラーが発生');
            end
        end
        
        %% 結果検証メソッド
        function validateResults(~, results)
            % 結果構造体の妥当性を検証
            
            if ~isstruct(results)
                warning('結果が構造体ではありません');
                return;
            end
            
            % 必須フィールドのチェック
            requiredFields = {'model', 'performance', 'trainInfo', 'params'};
            for i = 1:length(requiredFields)
                if ~isfield(results, requiredFields{i})
                    warning('結果構造体に必須フィールド「%s」がありません', requiredFields{i});
                end
            end
            
            % paramsフィールドのチェック
            if isfield(results, 'params')
                if length(results.params) ~= 6
                    warning('paramsフィールドの長さが期待値と異なります: %d (期待値: 6)', length(results.params));
                end
            end
            
            % パフォーマンスフィールドのチェック
            if isfield(results, 'performance')
                if ~isfield(results.performance, 'accuracy')
                    warning('performance構造体にaccuracyフィールドがありません');
                end
            end
        end
        
        %% 構造体の深いコピー作成
        function copy = createDeepCopy(obj, original)
            % 構造体のコピーを作成
            
            if ~isstruct(original)
                copy = original;
                return;
            end
            
            % 新しい構造体を作成
            copy = struct();
            
            % 各フィールドを再帰的にコピー
            fields = fieldnames(original);
            for i = 1:length(fields)
                field = fields{i};
                if isstruct(original.(field))
                    % 構造体の場合は再帰的にコピー
                    copy.(field) = obj.createDeepCopy(original.(field));
                elseif iscell(original.(field))
                    % セル配列の場合
                    cellArray = original.(field);
                    newCellArray = cell(size(cellArray));
                    for j = 1:numel(cellArray)
                        if isstruct(cellArray{j})
                            newCellArray{j} = obj.createDeepCopy(cellArray{j});
                        else
                            newCellArray{j} = cellArray{j};
                        end
                    end
                    copy.(field) = newCellArray;
                else
                    % その他のデータ型はそのままコピー
                    copy.(field) = original.(field);
                end
            end
        end

        %% 結果表示メソッド
        function displayResults(obj)
            % 総合的な結果サマリーの表示
            
            try
                fprintf('\n=== LSTM分類結果サマリー ===\n');
                
                if ~isempty(obj.performance)
                    % 精度情報
                    fprintf('全体精度: %.2f%%\n', obj.performance.accuracy * 100);

                    if isfield(obj.performance, 'auc')
                        fprintf('AUC: %.3f\n', obj.performance.auc);
                    end

                    % 混同行列
                    if ~isempty(obj.performance.confusionMat)
                        fprintf('\n混同行列:\n');
                        disp(obj.performance.confusionMat);
                    end

                    % クラスごとの性能
                    if isfield(obj.performance, 'classwise') && ~isempty(obj.performance.classwise)
                        fprintf('\nクラスごとの性能:\n');
                        for i = 1:length(obj.performance.classwise)
                            fprintf('クラス %d:\n', i);
                            fprintf('  - 精度: %.2f%%\n', ...
                                obj.performance.classwise(i).precision * 100);
                            fprintf('  - 再現率: %.2f%%\n', ...
                                obj.performance.classwise(i).recall * 100);
                            fprintf('  - F1スコア: %.2f\n', ...
                                obj.performance.classwise(i).f1score);
                        end
                    end
                end

                % 過学習分析結果の表示
                if ~isempty(obj.overfitMetrics) && isstruct(obj.overfitMetrics)
                    fprintf('\n過学習分析:\n');
                    fprintf('  - 性能ギャップ: %.2f%%\n', obj.overfitMetrics.performanceGap);
                    fprintf('  - 重大度: %s\n', obj.overfitMetrics.severity);
                    
                    fprintf('\n学習カーブ分析:\n');
                    
                    % 検証傾向
                    if isfield(obj.overfitMetrics, 'validationTrend')
                        trend = obj.overfitMetrics.validationTrend;
                        fprintf('  - 検証平均変化率: %.4f\n', trend.mean_change);
                        fprintf('  - 検証変動性: %.4f\n', trend.volatility);
                        fprintf('  - 検証上昇率: %.2f%%\n', trend.increasing_ratio * 100);
                    end
                    
                    % 最適エポック
                    if isfield(obj.overfitMetrics, 'optimalEpoch') && ...
                       isfield(obj.overfitMetrics, 'totalEpochs')
                        fprintf('  - 最適エポック: %d/%d (%.2f%%)\n', ...
                            obj.overfitMetrics.optimalEpoch, ...
                            obj.overfitMetrics.totalEpochs, ...
                            (obj.overfitMetrics.optimalEpoch / ...
                             max(obj.overfitMetrics.totalEpochs, 1)) * 100);
                    end
                    
                    % Early Stopping効果
                    if isfield(obj.overfitMetrics, 'earlyStoppingEffect')
                        effect = obj.overfitMetrics.earlyStoppingEffect;
                        fprintf('  - Early Stopping効率: %.2f\n', effect.stopping_efficiency);
                        if effect.potential_savings > 0
                            fprintf('  - 潜在的エポック節約: %d\n', effect.potential_savings);
                        end
                    end
                end

            catch ME
                fprintf('結果表示でエラーが発生: %s', ME.message);
            end
        end
    end
end