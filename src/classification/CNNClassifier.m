classdef CNNClassifier < handle
    %% CNNClassifier - EEGデータ用畳み込みニューラルネットワーク分類器
    %
    % このクラスはEEGデータに対して畳み込みニューラルネットワーク(CNN)ベースの
    % 分類を行います。脳波特有のデータ構造に適したアーキテクチャを使用し、
    % データ前処理、モデル学習、評価、予測の機能を提供します。
    %
    % 主な機能:
    %   - EEGデータの前処理と次元変換
    %   - データ拡張と正規化
    %   - EEG特化型CNNアーキテクチャの構築
    %   - モデルのトレーニングと評価
    %   - 過学習の検出と詳細分析
    %   - 交差検証
    %   - オンライン予測
    
    properties (Access = private)
        params              % システム設定パラメータ
        net                 % 学習済みCNNネットワーク
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
        function obj = CNNClassifier(params)
            % CNNClassifierのインスタンスを初期化
            %
            % 入力:
            %   params - 設定パラメータ
            
            % 基本パラメータの設定
            obj.params = params;
            obj.isInitialized = false;
            obj.useGPU = params.classifier.cnn.gpu;
            
            % プロパティの初期化
            obj.initializeProperties();
            
            % コンポーネントの初期化
            obj.dataAugmenter = DataAugmenter(params);
            obj.normalizer = EEGNormalizer(params);
        end

        %% CNN学習メソッド - モデルの学習と評価を実行
        function results = trainCNN(obj, processedData, processedLabel)
            % EEGデータを使用してCNNモデルを学習
            %
            % 入力:
            %   processedData - 前処理済みEEGデータ [チャンネル x サンプル x エポック]
            %   processedLabel - クラスラベル [エポック x 1]
            %
            % 出力:
            %   results - 学習結果を含む構造体（モデル、性能評価、正規化パラメータなど）
            try                
                fprintf('\n=== CNN学習処理を開始 ===\n');
                
                % データの次元を確認し、必要に応じて調整
                [processedData, ~] = obj.validateAndPrepareData(processedData);
                
                % データを学習・検証・テストセットに分割
                [trainData, trainLabels, valData, valLabels, testData, testLabels] = obj.splitDataset(processedData, processedLabel);

                % 学習データの拡張処理
                [trainData, trainLabels, normParams] = obj.preprocessTrainingData(trainData, trainLabels);
                
                % 検証・テストデータも同じ正規化パラメータで処理
                valData = obj.normalizer.normalizeOnline(valData, normParams);
                testData = obj.normalizer.normalizeOnline(testData, normParams);
                
                % CNNに適したデータ形式への変換
                prepTrainData = obj.prepareDataForCNN(trainData);
                prepValData = obj.prepareDataForCNN(valData);
                prepTestData = obj.prepareDataForCNN(testData);
                
                % モデルの学習
                [cnnModel, trainInfo] = obj.trainCNNModel(prepTrainData, trainLabels, prepValData, valLabels);
                
                % テストデータでの最終評価
                testMetrics = obj.evaluateModel(cnnModel, prepTestData, testLabels);
                
                % 過学習の分析
                [~, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);

                % 性能指標の更新
                 obj.performance = testMetrics;
                
                % 結果構造体の構築
                results = obj.buildResultsStruct(cnnModel, testMetrics, trainInfo, normParams);
                
                fprintf('\n=== CNN学習処理が完了しました ===\n');

            catch ME
                % エラー発生時の詳細情報出力
                fprintf('\n=== CNN学習中にエラーが発生しました ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラー発生場所:\n');
                for i = 1:length(ME.stack)
                    fprintf('  ファイル: %s\n  行: %d\n  関数: %s\n\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end

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
                testAcc = testMetrics.accuracy*100;
                
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
                isOverfit = gapOverfit || isCompletelyBiased || (~isLearningProgressing && ~strcmp(severity, 'none'));
                fprintf('過学習判定: %s (重大度: %s)\n', mat2str(isOverfit), severity);

                if isOverfit
                    fprintf('\n警告: モデルに過学習の兆候が検出されました (%s)\n', metrics.severity);
                end
                obj.overfitMetrics = metrics;

            catch ME
                fprintf('過学習検証でエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
            end
        end

        %% オンライン予測メソッド - 新しいデータの分類を実行
        function [label, score] = predictOnline(obj, data, cnn)
            % 学習済みモデルを使用して新しいEEGデータを分類
            %
            % 入力:
            %   data - 分類するEEGデータ [チャンネル x サンプル]
            %   cnnModel - 学習済みCNNモデル
            %
            % 出力:
            %   label - 予測クラスラベル
            %   score - 予測確率スコア
            try
                % 正規化パラメータを取得して正規化を実行
                if isfield(cnn, 'normParams') && ~isempty(cnn.normParams)
                    data = obj.normalizer.normalizeOnline(data, cnn.normParams);
                else
                    warning('CNNClassifier:NoNormParams', ...
                        '正規化パラメータが見つかりません。正規化をスキップします。');
                end

                % データの形状変換（必要に応じて）
                if ndims(data) ~= 4
                    prepData = obj.prepareDataForCNN(data);
                else
                    prepData = data;
                end
                
                % モデルの存在確認
                if isempty(cnn.model)
                    error('CNN model is not available');
                end
                
                % 予測の実行
                [label, score] = classify(cnn.model, prepData);
        
            catch ME
                fprintf('Error in CNN online prediction: %s\n', ME.message);
                fprintf('Error details:\n');
                disp(getReport(ME, 'extended'));
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
        
        %% データ分割（学習データ・検証データ・テストデータ）
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
                % testRatio = 1/(2*k);    % 0.5/k
        
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
                
                % クラスごとに分割するためのインデックス配列を事前に割り当て
                totalSamples = sum(cellfun(@length, classIndices));
                trainIdx = zeros(totalSamples, 1);
                valIdx = zeros(totalSamples, 1);
                testIdx = zeros(totalSamples, 1);
                
                % カウンタ初期化
                trainCount = 0;
                valCount = 0;
                testCount = 0;
                
                % 各クラスごとに分割
                for i = 1:numClasses
                    currentIndices = classIndices{i};
                    
                    % インデックスをランダムに並べ替え（毎回異なる結果になる）
                    randomOrder = randperm(length(currentIndices));
                    shuffledIndices = currentIndices(randomOrder);
                    
                    % 分割数の計算
                    numTrain = round(length(shuffledIndices) * trainRatio);
                    numVal = round(length(shuffledIndices) * valRatio);
                    numTest = length(shuffledIndices) - numTrain - numVal;
                    
                    % インデックスを効率的に追加
                    trainIdx(trainCount+1:trainCount+numTrain) = shuffledIndices(1:numTrain);
                    valIdx(valCount+1:valCount+numVal) = shuffledIndices(numTrain+1:numTrain+numVal);
                    testIdx(testCount+1:testCount+numTest) = shuffledIndices(numTrain+numVal+1:end);
                    
                    % カウンタを更新
                    trainCount = trainCount + numTrain;
                    valCount = valCount + numVal;
                    testCount = testCount + numTest;
                end
                
                % 実際に使用したサイズに切り詰める
                trainIdx = trainIdx(1:trainCount);
                valIdx = valIdx(1:valCount);
                testIdx = testIdx(1:testCount);
        
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

        %% CNNモデル学習メソッド
        function [cnnModel, trainInfo] = trainCNNModel(obj, trainData, trainLabels, valData, valLabels)
            % CNNモデルの構築と学習を実行
            %
            % 入力:
            %   trainData - 学習データ
            %   trainLabels - 学習ラベル
            %   valData - 検証データ
            %   valLabels - 検証ラベル
            %
            % 出力:
            %   cnnModel - 学習済みCNNモデル
            %   trainInfo - 学習に関する情報
            
            try        
                fprintf('\n=== CNNモデルの学習開始 ===\n');                
                % データの形状確認
                if ndims(trainData) ~= 4
                    trainData = obj.prepareDataForCNN(trainData);
                end
        
                % ラベルのカテゴリカル変換
                uniqueLabels = unique(trainLabels);
                trainLabels = categorical(trainLabels, uniqueLabels);
        
                % 検証データの処理
                valDS = {};
                if ~isempty(valData)
                    if ndims(valData) ~= 4
                        valData = obj.prepareDataForCNN(valData);
                    end
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
        
                % トレーニングオプションの設定
                options = trainingOptions(obj.params.classifier.cnn.training.optimizer.type, ...
                    'InitialLearnRate', obj.params.classifier.cnn.training.optimizer.learningRate, ...
                    'MaxEpochs', obj.params.classifier.cnn.training.maxEpochs, ...
                    'MiniBatchSize', obj.params.classifier.cnn.training.miniBatchSize, ...
                    'Plots', 'none', ...
                    'Shuffle', obj.params.classifier.cnn.training.shuffle, ...
                    'ExecutionEnvironment', executionEnvironment, ...
                    'OutputNetwork', 'best-validation', ...
                    'Verbose', false, ...
                    'ValidationData', valDS, ...
                    'ValidationFrequency', obj.params.classifier.cnn.training.frequency, ...
                    'ValidationPatience', obj.params.classifier.cnn.training.patience, ...
                    'GradientThreshold', obj.params.classifier.cnn.training.patience);

                layers = obj.buildCNNLayers(trainData);
                
                % モデルの学習
                [cnnModel, trainHistory] = trainNetwork(trainData, trainLabels, layers, options);
        
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
        
                fprintf('学習完了: %d エポック\n', trainInfo.FinalEpoch);
                fprintf('  - 最終学習損失: %.4f\n', trainHistory.TrainingLoss(end));
                fprintf('  - 最終検証損失: %.4f\n', trainHistory.ValidationLoss(end));
                fprintf('  - 最終学習精度: %.2f%%\n', trainHistory.TrainingAccuracy(end));
                fprintf('  - 最終検証精度: %.2f%%\n', trainHistory.ValidationAccuracy(end));
                
                % 最良の検証精度
                [bestValAcc, bestEpoch] = max(trainHistory.ValidationAccuracy);
                fprintf('  - 最良検証精度: %.2f%% (エポック %d)\n', bestValAcc, bestEpoch);
        
            catch ME
                fprintf('CNNモデル学習でエラーが発生: %s\n', ME.message);
                rethrow(ME);
            end
        end

        %% CNNレイヤー構築メソッド
        function layers = buildCNNLayers(obj, data)
            % 入力データに適したCNNアーキテクチャを構築
            %
            % 入力:
            %   data - 入力データ
            %
            % 出力:
            %   layers - CNNレイヤー構造
            
            try
                % 入力サイズの取得と調整
                layerInputSize = obj.determineInputSize(data);
                
                % アーキテクチャパラメータの取得
                arch = obj.params.classifier.cnn.architecture;
                
                % チャンネル数の取得
                numChannels = obj.params.device.channelCount;
                
                % チャンネル数に応じたアーキテクチャの選択
                if numChannels <= 10
                    layers = obj.buildCompactCNN(layerInputSize, arch);
                else
                    layers = obj.buildStandardCNN(layerInputSize, arch, numChannels);
                end
                
            catch ME
                fprintf('CNNレイヤー構築でエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end
        
        %% 入力サイズ決定メソッド
        function layerInputSize = determineInputSize(~, data)
            % データの次元に基づいて入力サイズを決定
            
            if ndims(data) == 4
                % 4次元データ [サンプル x チャンネル x 1 x エポック]
                inputSize = size(data);
                layerInputSize = [inputSize(1), inputSize(2), 1];
            elseif ndims(data) == 3
                % 3次元データ [サンプル x チャンネル x エポック]
                inputSize = size(data);
                layerInputSize = [inputSize(1), inputSize(2), 1];
            elseif ismatrix(data)
                % 2次元データ [チャンネル x サンプル]
                inputSize = size(data);
                layerInputSize = [inputSize(2), inputSize(1), 1];
            else
                error('対応していないデータ次元数: %d', ndims(data));
            end
            
            return;
        end
        
        %% コンパクトCNNアーキテクチャ構築
        function layers = buildCompactCNN(~, layerInputSize, arch)
            % チャンネル数が少ない場合の最適化されたCNN構造
            
            % 入力層
            layers = [
                imageInputLayer(layerInputSize, 'Normalization', 'none', 'Name', 'input')
                
                % 第1畳み込みブロック - 小さいフィルタサイズを使用
                convolution2dLayer([3 3], 16, 'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                % プーリングサイズを小さく設定 (2,1) - チャンネル次元は維持
                maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')
                dropoutLayer(0.3, 'Name', 'dropout1')
                
                % 第2畳み込みブロック
                convolution2dLayer([3 3], 32, 'Padding', 'same', 'Name', 'conv2')
                batchNormalizationLayer('Name', 'bn2')
                reluLayer('Name', 'relu2')
                maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')
                dropoutLayer(0.4, 'Name', 'dropout2')
                
                % グローバルプーリング
                globalAveragePooling2dLayer('Name', 'globalPool')
                
                % 全結合層
                fullyConnectedLayer(64, 'Name', 'fc1')
                reluLayer('Name', 'relu_fc1')
                dropoutLayer(0.5, 'Name', 'dropout_fc')
                
                % 出力層
                fullyConnectedLayer(arch.numClasses, 'Name', 'fc_output')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];
        end
        
        %% 標準CNNアーキテクチャ構築
        function layers = buildStandardCNN(~, layerInputSize, arch, numChannels)
            % 標準的なCNNアーキテクチャ（チャンネル数が十分な場合）  
            
            % 初期フィルタ数の決定 (チャンネル数によって調整)
            initialFilters = min(32, max(16, ceil(numChannels * 1.5)));
            
            % レイヤー数の計算
            numConvLayers = length(fieldnames(arch.convLayers));
            numFCLayers = length(arch.fullyConnected);
            totalLayers = numConvLayers * 5 + numFCLayers * 2 + 4; 

            % レイヤー配列の事前割り当て
            layers = cell(totalLayers, 1);
            layerIdx = 1;

            % 入力層
            layers{layerIdx} = imageInputLayer(layerInputSize, 'Normalization', 'none', 'Name', 'input');
            layerIdx = layerIdx + 1;

            % 畳み込み層、プーリング層、ドロップアウト層の追加
            convLayers = fieldnames(arch.convLayers);
            for i = 1:length(convLayers)
                convName = convLayers{i};
                convParams = arch.convLayers.(convName);
                
                % プーリング層パラメータの取得
                if isfield(arch.poolLayers, ['pool' num2str(i)])
                    poolParams = arch.poolLayers.(['pool' num2str(i)]);
                else
                    % デフォルト値
                    poolParams = struct('size', 2, 'stride', 2);
                end
                
                % ドロップアウト率の取得
                if isfield(arch.dropoutLayers, ['dropout' num2str(i)])
                    dropoutRate = arch.dropoutLayers.(['dropout' num2str(i)]);
                else
                    % デフォルト値
                    dropoutRate = 0.5;
                end

                % フィルタ数の計算（各層でサイズを2倍に）
                filterNum = initialFilters * (2^(i-1));
                filterSize = min(convParams.size);
                
                % パディングの決定
                paddingMode = 'same';

                layers{layerIdx} = convolution2dLayer([filterSize filterSize], filterNum, ...
                    'Stride', convParams.stride, 'Padding', paddingMode, ...
                    'Name', ['conv' num2str(i)]);
                layerIdx = layerIdx + 1;

                if arch.batchNorm
                    layers{layerIdx} = batchNormalizationLayer('Name', ['bn' num2str(i)]);
                    layerIdx = layerIdx + 1;
                end

                layers{layerIdx} = reluLayer('Name', ['relu' num2str(i)]);
                layerIdx = layerIdx + 1;
                
                % プーリングサイズのチェックと調整
                poolSize = poolParams.size;
                poolStride = poolParams.stride;
                
                % チャンネル数が少ない場合、チャンネル方向へのプーリングを制限
                if i == 1 && numChannels <= 32
                    poolSize = min(poolSize, 2);
                    layers{layerIdx} = maxPooling2dLayer([poolSize 1], ...
                        'Stride', [poolStride 1], 'Name', ['pool' num2str(i)]);
                else
                    layers{layerIdx} = maxPooling2dLayer(poolSize, ...
                        'Stride', poolStride, 'Name', ['pool' num2str(i)]);
                end
                layerIdx = layerIdx + 1;
                
                layers{layerIdx} = dropoutLayer(dropoutRate, 'Name', ['dropout' num2str(i)]);
                layerIdx = layerIdx + 1;
            end
            
            % グローバルプーリングの追加
            layers{layerIdx} = globalAveragePooling2dLayer('Name', 'globalPool');
            layerIdx = layerIdx + 1;

            % 全結合層の追加
            for i = 1:length(arch.fullyConnected)
                layers{layerIdx} = fullyConnectedLayer(arch.fullyConnected(i), ...
                    'Name', ['fc' num2str(i)]);
                layerIdx = layerIdx + 1;
                layers{layerIdx} = reluLayer('Name', ['relu_fc' num2str(i)]);
                layerIdx = layerIdx + 1;
            end

            % 出力層の追加
            layers{layerIdx} = fullyConnectedLayer(arch.numClasses, 'Name', 'fc_output');
            layerIdx = layerIdx + 1;
            layers{layerIdx} = softmaxLayer('Name', 'softmax');
            layerIdx = layerIdx + 1;
            layers{layerIdx} = classificationLayer('Name', 'output');

            % セル配列を層配列に変換
            layers = layers(~cellfun('isempty', layers));
            layers = cat(1, layers{:});
        end

        %% データのCNN形式への変換
        function preparedData = prepareDataForCNN(~, data)
            % データをCNNに適した形式に変換
            %
            % 入力:
            %   data - 入力データ
            %
            % 出力:
            %   preparedData - CNN用に整形されたデータ
            
            try
                % データの次元数に基づいて処理を分岐
                if ndims(data) == 3
                    % 3次元データ（チャンネル x サンプル x エポック）の場合
                    [channels, samples, epochs] = size(data);
                    preparedData = zeros(samples, channels, 1, epochs);
                    
                    for i = 1:epochs
                        preparedData(:,:,1,i) = data(:,:,i)';
                    end
                        
                elseif ismatrix(data)
                    % 2次元データ（チャンネル x サンプル）の場合
                    [channels, samples] = size(data);
                    preparedData = permute(data, [2, 1, 3]);
                    preparedData = reshape(preparedData, [samples, channels, 1, 1]);
                        
                else
                    % その他の次元数は対応外
                    error('対応していないデータ次元数: %d', ndims(data));
                end
                
                % NaN/Infチェック
                if any(isnan(preparedData(:)))
                    error('変換後のデータにNaN値が含まれています');
                end
                
                if any(isinf(preparedData(:)))
                    error('変換後のデータにInf値が含まれています');
                end
                
            catch ME
                fprintf('データ形式変換でエラーが発生: %s\n', ME.message);
                fprintf('入力データサイズ: [%s]\n', num2str(size(data)));
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% モデル評価メソッド
        function metrics = evaluateModel(~, model, testData, testLabels)
            % 学習済みモデルの性能を評価
            %
            % 入力:
            %   model - 学習済みCNNモデル
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
        
        %% 学習曲線分析メソッド
        function [trainTrend, valTrend] = analyzeLearningCurves(obj, trainAcc, valAcc)
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
                
                % 移動平均の計算
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
                    'plateau_detected', obj.detectPlateau(trainSmooth), ...
                    'oscillation_strength', obj.calculateOscillation(trainDiff), ...
                    'convergence_epoch', obj.estimateConvergenceEpoch(trainSmooth) ...
                );
                
                valTrend = struct(...
                    'mean_change', mean(valDiff), ...
                    'volatility', std(valDiff), ...
                    'increasing_ratio', sum(valDiff > 0) / max(length(valDiff), 1), ...
                    'plateau_detected', obj.detectPlateau(valSmooth), ...
                    'oscillation_strength', obj.calculateOscillation(valDiff), ...
                    'convergence_epoch', obj.estimateConvergenceEpoch(valSmooth) ...
                );
                
            catch ME
                fprintf('学習曲線分析でエラーが発生: %s\n', ME.message);
                
                % エラー時のフォールバック値を設定
                trainTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, ...
                    'plateau_detected', false, 'oscillation_strength', 0, 'convergence_epoch', 0);
                valTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, ...
                    'plateau_detected', false, 'oscillation_strength', 0, 'convergence_epoch', 0);
            end
        end
        
        %% プラトー検出メソッド
        function isPlateau = detectPlateau(~, smoothedCurve)
            % 学習曲線でのプラトー（停滞）状態を検出
            %
            % 入力:
            %   smoothedCurve - 平滑化された学習曲線
            %
            % 出力:
            %   isPlateau - プラトー検出フラグ
            
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

        %% 振動強度計算メソッド
        function oscillation = calculateOscillation(~, diffValues)
            % 学習曲線の振動強度を計算
            %
            % 入力:
            %   diffValues - 変化率データ
            %
            % 出力:
            %   oscillation - 振動強度 (0-1)
            
            % 最小必要長のチェック
            if length(diffValues) < 2
                oscillation = 0;
                return;
            end
            
            % 符号変化の数をカウント（振動の指標）
            signChanges = sum(diff(sign(diffValues)) ~= 0);
            oscillation = signChanges / (length(diffValues) - 1);  % 正規化
        end

        %% 収束エポック推定メソッド
        function convergenceEpoch = estimateConvergenceEpoch(~, smoothedCurve)
            % 学習が収束したエポックを推定
            %
            % 入力:
            %   smoothedCurve - 平滑化された学習曲線
            %
            % 出力:
            %   convergenceEpoch - 推定収束エポック
            
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
            scaleFactor = min(1, sqrt(dataSize / 1000));  % データサイズに基づく調整
            
            % 正規化された差（z-スコア的アプローチ）
            normalizedGap = abs(meanValAcc - testAcc) / max(stdValAcc, 1);
            
            % スケール調整されたギャップ
            adjustedGap = normalizedGap * scaleFactor;
            
            % 絶対的な差の計算
            rawGap = abs(meanValAcc - testAcc);
            
            % 絶対差に基づく過学習判定
            % rawGapが0.15 (15%)より大きい場合は過学習と判断
            absoluteOverfit = (rawGap > 15.0);
            
            % 統計的アプローチに基づく過学習判定
            statisticalOverfit = false;
            statSeverity = 'none';
            
            % 調整済みギャップに基づく重大度判定（統計的アプローチ）
            if adjustedGap > 3
                statSeverity = 'critical';     % 3標準偏差以上
                statisticalOverfit = true;
            elseif adjustedGap > 2
                statSeverity = 'severe';       % 2標準偏差以上
                statisticalOverfit = true;
            elseif adjustedGap > 1.5
                statSeverity = 'moderate';     % 1.5標準偏差以上
                statisticalOverfit = true;
            elseif adjustedGap > 1
                statSeverity = 'mild';         % 1標準偏差以上
                statisticalOverfit = true;
            end
            
            % 絶対差に基づく重大度判定
            absSeverity = 'none';
            if rawGap > 20.0
                absSeverity = 'critical';      % 20%以上の差は非常に深刻
            elseif rawGap > 15.0
                absSeverity = 'severe';        % 15%以上の差は深刻
            elseif rawGap > 10.0
                absSeverity = 'moderate';      % 10%以上の差は中程度
            elseif rawGap > 5.0
                absSeverity = 'mild';          % 5%以上の差は軽度
            end
            
            % 最終的な重大度判定（より厳しい方を採用）
            severityOrder = {'none', 'mild', 'moderate', 'severe', 'critical'};
            statIdx = find(strcmp(severityOrder, statSeverity));
            absIdx = find(strcmp(severityOrder, absSeverity));
            
            if isempty(statIdx)
                statIdx = 1; % デフォルトは'none'
            end
            if isempty(absIdx)
                absIdx = 1; % デフォルトは'none'
            end
            
            if absIdx >= statIdx
                severity = absSeverity;
            else
                severity = statSeverity;
            end
            
            % 最終的な過学習判定（どちらかが過学習と判断すれば過学習）
            isOverfit = statisticalOverfit || absoluteOverfit;
            
            % 結果の格納
            metrics = struct(...
                'rawGap', rawGap, ...
                'normalizedGap', normalizedGap, ...
                'adjustedGap', adjustedGap, ...
                'meanValAcc', meanValAcc, ...
                'stdValAcc', stdValAcc, ...
                'testAcc', testAcc, ...
                'severity', severity, ...
                'statisticalOverfit', statisticalOverfit, ...
                'absoluteOverfit', absoluteOverfit, ...
                'statSeverity', statSeverity, ...
                'absSeverity', absSeverity ...
            );
            
            % 結果の表示
            fprintf('\n=== 検証-テスト精度差分析 ===\n');
            fprintf('  検証精度: %.2f%% (±%.2f%%)\n', meanValAcc, stdValAcc);
            fprintf('  テスト精度: %.2f%%\n', testAcc);
            fprintf('  検証-テスト精度差: %.2f%%\n', rawGap);
            fprintf('  統計的判定: %s (重大度: %s)\n', mat2str(statisticalOverfit), statSeverity);
            fprintf('  絶対差判定: %s (重大度: %s)\n', mat2str(absoluteOverfit), absSeverity);
            fprintf('  最終判定結果: %s (重大度: %s)\n', mat2str(isOverfit), severity);
        end
        
        %% 過学習重大度判定メソッド
        function severity = determineOverfittingSeverity(~, perfGap, isCompletelyBiased, ... 
            isLearningProgressing, params)
            % 過学習の重大度を判定
            %
            % 入力:
            %   perfGap - 性能ギャップ（検証とテストの差）
            %   isCompletelyBiased - 完全なバイアスの有無
            %   isLearningProgressing - 学習の進行状態
            %   params - 判定パラメータ
            %
            % 出力:
            %   severity - 過学習の重大度 ('none', 'mild', 'moderate', 'severe', 'critical', 'failed')
            
            if isCompletelyBiased
                severity = 'critical';  % 特定のクラスに完全に偏っている
            elseif ~isLearningProgressing
                severity = 'failed';    % 学習が進行していない
            elseif perfGap > params.critical_gap
                severity = 'critical';  % 非常に大きな性能ギャップ
            elseif perfGap > params.severe_gap
                severity = 'severe';    % 大きな性能ギャップ
            elseif perfGap > params.moderate_gap
                severity = 'moderate';  % 中程度の性能ギャップ
            elseif perfGap > params.moderate_gap / 2
                severity = 'mild';      % 軽度の性能ギャップ
            else
                severity = 'none';      % 過学習は検出されない
            end
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
        
        %% 過学習メトリクス構築メソッド
        function metrics = buildOverfitMetrics(~, perfGap, isCompletelyBiased, isLearningProgressing, ...
            trainTrend, valTrend, severity, optimalEpoch, totalEpochs, earlyStoppingEffect)
            % 過学習分析の詳細メトリクスを構築
            
            metrics = struct(...
                'performanceGap', perfGap, ...
                'isCompletelyBiased', isCompletelyBiased, ...
                'isLearningProgressing', isLearningProgressing, ...
                'validationTrend', valTrend, ...
                'trainingTrend', trainTrend, ...
                'severity', severity, ...
                'optimalEpoch', optimalEpoch, ...
                'totalEpochs', totalEpochs, ...
                'earlyStoppingEffect', earlyStoppingEffect);
        end
        
        %% トレーニング情報検証メソッド
        function validateTrainInfo(~, trainInfo)
            % trainInfo構造体の妥当性を検証
            
            if ~isstruct(trainInfo)
                warning('trainInfoが構造体ではありません');
                return;
            end
            
            % 必須フィールドの確認
            requiredFields = {'History', 'FinalEpoch'};
            for i = 1:length(requiredFields)
                if ~isfield(trainInfo, requiredFields{i})
                    warning('trainInfoに必須フィールド「%s」がありません', requiredFields{i});
                end
            end
            
            % History構造体の検証
            if isfield(trainInfo, 'History')
                historyFields = {'TrainingLoss', 'ValidationLoss', 'TrainingAccuracy', 'ValidationAccuracy'};
                for i = 1:length(historyFields)
                    if ~isfield(trainInfo.History, historyFields{i})
                        warning('History構造体に「%s」フィールドがありません', historyFields{i});
                    elseif isempty(trainInfo.History.(historyFields{i}))
                        warning('History構造体の「%s」フィールドが空です', historyFields{i});
                    end
                end
            end
            
            % FinalEpochの妥当性検証
            if isfield(trainInfo, 'FinalEpoch')
                if trainInfo.FinalEpoch <= 0
                    warning('FinalEpochが0以下です: %d', trainInfo.FinalEpoch);
                end
            end
        end

        %%
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
                    copy.(field) = obj.createDeepCopy(original.(field));  % objを使用
                elseif iscell(original.(field))
                    % セル配列の場合
                    cellArray = original.(field);
                    newCellArray = cell(size(cellArray));
                    for j = 1:numel(cellArray)
                        if isstruct(cellArray{j})
                            newCellArray{j} = obj.createDeepCopy(cellArray{j});  % objを使用
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

        %%
        function extractedParams = extractParams(obj)
            % CNNのパラメータを抽出して7要素の配列として返す
            
            % デフォルト値で初期化
            extractedParams = [0, 0, 0, 0, 0, 0, 0];
            
            try
                % 1. 学習率
                if isfield(obj.params.classifier.cnn.training.optimizer, 'learningRate')
                    extractedParams(1) = obj.params.classifier.cnn.training.optimizer.learningRate;
                end
                
                % 2. バッチサイズ
                if isfield(obj.params.classifier.cnn.training, 'miniBatchSize')
                    extractedParams(2) = obj.params.classifier.cnn.training.miniBatchSize;
                end
                
                % 3. 畳み込み層数
                if isfield(obj.params.classifier.cnn.architecture, 'convLayers')
                    extractedParams(3) = length(fieldnames(obj.params.classifier.cnn.architecture.convLayers));
                end
                
                % 4. フィルタサイズと5. フィルタ数
                if isfield(obj.params.classifier.cnn.architecture, 'convLayers')
                    convFields = fieldnames(obj.params.classifier.cnn.architecture.convLayers);
                    if ~isempty(convFields)
                        firstConv = obj.params.classifier.cnn.architecture.convLayers.(convFields{1});
                        if isfield(firstConv, 'size')
                            if length(firstConv.size) >= 1
                                extractedParams(4) = firstConv.size(1);
                            end
                        end
                        if isfield(firstConv, 'filters')
                            extractedParams(5) = firstConv.filters;
                        end
                    end
                end
                
                % 6. ドロップアウト率
                if isfield(obj.params.classifier.cnn.architecture, 'dropoutLayers')
                    dropout = obj.params.classifier.cnn.architecture.dropoutLayers;
                    dropoutFields = fieldnames(dropout);
                    if ~isempty(dropoutFields)
                        extractedParams(6) = dropout.(dropoutFields{1});
                    end
                end
                
                % 7. 全結合層ユニット数
                if isfield(obj.params.classifier.cnn.architecture, 'fullyConnected')
                    fc = obj.params.classifier.cnn.architecture.fullyConnected;
                    if ~isempty(fc)
                        extractedParams(7) = fc(1);
                    end
                end
            catch ME
                warning('パラメータ抽出中にエラーが発生');
            end
        end

        %%
        function validateResults(obj, results)
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
                if length(results.params) ~= 7
                    warning('paramsフィールドの長さが期待値と異なります: %d (期待値: 7)', length(results.params));
                end
            end
            
            % trainInfoフィールドのチェック
            if isfield(results, 'trainInfo')
                obj.validateTrainInfo(results.trainInfo);
            end
            
            % パフォーマンスフィールドのチェック
            if isfield(results, 'performance')
                if ~isfield(results.performance, 'accuracy')
                    warning('performance構造体にaccuracyフィールドがありません');
                end
            end
        end

        %% 結果構造体構築メソッド
        function results = buildResultsStruct(obj, cnnModel, metrics, trainInfo, normParams)
            % 結果構造体の構築
            
            % trainInfoの検証と安全なディープコピー
            obj.validateTrainInfo(trainInfo);
            trainInfoCopy = obj.createDeepCopy(trainInfo);
            
            % パラメータ情報の抽出
            extractedParams = obj.extractParams();
            
            % 結果構造体の構築
            results = struct(...
                'model', cnnModel, ...
                'performance', metrics, ...
                'trainInfo', trainInfoCopy, ...
                'overfitting', obj.overfitMetrics, ...
                'normParams', normParams, ...
                'params', extractedParams ...
            );
            
            % 出力前に結果の検証
            obj.validateResults(results);
        end
    end
end