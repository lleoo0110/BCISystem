classdef HybridClassifier < handle
    %% HybridClassifier - CNN+LSTM統合ハイブリッド分類器
    %
    % このクラスはEEGデータに対してCNNとLSTMを組み合わせたハイブリッド深層学習モデルを
    % 実装します。CNNで空間的特徴を、LSTMで時間的特徴を抽出し、これらを統合して
    % 高精度な分類を実現します。
    %
    % 主な機能:
    %   - EEGデータの前処理と次元変換
    %   - CNN・LSTMの並列学習と特徴抽出
    %   - 特徴融合とAdaBoost統合分類
    %   - 学習済みモデルによるオンライン予測
    %   - 過学習検出と詳細な性能評価
    %   - リソース管理（GPU）の最適化
    %   - verbosityレベルによる出力制御
    %
    % 使用例:
    %   params = getConfig('epocx', 'preset', 'magic');
    %   hybrid = HybridClassifier(params);  % デフォルトverbosity=1
    %   hybrid = HybridClassifier(params, 2);  % 詳細出力モード
    %   results = hybrid.trainModel(processedData, processedLabel);
    %   [label, score] = hybrid.predictOnline(newData, results.model);
    
    properties (Access = private)
        params              % システム設定パラメータ
        netCNN              % 学習済みCNNネットワーク
        netLSTM             % 学習済みLSTMネットワーク
        isInitialized       % 初期化完了フラグ
        useGPU              % GPU使用フラグ
        verbosity           % 出力詳細度 (0:最小限, 1:通常, 2:詳細, 3:デバッグ)
        
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
        function obj = HybridClassifier(params, verbosity)
            % HybridClassifierのインスタンスを初期化
            %
            % 入力:
            %   params - 設定パラメータ（getConfig関数から取得）
            %   verbosity - 出力詳細度 (0:最小限, 1:通常, 2:詳細, 3:デバッグ)
            
            % 基本パラメータの設定
            obj.params = params;
            obj.isInitialized = false;
            obj.useGPU = params.classifier.hybrid.gpu;
            
            % verbosityレベルの設定（デフォルトは1）
            if nargin < 2
                obj.verbosity = 1;
            else
                obj.verbosity = verbosity;
            end
            
            % プロパティの初期化
            obj.initializeProperties();
            
            % コンポーネントの初期化
            obj.dataAugmenter = DataAugmenter(params);
            obj.normalizer = EEGNormalizer(params);
            
            obj.logMessage(1, 'HybridClassifier初期化完了: GPU=%s\n', mat2str(obj.useGPU));
        end

        %% ハイブリッドモデル学習メソッド - モデルの学習と評価を実行
        function results = trainModel(obj, processedData, processedLabel)
            % EEGデータを使用してハイブリッドモデルを学習
            %
            % 入力:
            %   processedData - 前処理済みEEGデータ [チャンネル x サンプル x エポック]
            %   processedLabel - クラスラベル [エポック x 1]
            %
            % 出力:
            %   results - 学習結果を含む構造体（モデル、性能評価、正規化パラメータなど）
            
            try                
                obj.logMessage(1, '\n=== ハイブリッドモデル学習処理を開始 ===\n');
                
                % データの次元を確認し、必要に応じて調整
                [processedData, ~] = obj.validateAndPrepareData(processedData);
                
                % データを学習・検証・テストセットに分割
                [trainData, trainLabels, valData, valLabels, testData, testLabels] = obj.splitDataset(processedData, processedLabel);

                % 学習データの拡張と正規化処理
                [trainData, trainLabels, normParams] = obj.preprocessTrainingData(trainData, trainLabels);
                
                % 検証・テストデータも同じ正規化パラメータで処理
                valData = obj.normalizer.normalizeOnline(valData, normParams);
                testData = obj.normalizer.normalizeOnline(testData, normParams);
                
                % CNN用にデータ形式を変換
                prepCnnTrainData = obj.prepareDataForCNN(trainData);
                prepCnnValData = obj.prepareDataForCNN(valData);
                prepCnnTestData = obj.prepareDataForCNN(testData);
                
                % LSTM用にデータ形式を変換
                prepLstmTrainData = obj.prepareDataForLSTM(trainData);
                prepLstmValData = obj.prepareDataForLSTM(valData);
                prepLstmTestData = obj.prepareDataForLSTM(testData);
                
                % GPU使用メモリの確認
                if obj.useGPU
                    obj.checkGPUMemory();
                end
                
                % モデルの学習
                [hybridModel, trainInfo] = obj.trainHybridModel( ...
                    prepCnnTrainData, prepLstmTrainData, trainLabels, ...
                    prepCnnValData, prepLstmValData, valLabels);
                
                % テストデータでの最終評価
                testMetrics = obj.evaluateModel(hybridModel, prepCnnTestData, prepLstmTestData, testLabels);
                
                % 過学習の分析
                [~, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);

                % 性能指標の更新
                obj.performance = testMetrics;
                
                % 結果構造体の構築
                results = obj.buildResultsStruct(hybridModel, testMetrics, trainInfo, normParams);
                
                % 使用リソースのクリーンアップ
                obj.resetGPUMemory();
                
                obj.logMessage(1, '\n=== ハイブリッドモデル学習処理が完了しました ===\n');

            catch ME
                % エラー発生時の詳細情報出力
                obj.logMessage(0, '\n=== ハイブリッドモデル学習中にエラーが発生しました ===\n');
                obj.logMessage(0, 'エラーメッセージ: %s\n', ME.message);
                obj.logMessage(0, 'エラー発生場所:\n');
                for i = 1:length(ME.stack)
                    obj.logMessage(0, '  ファイル: %s\n  行: %d\n  関数: %s\n\n', ...
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
            
            obj.logMessage(1, '\n=== 過学習検証の実行 ===\n');
            
            try
                % trainInfoの構造を検証
                obj.validateTrainInfo(trainInfo);
                
                % CNN学習曲線の分析
                [cnnTrainTrend, cnnValTrend, cnnOptimalEpoch, cnnTotalEpochs] = obj.analyzeCNNTraining(trainInfo);
                
                % LSTM学習曲線の分析
                [lstmTrainTrend, lstmValTrend, lstmOptimalEpoch, lstmTotalEpochs] = obj.analyzeLSTMTraining(trainInfo);
                
                % ハイブリッドモデル全体の検証-テスト精度ギャップ分析
                [gapOverfit, gapMetrics] = obj.analyzeValidationTestGap(trainInfo, testMetrics);
                
                % バイアスの検出（特定クラスへの偏り）
                isCompletelyBiased = obj.detectClassificationBias(testMetrics);
                
                % 学習進行の評価
                isLearningProgressing = (cnnTrainTrend.mean_change > 0.001) || (lstmTrainTrend.mean_change > 0.001) || ...
                                       (cnnValTrend.mean_change > 0.001) || (lstmValTrend.mean_change > 0.001);
                
                % Early Stoppingの効果分析
                cnnEarlyStoppingEffect = struct('optimal_epoch', cnnOptimalEpoch, 'total_epochs', cnnTotalEpochs);
                lstmEarlyStoppingEffect = struct('optimal_epoch', lstmOptimalEpoch, 'total_epochs', lstmTotalEpochs);
                
                % 複合判定 - バイアス検出を優先
                if isCompletelyBiased
                    severity = 'critical';  % 完全な偏りがある場合は最重度の過学習と判定
                    obj.logMessage(1, '完全な分類バイアスが検出されたため、過学習を「%s」と判定\n', severity);
                else
                    % 通常のギャップベース判定を使用
                    severity = gapMetrics.severity;
                end
                
                % 過学習判定 - 複数条件の組み合わせ
                isOverfit = gapOverfit || isCompletelyBiased || (~isLearningProgressing && ~strcmp(severity, 'none'));
                
                % メトリクスの構築
                metrics = struct(...
                    'gapMetrics', gapMetrics, ...
                    'performanceGap', gapMetrics.rawGap, ...
                    'isCompletelyBiased', isCompletelyBiased, ...
                    'isLearningProgressing', isLearningProgressing, ...
                    'cnnValidationTrend', cnnValTrend, ...
                    'cnnTrainingTrend', cnnTrainTrend, ...
                    'lstmValidationTrend', lstmValTrend, ...
                    'lstmTrainingTrend', lstmTrainTrend, ...
                    'severity', severity, ...
                    'cnnOptimalEpoch', cnnOptimalEpoch, ...
                    'cnnTotalEpochs', cnnTotalEpochs, ...
                    'lstmOptimalEpoch', lstmOptimalEpoch, ...
                    'lstmTotalEpochs', lstmTotalEpochs, ...
                    'cnnEarlyStoppingEffect', cnnEarlyStoppingEffect, ...
                    'lstmEarlyStoppingEffect', lstmEarlyStoppingEffect, ...
                    'isOverfit', isOverfit ...
                );
                
                obj.logMessage(1, '過学習判定: %s (重大度: %s)\n', mat2str(isOverfit), severity);
                
                if isOverfit
                    obj.logMessage(1, '\n警告: モデルに過学習の兆候が検出されました (%s)\n', metrics.severity);
                end
                
            catch ME
                obj.logMessage(1, '過学習検証でエラーが発生: %s\n', ME.message);
                obj.logMessage(2, 'エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                
                % エラー時のフォールバック値を設定
                metrics = obj.createFallbackOverfitMetrics();
                isOverfit = true;
                metrics.isOverfit = isOverfit;
            end
        end

        %% オンライン予測メソッド - 新しいデータの分類を実行
        function [label, score] = predictOnline(obj, data, hybrid)
            % 学習済みモデルを使用して新しいEEGデータを分類
            %
            % 入力:
            %   data - 分類するEEGデータ [チャンネル x サンプル]
            %   hybrid - 学習済みハイブリッドモデル構造体
            %
            % 出力:
            %   label - 予測クラスラベル
            %   score - 予測確率スコア
            
            try
                % 正規化パラメータを使用してデータを正規化
                if isfield(hybrid, 'normParams') && ~isempty(hybrid.normParams)
                    data = obj.normalizer.normalizeOnline(data, hybrid.normParams);
                else
                    warning('HybridClassifier:NoNormParams', ...
                        '正規化パラメータが見つかりません。正規化をスキップします。');
                end
        
                % CNN用とLSTM用にデータを変換
                cnnData = obj.prepareDataForCNN(data);
                lstmData = obj.prepareDataForLSTM(data);
                
                % 各モデルから特徴抽出
                cnnFeatures = activations(hybrid.model.netCNN, cnnData, 'fc_cnn', 'OutputAs', 'rows');
                
                % LSTMからの特徴抽出
                if iscell(lstmData)
                    % 事前に配列サイズを決定して動的サイズ変更を回避
                    numSamples = length(lstmData);
                    % 最初のサンプルで特徴量次元を確認
                    tempOut = predict(hybrid.model.netLSTM, lstmData{1}, 'MiniBatchSize', 1);
                    featureSize = size(tempOut, 2);
                    
                    % 事前に配列を割り当て
                    lstmFeatures = zeros(numSamples, featureSize);
                    lstmFeatures(1,:) = tempOut;
                    
                    % 残りのサンプルを処理
                    for i = 2:numSamples
                        lstmOut = predict(hybrid.model.netLSTM, lstmData{i}, 'MiniBatchSize', 1);
                        lstmFeatures(i,:) = lstmOut;
                    end
                else
                    lstmOut = predict(hybrid.model.netLSTM, lstmData, 'MiniBatchSize', 1);
                    lstmFeatures = lstmOut;
                end
                
                % 特徴を結合
                combinedFeatures = [cnnFeatures, lstmFeatures];
                
                % AdaBoostで最終予測
                [label, score] = predict(hybrid.model.adaModel, combinedFeatures);

                % スコアを確率に変換
                score = obj.convertScoreToProbability(score);
        
            catch ME
                obj.logMessage(0, 'Error in Hybrid online prediction: %s\n', ME.message);
                obj.logMessage(2, 'Error details:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end
        
        %% ログ出力メソッド
        function logMessage(obj, level, format, varargin)
            % 指定されたverbosityレベル以上の場合にメッセージを出力
            %
            % 入力:
            %   level - メッセージの重要度 (0:エラー, 1:警告/通常, 2:情報, 3:デバッグ)
            %   format - fprintf形式の文字列
            %   varargin - 追加パラメータ
            
            if obj.verbosity >= level
                fprintf(format, varargin{:});
            end
        end
    end
    
    methods (Access = private)
        %% プロパティ初期化メソッド
        function initializeProperties(obj)
            % クラスプロパティの初期化
            % すべての内部プロパティを適切な初期値で設定する
            
            obj.trainingHistory = struct('loss', [], 'accuracy', []);
            obj.validationHistory = struct('loss', [], 'accuracy', []);
            obj.bestValAccuracy = 0;
            obj.patienceCounter = 0;
            obj.currentEpoch = 0;
            obj.overfitMetrics = struct();
            obj.gpuMemory = struct('total', 0, 'used', 0, 'peak', 0);
        end
        
        %% データ検証と準備メソッド
        function [validatedData, infoMsg] = validateAndPrepareData(obj, data)
            % 入力データの検証と適切な形式への変換
            %
            % 入力:
            %   data - 入力EEGデータ
            % 
            % 出力:
            %   validatedData - 検証・調整済みデータ
            %   infoMsg - データ変換に関する情報メッセージ
            
            % データの次元と形状を確認
            if isempty(data)
                error('データが空です。サイズを決定できません。');
            end
            
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
                obj.logMessage(2, '%s\n', infoMsg);
            else
                [channels, samples, epochs] = size(data);
                infoMsg = sprintf('3次元データを検証 [%d×%d×%d]', channels, samples, epochs);
                obj.logMessage(2, '%s\n', infoMsg);
            end
            
            % データの妥当性検証 - NaN/Infinityチェック
            validateattributes(validatedData, {'numeric'}, {'finite', 'nonnan'}, ...
                'validateAndPrepareData', 'data');
            
            return;
        end
        
        %% 学習データの前処理メソッド
        function [procTrainData, procTrainLabels, normParams] = preprocessTrainingData(obj, trainData, trainLabels)
            % 学習データの拡張と正規化を実行
            %
            % 入力:
            %   trainData - 学習データ
            %   trainLabels - 学習ラベル
            %
            % 出力:
            %   procTrainData - 処理後の学習データ
            %   procTrainLabels - 処理後のラベル
            %   normParams - 正規化パラメータ（テスト時に再利用）
            
            procTrainData = trainData;
            procTrainLabels = trainLabels;
            normParams = [];
            
            % データ拡張処理
            if obj.params.classifier.augmentation.enable
                obj.logMessage(1, '\nデータ拡張を実行...\n');
                [procTrainData, procTrainLabels, ~] = obj.dataAugmenter.augmentData(trainData, trainLabels);
                obj.logMessage(1, '  - 拡張前: %d サンプル\n', length(trainLabels));
                obj.logMessage(1, '  - 拡張後: %d サンプル (%.1f倍)\n', length(procTrainLabels), ... 
                    length(procTrainLabels)/length(trainLabels));
            end
            
            % 正規化処理
            if obj.params.classifier.normalize.enable
                obj.logMessage(1, '\nデータ正規化を実行...\n');
                [procTrainData, normParams] = obj.normalizer.normalize(procTrainData);
                
                % 正規化パラメータの検証
                obj.validateNormalizationParams(normParams);
                obj.logMessage(1, '  - 正規化方法: %s\n', normParams.method);
            end
            
            return;
        end
        
        %% 正規化パラメータの検証メソッド
        function validateNormalizationParams(obj, params)
            % 正規化パラメータの妥当性を検証
            %
            % 入力:
            %   params - 正規化パラメータ構造体
            
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
                        obj.logMessage(1, '警告: 標準偏差が0のチャンネルがあります。正規化で問題が発生する可能性があります\n');
                    end
                    
                case 'minmax'
                    % MinMax正規化パラメータの検証
                    if ~isfield(params, 'min') || ~isfield(params, 'max')
                        error('MinMax正規化に必要なパラメータ(min, max)が不足しています');
                    end
                    
                    if any(params.max - params.min < eps)
                        obj.logMessage(1, '警告: 最大値と最小値がほぼ同じチャンネルがあります。正規化で問題が発生する可能性があります\n');
                    end
                    
                case 'robust'
                    % Robust正規化パラメータの検証
                    if ~isfield(params, 'median') || ~isfield(params, 'mad')
                        error('Robust正規化に必要なパラメータ(median, mad)が不足しています');
                    end
                    
                    if any(params.mad < eps)
                        obj.logMessage(1, '警告: MADが極めて小さいチャンネルがあります。正規化で問題が発生する可能性があります\n');
                    end
                    
                otherwise
                    obj.logMessage(1, '警告: 未知の正規化方法: %s\n', params.method);
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
                % 分割数の取得（CNNの設定から取得）
                k = obj.params.classifier.cnn.training.validation.kfold;
                
                % kの値をチェック
                if k <= 0
                    error('k値は正の値である必要があります。');
                end

                % データサイズの取得
                [~, ~, numEpochs] = size(data);
                obj.logMessage(1, '\nデータセット分割:\n');
                obj.logMessage(1, '  - 総エポック数: %d\n', numEpochs);

                % 分割比率の計算（k-fold交差検証の考え方に基づく）
                trainRatio = (k-1)/k;  % (k-1)/k
                valRatio = 1/(2*k);    % 0.5/k
                % testRatio = 1/(2*k);   % 0.5/k
        
                % クラスバランスを維持するため、層別化サンプリングを使用
                uniqueLabels = unique(labels);
                numClasses = length(uniqueLabels);
                
                obj.logMessage(1, '  - クラス数: %d\n', numClasses);
                
                % クラスごとのインデックスを取得
                classIndices = cell(numClasses, 1);
                for i = 1:numClasses
                    classIndices{i} = find(labels == uniqueLabels(i));
                    obj.logMessage(1, '  - クラス %d: %d サンプル\n', uniqueLabels(i), length(classIndices{i}));
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
                obj.logMessage(1, '  - 学習データ: %d サンプル (%.1f%%)\n', ...
                    length(trainIdx), (length(trainIdx)/numEpochs)*100);
                obj.logMessage(1, '  - 検証データ: %d サンプル (%.1f%%)\n', ...
                    length(valIdx), (length(valIdx)/numEpochs)*100);
                obj.logMessage(1, '  - テストデータ: %d サンプル (%.1f%%)\n', ...
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
        function checkClassDistribution(obj, setName, labels)
            % データセット内のクラス分布を解析して表示
            %
            % 入力:
            %   setName - データセット名（表示用）
            %   labels - クラスラベル
            
            uniqueLabels = unique(labels);
            obj.logMessage(1, '\n%sデータのクラス分布:\n', setName);
            
            for i = 1:length(uniqueLabels)
                count = sum(labels == uniqueLabels(i));
                obj.logMessage(1, '  - クラス %d: %d サンプル (%.1f%%)\n', ...
                    uniqueLabels(i), count, (count/length(labels))*100);
            end
            
            % クラス不均衡の評価
            maxCount = max(histcounts(labels));
            minCount = min(histcounts(labels));
            imbalanceRatio = maxCount / max(minCount, 1);
            
            if imbalanceRatio > 3
                obj.logMessage(1, '警告: %sデータセットのクラス不均衡が大きいです (比率: %.1f:1)\n', ...
                    setName, imbalanceRatio);
            end
        end

        %% データのCNN形式への変換メソッド
        function preparedData = prepareDataForCNN(obj, data)
            % データをCNNに適した形式に変換
            % CNN用データ形式: [サンプル数 x チャンネル数 x 1 x エポック数]
            %
            % 入力:
            %   data - 入力データ
            %
            % 出力:
            %   preparedData - CNN用に整形されたデータ
            
            try
                % データ次元数の確認
                dimCount = ndims(data);
                
                if dimCount == 3
                    % 3次元データ（チャンネル x サンプル x エポック）の場合
                    [channels, samples, epochs] = size(data);
                    
                    % パフォーマンス向上のため、事前にメモリ割り当て
                    preparedData = zeros(samples, channels, 1, epochs, 'like', data);
                    
                    for i = 1:epochs
                        % [サンプル x チャンネル x 1] の形式に変換（転置して次元を入れ替え）
                        preparedData(:,:,1,i) = data(:,:,i)';
                    end
                    
                    obj.logMessage(3, 'データ形式変換: 3次元→4次元 [%d×%d×%d] → [%d×%d×1×%d]\n', ...
                        channels, samples, epochs, samples, channels, epochs);
                        
                elseif ismatrix(data)
                    % 2次元データ（チャンネル x サンプル）の場合
                    [channels, samples] = size(data);
                    
                    % 転置して次元を入れ替え、4次元データに変換
                    % [サンプル x チャンネル x 1 x 1] 形式
                    preparedData = permute(data, [2, 1, 3]);
                    preparedData = reshape(preparedData, [samples, channels, 1, 1]);
                    
                    obj.logMessage(3, 'データ形式変換: 2次元→4次元 [%d×%d] → [%d×%d×1×1]\n', ...
                        channels, samples, samples, channels);
                        
                elseif dimCount == 4
                    % すでに4次元の場合はそのまま使用
                    preparedData = data;
                    obj.logMessage(3, 'データ形式: すでに4次元形式のため変換なし [%d×%d×%d×%d]\n', ...
                        size(data,1), size(data,2), size(data,3), size(data,4));
                        
                else
                    % その他の次元数は対応外
                    error('対応していないデータ次元数: %d', dimCount);
                end
                
                % NaN/Infチェック
                if any(isnan(preparedData(:)))
                    error('変換後のデータにNaN値が含まれています');
                end
                
                if any(isinf(preparedData(:)))
                    error('変換後のデータにInf値が含まれています');
                end
                
            catch ME
                obj.logMessage(0, 'CNN用データ形式変換でエラーが発生: %s\n', ME.message);
                obj.logMessage(0, '入力データサイズ: [%s]\n', num2str(size(data)));
                obj.logMessage(2, 'エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% データのLSTM形式への変換メソッド
        function preparedData = prepareDataForLSTM(obj, data)
            % データをLSTMに適した形式に変換
            % LSTM用データ形式: セル配列 {[特徴量 × 時間ステップ], ...}
            %
            % 入力:
            %   data - 入力データ
            %
            % 出力:
            %   preparedData - LSTM用に整形されたデータ
            
            try
                if iscell(data)
                    % 入力が既にセル配列の場合
                    trials = numel(data);
                    preparedData = cell(trials, 1);
                    
                    for i = 1:length(data)
                        currentData = data{i};
                        if ~isa(currentData, 'double')
                            currentData = double(currentData);
                        end
                        
                        % MATLABのLSTMでは [特徴量 × 時間ステップ] の形式が必要
                        [dim1, dim2] = size(currentData);
                        % 時間ステップの方が特徴量より多いことが一般的
                        if dim1 > dim2
                            % 転置して [特徴量 × 時間ステップ] の形式に変換
                            currentData = currentData';
                        end
                        
                        % NaN/Inf値の検出と補間処理
                        currentData = obj.interpolateInvalidValues(currentData, i);
                        preparedData{i} = currentData;
                    end
                    
                    obj.logMessage(3, 'LSTM用データ形式変換: セル配列から処理（%d試行）\n', trials);
                    
                elseif ndims(data) == 3
                    % 3次元数値配列の処理 [チャンネル × 時間ポイント × 試行]
                    [channels, timepoints, trials] = size(data);
                    preparedData = cell(trials, 1);
                    
                    for i = 1:trials
                        % 各試行のデータを取得
                        currentData = data(:, :, i);
                        if ~isa(currentData, 'double')
                            currentData = double(currentData);
                        end
                        
                        % NaN/Inf値の検出と補間処理
                        currentData = obj.interpolateInvalidValues(currentData, i);
                        preparedData{i} = currentData;
                    end
                    
                    obj.logMessage(3, 'LSTM用データ形式変換: 3次元→セル配列 [%d×%d×%d] → {%d要素}\n', ...
                        channels, timepoints, trials, trials);
                    
                elseif ismatrix(data)
                    % 2次元データ（単一試行）の処理
                    [dim1, dim2] = size(data);
                    
                    currentData = data;
                    if ~isa(currentData, 'double')
                        currentData = double(currentData);
                    end
                    
                    % チャンネル数が時点数より少ないことが一般的
                    if dim1 > dim2
                        % 転置して [特徴量 × 時間ステップ] の形式に変換
                        currentData = currentData';
                    end
                    
                    % NaN/Inf値の検出と補間処理
                    currentData = obj.interpolateInvalidValues(currentData, 1);
                    preparedData = {currentData};
                    
                    obj.logMessage(3, 'LSTM用データ形式変換: 2次元→セル配列 [%d×%d] → {1要素}\n', dim1, dim2);
                    
                else
                    error('対応していないデータ次元数: %d (最大3次元まで対応)', ndims(data));
                end
                
                % 結果検証
                if isempty(preparedData)
                    error('LSTM用データ変換後のデータが空です');
                end
                
            catch ME
                obj.logMessage(0, 'LSTM用データ準備でエラーが発生: %s\n', ME.message);
                obj.logMessage(0, '入力データサイズ: [%s]\n', num2str(size(data)));
                obj.logMessage(2, 'エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end
        
        %% 無効値の補間処理メソッド
        function processedData = interpolateInvalidValues(obj, data, trialIndex)
            % NaN/Infinityなどの無効値を線形補間で処理
            % MATLABのLSTMではNaN/Infが含まれるデータは処理できないため
            %
            % 入力:
            %   data - 処理対象データ [特徴量 × 時間ステップ]
            %   trialIndex - 試行インデックス（ログ用）
            %
            % 出力:
            %   processedData - 補間処理後のデータ
            
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
                            obj.logMessage(1, '警告: 試行 %d, 特徴量 %d の全データポイントが無効です。0で置換します。\n', ...
                                trialIndex, f);
                        end
                        featureData(invalidIndices) = replacementValue;
                    end
                    
                    processedData(f, :) = featureData;
                end
            end
            
            % 無効値があった場合に情報を表示
            if hasInvalidData
                obj.logMessage(2, '試行 %d: %d個の無効値を検出し補間処理しました (%.1f%%)\n', ...
                    trialIndex, invalidCount, (invalidCount/(features*timepoints))*100);
            end
        end

        %% ハイブリッドモデル学習メソッド
        function [hybridModel, trainInfo] = trainHybridModel(obj, cnnTrainData, lstmTrainData, trainLabels, cnnValData, lstmValData, valLabels)
            % CNN, LSTMモデルを学習し、特徴を統合するハイブリッドモデルを構築
            %
            % 入力:
            %   cnnTrainData - CNN用学習データ
            %   lstmTrainData - LSTM用学習データ
            %   trainLabels - 学習ラベル
            %   cnnValData - CNN用検証データ
            %   lstmValData - LSTM用検証データ
            %   valLabels - 検証ラベル
            %
            % 出力:
            %   hybridModel - 学習済みハイブリッドモデル構造体
            %   trainInfo - 学習情報構造体
            
            try        
                obj.logMessage(1, '\n=== ハイブリッドモデルの学習開始 ===\n');
                
                % GPUメモリの確認
                if obj.useGPU
                    obj.checkGPUMemory();
                end
                
                % データの構造確認と検証
                if ~isempty(lstmTrainData) && iscell(lstmTrainData) && ~isempty(lstmTrainData{1})
                    sampleLstmData = lstmTrainData{1};
                    [numFeatures, ~] = size(sampleLstmData);
                    obj.logMessage(2, 'LSTM入力特徴量数: %d\n', numFeatures);
                else
                    error('不正なLSTM学習データ形式');
                end
        
                % ラベルのカテゴリカル変換
                uniqueLabels = unique(trainLabels);
                trainLabels = categorical(trainLabels, uniqueLabels);
                valLabels = categorical(valLabels, uniqueLabels);
                
                % --- CNNモデル学習 ---
                obj.logMessage(1, '\n--- CNNモデル学習開始 ---\n');
                
                % CNNアーキテクチャの構築
                cnnArchitecture = obj.params.classifier.hybrid.architecture.cnn;
                cnnLayers = obj.buildCNNLayers(cnnTrainData, cnnArchitecture);
                
                % CNN学習オプションの設定
                cnnTrainOptions = obj.getCNNTrainingOptions(cnnValData, valLabels);
                
                % CNNモデルの学習
                [cnnModel, cnnTrainInfo] = trainNetwork(cnnTrainData, trainLabels, cnnLayers, cnnTrainOptions);
                obj.logMessage(1, 'CNNモデル学習完了: %d エポック\n', length(cnnTrainInfo.TrainingLoss));
                
                % --- LSTMモデル学習 ---
                obj.logMessage(1, '\n--- LSTMモデル学習開始 ---\n');
                
                % LSTMアーキテクチャの構築 - 正しい特徴量次元を渡す
                lstmLayers = obj.buildLSTMLayers(numFeatures);
                
                % LSTM学習オプションの設定
                lstmTrainOptions = obj.getLSTMTrainingOptions(lstmValData, valLabels);
                
                % LSTMモデルの学習
                [lstmModel, lstmTrainInfo] = trainNetwork(lstmTrainData, trainLabels, lstmLayers, lstmTrainOptions);
                obj.logMessage(1, 'LSTMモデル学習完了: %d エポック\n', length(lstmTrainInfo.TrainingLoss));
                
                % --- 特徴抽出と統合モデルの学習 ---
                obj.logMessage(1, '\n--- 特徴統合と最終分類器の学習 ---\n');
                
                % CNNからの特徴抽出
                cnnFeatures = activations(cnnModel, cnnTrainData, 'fc_cnn', 'OutputAs', 'rows');
                obj.logMessage(2, 'CNN特徴抽出: [%d × %d]\n', size(cnnFeatures, 1), size(cnnFeatures, 2));
                
                % LSTMからの特徴抽出 - 動的配列サイズ変更を回避
                lstmFeatures = obj.extractLSTMFeatures(lstmModel, lstmTrainData);
                obj.logMessage(2, 'LSTM特徴抽出: [%d × %d]\n', size(lstmFeatures, 1), size(lstmFeatures, 2));
                
                % 特徴の統合
                combinedFeatures = [cnnFeatures, lstmFeatures];
                obj.logMessage(1, '統合特徴量: [%d × %d]\n', size(combinedFeatures, 1), size(combinedFeatures, 2));
                
                % AdaBoost分類器のパラメータ設定
                adaParams = obj.params.classifier.hybrid.adaBoost;
                
                % クラス数を確認
                numClasses = length(uniqueLabels);
                
                obj.logMessage(1, '%dクラス分類のために AdaBoostM2 を使用します\n', numClasses);
                
                % AdaBoostM2 分類器の学習 (マルチクラス対応)
                adaModel = fitcensemble(combinedFeatures, trainLabels, ...
                    'Method', 'AdaBoostM2', ...
                    'NumLearningCycles', adaParams.numLearners, ...
                    'Learners', 'tree', ...
                    'LearnRate', adaParams.learnRate);
                
                obj.logMessage(1, 'AdaBoost 統合分類器の学習完了: %d 弱学習器\n', adaParams.numLearners);
                
                % --- 検証データでのハイブリッドモデル全体の評価 ---
                obj.logMessage(1, '\n--- 検証データでのハイブリッドモデル評価 ---\n');
                
                % 検証データでの特徴抽出と評価
                hybridValMetrics = obj.evaluateOnValidation(cnnModel, lstmModel, adaModel, ...
                    cnnValData, lstmValData, valLabels);
                
                obj.logMessage(1, '検証精度: %.2f%%\n', hybridValMetrics.accuracy * 100);
                
                % ハイブリッドモデル情報を構築
                hybridModel = struct(...
                    'netCNN', cnnModel, ...
                    'netLSTM', lstmModel, ...
                    'adaModel', adaModel, ...
                    'lstmFeatureSize', size(lstmFeatures, 2));
                
                % 学習履歴情報の構築
                trainInfo = struct(...
                    'cnnHistory', cnnTrainInfo, ...
                    'lstmHistory', lstmTrainInfo, ...
                    'hybridValAccuracy', hybridValMetrics.accuracy, ...
                    'hybridValMetrics', hybridValMetrics, ...
                    'FinalEpoch', max(length(cnnTrainInfo.TrainingLoss), length(lstmTrainInfo.TrainingLoss)));
                
                obj.logMessage(1, '\n=== ハイブリッドモデル学習が完了しました ===\n');
                
            catch ME
                obj.logMessage(0, '\n=== ハイブリッドモデル学習中にエラーが発生 ===\n');
                obj.logMessage(0, 'エラーメッセージ: %s\n', ME.message);
                obj.logMessage(2, 'エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                
                % GPU使用時はメモリをクリア
                obj.resetGPUMemory();
                rethrow(ME);
            end
        end

        %% LSTM特徴抽出メソッド
        function features = extractLSTMFeatures(~, lstmModel, lstmData)
            % LSTMモデルから特徴を抽出
            %
            % 入力:
            %   lstmModel - 学習済みLSTMモデル
            %   lstmData - LSTM用データ（セル配列）
            %
            % 出力:
            %   features - 抽出された特徴量行列
            
            % 事前に特徴量次元を確認してメモリを事前割り当て
            numSamples = length(lstmData);
            % 最初のサンプルで特徴量次元を確認
            tempOut = predict(lstmModel, lstmData{1}, 'MiniBatchSize', 1);
            featureSize = size(tempOut, 2);
            
            % 事前に配列を割り当て
            features = zeros(numSamples, featureSize);
            features(1,:) = tempOut;
            
            % 残りのサンプルを処理
            for i = 2:numSamples
                lstmOut = predict(lstmModel, lstmData{i}, 'MiniBatchSize', 1);
                features(i,:) = lstmOut;
            end
        end

        %% 検証データ評価メソッド
        function metrics = evaluateOnValidation(obj, cnnModel, lstmModel, adaModel, cnnValData, lstmValData, valLabels)
            % 検証データでハイブリッドモデルを評価
            %
            % 入力:
            %   cnnModel - 学習済みCNNモデル
            %   lstmModel - 学習済みLSTMモデル
            %   adaModel - 学習済み統合分類器
            %   cnnValData - CNN用検証データ
            %   lstmValData - LSTM用検証データ
            %   valLabels - 検証ラベル
            %
            % 出力:
            %   metrics - 評価メトリクス
            
            % 検証データからの特徴抽出 (CNN)
            cnnValFeatures = activations(cnnModel, cnnValData, 'fc_cnn', 'OutputAs', 'rows');
            
            % 検証データからの特徴抽出 (LSTM)
            lstmValFeatures = obj.extractLSTMFeatures(lstmModel, lstmValData);
            
            % 特徴の統合 (検証データ)
            combinedValFeatures = [cnnValFeatures, lstmValFeatures];
            
            % 検証データでの予測
            [valPred, valScores] = predict(adaModel, combinedValFeatures);
            
            % 検証精度の計算
            accuracy = mean(valPred == valLabels);
            
            % 混同行列
            confusionMat = confusionmat(valLabels, valPred);
            
            % クラスごとの評価指標
            classes = unique(valLabels);
            classwise = struct('precision', zeros(1,length(classes)), ...
                              'recall', zeros(1,length(classes)), ...
                              'f1score', zeros(1,length(classes)));
            
            for i = 1:length(classes)
                className = classes(i);
                classIdx = (valLabels == className);
                
                % クラスごとの指標計算
                TP = sum(valPred(classIdx) == className);
                FP = sum(valPred == className) - TP;
                FN = sum(classIdx) - TP;
                
                % ゼロ除算防止
                precision = 0;
                recall = 0;
                f1score = 0;
                
                if (TP + FP) > 0
                    precision = TP / (TP + FP);
                end
                
                if (TP + FN) > 0
                    recall = TP / (TP + FN);
                end
                
                if (precision + recall) > 0
                    f1score = 2 * (precision * recall) / (precision + recall);
                end
                
                % クラス指標の保存
                classwise(i).precision = precision;
                classwise(i).recall = recall;
                classwise(i).f1score = f1score;
                
                obj.logMessage(2, '  - クラス %d:\n', double(classes(i)));
                obj.logMessage(2, '    - 精度 (Precision): %.2f%%\n', precision * 100);
                obj.logMessage(2, '    - 再現率 (Recall): %.2f%%\n', recall * 100);
                obj.logMessage(2, '    - F1スコア: %.2f\n', f1score);
            end
            
            % 混同行列の表示
            if obj.verbosity >= 2
                obj.logMessage(2, '\n検証データの混同行列:\n');
                disp(confusionMat);
            end
            
            % 検証結果の詳細構造体
            metrics = struct(...
                'accuracy', accuracy, ...
                'prediction', valPred, ...
                'score', valScores, ...
                'confusionMat', confusionMat, ...
                'classwise', classwise);
        end

        %% CNNレイヤー構築メソッド
        function layers = buildCNNLayers(obj, data, architecture)
            % ハイブリッドモデルのCNN部分のアーキテクチャを構築
            %
            % 入力:
            %   data - 入力データサンプル（サイズ取得用）
            %   architecture - CNNアーキテクチャ構造体（設定ファイルから）
            %
            % 出力:
            %   layers - 構築されたCNNレイヤー
            
            obj.logMessage(2, 'CNNアーキテクチャを構築中...\n');
            
            % 入力サイズの決定
            layerInputSize = obj.determineInputSize(data);
            
            % レイヤーの数を事前に計算
            convFields = fieldnames(architecture.convLayers);
            numConvLayers = length(convFields);
            
            % 各畳み込みブロックにつき最大6層（conv, bn, relu, pool, dropout, 他）
            % 最後のブロックで8層（gap, fc, relu, fc_output, softmax, classification）
            maxLayers = 1 + numConvLayers * 6 + 8;
            
            % 事前にセル配列を割り当て
            layerArray = cell(maxLayers, 1);
            layerIdx = 1;
            
            % 入力層
            layerArray{layerIdx} = imageInputLayer(layerInputSize, 'Name', 'input', 'Normalization', 'none');
            layerIdx = layerIdx + 1;
            
            % 畳み込み層の追加
            for i = 1:numConvLayers
                convName = convFields{i};
                convParams = architecture.convLayers.(convName);
                
                % 畳み込み層
                layerArray{layerIdx} = convolution2dLayer(...
                    convParams.size, convParams.filters, ...
                    'Stride', convParams.stride, ...
                    'Padding', convParams.padding, ...
                    'Name', convName);
                layerIdx = layerIdx + 1;
                
                % バッチ正規化層（設定に応じて）
                if architecture.batchNorm
                    layerArray{layerIdx} = batchNormalizationLayer('Name', ['bn_' convName]);
                    layerIdx = layerIdx + 1;
                end
                
                % 活性化関数
                layerArray{layerIdx} = reluLayer('Name', ['relu_' convName]);
                layerIdx = layerIdx + 1;
                
                % プーリング層
                poolName = ['pool' num2str(i)];
                if isfield(architecture.poolLayers, poolName)
                    poolParams = architecture.poolLayers.(poolName);
                    layerArray{layerIdx} = maxPooling2dLayer(...
                        poolParams.size, 'Stride', poolParams.stride, ...
                        'Name', poolName);
                    layerIdx = layerIdx + 1;
                end
                
                % ドロップアウト層
                dropoutName = ['dropout' num2str(i)];
                if isfield(architecture.dropoutLayers, dropoutName)
                    dropoutRate = architecture.dropoutLayers.(dropoutName);
                    layerArray{layerIdx} = dropoutLayer(dropoutRate, 'Name', dropoutName);
                    layerIdx = layerIdx + 1;
                end
            end
            
            % 全結合層（特徴出力用）
            layerArray{layerIdx} = globalAveragePooling2dLayer('Name', 'gap');
            layerIdx = layerIdx + 1;
            
            layerArray{layerIdx} = fullyConnectedLayer(architecture.fullyConnected, 'Name', 'fc_cnn');
            layerIdx = layerIdx + 1;
            
            layerArray{layerIdx} = reluLayer('Name', 'relu_fc');
            layerIdx = layerIdx + 1;
            
            % 出力層
            layerArray{layerIdx} = fullyConnectedLayer(obj.params.classifier.hybrid.architecture.numClasses, 'Name', 'fc_output');
            layerIdx = layerIdx + 1;
            
            layerArray{layerIdx} = softmaxLayer('Name', 'softmax');
            layerIdx = layerIdx + 1;
            
            layerArray{layerIdx} = classificationLayer('Name', 'output');
            layerIdx = layerIdx + 1;
            
            % 実際に使用した分だけ切り取り、配列に変換
            layers = layerArray(1:layerIdx-1);
            layers = cat(1, layers{:});
            
            obj.logMessage(3, 'CNNアーキテクチャ構築完了: %dレイヤー\n', length(layers));
        end

        %% 入力サイズ決定メソッド
        function layerInputSize = determineInputSize(obj, data)
            % データの次元に基づいて入力サイズを決定
            %
            % 入力:
            %   data - 入力データ
            %
            % 出力:
            %   layerInputSize - レイヤー入力サイズ [高さ, 幅, チャンネル]
            
            % データの初期サイズと次元数を記録
            originalSize = size(data);
            dimCount = ndims(data);
            
            if dimCount == 4
                % 4次元データ [サンプル x チャンネル x 1 x エポック]
                layerInputSize = [originalSize(1), originalSize(2), 1];
                obj.logMessage(3, '4次元データの入力サイズ: [%d, %d, %d]\n', ...
                    layerInputSize(1), layerInputSize(2), layerInputSize(3));
            elseif dimCount == 3
                % 3次元データ [チャンネル x サンプル x エポック]
                layerInputSize = [originalSize(2), originalSize(1), 1];
                obj.logMessage(3, '3次元データの入力サイズ: [%d, %d, %d]\n', ...
                    layerInputSize(1), layerInputSize(2), layerInputSize(3));
            elseif ismatrix(data)
                % 2次元データ [チャンネル x サンプル]
                layerInputSize = [originalSize(2), originalSize(1), 1];
                obj.logMessage(3, '2次元データの入力サイズ: [%d, %d, %d]\n', ...
                    layerInputSize(1), layerInputSize(2), layerInputSize(3));
            else
                error('対応していないデータ次元数: %d', dimCount);
            end
            
            % サイズの妥当性確認
            if any(layerInputSize <= 0)
                error('無効な入力サイズ: [%s]', num2str(layerInputSize));
            end
            
            return;
        end

        %% LSTMレイヤー構築メソッド
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
                arch = obj.params.classifier.hybrid.architecture;
                numClasses = arch.numClasses;
                
                obj.logMessage(2, 'LSTMレイヤー構築: 入力特徴量=%d, クラス数=%d\n', inputSize, numClasses);
                
                % LSTM固有のパラメータを取得
                lstmArch = arch.lstm;
                if ~isfield(lstmArch, 'lstmLayers') || ~isfield(lstmArch, 'dropoutLayers')
                    error('LSTM構造が不完全です。lstmLayers/dropoutLayersが見つかりません。');
                end
                
                % レイヤー数を計算
                numLSTMLayers = length(fieldnames(lstmArch.lstmLayers));
                
                obj.logMessage(3, 'LSTM構成: LSTM層=%d\n', numLSTMLayers);
                
                % レイヤーの数を事前に計算
                % 入力層: 1
                % 各LSTM層: 3層（lstm, bn（オプション）, dropout）
                % 全結合層: 2層（fc, relu）
                % 出力層: 3層（fc_output, softmax, classification）
                maxLayers = 1 + numLSTMLayers * 3 + 2 + 3;
                
                % バッチ正規化が有効な場合の追加
                if isfield(lstmArch, 'batchNorm') && lstmArch.batchNorm
                    maxLayers = maxLayers + numLSTMLayers;
                end
                
                % 事前にセル配列を割り当て
                layerArray = cell(maxLayers, 1);
                layerIdx = 1;
                
                % 入力層 - ここで正しい入力サイズを指定することが重要
                layerArray{layerIdx} = sequenceInputLayer(inputSize, ...
                    'Normalization', 'none', ...
                    'Name', 'input');
                layerIdx = layerIdx + 1;
                
                % LSTM層、バッチ正規化、ドロップアウト層の追加
                lstmLayerNames = fieldnames(lstmArch.lstmLayers);
                for i = 1:length(lstmLayerNames)
                    lstmParams = lstmArch.lstmLayers.(lstmLayerNames{i});
                    
                    % ドロップアウト率の取得
                    if isfield(lstmArch.dropoutLayers, ['dropout' num2str(i)])
                        dropoutRate = lstmArch.dropoutLayers.(['dropout' num2str(i)]);
                    else
                        dropoutRate = 0.5;  % デフォルト値
                    end
                    
                    % LSTM層の追加
                    layerArray{layerIdx} = lstmLayer(lstmParams.numHiddenUnits, ...
                        'OutputMode', lstmParams.OutputMode, ...
                        'Name', ['lstm' num2str(i)]);
                    layerIdx = layerIdx + 1;
                    
                    % バッチ正規化層（オプション）
                    if isfield(lstmArch, 'batchNorm') && lstmArch.batchNorm
                        layerArray{layerIdx} = batchNormalizationLayer('Name', ['bn' num2str(i)]);
                        layerIdx = layerIdx + 1;
                    end
                    
                    % ドロップアウト層
                    layerArray{layerIdx} = dropoutLayer(dropoutRate, 'Name', ['dropout' num2str(i)]);
                    layerIdx = layerIdx + 1;
                end
                
                % 全結合層とReLU層の追加
                if isfield(lstmArch, 'fullyConnected') && ~isempty(lstmArch.fullyConnected)
                    % 単一値の場合の処理
                    if isscalar(lstmArch.fullyConnected)
                        layerArray{layerIdx} = fullyConnectedLayer(lstmArch.fullyConnected, ...
                            'Name', 'fc1');
                        layerIdx = layerIdx + 1;
                        
                        layerArray{layerIdx} = reluLayer('Name', 'relu_fc1');
                        layerIdx = layerIdx + 1;
                    else
                        % 配列の場合の処理
                        for i = 1:length(lstmArch.fullyConnected)
                            layerArray{layerIdx} = fullyConnectedLayer(lstmArch.fullyConnected(i), ...
                                'Name', ['fc' num2str(i)]);
                            layerIdx = layerIdx + 1;
                            
                            layerArray{layerIdx} = reluLayer('Name', ['relu_fc' num2str(i)]);
                            layerIdx = layerIdx + 1;
                        end
                    end
                else
                    % デフォルトの全結合層（設定がない場合）
                    layerArray{layerIdx} = fullyConnectedLayer(64, ...
                        'Name', 'fc1');
                    layerIdx = layerIdx + 1;
                    
                    layerArray{layerIdx} = reluLayer('Name', 'relu_fc1');
                    layerIdx = layerIdx + 1;
                end
                
                % 出力層
                layerArray{layerIdx} = fullyConnectedLayer(numClasses, 'Name', 'fc_output');
                layerIdx = layerIdx + 1;
                
                layerArray{layerIdx} = softmaxLayer('Name', 'softmax');
                layerIdx = layerIdx + 1;
                
                layerArray{layerIdx} = classificationLayer('Name', 'output');
                layerIdx = layerIdx + 1;
                
                % 実際に使用した分だけ切り取り、配列に変換
                layers = layerArray(1:layerIdx-1);
                layers = cat(1, layers{:});
                
                obj.logMessage(3, 'LSTMアーキテクチャ構築完了: %dレイヤー\n', length(layers));
                
            catch ME
                obj.logMessage(0, 'LSTMレイヤー構築でエラーが発生: %s\n', ME.message);
                obj.logMessage(2, 'エラースタック:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% CNNトレーニングオプション設定メソッド
        function options = getCNNTrainingOptions(obj, valData, valLabels)
            % CNNのトレーニングオプションを設定
            %
            % 入力:
            %   valData - 検証データ
            %   valLabels - 検証ラベル
            %
            % 出力:
            %   options - トレーニングオプション
            
            % 実行環境の選択
            executionEnvironment = 'cpu';
            if obj.useGPU
                executionEnvironment = 'gpu';
            end
            
            % 検証データの準備
            valDS = {valData, valLabels};
            
            % verbosityレベルに応じたverbose設定
            verboseFlag = false;
            if obj.verbosity >= 2
                verboseFlag = true;
            end
            
            % トレーニングオプションの設定
            options = trainingOptions(obj.params.classifier.hybrid.training.optimizer.type, ...
                'InitialLearnRate', obj.params.classifier.hybrid.training.optimizer.learningRate, ...
                'MaxEpochs', obj.params.classifier.hybrid.training.maxEpochs, ...
                'MiniBatchSize', obj.params.classifier.hybrid.training.miniBatchSize, ...
                'Plots', 'none', ...
                'Shuffle', obj.params.classifier.hybrid.training.shuffle, ...
                'ExecutionEnvironment', executionEnvironment, ...
                'OutputNetwork', 'best-validation', ...
                'Verbose', verboseFlag, ...
                'ValidationData', valDS, ...
                'ValidationFrequency', obj.params.classifier.hybrid.training.frequency, ...
                'ValidationPatience', obj.params.classifier.hybrid.training.patience, ...
                'GradientThreshold', 1);
        end

        %% LSTMトレーニングオプション設定メソッド
        function options = getLSTMTrainingOptions(obj, valData, valLabels)
            % LSTMのトレーニングオプションを設定
            %
            % 入力:
            %   valData - 検証データ
            %   valLabels - 検証ラベル
            %
            % 出力:
            %   options - トレーニングオプション
            
            % 実行環境の選択
            executionEnvironment = 'cpu';
            if obj.useGPU
                executionEnvironment = 'gpu';
            end
            
            % 検証データの準備
            valDS = {valData, valLabels};
            
            % verbosityレベルに応じたverbose設定
            verboseFlag = false;
            if obj.verbosity >= 2
                verboseFlag = true;
            end
            
            % トレーニングオプションの設定
            options = trainingOptions(obj.params.classifier.hybrid.training.optimizer.type, ...
                'InitialLearnRate', obj.params.classifier.hybrid.training.optimizer.learningRate, ...
                'MaxEpochs', obj.params.classifier.hybrid.training.maxEpochs, ...
                'MiniBatchSize', obj.params.classifier.hybrid.training.miniBatchSize, ...
                'Plots', 'none', ...
                'Shuffle', obj.params.classifier.hybrid.training.shuffle, ...
                'ExecutionEnvironment', executionEnvironment, ...
                'OutputNetwork', 'best-validation', ...
                'Verbose', verboseFlag, ...
                'ValidationData', valDS, ...
                'ValidationFrequency', obj.params.classifier.hybrid.training.frequency, ...
                'ValidationPatience', obj.params.classifier.hybrid.training.patience, ...
                'GradientThreshold', obj.params.classifier.hybrid.training.optimizer.gradientThreshold);
        end

        %% モデル評価メソッド
        function metrics = evaluateModel(obj, model, cnnTestData, lstmTestData, testLabels)
            % 学習済みハイブリッドモデルの性能を評価
            %
            % 入力:
            %   model - 学習済みハイブリッドモデル
            %   cnnTestData - CNN用テストデータ
            %   lstmTestData - LSTM用テストデータ
            %   testLabels - テストラベル
            %
            % 出力:
            %   metrics - 詳細な評価メトリクス
            
            obj.logMessage(1, '\n=== モデル評価を実行 ===\n');
            metrics = struct(...
                'accuracy', [], ...
                'score', [], ...
                'confusionMat', [], ...
                'classwise', [], ...
                'roc', [], ...
                'auc', [] ...
            );
        
            try                
                % CNNからの特徴抽出
                cnnFeatures = activations(model.netCNN, cnnTestData, 'fc_cnn', 'OutputAs', 'rows');
                
                % LSTMからの特徴抽出
                lstmFeatures = obj.extractLSTMFeatures(model.netLSTM, lstmTestData);
                
                % 特徴の統合
                combinedFeatures = [cnnFeatures, lstmFeatures];
                
                % AdaBoostによる予測
                [pred, score] = predict(model.adaModel, combinedFeatures);
                % スコアを確率に変換
                metrics.score = obj.convertScoreToProbability(score);
                
                % テストラベルをcategorical型に変換
                testLabels_cat = categorical(testLabels);
                
                % 基本的な指標の計算 
                metrics.accuracy = mean(pred == testLabels_cat);
                metrics.confusionMat = confusionmat(testLabels_cat, pred);
                
                obj.logMessage(1, 'テスト精度: %.2f%%\n', metrics.accuracy * 100);
                
                % クラスごとの性能評価
                classes = unique(testLabels_cat);
                metrics.classwise = struct('precision', zeros(1,length(classes)), ...
                                        'recall', zeros(1,length(classes)), ...
                                        'f1score', zeros(1,length(classes)));
                
                obj.logMessage(1, '\nクラスごとの評価:\n');
                for i = 1:length(classes)
                    className = classes(i);
                    classIdx = (testLabels_cat == className);
                    
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
                    
                    obj.logMessage(1, '  - クラス %d:\n', double(className));
                    obj.logMessage(1, '    - 精度 (Precision): %.2f%%\n', precision * 100);
                    obj.logMessage(1, '    - 再現率 (Recall): %.2f%%\n', recall * 100);
                    obj.logMessage(1, '    - F1スコア: %.2f\n', f1);
                end
                
                % 2クラス分類の場合のROC曲線とAUC
                if length(classes) == 2
                    [X, Y, T, AUC] = perfcurve(testLabels_cat, score(:,2), classes(2));
                    metrics.roc = struct('X', X, 'Y', Y, 'T', T);
                    metrics.auc = AUC;
                    obj.logMessage(1, '\nAUC: %.3f\n', AUC);
                end
                
                % 混同行列の表示
                obj.logMessage(1, '\n混同行列:\n');
                disp(metrics.confusionMat);
                
            catch ME
                obj.logMessage(0, 'モデル評価でエラーが発生: %s\n', ME.message);
                obj.logMessage(2, 'エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% CNN学習分析メソッド
        function [trainTrend, valTrend, optimalEpoch, totalEpochs] = analyzeCNNTraining(obj, trainInfo)
            % CNN学習曲線の分析
            %
            % 入力:
            %   trainInfo - 学習情報
            %
            % 出力:
            %   trainTrend - 学習傾向
            %   valTrend - 検証傾向
            %   optimalEpoch - 最適エポック
            %   totalEpochs - 総エポック数
            
            if isfield(trainInfo, 'cnnHistory') && ...
               isfield(trainInfo.cnnHistory, 'TrainingAccuracy') && ...
               isfield(trainInfo.cnnHistory, 'ValidationAccuracy')
                
                cnnTrainAcc = trainInfo.cnnHistory.TrainingAccuracy;
                cnnValAcc = trainInfo.cnnHistory.ValidationAccuracy;
                
                % NaN値を除去
                cnnTrainAcc = cnnTrainAcc(~isnan(cnnTrainAcc));
                cnnValAcc = cnnValAcc(~isnan(cnnValAcc));
                
                % CNN学習曲線の詳細分析
                [trainTrend, valTrend] = obj.analyzeLearningCurves(cnnTrainAcc, cnnValAcc);

                % 収束エポック
                if ~isempty(cnnValAcc)
                    [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(cnnValAcc);
                else
                    optimalEpoch = 0;
                    totalEpochs = 0;
                end
                
                obj.logMessage(2, 'CNN学習曲線分析: 平均変化率=%.4f, 変動性=%.4f\n', ...
                    trainTrend.mean_change, trainTrend.volatility);
            else
                obj.logMessage(2, 'CNN学習曲線データが不足しています\n');
                trainTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false);
                valTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false);
                optimalEpoch = 0;
                totalEpochs = 0;
            end
        end

        %% LSTM学習分析メソッド
        function [trainTrend, valTrend, optimalEpoch, totalEpochs] = analyzeLSTMTraining(obj, trainInfo)
            % LSTM学習曲線の分析
            %
            % 入力:
            %   trainInfo - 学習情報
            %
            % 出力:
            %   trainTrend - 学習傾向
            %   valTrend - 検証傾向
            %   optimalEpoch - 最適エポック
            %   totalEpochs - 総エポック数
            
            if isfield(trainInfo, 'lstmHistory') && ...
               isfield(trainInfo.lstmHistory, 'TrainingAccuracy') && ...
               isfield(trainInfo.lstmHistory, 'ValidationAccuracy')
                
                lstmTrainAcc = trainInfo.lstmHistory.TrainingAccuracy;
                lstmValAcc = trainInfo.lstmHistory.ValidationAccuracy;
                
                % NaN値を除去
                lstmTrainAcc = lstmTrainAcc(~isnan(lstmTrainAcc));
                lstmValAcc = lstmValAcc(~isnan(lstmValAcc));
                
                % LSTM学習曲線の詳細分析
                [trainTrend, valTrend] = obj.analyzeLearningCurves(lstmTrainAcc, lstmValAcc);
                
                % 収束エポック
                if ~isempty(lstmValAcc)
                    [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(lstmValAcc);
                else
                    optimalEpoch = 0;
                    totalEpochs = 0;
                end
                
                obj.logMessage(2, 'LSTM学習曲線分析: 平均変化率=%.4f, 変動性=%.4f\n', ...
                    trainTrend.mean_change, trainTrend.volatility);
            else
                obj.logMessage(2, 'LSTM学習曲線データが不足しています\n');
                trainTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false);
                valTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false);
                optimalEpoch = 0;
                totalEpochs = 0;
            end
        end

        %% 検証-テストギャップ分析メソッド
        function [gapOverfit, gapMetrics] = analyzeValidationTestGap(obj, trainInfo, testMetrics)
            % ハイブリッドモデル全体の検証-テスト精度ギャップ分析
            %
            % 入力:
            %   trainInfo - 学習情報
            %   testMetrics - テスト評価結果
            %
            % 出力:
            %   gapOverfit - ギャップによる過学習判定
            %   gapMetrics - ギャップメトリクス
            
            if isfield(trainInfo, 'hybridValMetrics')
                hybridValAcc = trainInfo.hybridValMetrics.accuracy * 100; % パーセントに変換
                testAcc = testMetrics.accuracy * 100; % パーセントに変換
                
                % データサイズの取得（混同行列のサイズから推定）
                if isfield(testMetrics, 'confusionMat')
                    dataSize = sum(sum(testMetrics.confusionMat));
                else
                    dataSize = 100; % デフォルト値
                end
                
                % ギャップ分析を実行
                [gapOverfit, gapMetrics] = obj.validateTestValidationGap(hybridValAcc, testAcc, dataSize);
                
                obj.logMessage(1, 'ハイブリッド検証-テストギャップ: %.2f%% (重大度: %s)\n', ...
                    gapMetrics.rawGap, gapMetrics.severity);
            else
                % フォールバック処理
                gapOverfit = false;
                gapMetrics = struct('rawGap', 0, 'normalizedGap', 0, 'adjustedGap', 0, ...
                    'meanValAcc', 0, 'stdValAcc', 0, 'testAcc', testMetrics.accuracy * 100, ...
                    'severity', 'unknown');
                    
                obj.logMessage(1, '警告: 検証精度データが見つかりません\n');
            end
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
            
            obj.logMessage(3, '\n学習曲線の分析を実行...\n');
            
            % データの検証
            if isempty(trainAcc) || isempty(valAcc)
                trainTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, ...
                    'plateau_detected', false, 'oscillation_strength', 0, 'convergence_epoch', 0);
                valTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, ...
                    'plateau_detected', false, 'oscillation_strength', 0, 'convergence_epoch', 0);
                return;
            end
            
            try                
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
                
                obj.logMessage(3, '学習曲線分析結果:\n');
                obj.logMessage(3, '  - 学習曲線平均変化率: %.5f\n', trainTrend.mean_change);
                obj.logMessage(3, '  - 検証曲線平均変化率: %.5f\n', valTrend.mean_change);
                obj.logMessage(3, '  - 学習曲線変動性: %.5f\n', trainTrend.volatility);
                obj.logMessage(3, '  - 検証曲線変動性: %.5f\n', valTrend.volatility);
                obj.logMessage(3, '  - 学習プラトー検出: %s\n', mat2str(trainTrend.plateau_detected));
                obj.logMessage(3, '  - 検証プラトー検出: %s\n', mat2str(valTrend.plateau_detected));
                
            catch ME
                obj.logMessage(1, '学習曲線分析でエラーが発生: %s\n', ME.message);
                
                % エラー時のフォールバック値を設定
                trainTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, ...
                    'plateau_detected', false, 'oscillation_strength', 0, 'convergence_epoch', 0);
                valTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, ...
                    'plateau_detected', false, 'oscillation_strength', 0, 'convergence_epoch', 0);
            end
        end
        
        %% プラトー検出メソッド
        function isPlateau = detectPlateau(obj, smoothedCurve)
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
            
            obj.logMessage(3, '後半部分の平均変化率: %.6f (プラトー判定: %s)\n', ...
                mean(abs(segmentDiff)), mat2str(isPlateau));
        end

        %% 振動強度計算メソッド
        function oscillation = calculateOscillation(obj, diffValues)
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
            
            obj.logMessage(3, '振動強度: %.3f (符号変化: %d/%d)\n', ...
                oscillation, signChanges, length(diffValues) - 1);
        end

        %% 収束エポック推定メソッド
        function convergenceEpoch = estimateConvergenceEpoch(obj, smoothedCurve)
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
            
            obj.logMessage(3, '収束エポック推定: %d/%d\n', convergenceEpoch, length(smoothedCurve));
        end
        
        %% 最適エポック検出メソッド
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(obj, valAcc)
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
                
                obj.logMessage(2, '最適エポック: %d / %d\n', optimalEpoch, totalEpochs);
                
                % 最適エポックが最後のエポックの場合、改善の余地がある可能性
                if optimalEpoch == totalEpochs
                    obj.logMessage(1, '警告: 最適エポックが最終エポックと一致。より長い学習が有益かもしれません。\n');
                end
            catch ME
                obj.logMessage(1, '最適エポック検出でエラーが発生: %s\n', ME.message);
                optimalEpoch = 0;
                totalEpochs = 0;
            end
        end
        
        %% 分類バイアス検出メソッド
        function isCompletelyBiased = detectClassificationBias(obj, testMetrics)
            % 混同行列から分類バイアスを検出
            %
            % 入力:
            %   testMetrics - テスト評価指標
            %
            % 出力:
            %   isCompletelyBiased - バイアス検出フラグ
            
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
                    obj.logMessage(1, '\n警告: 分類に完全な偏りが検出されました\n');
                    obj.logMessage(1, '  - 分類された実際のクラス数: %d / %d\n', sum(rowSums > 0), size(cm, 1));
                    obj.logMessage(1, '  - 予測されたクラス数: %d / %d\n', predictedClassCount, size(cm, 2));
                    
                    % 混同行列の出力
                    obj.logMessage(2, '  混同行列:\n');
                    disp(cm);
                end
            end
        end
        
        %% 検証-テスト精度ギャップ分析メソッド
        function [isOverfit, metrics] = validateTestValidationGap(obj, valAcc, testAcc, dataSize)
            % 検証精度とテスト精度の差を統計的に評価
            %
            % 入力:
            %   valAcc - 検証精度
            %   testAcc - テスト精度
            %   dataSize - データサイズ（サンプル数）
            %
            % 出力:
            %   isOverfit - 過学習の有無（論理値）
            %   metrics - 詳細な評価メトリクス
        
            % NaNチェックを追加
            if isempty(valAcc) || isnan(valAcc)
                obj.logMessage(1, '警告: 有効な検証精度履歴がありません\n');
                metrics = struct(...
                    'rawGap', NaN, ...
                    'normalizedGap', NaN, ...
                    'adjustedGap', NaN, ...
                    'meanValAcc', NaN, ...
                    'stdValAcc', NaN, ...
                    'testAcc', testAcc, ...
                    'severity', 'unknown' ...
                );
                isOverfit = true;
                return;
            end
            
            % 検証精度と標準偏差の計算
            meanValAcc = valAcc;
            stdValAcc = 0.01 * meanValAcc;  % デフォルト値として平均の1%
            
            % ゼロ除算防止
            if stdValAcc < 0.001
                stdValAcc = 0.001; 
            end
            
            % スケールされたギャップ計算（データサイズで調整）
            scaleFactor = min(1, sqrt(dataSize / 1000));
            
            % 精度の差と正規化された差
            rawGap = abs(meanValAcc - testAcc);
            normalizedGap = rawGap / stdValAcc;
            
            % スケール調整されたギャップ
            adjustedGap = normalizedGap * scaleFactor;
            
            % 絶対差に基づく過学習判定
            absoluteOverfit = (rawGap > 15.0);  % 15%以上の差
            
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
                statIdx = 1;
            end
            if isempty(absIdx)
                absIdx = 1;
            end
            
            if absIdx >= statIdx
                severity = absSeverity;
            else
                severity = statSeverity;
            end
            
            % 最終的な過学習判定
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
            obj.logMessage(1, '\n=== 検証-テスト精度差分析 ===\n');
            obj.logMessage(1, '  検証精度: %.2f%% (±%.2f%%)\n', meanValAcc, stdValAcc);
            obj.logMessage(1, '  テスト精度: %.2f%%\n', testAcc);
            obj.logMessage(1, '  検証-テスト精度差: %.2f%%\n', rawGap);
            obj.logMessage(1, '  統計的判定: %s (重大度: %s)\n', mat2str(statisticalOverfit), statSeverity);
            obj.logMessage(1, '  絶対差判定: %s (重大度: %s)\n', mat2str(absoluteOverfit), absSeverity);
            obj.logMessage(1, '  最終判定結果: %s (重大度: %s)\n', mat2str(isOverfit), severity);
        end

        %% スコアを確率に変換するメソッド
        function probability = convertScoreToProbability(obj, score)
            % AdaBoostのスコアを確率に変換
            %
            % 入力:
            %   score - AdaBoostの生スコア
            %
            % 出力:
            %   probability - 正規化された確率
            
            try
                % AdaBoostのスコアをソフトマックス関数で確率に変換
                if size(score, 2) > 1
                    % 多クラス分類の場合
                    % スコアが非常に大きい場合の数値安定性を確保
                    maxScore = max(score, [], 2);
                    expScore = exp(score - maxScore);
                    sumExpScore = sum(expScore, 2);
                    probability = expScore ./ sumExpScore;
                else
                    % 2クラス分類の場合（通常はこちらが使われる）
                    % シグモイド関数を適用
                    probability = 1 ./ (1 + exp(-score));
                    % 2クラス分類では補数も計算
                    probability = [1-probability, probability];
                end
                
                % 確率の合計が1になるように正規化（数値誤差対策）
                probability = probability ./ sum(probability, 2);
                
                % NaN/Infチェック
                if any(isnan(probability(:))) || any(isinf(probability(:)))
                    obj.logMessage(1, '警告: 確率計算でNaN/Infが検出されました。均等確率にフォールバックします。\n');
                    numClasses = size(score, 2);
                    if numClasses == 1
                        numClasses = 2; % 2クラス分類
                    end
                    probability = ones(size(score, 1), numClasses) / numClasses;
                end
                
            catch ME
                obj.logMessage(1, 'スコア→確率変換でエラー: %s\n', ME.message);
                % エラー時は均等確率を返す
                numClasses = size(score, 2);
                if numClasses == 1
                    numClasses = 2; % 2クラス分類
                end
                probability = ones(size(score, 1), numClasses) / numClasses;
            end
        end
        
        %% トレーニング情報検証メソッド
        function validateTrainInfo(obj, trainInfo)
            % trainInfo構造体の妥当性を検証
            %
            % 入力:
            %   trainInfo - 学習情報構造体
            
            if ~isstruct(trainInfo)
                obj.logMessage(1, '警告: trainInfoが構造体ではありません\n');
                return;
            end
            
            % 必須フィールドの確認
            requiredFields = {'cnnHistory', 'lstmHistory'};
            for i = 1:length(requiredFields)
                if ~isfield(trainInfo, requiredFields{i})
                    obj.logMessage(1, '警告: trainInfoに必須フィールド「%s」がありません\n', requiredFields{i});
                end
            end
            
            % CNN History構造体の検証
            if isfield(trainInfo, 'cnnHistory')
                historyFields = {'TrainingLoss', 'ValidationLoss', 'TrainingAccuracy', 'ValidationAccuracy'};
                for i = 1:length(historyFields)
                    if ~isfield(trainInfo.cnnHistory, historyFields{i})
                        obj.logMessage(1, '警告: cnnHistory構造体に「%s」フィールドがありません\n', historyFields{i});
                    elseif isempty(trainInfo.cnnHistory.(historyFields{i}))
                        obj.logMessage(1, '警告: cnnHistory構造体の「%s」フィールドが空です\n', historyFields{i});
                    end
                end
            end
            
            % LSTM History構造体の検証
            if isfield(trainInfo, 'lstmHistory')
                historyFields = {'TrainingLoss', 'ValidationLoss', 'TrainingAccuracy', 'ValidationAccuracy'};
                for i = 1:length(historyFields)
                    if ~isfield(trainInfo.lstmHistory, historyFields{i})
                        obj.logMessage(1, '警告: lstmHistory構造体に「%s」フィールドがありません\n', historyFields{i});
                    elseif isempty(trainInfo.lstmHistory.(historyFields{i}))
                        obj.logMessage(1, '警告: lstmHistory構造体の「%s」フィールドが空です\n', historyFields{i});
                    end
                end
            end
            
            % FinalEpochの妥当性検証
            if isfield(trainInfo, 'FinalEpoch')
                if trainInfo.FinalEpoch <= 0
                    obj.logMessage(1, '警告: FinalEpochが0以下です: %d\n', trainInfo.FinalEpoch);
                end
            else
                obj.logMessage(1, '警告: FinalEpochフィールドがありません\n');
            end
            
            % hybridValAccuracyの検証
            if isfield(trainInfo, 'hybridValAccuracy')
                if trainInfo.hybridValAccuracy < 0 || trainInfo.hybridValAccuracy > 1
                    obj.logMessage(1, '警告: hybridValAccuracyの値が範囲外です: %.4f (期待範囲: 0-1)\n', ...
                        trainInfo.hybridValAccuracy);
                end
            else
                obj.logMessage(1, '警告: hybridValAccuracyフィールドがありません\n');
            end
        end
        
        %% フォールバック過学習メトリクス作成メソッド
        function metrics = createFallbackOverfitMetrics(~)
            % エラー発生時のフォールバック過学習メトリクスを作成
            % 分析失敗時でもシステムが動作を続けられるようデフォルト値を提供
            
            metrics = struct(...
                'performanceGap', Inf, ...
                'isCompletelyBiased', true, ...
                'isLearningProgressing', false, ...
                'cnnValidationTrend', struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0), ...
                'cnnTrainingTrend', struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0), ...
                'lstmValidationTrend', struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0), ...
                'lstmTrainingTrend', struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0), ...
                'severity', 'error', ...
                'cnnOptimalEpoch', 0, ...
                'cnnTotalEpochs', 0, ...
                'lstmOptimalEpoch', 0, ...
                'lstmTotalEpochs', 0, ...
                'cnnEarlyStoppingEffect', struct('optimal_epoch', 0, 'total_epochs', 0), ...
                'lstmEarlyStoppingEffect', struct('optimal_epoch', 0, 'total_epochs', 0) ...
            );
        end
        
        %% GPU使用状況確認メソッド
        function checkGPUMemory(obj)
            % GPU使用状況の確認とメモリ使用率の監視
            % 高メモリ使用時は自動的にバッチサイズを調整
            
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
                    
                    obj.logMessage(1, 'GPU使用状況: %.2f/%.2f GB (%.1f%%)\n', ...
                        usedMem, totalMem, (usedMem/totalMem)*100);
                    
                    % メモリ使用率が高い場合は警告と対応
                    if usedMem/totalMem > 0.8
                        obj.logMessage(1, '警告: GPU使用率が80%%を超えています。パフォーマンスに影響する可能性があります。\n');
                        
                        % 適応的なバッチサイズ調整
                        if obj.params.classifier.hybrid.training.miniBatchSize > 32
                            newBatchSize = max(16, floor(obj.params.classifier.hybrid.training.miniBatchSize / 2));
                            obj.logMessage(1, '高メモリ使用率のため、バッチサイズを%dから%dに削減します\n', ...
                                obj.params.classifier.hybrid.training.miniBatchSize, newBatchSize);
                            obj.params.classifier.hybrid.training.miniBatchSize = newBatchSize;
                        end
                    end
                catch ME
                    obj.logMessage(1, 'GPUメモリチェックでエラー: %s\n', ME.message);
                end
            end
        end
        
        %% GPUメモリ解放メソッド
        function resetGPUMemory(obj)
            % GPUメモリのリセットと解放
            % 学習終了時やエラー発生時にリソースを適切に解放する
            
            if obj.useGPU
                try
                    % GPUデバイスのリセット
                    currentDevice = gpuDevice();
                    reset(currentDevice);
                    
                    % リセット後のメモリ状況
                    availMem = currentDevice.AvailableMemory / 1e9;  % GB
                    totalMem = currentDevice.TotalMemory / 1e9;  % GB
                    obj.logMessage(1, 'GPUメモリをリセットしました (利用可能: %.2f/%.2f GB)\n', ...
                        availMem, totalMem);
                        
                    % 使用状況の更新
                    obj.gpuMemory.used = totalMem - availMem;
                    
                catch ME
                    obj.logMessage(1, 'GPUメモリのリセットに失敗: %s\n', ME.message);
                end
            end
        end

        %% 結果構造体構築メソッド
        function results = buildResultsStruct(obj, hybridModel, metrics, trainInfo, normParams)
            % 結果構造体の構築
            %
            % 入力:
            %   hybridModel - 学習済みハイブリッドモデル
            %   metrics - 評価メトリクス
            %   trainInfo - 学習情報
            %   normParams - 正規化パラメータ
            %
            % 出力:
            %   results - 結果構造体
            
            try
                % trainInfoの検証
                obj.validateTrainInfo(trainInfo);
                
                % パラメータ情報の抽出（ハイブリッドモデル用に調整）
                extractedParams = obj.extractParams();
                
                % 結果構造体の構築
                results = struct(...
                    'model', hybridModel, ...
                    'performance', metrics, ...
                    'trainInfo', trainInfo, ...
                    'overfitting', obj.overfitMetrics, ...
                    'normParams', normParams, ...
                    'params', extractedParams, ...
                    'timestamp', datetime('now') ...
                );
                
                % モデル情報の確認
                if ~isfield(hybridModel, 'netCNN') || isempty(hybridModel.netCNN)
                    obj.logMessage(1, '警告: CNN部分が欠損しています\n');
                end
                
                if ~isfield(hybridModel, 'netLSTM') || isempty(hybridModel.netLSTM)
                    obj.logMessage(1, '警告: LSTM部分が欠損しています\n');
                end
                
                if ~isfield(hybridModel, 'adaModel') || isempty(hybridModel.adaModel)
                    obj.logMessage(1, '警告: 統合分類器が欠損しています\n');
                end
                
                obj.logMessage(2, '結果構造体の構築完了\n');
                
            catch ME
                obj.logMessage(0, '結果構造体の構築中にエラーが発生: %s\n', ME.message);
                obj.logMessage(2, 'エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                
                % エラー時でも最低限の結果を返す
                results = struct(...
                    'model', hybridModel, ...
                    'performance', metrics, ...
                    'trainInfo', struct(), ...
                    'overfitting', struct('severity', 'unknown'), ...
                    'normParams', normParams, ...
                    'params', [], ...
                    'error', true, ...
                    'errorMessage', ME.message ...
                );
            end
        end

        %% パラメータ抽出メソッド
        function extractedParams = extractParams(obj)
            % ハイブリッドモデルのパラメータを抽出
            %
            % 出力:
            %   extractedParams - 抽出されたパラメータ配列
            
            % デフォルト値で初期化（ハイブリッドモデル用に拡張）
            extractedParams = zeros(1, 10);  % 10要素の配列
            
            try
                % 1. 学習率
                if isfield(obj.params.classifier.hybrid.training.optimizer, 'learningRate')
                    extractedParams(1) = obj.params.classifier.hybrid.training.optimizer.learningRate;
                end
                
                % 2. バッチサイズ
                if isfield(obj.params.classifier.hybrid.training, 'miniBatchSize')
                    extractedParams(2) = obj.params.classifier.hybrid.training.miniBatchSize;
                end
                
                % 3. CNN畳み込み層数
                if isfield(obj.params.classifier.hybrid.architecture.cnn, 'convLayers')
                    extractedParams(3) = length(fieldnames(obj.params.classifier.hybrid.architecture.cnn.convLayers));
                end
                
                % 4. CNNフィルタサイズと5. CNNフィルタ数
                if isfield(obj.params.classifier.hybrid.architecture.cnn, 'convLayers')
                    convFields = fieldnames(obj.params.classifier.hybrid.architecture.cnn.convLayers);
                    if ~isempty(convFields)
                        firstConv = obj.params.classifier.hybrid.architecture.cnn.convLayers.(convFields{1});
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
                
                % 6. CNNドロップアウト率
                if isfield(obj.params.classifier.hybrid.architecture.cnn, 'dropoutLayers')
                    dropout = obj.params.classifier.hybrid.architecture.cnn.dropoutLayers;
                    dropoutFields = fieldnames(dropout);
                    if ~isempty(dropoutFields)
                        extractedParams(6) = dropout.(dropoutFields{1});
                    end
                end
                
                % 7. CNN全結合層ユニット数
                if isfield(obj.params.classifier.hybrid.architecture.cnn, 'fullyConnected')
                    extractedParams(7) = obj.params.classifier.hybrid.architecture.cnn.fullyConnected;
                end
                
                % 8. LSTM隠れユニット数
                if isfield(obj.params.classifier.hybrid.architecture.lstm, 'lstmLayers')
                    lstmFields = fieldnames(obj.params.classifier.hybrid.architecture.lstm.lstmLayers);
                    if ~isempty(lstmFields)
                        firstLstm = obj.params.classifier.hybrid.architecture.lstm.lstmLayers.(lstmFields{1});
                        if isfield(firstLstm, 'numHiddenUnits')
                            extractedParams(8) = firstLstm.numHiddenUnits;
                        end
                    end
                end
                
                % 9. LSTM層数
                if isfield(obj.params.classifier.hybrid.architecture.lstm, 'lstmLayers')
                    extractedParams(9) = length(fieldnames(obj.params.classifier.hybrid.architecture.lstm.lstmLayers));
                end
                
                % 10. AdaBoost学習器数
                if isfield(obj.params.classifier.hybrid, 'adaBoost') && ...
                   isfield(obj.params.classifier.hybrid.adaBoost, 'numLearners')
                    extractedParams(10) = obj.params.classifier.hybrid.adaBoost.numLearners;
                end
                
            catch ME
                obj.logMessage(1, '警告: パラメータ抽出中にエラーが発生: %s\n', ME.message);
            end
        end
    end
end