classdef HybridClassifier < handle
    %% HybridClassifier - CNN+LSTM統合ハイブリッド分類器
    %
    % このクラスはEEGデータに対してCNNとLSTMを組み合わせた
    % ハイブリッド深層学習モデルを実装します。時空間的特徴の両方を
    % 効果的に捉え、高精度な分類を実現します。
    %
    % 主な機能:
    %   - EEGデータの前処理と変換
    %   - CNN・LSTMの並列学習とモデル統合
    %   - 学習済みモデルによるオンライン予測
    %   - 過学習検出と詳細な性能評価
    %   - ハイパーパラメータの最適化サポート
    %
    % 使用例:
    %   params = getConfig('epocx');
    %   hybrid = HybridClassifier(params);
    %   results = hybrid.trainHybrid(processedData, processedLabel);
    %   [label, score] = hybrid.predictOnline(newData, results.model);
    
    properties (Access = private)
        params              % システム設定パラメータ
        netCNN              % 学習済みCNNネットワーク
        netLSTM             % 学習済みLSTMネットワーク
        isEnabled           % 有効/無効フラグ
        isInitialized       % 初期化フラグ
        useGPU              % GPU使用の有無

        % 学習進捗の追跡用
        trainingHistory     % 学習履歴
        validationHistory   % 検証履歴
        bestValAccuracy     % 最良の検証精度
        patienceCounter     % Early Stopping用カウンター
        currentEpoch        % 現在のエポック

        % 過学習監視用
        overfitMetrics      % 過学習メトリクス

        % コンポーネント
        dataAugmenter       % データ拡張処理
        normalizer          % 正規化処理
        
        % パフォーマンス監視
        gpuMemory           % GPU使用メモリ監視
    end
    
    properties (Access = public)
        performance         % 性能評価指標
    end
    
    methods (Access = public)
        %% コンストラクタ - 初期化処理
        function obj = HybridClassifier(params)
            % HybridClassifierのインスタンスを初期化
            %
            % 入力:
            %   params - 設定パラメータ（getConfig関数から取得）
            
            % 基本パラメータの設定
            obj.params = params;
            obj.isInitialized = false;
            obj.useGPU = params.classifier.hybrid.gpu;
            
            % プロパティの初期化
            obj.initializeProperties();
            
            % コンポーネントの初期化
            obj.dataAugmenter = DataAugmenter(params);
            obj.normalizer = EEGNormalizer(params);
            
            % GPU利用可能性のチェック
            if obj.useGPU
                try
                    gpuInfo = gpuDevice();
                    fprintf('GPUが検出されました: %s (メモリ: %.2f GB)\n', ...
                        gpuInfo.Name, gpuInfo.TotalMemory/1e9);
                catch
                    warning('GPU使用が指定されていますが、GPUが利用できません。CPUで実行します。');
                    obj.useGPU = false;
                end
            end
        end

        %% ハイブリッドモデル学習メソッド
        function results = trainHybrid(obj, processedData, processedLabel)
            % EEGデータを使用してハイブリッドモデルを学習
            %
            % 入力:
            %   processedData - 前処理済みEEGデータ [チャンネル x サンプル x エポック]
            %   processedLabel - クラスラベル [エポック x 1]
            %
            % 出力:
            %   results - 学習結果を含む構造体（モデル、性能評価、正規化パラメータなど）
            try                
                fprintf('\n=== ハイブリッドモデル学習処理を開始 ===\n');
                
                % データの次元を確認し、必要に応じて調整
                [processedData, processInfo] = obj.validateAndPrepareData(processedData);
                
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
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);
                
                if isOverfit
                    fprintf('\n警告: モデルに過学習の兆候が検出されました (%s)\n', obj.overfitMetrics.severity);
                end

                % 性能指標の更新
                obj.updatePerformanceMetrics(testMetrics);
                
                % 結果構造体の構築
                results = obj.buildResultsStruct(hybridModel, testMetrics, trainInfo, normParams);

                % 結果のサマリー表示
                obj.displayResults();
                
                % 使用リソースのクリーンアップ
                obj.resetGPUMemory();
                fprintf('\n=== ハイブリッドモデル学習処理が完了しました ===\n');

            catch ME
                % エラー発生時の詳細情報出力
                fprintf('\n=== ハイブリッドモデル学習中にエラーが発生しました ===\n');
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
                
                % --- CNN学習曲線の分析 ---
                cnnTrainAcc = [];
                cnnValAcc = [];
                
                if isfield(trainInfo, 'cnnHistory') && ...
                   isfield(trainInfo.cnnHistory, 'TrainingAccuracy') && ...
                   isfield(trainInfo.cnnHistory, 'ValidationAccuracy')
                    
                    cnnTrainAcc = trainInfo.cnnHistory.TrainingAccuracy;
                    cnnValAcc = trainInfo.cnnHistory.ValidationAccuracy;
                    
                    % NaN値を除去
                    cnnTrainAcc = cnnTrainAcc(~isnan(cnnTrainAcc));
                    cnnValAcc = cnnValAcc(~isnan(cnnValAcc));
                    
                    % CNN学習曲線の詳細分析
                    [cnnTrainTrend, cnnValTrend] = obj.analyzeLearningCurves(cnnTrainAcc, cnnValAcc);
                    
                    fprintf('\nCNN学習曲線の分析:\n');
                    fprintf('  - 学習精度の平均変化率: %.4f/エポック\n', cnnTrainTrend.mean_change);
                    fprintf('  - 学習精度の変動性: %.4f\n', cnnTrainTrend.volatility);
                    fprintf('  - 検証精度の平均変化率: %.4f/エポック\n', cnnValTrend.mean_change);
                    fprintf('  - 検証精度の変動性: %.4f\n', cnnValTrend.volatility);
                    
                    % 収束エポック
                    if ~isempty(cnnValAcc)
                        [cnnOptimalEpoch, cnnTotalEpochs] = obj.findOptimalEpoch(cnnValAcc);
                        fprintf('  - CNN最適エポック: %d/%d (%.1f%%)\n', ...
                            cnnOptimalEpoch, cnnTotalEpochs, (cnnOptimalEpoch/cnnTotalEpochs)*100);
                    else
                        cnnOptimalEpoch = 0;
                        cnnTotalEpochs = 0;
                    end
                else
                    fprintf('CNN学習曲線データが不足しています\n');
                    cnnTrainTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false);
                    cnnValTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false);
                    cnnOptimalEpoch = 0;
                    cnnTotalEpochs = 0;
                end
                
                % --- LSTM学習曲線の分析 ---
                lstmTrainAcc = [];
                lstmValAcc = [];
                
                if isfield(trainInfo, 'lstmHistory') && ...
                   isfield(trainInfo.lstmHistory, 'TrainingAccuracy') && ...
                   isfield(trainInfo.lstmHistory, 'ValidationAccuracy')
                    
                    lstmTrainAcc = trainInfo.lstmHistory.TrainingAccuracy;
                    lstmValAcc = trainInfo.lstmHistory.ValidationAccuracy;
                    
                    % NaN値を除去
                    lstmTrainAcc = lstmTrainAcc(~isnan(lstmTrainAcc));
                    lstmValAcc = lstmValAcc(~isnan(lstmValAcc));
                    
                    % LSTM学習曲線の詳細分析
                    [lstmTrainTrend, lstmValTrend] = obj.analyzeLearningCurves(lstmTrainAcc, lstmValAcc);
                    
                    fprintf('\nLSTM学習曲線の分析:\n');
                    fprintf('  - 学習精度の平均変化率: %.4f/エポック\n', lstmTrainTrend.mean_change);
                    fprintf('  - 学習精度の変動性: %.4f\n', lstmTrainTrend.volatility);
                    fprintf('  - 検証精度の平均変化率: %.4f/エポック\n', lstmValTrend.mean_change);
                    fprintf('  - 検証精度の変動性: %.4f\n', lstmValTrend.volatility);
                    
                    % 収束エポック
                    if ~isempty(lstmValAcc)
                        [lstmOptimalEpoch, lstmTotalEpochs] = obj.findOptimalEpoch(lstmValAcc);
                        fprintf('  - LSTM最適エポック: %d/%d (%.1f%%)\n', ...
                            lstmOptimalEpoch, lstmTotalEpochs, (lstmOptimalEpoch/lstmTotalEpochs)*100);
                    else
                        lstmOptimalEpoch = 0;
                        lstmTotalEpochs = 0;
                    end
                else
                    fprintf('LSTM学習曲線データが不足しています\n');
                    lstmTrainTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false);
                    lstmValTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false);
                    lstmOptimalEpoch = 0;
                    lstmTotalEpochs = 0;
                end
                
                % --- ハイブリッドモデル全体の検証-テスト精度ギャップ分析 ---
                if isfield(trainInfo, 'hybridValMetrics')
                    hybridValAcc = trainInfo.hybridValMetrics.accuracy * 100; % パーセントに変換
                    testAcc = testMetrics.accuracy * 100;  % パーセントに変換
                    
                    % データサイズの取得（混同行列のサイズから推定）
                    if isfield(testMetrics, 'confusionMat')
                        dataSize = sum(sum(testMetrics.confusionMat));
                    else
                        dataSize = 100; % デフォルト値
                    end
                    
                    % ギャップ分析を実行
                    [gapOverfit, gapMetrics] = obj.validateTestValidationGap(hybridValAcc, testAcc, dataSize);
                else
                    % CNN検証精度とLSTM検証精度を平均化
                    if ~isempty(cnnValAcc) && ~isempty(lstmValAcc)
                        meanCnnValAcc = mean(cnnValAcc(max(1, end-5):end)) * 100; % 最後の5エポックの平均
                        meanLstmValAcc = mean(lstmValAcc(max(1, end-5):end)) * 100;
                        hybridValAcc = (meanCnnValAcc + meanLstmValAcc) / 2;
                    elseif ~isempty(cnnValAcc)
                        hybridValAcc = mean(cnnValAcc(max(1, end-5):end)) * 100;
                    elseif ~isempty(lstmValAcc)
                        hybridValAcc = mean(lstmValAcc(max(1, end-5):end)) * 100;
                    else
                        hybridValAcc = 0;
                    end
                    
                    testAcc = testMetrics.accuracy * 100;
                    
                    % データサイズ
                    if isfield(testMetrics, 'confusionMat')
                        dataSize = sum(sum(testMetrics.confusionMat));
                    else
                        dataSize = 100; % デフォルト値
                    end
                    
                    if hybridValAcc > 0
                        [gapOverfit, gapMetrics] = obj.validateTestValidationGap(hybridValAcc, testAcc, dataSize);
                    else
                        gapOverfit = false;
                        gapMetrics = struct('rawGap', 0, 'normalizedGap', 0, 'adjustedGap', 0, ...
                            'meanValAcc', 0, 'stdValAcc', 0, 'testAcc', testAcc, 'severity', 'unknown');
                    end
                end
                
                % バイアスの検出（特定クラスへの偏り）
                isCompletelyBiased = obj.detectClassificationBias(testMetrics);
                
                % 学習進行の評価
                isLearningProgressing = (cnnTrainTrend.mean_change > 0.001) || (lstmTrainTrend.mean_change > 0.001) || ...
                                       (cnnValTrend.mean_change > 0.001) || (lstmValTrend.mean_change > 0.001);
                
                % Early Stoppingの効果分析（CNNとLSTMの平均）
                cnnEarlyStoppingEffect = struct('optimal_epoch', cnnOptimalEpoch, 'total_epochs', cnnTotalEpochs);
                lstmEarlyStoppingEffect = struct('optimal_epoch', lstmOptimalEpoch, 'total_epochs', lstmTotalEpochs);
                
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
                    'lstmEarlyStoppingEffect', lstmEarlyStoppingEffect ...
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
        function [label, score] = predictOnline(obj, data, hybrid)
            % 学習済みモデルを使用して新しいEEGデータを分類
            %
            % 入力:
            %   data - 分類するEEGデータ [チャンネル x サンプル]
            %   hybridModel - 学習済みハイブリッドモデル
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
                cnnFeatures = activations(hybrid.netCNN, cnnData, 'fc_cnn', 'OutputAs', 'rows');
                
                % LSTMからの特徴抽出
                if iscell(lstmData)
                    lstmFeatures = [];
                    for i = 1:length(lstmData)
                        lstmOut = predict(hybrid.netLSTM, lstmData{i}, 'MiniBatchSize', 1);
                        
                        % 最初のサンプルで配列を初期化
                        if i == 1
                            lstmFeatures = zeros(length(lstmData), size(lstmOut, 2));
                        end
                        
                        lstmFeatures(i,:) = lstmOut;
                    end
                else
                    lstmOut = predict(hybrid.netLSTM, lstmData, 'MiniBatchSize', 1);
                    lstmFeatures = lstmOut;
                end
                
                % 特徴を結合
                combinedFeatures = [cnnFeatures, lstmFeatures];
                
                % AdaBoostで最終予測
                [label, score] = predict(hybrid.adaModel, combinedFeatures);
        
            catch ME
                fprintf('ハイブリッドモデルのオンライン予測でエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
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
        
        %% データ検証と準備メソッド
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
                fprintf('\nデータ正規化を実行...\n');
                [procTrainData, normParams] = obj.normalizer.normalize(procTrainData);
                
                % 正規化パラメータの検証
                obj.validateNormalizationParams(normParams);
                fprintf('  - 正規化方法: %s\n', normParams.method);
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
                % データ次元数の確認
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
                fprintf('CNN用データ形式変換でエラーが発生: %s\n', ME.message);
                fprintf('入力データサイズ: [%s]\n', num2str(size(data)));
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% データのLSTM形式への変換
        function preparedData = prepareDataForLSTM(obj, data)
            % データをLSTMに適した形式に変換
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
                    currentData = obj.interpolateInvalidValues(currentData, 1);
                    
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
        
        %% 無効値の補間処理
        function processedData = interpolateInvalidValues(~, data, trialIndex)
            % NaN/Infなどの無効値を線形補間で処理
            % MATLABのLSTMでは、データは [特徴量 × 時間ステップ] の形式
            
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

        %% ハイブリッドモデル学習メソッド
        function [hybridModel, trainInfo] = trainHybridModel(obj, cnnTrainData, lstmTrainData, trainLabels, cnnValData, lstmValData, valLabels)
            try        
                fprintf('\n=== ハイブリッドモデルの学習開始 ===\n');
                
                % GPUメモリの確認
                obj.checkGPUMemory();
                
                % データの構造確認と検証
                if ~isempty(lstmTrainData) && iscell(lstmTrainData) && ~isempty(lstmTrainData{1})
                    sampleData = lstmTrainData{1};
                    [numFeatures, ~] = size(sampleData);
                else
                    error('不正なLSTM学習データ形式');
                end
        
                % ラベルのカテゴリカル変換
                uniqueLabels = unique(trainLabels);
                trainLabels = categorical(trainLabels, uniqueLabels);
        
                % 検証データの処理
                valLabels = categorical(valLabels, uniqueLabels);
                
                % --- CNNモデル学習 ---
                fprintf('\n--- CNNモデル学習開始 ---\n');
                
                % CNNアーキテクチャの構築
                cnnArchitecture = obj.params.classifier.hybrid.architecture.cnn;
                cnnLayers = obj.buildCNNLayers(cnnTrainData, cnnArchitecture);
                
                % CNN学習オプションの設定
                cnnTrainOptions = obj.getCNNTrainingOptions(cnnValData, valLabels);
                
                % CNNモデルの学習
                [cnnModel, cnnTrainInfo] = trainNetwork(cnnTrainData, trainLabels, cnnLayers, cnnTrainOptions);
                fprintf('CNNモデル学習完了\n');
                
                % --- LSTMモデル学習 ---
                fprintf('\n--- LSTMモデル学習開始 ---\n');
                
                % LSTMアーキテクチャの構築 - ここで正しい特徴量次元を渡す
                lstmLayers = obj.buildLSTMLayers(numFeatures);
                
                % LSTM学習オプションの設定
                lstmTrainOptions = obj.getLSTMTrainingOptions(lstmValData, valLabels);
                
                % LSTMモデルの学習
                [lstmModel, lstmTrainInfo] = trainNetwork(lstmTrainData, trainLabels, lstmLayers, lstmTrainOptions);
                fprintf('LSTMモデル学習完了\n');
                
                % --- 特徴抽出と統合モデルの学習 ---
                fprintf('\n--- 特徴統合と最終分類器の学習 ---\n');
                
                % CNNからの特徴抽出
                cnnFeatures = activations(cnnModel, cnnTrainData, 'fc_cnn', 'OutputAs', 'rows');
                
                % LSTMからの特徴抽出 - 動的にサイズを決定
                lstmFeatures = [];
                for i = 1:length(lstmTrainData)
                    % LSTM予測実行
                    lstmOut = predict(lstmModel, lstmTrainData{i}, 'MiniBatchSize', 1);
                    
                    % 最初のサンプルで配列を初期化
                    if i == 1
                        lstmFeatureSize = size(lstmOut, 2);
                        lstmFeatures = zeros(length(lstmTrainData), lstmFeatureSize);
                    end
                    
                    % 特徴の格納
                    lstmFeatures(i,:) = lstmOut;
                end
                
                % 特徴の統合
                combinedFeatures = [cnnFeatures, lstmFeatures];
                
                % AdaBoost分類器のパラメータ設定
                adaParams = obj.params.classifier.hybrid.adaBoost;
                
                % クラス数を確認
                uniqueLabels = unique(trainLabels);
                numClasses = length(uniqueLabels);
                
                % クラス数に応じてアルゴリズムを選択
                fprintf('%dクラス分類のために AdaBoostM2 を使用します\n', numClasses);
                
                % AdaBoostM2 分類器の学習
                adaModel = fitcensemble(combinedFeatures, trainLabels, ...
                    'Method', 'AdaBoostM2', ...
                    'NumLearningCycles', adaParams.numLearners, ...
                    'Learners', 'tree', ...
                    'LearnRate', adaParams.learnRate);
                
                fprintf('AdaBoost 統合分類器の学習完了\n');
                
                % --- 検証データでのハイブリッドモデル全体の評価 ---
                fprintf('\n--- 検証データでのハイブリッドモデル評価 ---\n');
                
                % 検証データからの特徴抽出 (CNN)
                cnnValFeatures = activations(cnnModel, cnnValData, 'fc_cnn', 'OutputAs', 'rows');
                
                % 検証データからの特徴抽出 (LSTM)
                lstmValFeatures = [];
                for i = 1:length(lstmValData)
                    lstmOut = predict(lstmModel, lstmValData{i}, 'MiniBatchSize', 1);
                    
                    % 最初のサンプルで配列を初期化
                    if i == 1
                        lstmValFeatures = zeros(length(lstmValData), size(lstmOut, 2));
                    end
                    
                    % 特徴の格納
                    lstmValFeatures(i,:) = lstmOut;
                end
                
                % 特徴の統合 (検証データ)
                combinedValFeatures = [cnnValFeatures, lstmValFeatures];
                
                % 検証データでの予測
                [valPred, valScores] = predict(adaModel, combinedValFeatures);
                
                % 検証精度の計算
                hybridValAccuracy = mean(valPred == valLabels) * 100;  % パーセントに変換
                fprintf('検証精度: %.2f%%\n', hybridValAccuracy);
                
                % クラスごとの評価指標 (検証データ)
                classes = unique(valLabels);
                classwise_val = struct('precision', zeros(1,length(classes)), ...
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
                    classwise_val(i).precision = precision;
                    classwise_val(i).recall = recall;
                    classwise_val(i).f1score = f1score;
                    
                    fprintf('  - クラス %d:\n', classes(i));
                    fprintf('    - 精度 (Precision): %.2f%%\n', precision * 100);
                    fprintf('    - 再現率 (Recall): %.2f%%\n', recall * 100);
                    fprintf('    - F1スコア: %.2f\n', f1score);
                end
                
                % 検証混同行列の計算
                valConfMat = confusionmat(valLabels, valPred);
                fprintf('\n検証データの混同行列:\n');
                disp(valConfMat);
                
                % 検証結果の詳細構造体
                hybridValMetrics = struct(...
                    'accuracy', hybridValAccuracy / 100, ...  % 比率に戻す
                    'prediction', valPred, ...
                    'score', valScores, ...
                    'confusionMat', valConfMat, ...
                    'classwise', classwise_val);
                
                % ハイブリッドモデル情報を構築
                hybridModel = struct(...
                    'netCNN', cnnModel, ...
                    'netLSTM', lstmModel, ...
                    'adaModel', adaModel, ...
                    'lstmFeatureSize', lstmFeatureSize);
                
                % 学習履歴情報の構築
                trainInfo = struct(...
                    'cnnHistory', cnnTrainInfo, ...
                    'lstmHistory', lstmTrainInfo, ...
                    'hybridValAccuracy', hybridValAccuracy, ...
                    'hybridValMetrics', hybridValMetrics, ...
                    'FinalEpoch', max(length(cnnTrainInfo.TrainingLoss), length(lstmTrainInfo.TrainingLoss)));
                
                fprintf('\n=== ハイブリッドモデル学習が完了しました ===\n');
                
            catch ME
                fprintf('\n=== ハイブリッドモデル学習中にエラーが発生 ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                obj.resetGPUMemory();
                rethrow(ME);
            end
        end

        %% CNNレイヤー構築メソッド
        function layers = buildCNNLayers(obj, data, architecture)
            % ハイブリッドモデルのCNN部分のアーキテクチャを構築
            
            fprintf('CNNアーキテクチャを構築中...\n');
            
            % 入力サイズの決定
            inputSize = size(data);
            if length(inputSize) >= 4
                layerInputSize = inputSize(1:3);  % [サンプル数, チャンネル数, 1]
            else
                error('CNNデータの形式が不正です');
            end
            
            % 入力層
            layers = [
                imageInputLayer(layerInputSize, 'Name', 'input', 'Normalization', 'none')
            ];
            
            % 畳み込み層の追加
            convFields = fieldnames(architecture.convLayers);
            for i = 1:length(convFields)
                convName = convFields{i};
                convParams = architecture.convLayers.(convName);
                
                % 畳み込み層
                layers = [layers
                    convolution2dLayer(...
                        convParams.size, convParams.filters, ...
                        'Stride', convParams.stride, ...
                        'Padding', convParams.padding, ...
                        'Name', convName)
                ];
                
                % バッチ正規化層（設定に応じて）
                if architecture.batchNorm
                    layers = [layers
                        batchNormalizationLayer('Name', ['bn_' convName])
                    ];
                end
                
                % 活性化関数
                layers = [layers
                    reluLayer('Name', ['relu_' convName])
                ];
                
                % プーリング層
                poolName = ['pool' num2str(i)];
                if isfield(architecture.poolLayers, poolName)
                    poolParams = architecture.poolLayers.(poolName);
                    layers = [layers
                        maxPooling2dLayer(...
                            poolParams.size, 'Stride', poolParams.stride, ...
                            'Name', poolName)
                    ];
                end
                
                % ドロップアウト層
                dropoutName = ['dropout' num2str(i)];
                if isfield(architecture.dropoutLayers, dropoutName)
                    dropoutRate = architecture.dropoutLayers.(dropoutName);
                    layers = [layers
                        dropoutLayer(dropoutRate, 'Name', dropoutName)
                    ];
                end
            end
            
            % 全結合層（特徴出力用）
            layers = [layers
                globalAveragePooling2dLayer('Name', 'gap')
                fullyConnectedLayer(architecture.fullyConnected, 'Name', 'fc_cnn')
                reluLayer('Name', 'relu_fc')
            ];
            
            % 出力層
            layers = [layers
                fullyConnectedLayer(obj.params.classifier.hybrid.architecture.numClasses, 'Name', 'fc_output')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];
            
            fprintf('CNNアーキテクチャ構築完了: %dレイヤー\n', length(layers));
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
                
                fprintf('LSTMレイヤー構築: 入力特徴量=%d, クラス数=%d\n', inputSize, numClasses);
                
                % レイヤー数の計算 - 修正: arch.lstm を参照
                numLSTMLayers = length(fieldnames(arch.lstm.lstmLayers));
                numFCLayers = length(arch.lstm.fullyConnected);
                
                % セル配列でレイヤーを構築
                layers = cell(1, numLSTMLayers * 3 + numFCLayers * 2 + 3);
                layerIdx = 1;
                
                % 入力層 - ここで正しい入力サイズを指定することが重要
                layers{layerIdx} = sequenceInputLayer(inputSize, ...
                    'Normalization', 'none', ...
                    'Name', 'input');
                layerIdx = layerIdx + 1;
                
                % LSTM層、バッチ正規化、ドロップアウト層の追加 - 修正: arch.lstm を参照
                lstmLayerNames = fieldnames(arch.lstm.lstmLayers);
                for i = 1:length(lstmLayerNames)
                    lstmParams = arch.lstm.lstmLayers.(lstmLayerNames{i});
                    
                    % ドロップアウト率の取得 - 修正: arch.lstm を参照
                    if isfield(arch.lstm.dropoutLayers, ['dropout' num2str(i)])
                        dropoutRate = arch.lstm.dropoutLayers.(['dropout' num2str(i)]);
                    else
                        dropoutRate = 0.5;  % デフォルト値
                    end
                    
                    % LSTM層の追加
                    layers{layerIdx} = lstmLayer(lstmParams.numHiddenUnits, ...
                        'OutputMode', lstmParams.OutputMode, ...
                        'Name', ['lstm' num2str(i)]);
                    layerIdx = layerIdx + 1;
                    
                    % バッチ正規化層（オプション） - 修正: arch.lstm を参照
                    if arch.lstm.batchNorm
                        layers{layerIdx} = batchNormalizationLayer('Name', ['bn' num2str(i)]);
                        layerIdx = layerIdx + 1;
                    end
                    
                    % ドロップアウト層
                    layers{layerIdx} = dropoutLayer(dropoutRate, 'Name', ['dropout' num2str(i)]);
                    layerIdx = layerIdx + 1;
                end
                
                % 全結合層とReLU層の追加 - 修正: arch.lstm を参照
                for i = 1:length(arch.lstm.fullyConnected)
                    layers{layerIdx} = fullyConnectedLayer(arch.lstm.fullyConnected(i), ...
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

        %% CNNトレーニングオプション設定メソッド
        function options = getCNNTrainingOptions(obj, valData, valLabels)
            % CNNのトレーニングオプションを設定
            
            % 実行環境の選択
            executionEnvironment = 'cpu';
            if obj.useGPU
                executionEnvironment = 'gpu';
            end
            
            % 検証データの準備
            valLabels_cat = categorical(valLabels, unique(valLabels));
            valDS = {valData, valLabels_cat};
            
            % トレーニングオプションの設定
            options = trainingOptions(obj.params.classifier.hybrid.training.optimizer.type, ...
                'InitialLearnRate', obj.params.classifier.hybrid.training.optimizer.learningRate, ...
                'MaxEpochs', obj.params.classifier.hybrid.training.maxEpochs, ...
                'MiniBatchSize', obj.params.classifier.hybrid.training.miniBatchSize, ...
                'Plots', 'none', ...
                'Shuffle', obj.params.classifier.hybrid.training.shuffle, ...
                'ExecutionEnvironment', executionEnvironment, ...
                'OutputNetwork', 'best-validation', ...
                'Verbose', true, ...
                'ValidationData', valDS, ...
                'ValidationFrequency', obj.params.classifier.hybrid.training.frequency, ...
                'ValidationPatience', obj.params.classifier.hybrid.training.patience, ...
                'GradientThreshold', 1);
        end

        %% LSTMトレーニングオプション設定メソッド
        function options = getLSTMTrainingOptions(obj, valData, valLabels)
            % LSTMのトレーニングオプションを設定
            
            % 実行環境の選択
            executionEnvironment = 'cpu';
            if obj.useGPU
                executionEnvironment = 'gpu';
            end
            
            % 検証データの準備
            valLabels_cat = categorical(valLabels, unique(valLabels));
            valDS = {valData, valLabels_cat};
            
            % トレーニングオプションの設定
            options = trainingOptions(obj.params.classifier.hybrid.training.optimizer.type, ...
                'InitialLearnRate', obj.params.classifier.hybrid.training.optimizer.learningRate, ...
                'MaxEpochs', obj.params.classifier.hybrid.training.maxEpochs, ...
                'MiniBatchSize', obj.params.classifier.hybrid.training.miniBatchSize, ...
                'Plots', 'none', ...
                'Shuffle', obj.params.classifier.hybrid.training.shuffle, ...
                'ExecutionEnvironment', executionEnvironment, ...
                'OutputNetwork', 'best-validation', ...
                'Verbose', true, ...
                'ValidationData', valDS, ...
                'ValidationFrequency', obj.params.classifier.hybrid.training.frequency, ...
                'ValidationPatience', obj.params.classifier.hybrid.training.patience, ...
                'GradientThreshold', obj.params.classifier.hybrid.training.optimizer.gradientThreshold);
        end

        %% モデル評価メソッド
        function metrics = evaluateModel(~, model, cnnTestData, lstmTestData, testLabels)
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
            
            fprintf('\n=== モデル評価を実行 ===\n');
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
                lstmFeatures = [];
                for i = 1:length(lstmTestData)
                    lstmOut = predict(model.netLSTM, lstmTestData{i}, 'MiniBatchSize', 1);
                    
                    % 最初のサンプルで配列を初期化
                    if i == 1
                        lstmFeatures = zeros(length(lstmTestData), size(lstmOut, 2));
                    end
                    
                    lstmFeatures(i,:) = lstmOut;
                end
                
                % 特徴の統合
                combinedFeatures = [cnnFeatures, lstmFeatures];
                
                % AdaBoostによる予測
                [pred, score] = predict(model.adaModel, combinedFeatures);
                metrics.score = score;
                
                % テストラベルをcategorical型に変換 - ここを修正
                testLabels_cat = categorical(testLabels);
                
                % 基本的な指標の計算 - ここを修正
                metrics.accuracy = mean(pred == testLabels_cat);
                metrics.confusionMat = confusionmat(testLabels_cat, pred);
                
                fprintf('テスト精度: %.2f%%\n', metrics.accuracy * 100);
                
                % クラスごとの性能評価
                classes = unique(testLabels_cat);
                metrics.classwise = struct('precision', zeros(1,length(classes)), ...
                                        'recall', zeros(1,length(classes)), ...
                                        'f1score', zeros(1,length(classes)));
                
                fprintf('\nクラスごとの評価:\n');
                for i = 1:length(classes)
                    className = classes(i);
                    classIdx = (testLabels_cat == className);
                    
                    % 各クラスの指標計算
                    TP = sum(pred(classIdx) == className);
                    FP = sum(pred == className) - TP;
                    FN = sum(classIdx) - TP;
                    
                    % 0による除算を回避
                    precision = 0;
                    recall = 0;
                    f1 = 0;
                    
                    if (TP + FP) > 0
                        precision = TP / (TP + FP);
                    end
                    
                    if (TP + FN) > 0
                        recall = TP / (TP + FN);
                    end
                    
                    if (precision + recall) > 0
                        f1 = 2 * (precision * recall) / (precision + recall);
                    end
                    
                    metrics.classwise(i).precision = precision;
                    metrics.classwise(i).recall = recall;
                    metrics.classwise(i).f1score = f1;
                    
                    fprintf('  - クラス %d:\n', className);
                    fprintf('    - 精度 (Precision): %.2f%%\n', precision * 100);
                    fprintf('    - 再現率 (Recall): %.2f%%\n', recall * 100);
                    fprintf('    - F1スコア: %.2f\n', f1);
                end
                
                % 2クラス分類の場合のROC曲線とAUC
                if length(classes) == 2
                    [X, Y, T, AUC] = perfcurve(testLabels_cat, score(:,2), classes(2));
                    metrics.roc = struct('X', X, 'Y', Y, 'T', T);
                    metrics.auc = AUC;
                    fprintf('\nAUC: %.3f\n', AUC);
                end
                
                % 混同行列の表示
                fprintf('\n混同行列:\n');
                disp(metrics.confusionMat);
                
            catch ME
                fprintf('モデル評価でエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
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
            
            % データの検証
            if isempty(trainAcc) || isempty(valAcc)
                trainTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false, 'oscillation_strength', 0);
                valTrend = struct('mean_change', 0, 'volatility', 0, 'increasing_ratio', 0, 'plateau_detected', false, 'oscillation_strength', 0);
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
        
        %% 分類バイアス検出メソッド
        function isCompletelyBiased = detectClassificationBias(~, testMetrics)
            % 混同行列から分類バイアスを検出
            
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
            if iscell(valAccHistory) || isvector(valAccHistory) && length(valAccHistory) > 1
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
            else
                % 単一値の場合
                meanValAcc = valAccHistory;
                stdValAcc = 0.01 * meanValAcc; % 仮の標準偏差として平均値の1%を使用
            end
            
            % ゼロ除算防止
            if stdValAcc < 0.001
                stdValAcc = 0.001; 
            end
            
            % スケールされたギャップ計算（データサイズで調整）
            % 小さいデータセットでは大きなギャップが生じやすい
            scaleFactor = min(1, sqrt(dataSize / 1000));  % データサイズに基づく調整
            
            % 正規化された差（z-スコア的アプローチ）
            rawGap = abs(meanValAcc - testAcc*100);
            normalizedGap = rawGap / max(stdValAcc, 1);
            
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
                'rawGap', rawGap, ...
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
        
        %% トレーニング情報検証メソッド
        function validateTrainInfo(~, trainInfo)
            % trainInfo構造体の妥当性を検証
            
            if ~isstruct(trainInfo)
                warning('trainInfoが構造体ではありません');
                return;
            end
            
            % 必須フィールドの確認
            requiredFields = {'cnnHistory', 'lstmHistory'};
            for i = 1:length(requiredFields)
                if ~isfield(trainInfo, requiredFields{i})
                    warning('trainInfoに必須フィールド「%s」がありません', requiredFields{i});
                end
            end
            
            % CNN History構造体の検証
            if isfield(trainInfo, 'cnnHistory')
                historyFields = {'TrainingLoss', 'ValidationLoss', 'TrainingAccuracy', 'ValidationAccuracy'};
                for i = 1:length(historyFields)
                    if ~isfield(trainInfo.cnnHistory, historyFields{i})
                        warning('cnnHistory構造体に「%s」フィールドがありません', historyFields{i});
                    elseif isempty(trainInfo.cnnHistory.(historyFields{i}))
                        warning('cnnHistory構造体の「%s」フィールドが空です', historyFields{i});
                    end
                end
            end
            
            % LSTM History構造体の検証
            if isfield(trainInfo, 'lstmHistory')
                historyFields = {'TrainingLoss', 'ValidationLoss', 'TrainingAccuracy', 'ValidationAccuracy'};
                for i = 1:length(historyFields)
                    if ~isfield(trainInfo.lstmHistory, historyFields{i})
                        warning('lstmHistory構造体に「%s」フィールドがありません', historyFields{i});
                    elseif isempty(trainInfo.lstmHistory.(historyFields{i}))
                        warning('lstmHistory構造体の「%s」フィールドが空です', historyFields{i});
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
        
        %% フォールバック過学習メトリクス作成メソッド
        function metrics = createFallbackOverfitMetrics(~)
            % エラー発生時のフォールバックメトリクスを作成
            
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
                        if obj.params.classifier.hybrid.training.miniBatchSize > 32
                            newBatchSize = max(16, floor(obj.params.classifier.hybrid.training.miniBatchSize / 2));
                            fprintf('高メモリ使用率のため、バッチサイズを%dから%dに削減します\n', ...
                                obj.params.classifier.hybrid.training.miniBatchSize, newBatchSize);
                            obj.params.classifier.hybrid.training.miniBatchSize = newBatchSize;
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
        
        %% 性能メトリクス更新メソッド
        function updatePerformanceMetrics(obj, testMetrics)
            % 評価結果から性能メトリクスを更新
            %
            % 入力:
            %   testMetrics - テストデータでの評価結果
            
            fprintf('\n=== 性能メトリクスの更新 ===\n');
            
            % メトリクスの更新
            obj.performance = testMetrics;
            
            % 基本的な指標を表示
            fprintf('更新された性能指標:\n');
            fprintf('  - 全体精度: %.2f%%\n', testMetrics.accuracy * 100);
            
            if isfield(testMetrics, 'auc') && ~isempty(testMetrics.auc)
                fprintf('  - AUC: %.3f\n', testMetrics.auc);
            end
            
            % クラスごとの指標の表示
            if isfield(testMetrics, 'classwise') && ~isempty(testMetrics.classwise)
                fprintf('クラスごとの性能指標が更新されました\n');
            end
        end
        
        %% 結果表示メソッド
        function displayResults(obj)
            % 総合的な結果サマリーの表示
            
            try
                fprintf('\n=== ハイブリッド分類結果サマリー ===\n');
                
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

                % モデル構造情報
                fprintf('\nモデル構造情報:\n');
                fprintf('  - CNN部分: ');
                if isfield(obj.params.classifier.hybrid.architecture.cnn, 'convLayers')
                    numCnnLayers = length(fieldnames(obj.params.classifier.hybrid.architecture.cnn.convLayers));
                    fprintf('%d層の畳み込みネットワーク\n', numCnnLayers);
                else
                    fprintf('情報なし\n');
                end
                
                fprintf('  - LSTM部分: ');
                if isfield(obj.params.classifier.hybrid.architecture.lstm, 'lstmLayers')
                    numLstmLayers = length(fieldnames(obj.params.classifier.hybrid.architecture.lstm.lstmLayers));
                    fprintf('%d層のリカレントネットワーク\n', numLstmLayers);
                else
                    fprintf('情報なし\n');
                end

                % 過学習分析結果の表示
                if ~isempty(obj.overfitMetrics) && isstruct(obj.overfitMetrics)
                    fprintf('\n過学習分析:\n');
                    fprintf('  - 性能ギャップ: %.2f%%\n', obj.overfitMetrics.performanceGap);
                    fprintf('  - 重大度: %s\n', obj.overfitMetrics.severity);
                    
                    if isfield(obj.overfitMetrics, 'isCompletelyBiased')
                        fprintf('  - バイアス検出: %s\n', string(obj.overfitMetrics.isCompletelyBiased));
                    end
                    
                    fprintf('\n学習カーブ分析:\n');
                    
                    % CNN検証傾向
                    if isfield(obj.overfitMetrics, 'cnnValidationTrend')
                        trend = obj.overfitMetrics.cnnValidationTrend;
                        fprintf('  - CNN検証平均変化率: %.4f\n', trend.mean_change);
                        fprintf('  - CNN検証変動性: %.4f\n', trend.volatility);
                    end
                    
                    % LSTM検証傾向
                    if isfield(obj.overfitMetrics, 'lstmValidationTrend')
                        trend = obj.overfitMetrics.lstmValidationTrend;
                        fprintf('  - LSTM検証平均変化率: %.4f\n', trend.mean_change);
                        fprintf('  - LSTM検証変動性: %.4f\n', trend.volatility);
                    end
                    
                    % 収束エポック情報
                    if isfield(obj.overfitMetrics, 'cnnOptimalEpoch') && ...
                       isfield(obj.overfitMetrics, 'cnnTotalEpochs')
                        fprintf('  - CNN最適エポック: %d/%d (%.2f%%)\n', ...
                            obj.overfitMetrics.cnnOptimalEpoch, ...
                            obj.overfitMetrics.cnnTotalEpochs, ...
                            (obj.overfitMetrics.cnnOptimalEpoch / ...
                             max(obj.overfitMetrics.cnnTotalEpochs, 1)) * 100);
                    end
                    
                    if isfield(obj.overfitMetrics, 'lstmOptimalEpoch') && ...
                       isfield(obj.overfitMetrics, 'lstmTotalEpochs')
                        fprintf('  - LSTM最適エポック: %d/%d (%.2f%%)\n', ...
                            obj.overfitMetrics.lstmOptimalEpoch, ...
                            obj.overfitMetrics.lstmTotalEpochs, ...
                            (obj.overfitMetrics.lstmOptimalEpoch / ...
                             max(obj.overfitMetrics.lstmTotalEpochs, 1)) * 100);
                    end
                end
                
                % GPU使用状況
                if obj.useGPU && obj.gpuMemory.peak > 0
                    fprintf('\nGPU使用状況:\n');
                    fprintf('  - 最大使用メモリ: %.2f GB\n', obj.gpuMemory.peak);
                    fprintf('  - 総メモリ: %.2f GB\n', obj.gpuMemory.total);
                    fprintf('  - 使用率: %.1f%%\n', (obj.gpuMemory.peak / max(obj.gpuMemory.total, 1)) * 100);
                end

            catch ME
                fprintf('結果表示でエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
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
                
                % 結果構造体の構築
                results = struct(...
                    'model', hybridModel, ...
                    'performance', metrics, ...
                    'trainInfo', trainInfo, ...
                    'overfitting', obj.overfitMetrics, ...
                    'normParams', normParams ...
                );
                
            catch ME
                fprintf('結果構造体の構築中にエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                
                % エラー時でも最低限の結果を返す
                results = struct(...
                    'model', hybridModel, ...
                    'performance', metrics, ...
                    'trainInfo', trainInfo, ...
                    'overfitting', struct('severity', 'unknown'), ...
                    'normParams', normParams ...
                );
            end
        end
    end
end