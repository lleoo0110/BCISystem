classdef LSTMClassifier < handle
    properties (Access = private)
        params              % パラメータ設定（構成情報）
        net                 % LSTMネットワーク
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
        dataAugmenter
        normalizer
    end

    properties (Access = public)
        performance         % 性能評価指標
    end

    methods (Access = public)
        %% コンストラクタ
        function obj = LSTMClassifier(params)
            obj.params = params;
            obj.isEnabled = params.classifier.lstm.enable;
            obj.isInitialized = false;
            obj.trainingHistory = struct('loss', [], 'accuracy', []);
            obj.validationHistory = struct('loss', [], 'accuracy', []);
            obj.overfitMetrics = struct();
            obj.useGPU = params.classifier.lstm.gpu;
            obj.dataAugmenter = DataAugmenter(params);
            obj.normalizer = EEGNormalizer(params);
        end

        %% LSTMの学習開始
        function results = trainLSTM(obj, processedData, processedLabel)
            if ~obj.isEnabled
                error('LSTM is disabled in configuration');
            end

            try
                fprintf('\n=== Starting LSTM Training ===\n');

                % データ分割
                [trainData, trainLabels, valData, valLabels, testData, testLabels] = ...
                    obj.splitDataset(processedData, processedLabel);

                % 学習データのみ拡張
                if obj.params.signal.preprocessing.augmentation.enable
                    [trainData, trainLabels, ~] = obj.dataAugmenter.augmentData(trainData, trainLabels);
                    fprintf('訓練データを拡張しました:\n');
                    fprintf('  訓練データ: %d サンプル\n', length(trainData));
                end

                % 正規化
                if obj.params.signal.preprocessing.normalize.enable
                    [trainData, normParams] = obj.normalizer.normalize(trainData);
                end

                % 検証データと評価データにも同じ正規化パラメータで正規化
                valData = obj.normalizer.normalizeOnline(valData, normParams);
                testData = obj.normalizer.normalizeOnline(testData, normParams);

                % データ前処理
                prepTrainData = obj.prepareDataForLSTM(trainData);
                prepValData   = obj.prepareDataForLSTM(valData);
                prepTestData  = obj.prepareDataForLSTM(testData);

                % モデルの学習
                [lstmModel, trainInfo] = obj.trainLSTMModel(prepTrainData, trainLabels, prepValData, valLabels);

                % テストデータでの評価
                testMetrics = obj.evaluateModel(lstmModel, prepTestData, testLabels);

                % 過学習の検証
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);
                if isOverfit
                    warning('Overfitting detected: %s severity', obj.overfitMetrics.severity);
                end

                % 性能指標の更新
                obj.updatePerformanceMetrics(testMetrics);

                % 交差検証の実施
                crossValidationResults = struct('meanAccuracy', [], 'stdAccuracy', []);
                if obj.params.classifier.lstm.training.validation.enable
                    crossValidationResults = obj.performCrossValidation(processedData, processedLabel);
                end

                % aucフィールドが存在しなければ空配列を設定
                aucValue = [];
                if isfield(testMetrics, 'auc')
                    aucValue = testMetrics.auc;
                end

                % 結果構造体の構築
                results = struct(...
                    'model', lstmModel, ...
                    'performance', struct(...
                        'overallAccuracy', testMetrics.accuracy, ...
                        'crossValidation', struct(...
                            'accuracy', crossValidationResults.meanAccuracy, ...
                            'std', crossValidationResults.stdAccuracy), ...
                        'precision', testMetrics.classwise(1).precision, ...
                        'recall', testMetrics.classwise(1).recall, ...
                        'f1score', testMetrics.classwise(1).f1score, ...
                        'auc', aucValue, ...
                        'confusionMatrix', testMetrics.confusionMat), ...
                    'trainInfo', trainInfo, ...
                    'overfitting', obj.overfitMetrics, ...
                    'normParams', normParams ...
                );

                obj.displayResults();

                % GPU使用時はGPUメモリをリセット
                if obj.useGPU
                    reset(gpuDevice);
                end

            catch ME
                fprintf('\n=== Error in LSTM Training ===\n');
                fprintf('Error message: %s\n', ME.message);
                disp(getReport(ME, 'extended'));

                if obj.useGPU
                    reset(gpuDevice);
                end

                rethrow(ME);
            end
        end

        function [label, score] = predictOnline(obj, data, lstmModel)
            if ~obj.isEnabled
                error('LSTM is disabled');
            end

            try
                % データの整形（時系列データ用）
                prepData = obj.prepareDataForLSTM(data);
                
                % モデルの存在確認
                if isempty(lstmModel)
                    error('LSTM model is not available');
                end
                
                % 予測の実行
                [label, scores] = classify(lstmModel, prepData);
                
                % クラス1（安静状態）の確率を取得
                score = scores(:,1);
            catch ME
                fprintf('Error in LSTM online prediction: %s\n', ME.message);
                fprintf('Error details:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% LSTM層の構築
        function layers = buildLSTMLayers(obj, inputSize)
            % アーキテクチャパラメータの取得
            arch = obj.params.classifier.lstm.architecture;
            fprintf('Building LSTM layers with input size: %d\n', inputSize);

            % レイヤー数の計算（大まかな目安）
            numLSTMLayers = length(fieldnames(arch.lstmLayers));
            numFCLayers = length(arch.fullyConnected);
            totalLayers = numLSTMLayers * 3 + numFCLayers * 2 + 4;

            % セル配列でレイヤーを構築
            layers = cell(totalLayers, 1);
            layerIdx = 1;

            % 入力層
            layers{layerIdx} = sequenceInputLayer(inputSize, ...
                'Normalization', arch.sequenceInputLayer.normalization, ...
                'Name', 'input');
            layerIdx = layerIdx + 1;

            % LSTM層、バッチ正規化、ドロップアウト層の追加
            lstmLayerNames = fieldnames(arch.lstmLayers);
            for i = 1:length(lstmLayerNames)
                lstmParams = arch.lstmLayers.(lstmLayerNames{i});
                dropoutRate = arch.dropoutLayers.(['dropout' num2str(i)]);

                layers{layerIdx} = lstmLayer(lstmParams.numHiddenUnits, ...
                    'OutputMode', lstmParams.OutputMode, ...
                    'Name', ['lstm' num2str(i)]);
                layerIdx = layerIdx + 1;

                if arch.batchNorm
                    layers{layerIdx} = batchNormalizationLayer('Name', ['bn' num2str(i)]);
                    layerIdx = layerIdx + 1;
                end

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

            % 出力層（全結合層 → softmax → 分類層）
            layers{layerIdx} = fullyConnectedLayer(arch.numClasses, 'Name', 'fc_output');
            layerIdx = layerIdx + 1;
            layers{layerIdx} = softmaxLayer('Name', 'softmax');
            layerIdx = layerIdx + 1;
            layers{layerIdx} = classificationLayer('Name', 'output');

            % 空セルを除去し、レイヤー配列へ変換
            layers = layers(~cellfun('isempty', layers));
            layers = [layers{:}];
        end

        %% トレーニングオプションの設定（検証データがある場合は自動設定）
        function options = getTrainingOptions(obj, valData, valLabels)
            if obj.useGPU
                executionEnvironment = 'cpu';
               if obj.useGPU
                   executionEnvironment = 'gpu';
               end
            end

            valDS = {valData, valLabels};
        
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
                   'GradientThreshold', 1 ...
               );
        end

        %% データセットの分割（訓練/検証/テスト）
        function [trainData, trainLabels, valData, valLabels, testData, testLabels] = splitDataset(obj, data, labels)
            try
                % 分割数の取得
                k = obj.params.classifier.evaluation.kfold;
                
                % データサイズの取得
                [~, ~, numEpochs] = size(data);
                fprintf('Total epochs: %d\n', numEpochs);
        
                % インデックスのシャッフル
                rng('default'); % 再現性のため
                shuffledIdx = randperm(numEpochs);
        
                % 分割比率の計算
                trainRatio = (k-1)/k;  % 1-k/k
                valRatio = 1/(2*k);    % k/2k
                % testRatio = 1/(2*k);   % k/2k
        
                % データ数の計算
                numTrain = floor(numEpochs * trainRatio);
                numVal = floor(numEpochs * valRatio);
                
                % インデックスの分割
                trainIdx = shuffledIdx(1:numTrain);
                valIdx = shuffledIdx(numTrain+1:numTrain+numVal);
                testIdx = shuffledIdx(numTrain+numVal+1:end);
        
                % データの分割
                trainData = data(:,:,trainIdx);
                trainLabels = labels(trainIdx);
                
                valData = data(:,:,valIdx);
                valLabels = labels(valIdx);
                
                testData = data(:,:,testIdx);
                testLabels = labels(testIdx);

                fprintf('データ分割 (k=%d):\n', k);
                fprintf('  訓練データ: %d サンプル (%.1f%%)\n', ...
                    length(trainIdx), (length(trainIdx)/numEpochs)*100);
                fprintf('  検証データ: %d サンプル (%.1f%%)\n', ...
                    length(valIdx), (length(valIdx)/numEpochs)*100);
                fprintf('  テストデータ: %d サンプル (%.1f%%)\n', ...
                    length(testIdx), (length(testIdx)/numEpochs)*100);
        
                % データの検証
                if isempty(trainData) || isempty(valData) || isempty(testData)
                    error('一つ以上のデータセットが空です');
                end
        
                % クラスの分布を確認
                obj.checkClassDistribution('訓練', trainLabels);
                obj.checkClassDistribution('検証', valLabels);
                obj.checkClassDistribution('テスト', testLabels);
        
            catch ME
                error('データ分割に失敗: %s', ME.message);
            end
        end
        
        % クラスの分布を確認するヘルパーメソッド
        function checkClassDistribution(~, setName, labels)
            uniqueLabels = unique(labels);
            fprintf('\n%sデータのクラス分布:\n', setName);
            for i = 1:length(uniqueLabels)
                count = sum(labels == uniqueLabels(i));
                fprintf('  クラス %d: %d サンプル (%.1f%%)\n', ...
                    uniqueLabels(i), count, (count/length(labels))*100);
            end
        end

        %% LSTM用のデータ前処理（セル配列へ変換）
        function preparedData = prepareDataForLSTM(~, data)
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
                        % NaN, Inf のチェックと置換
                        if any(isnan(currentData(:)))
                            warning('Trial %d contains NaN values. Replacing with zeros.', i);
                            currentData(isnan(currentData)) = 0;
                        end
                        if any(isinf(currentData(:)))
                            warning('Trial %d contains Inf values. Replacing with zeros.', i);
                            currentData(isinf(currentData)) = 0;
                        end
                        preparedData{i} = currentData;
                    end
                else
                    % 入力が数値配列の場合（3次元: channels x timepoints x trials）
                    [~, ~, trials] = size(data);
                    preparedData = cell(trials, 1);
                    for i = 1:trials
                        currentData = data(:, :, i);
                        if ~isa(currentData, 'double')
                            currentData = double(currentData);
                        end
                        % NaN, Inf のチェックと置換
                        if any(isnan(currentData(:)))
                            warning('Trial %d contains NaN values. Replacing with zeros.', i);
                            currentData(isnan(currentData)) = 0;
                        end
                        if any(isinf(currentData(:)))
                            warning('Trial %d contains Inf values. Replacing with zeros.', i);
                            currentData(isinf(currentData)) = 0;
                        end
                        preparedData{i} = currentData;
                    end
                end
        
            catch ME
                errorInfo = struct(...
                    'message', ME.message, ...
                    'stack', ME.stack, ...
                    'dataInfo', struct(...
                        'inputSize', size(data), ...
                        'dataType', class(data)));
                if exist('preparedData', 'var') && ~isempty(preparedData)
                    errorInfo.dataInfo.preparedDataSize = size(preparedData{1});
                    errorInfo.dataInfo.lastProcessedTrial = length(preparedData);
                end
                error('LSTM用データ準備に失敗: %s\nエラー情報: %s', ME.message, jsonencode(errorInfo));
            end
        end

        %% LSTMモデルの学習
        function [lstmModel, trainInfo] = trainLSTMModel(obj, trainData, trainLabels, valData, valLabels)
            try        
                % データの形状変換
                if ndims(trainData) ~= 4
                    trainData = obj.prepareDataForLSTM(trainData);
                end
        
                % データサイズの表示
                sampleSize = size(trainData{1});
                numFeatures = sampleSize(1);
                fprintf('\n=== LSTM モデル学習開始 ===\n');
                fprintf('入力特徴量数: %d\n', numFeatures);
                fprintf('時系列長: %d\n', sampleSize(2));
                fprintf('バッチサイズ: %d\n', obj.params.classifier.lstm.training.miniBatchSize);
        
                % ラベルのカテゴリカル変換
                uniqueLabels = unique(trainLabels);
                trainLabels = categorical(trainLabels, uniqueLabels);
        
                % GPU転送
                if obj.useGPU
                    fprintf('GPUを使用して学習を実行します\n');
                else
                    fprintf('CPUを使用して学習を実行します\n');
                end
        
                % 検証データの処理
                if ~isempty(valData)
                    if ndims(valData) ~= 4
                        valData = obj.prepareDataForLSTM(valData);
                    end
                    valLabels = categorical(valLabels, uniqueLabels);

                    fprintf('検証データを使用して学習を実行します\n');
                end
        
                % アーキテクチャ情報の表示
                arch = obj.params.classifier.lstm.architecture;
                fprintf('\nLSTMアーキテクチャ:\n');
                fprintf('  レイヤー数: %d\n', length(fieldnames(arch.lstmLayers)));
                fprintf('  隠れ層ユニット数: %d\n', arch.lstmLayers.lstm1.numHiddenUnits);
                fprintf('  全結合層: [%s]\n', strjoin(string(arch.fullyConnected), ', '));
                fprintf('  ドロップアウト率: %.2f\n', arch.dropoutLayers.dropout1);
        
                % トレーニング情報の初期化
                trainInfo = struct(...
                    'TrainingLoss', [], ...
                    'ValidationLoss', [], ...
                    'TrainingAccuracy', [], ...
                    'ValidationAccuracy', [], ...
                    'FinalEpoch', 0, ...
                    'History', [] ...
                );
        
                % 学習パラメータの表示
                fprintf('\n学習パラメータ:\n');
                fprintf('  最大エポック数: %d\n', obj.params.classifier.lstm.training.maxEpochs);
                fprintf('  学習率: %.6f\n', obj.params.classifier.lstm.training.optimizer.learningRate);
                
                layers = obj.buildLSTMLayers(numFeatures);
                options = obj.getTrainingOptions(valData, valLabels);
                
                fprintf('\n=== 学習開始 ===\n');
                [lstmModel, trainHistory] = trainNetwork(trainData, trainLabels, layers, options);
                
                % 学習履歴の保存
                if isfield(trainHistory, 'TrainingLoss')
                    trainInfo.History.TrainingLoss = trainHistory.TrainingLoss;
                    trainInfo.History.ValidationLoss = trainHistory.ValidationLoss;
                    trainInfo.History.TrainingAccuracy = trainHistory.TrainingAccuracy;
                    trainInfo.History.ValidationAccuracy = trainHistory.ValidationAccuracy;
                    trainInfo.FinalEpoch = length(trainHistory.TrainingLoss);
                end
                
                fprintf('\n学習完了: %d反復\n', trainInfo.FinalEpoch);
                if isfield(trainHistory, 'TrainingAccuracy')
                    fprintf('最終トレーニング精度: %.2f%%\n', trainHistory.TrainingAccuracy(end));
                end
                if isfield(trainHistory, 'ValidationAccuracy')
                    fprintf('最終検証精度: %.2f%%\n', trainHistory.ValidationAccuracy(end));
                end
        
            catch ME
                fprintf('\n=== 学習中にエラーが発生 ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラー位置:\n');
                for i = 1:length(ME.stack)
                    fprintf('  File: %s\n  Line: %d\n  Function: %s\n\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end
                rethrow(ME);
            end
        end

        %% モデル評価（精度・混同行列・各クラスの指標等）
        function metrics = evaluateModel(~, model, testData, testLabels)
            try
                % 予測実施
                [pred, scores] = classify(model, testData);
                testLabels = categorical(testLabels);

                metrics = struct();
                metrics.accuracy = mean(pred == testLabels);
                metrics.confusionMat = confusionmat(testLabels, pred);

                % 各クラスごとの性能評価
                classes = unique(testLabels);
                metrics.classwise = repmat(struct('precision', 0, 'recall', 0, 'f1score', 0), length(classes), 1);

                for i = 1:length(classes)
                    className = classes(i);
                    classIdx = (testLabels == className);

                    TP = sum(pred(classIdx) == className);
                    FP = sum(pred == className) - TP;
                    FN = sum(classIdx) - TP;

                    if (TP + FP) == 0
                        precision = 0;
                    else
                        precision = TP / (TP + FP);
                    end
                    if (TP + FN) == 0
                        recall = 0;
                    else
                        recall = TP / (TP + FN);
                    end
                    if (precision + recall) == 0
                        f1 = 0;
                    else
                        f1 = 2 * (precision * recall) / (precision + recall);
                    end

                    metrics.classwise(i).precision = precision;
                    metrics.classwise(i).recall = recall;
                    metrics.classwise(i).f1score = f1;
                end

                % 2クラス分類の場合のROCとAUC
                if length(classes) == 2
                    [X, Y, T, AUC] = perfcurve(testLabels, scores(:,2), classes(2));
                    metrics.roc = struct('X', X, 'Y', Y, 'T', T);
                    metrics.auc = AUC;
                end

            catch ME
                error('モデル評価に失敗: %s', ME.message);
            end
        end

        %% k-fold 交差検証の実施
        function results = performCrossValidation(obj, data, labels)
            try
                % k-fold cross validationのパラメータ取得
                k = obj.params.classifier.lstm.training.validation.kfold;
                fprintf('\n=== %d分割交差検証開始 ===\n', k);
        
                % 結果保存用構造体の初期化
                results = struct();
                results.folds = struct(...
                    'accuracy', zeros(1, k), ...
                    'confusionMat', cell(1, k), ...
                    'classwise', cell(1, k), ...
                    'validation_curve', cell(1, k) ...
                );
        
                % 各フォールドの処理
                for i = 1:k
                    fprintf('\nフォールド %d/%d の処理開始\n', i, k);
                    
                    try
                        % データの分割と前処理
                        [trainData, trainLabels, valData, valLabels, ~, ~] = obj.splitDataset(data, labels);
                        
                        prepTrainData = obj.prepareDataForLSTM(trainData);
                        prepValData = obj.prepareDataForLSTM(valData);
                        
                        % モデルの学習
                        [model, trainInfo] = obj.trainLSTMModel(prepTrainData, trainLabels, prepValData, valLabels);
                        
                        % 性能評価
                        metrics = obj.evaluateModel(model, prepValData, valLabels);
                        
                        % 結果の保存
                        results.folds.accuracy(i) = metrics.accuracy;
                        results.folds.confusionMat{i} = metrics.confusionMat;
                        results.folds.classwise{i} = metrics.classwise;
                        
                        if ~isempty(trainInfo.History)
                            results.folds.validation_curve{i} = struct(...
                                'train_accuracy', trainInfo.History.TrainingAccuracy, ...
                                'val_accuracy', trainInfo.History.ValidationAccuracy, ...
                                'train_loss', trainInfo.History.TrainingLoss, ...
                                'val_loss', trainInfo.History.ValidationLoss);
                        end
                        
                        fprintf('フォールド %d の精度: %.2f%%\n', i, results.folds.accuracy(i) * 100);
                        
                    catch ME
                        warning('フォールド %d でエラーが発生: %s', i, ME.message);
                        results.folds.accuracy(i) = 0;
                    end
                end
        
                % 統計量の計算と表示
                validFolds = results.folds.accuracy > 0;
                results.meanAccuracy = mean(results.folds.accuracy(validFolds));
                results.stdAccuracy = std(results.folds.accuracy(validFolds));
                
                fprintf('\n=== 交差検証結果 ===\n');
                fprintf('平均精度: %.2f%% (±%.2f%%)\n', ...
                    results.meanAccuracy * 100, results.stdAccuracy * 100);
                fprintf('有効フォールド数: %d/%d\n', sum(validFolds), k);
        
            catch ME
                fprintf('\n=== 交差検証中にエラーが発生 ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラースタック:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% 過学習検証（学習曲線とテスト精度の比較）
        function [isOverfit, metrics] = validateOverfitting(obj, trainInfo, testMetrics)
            try
                fprintf('\n=== 過学習の検証 ===\n');
                
                % trainInfoの検証
                if ~isstruct(trainInfo) || ~isfield(trainInfo, 'History')
                    error('学習履歴情報が不正です');
                end
                
                history = trainInfo.History;
                if ~isfield(history, 'TrainingAccuracy') || ~isfield(history, 'ValidationAccuracy')
                    warning('学習履歴に精度情報が含まれていません。過学習検証をスキップします。');
                    metrics = struct(...
                        'generalizationGap', NaN, ...
                        'performanceGap', NaN, ...
                        'isCompletelyBiased', true, ...
                        'isLearningProgressing', false, ...
                        'severity', 'unknown', ...
                        'optimalEpoch', 0, ...
                        'totalEpochs', 0);
                    isOverfit = false;
                    return;
                end
        
                % 精度データの取得と表示
                trainAcc = history.TrainingAccuracy;
                valAcc = history.ValidationAccuracy;
                testAcc = testMetrics.accuracy * 100;

                fprintf('Validation Accuracy: %.2f%%\n', max(valAcc));
                fprintf('Test Accuracy: %.2f%%\n', testAcc);
        
                % Performance Gapの計算（検証結果とテスト結果の差）
                perfGap = abs(max(valAcc) - testAcc);
                fprintf('Performance Gap: %.2f%%\n', perfGap);
        
                % 完全な偏りの検出
                if isfield(testMetrics, 'confusionMat')
                    cm = testMetrics.confusionMat;
                    % 各実際のクラス（行）のサンプル数を確認
                    missingActual = any(sum(cm, 2) == 0);
                    % 各予測クラス（列）の予測件数を確認
                    missingPredicted = any(sum(cm, 1) == 0);
                    % いずれかが true ならば、全く現れないクラスがあると判断
                    isCompletelyBiased = missingActual || missingPredicted;
                    
                    if isCompletelyBiased
                        fprintf('警告: 予測が特定のクラスに偏っています\n');
                        if missingActual
                            fprintf('  - 出現しない実際のクラスが存在\n');
                        end
                        if missingPredicted
                            fprintf('  - 予測されないクラスが存在\n');
                        end
                    end
                else
                    isCompletelyBiased = false;
                end
        
                % 学習進行の確認
                trainDiff = diff(trainAcc);
                isLearningProgressing = std(trainDiff) > 0.01;
                if ~isLearningProgressing
                    fprintf('警告: 学習が十分に進行していない可能性があります\n');
                    fprintf('  - 訓練精度の変化の標準偏差: %.4f\n', std(trainDiff));
                end
        
                % トレンド分析
                [trainTrend, valTrend] = obj.analyzeLearningCurves(trainAcc, valAcc);
                
                % 最適エポックの検出
                [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(valAcc);
                fprintf('Optimal Epoch: %d/%d\n', optimalEpoch, totalEpochs);
        
                % 過学習の重大度判定
                severity = obj.determineOverfittingSeverity(perfGap, isCompletelyBiased, isLearningProgressing);
        
                % メトリクスの構築
                metrics = struct(...
                    'performanceGap', perfGap, ...
                    'isCompletelyBiased', isCompletelyBiased, ...
                    'isLearningProgressing', isLearningProgressing, ...
                    'validationTrend', valTrend, ...
                    'trainingTrend', trainTrend, ...
                    'severity', severity, ...
                    'optimalEpoch', optimalEpoch, ...
                    'totalEpochs', totalEpochs);
                
                % 過学習判定
                isOverfit = ismember(severity, {'critical', 'severe', 'moderate'});
                fprintf('過学習判定: %s (重大度: %s)\n', mat2str(isOverfit), severity);
        
                % 詳細な分析結果の表示
                if isOverfit
                    fprintf('\n=== 詳細な過学習分析 ===\n');
                    fprintf('1. 汎化性能:\n');
                    fprintf('  - Performance Gap: %.2f%%\n', perfGap);
                    
                    fprintf('\n2. 学習の進行状況:\n');
                    fprintf('  - 訓練傾向の平均変化: %.4f\n', trainTrend.mean_change);
                    fprintf('  - 検証傾向の平均変化: %.4f\n', valTrend.mean_change);
                    fprintf('  - 訓練のボラティリティ: %.4f\n', trainTrend.volatility);
                    fprintf('  - 検証のボラティリティ: %.4f\n', valTrend.volatility);
                    
                    fprintf('\n3. モデルの偏り:\n');
                    fprintf('  - 完全な偏りが存在: %s\n', mat2str(isCompletelyBiased));
                    fprintf('  - 学習の進行: %s\n', mat2str(isLearningProgressing));
                    
                    fprintf('\n推奨される対策:\n');
                    if ~isLearningProgressing
                        fprintf('- 学習率の調整\n');
                        fprintf('- モデル容量の見直し\n');
                    end
                    if isCompletelyBiased
                        fprintf('- データバランスの改善\n');
                        fprintf('- クラス重み付けの導入\n');
                    end
                end
        
            catch ME
                fprintf('\n=== 過学習検証中にエラーが発生 ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラースタック:\n');
                disp(getReport(ME, 'extended'));
                
                metrics = struct(...
                    'generalizationGap', Inf, ...
                    'performanceGap', Inf, ...
                    'isCompletelyBiased', true, ...
                    'isLearningProgressing', false, ...
                    'severity', 'error', ...
                    'optimalEpoch', 0, ...
                    'totalEpochs', 0);
                isOverfit = true;
            end
        end

        function [trainTrend, valTrend] = analyzeLearningCurves(~, trainAcc, valAcc)
            % 学習曲線の変化率（傾き）とボラティリティ（変動性）を計算する
            if length(trainAcc) < 2 || length(valAcc) < 2
                trainTrend = struct('mean_change', NaN, 'volatility', NaN);
                valTrend = struct('mean_change', NaN, 'volatility', NaN);
                return;
            end
            
            % 各エポック間の変化量を計算
            trainDiff = diff(trainAcc);
            valDiff = diff(valAcc);
            
            % 平均変化量と標準偏差（ボラティリティ）を算出
            trainTrend.mean_change = mean(trainDiff);
            trainTrend.volatility = std(trainDiff);
            
            valTrend.mean_change = mean(valDiff);
            valTrend.volatility = std(valDiff);
        end
    
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(~, valAcc)
            % 検証精度が最大となるエポックを最適エポックとして返す
            totalEpochs = length(valAcc);
            [~, optimalEpoch] = max(valAcc);
        end

        %% 過学習の重症度判定
        function severity = determineOverfittingSeverity(~, perfGap, isCompletelyBiased, isLearningProgressing)
            if isCompletelyBiased
                severity = 'critical';
            elseif ~isLearningProgressing
                severity = 'failed';
            elseif perfGap > 15
                severity = 'severe';
            elseif perfGap > 8
                severity = 'moderate';
            elseif perfGap > 5
                severity = 'mild';
            else
                severity = 'none';
            end
        end

        %% 性能指標の更新
        function updatePerformanceMetrics(obj, testMetrics)
            obj.performance = testMetrics;
        end

        %% 結果の表示
        function displayResults(obj)
            try
                fprintf('\n=== LSTM Classification Results ===\n');
                if ~isempty(obj.performance)
                    fprintf('Overall Accuracy: %.2f%%\n', obj.performance.accuracy * 100);
                    if isfield(obj.performance, 'auc')
                        fprintf('AUC: %.3f\n', obj.performance.auc);
                    end
                    if ~isempty(obj.performance.confusionMat)
                        fprintf('\nConfusion Matrix:\n');
                        disp(obj.performance.confusionMat);
                    end
                    fprintf('\nClass-wise Performance:\n');
                    for i = 1:length(obj.performance.classwise)
                        fprintf('Class %d:\n', i);
                        fprintf('  Precision: %.2f%%\n', obj.performance.classwise(i).precision * 100);
                        fprintf('  Recall: %.2f%%\n', obj.performance.classwise(i).recall * 100);
                        fprintf('  F1-Score: %.2f%%\n', obj.performance.classwise(i).f1score * 100);
                    end
                end

                if ~isempty(obj.overfitMetrics)
                    fprintf('\nOverfitting Analysis:\n');
                    fprintf('Performance Gap: %.2f%%\n', obj.overfitMetrics.performanceGap * 100);
                    fprintf('Severity: %s\n', obj.overfitMetrics.severity);
                    if isfield(obj.overfitMetrics, 'validationTrend')
                        fprintf('\nValidation Trend:\n');
                        fprintf('Mean Change: %.4f\n', obj.overfitMetrics.validationTrend.mean_change);
                        fprintf('Volatility: %.4f\n', obj.overfitMetrics.validationTrend.volatility);
                    end
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
    end
end