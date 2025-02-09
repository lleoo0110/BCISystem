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
            obj.bestValAccuracy = 0;
            obj.patienceCounter = 0;
            obj.currentEpoch = 0;
            obj.overfitMetrics = struct();
            obj.useGPU = params.classifier.lstm.gpu;
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

                % データ前処理（LSTM用のセル配列に変換）
                prepTrainData = obj.prepareDataForLSTM(trainData);
                prepValData   = obj.prepareDataForLSTM(valData);
                prepTestData  = obj.prepareDataForLSTM(testData);

                % モデルの学習（内部で層構築やオプション設定を行う）
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

                % 交差検証の実施（有効な場合）
                crossValidationResults = struct('meanAccuracy', [], 'stdAccuracy', []);
                if obj.params.classifier.lstm.training.validation.enable
                    crossValidationResults = obj.performCrossValidation(processedData, processedLabel);
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
                        'auc', testMetrics.auc, ...
                        'confusionMatrix', testMetrics.confusionMat), ...
                    'trainInfo', trainInfo, ...
                    'overfitting', obj.overfitMetrics);

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
            training = obj.params.classifier.lstm.training;
            if obj.useGPU
                execEnv = 'gpu';
            else
                execEnv = 'cpu';
            end

            options = trainingOptions(training.optimizer.type, ...
                'InitialLearnRate', training.optimizer.learningRate, ...
                'MaxEpochs', training.maxEpochs, ...
                'MiniBatchSize', training.miniBatchSize, ...
                'GradientThreshold', training.optimizer.gradientThreshold, ...
                'Shuffle', training.shuffle, ...
                'Plots', 'training-progress', ...
                'ExecutionEnvironment', execEnv, ...
                'Verbose', true);

            % 検証データが提供され、検証が有効な場合のみ設定
            if training.validation.enable && ~isempty(valData) && ~isempty(valLabels)
                options.ValidationData = {valData, categorical(valLabels)};
                options.ValidationFrequency = training.validation.frequency;
                options.ValidationPatience = training.validation.patience;
            end
        end

        %% データセットの分割（訓練/検証/テスト）
        function [trainData, trainLabels, valData, valLabels, testData, testLabels] = splitDataset(obj, data, labels)
            try
                % k-foldのパラメータ取得
                k = obj.params.classifier.evaluation.kfold;
                [~, ~, numTrials] = size(data);
                fprintf('Total trials: %d\n', numTrials);

                % インデックスのシャッフル（再現性のため固定シード）
                rng('default');
                shuffledIdx = randperm(numTrials);

                % 分割比率の設定
                trainRatio = (k - 1) / k;  % 訓練データ
                valRatio = 1 / (2 * k);    % 検証データ（テストは残り）

                numTrain = floor(numTrials * trainRatio);
                numVal   = floor(numTrials * valRatio);

                trainIdx = shuffledIdx(1:numTrain);
                valIdx   = shuffledIdx(numTrain+1:numTrain+numVal);
                testIdx  = shuffledIdx(numTrain+numVal+1:end);

                % 各データの抽出
                trainData   = data(:, :, trainIdx);
                trainLabels = labels(trainIdx);

                valData   = data(:, :, valIdx);
                valLabels = labels(valIdx);

                testData   = data(:, :, testIdx);
                testLabels = labels(testIdx);

                fprintf('データ分割:\n');
                fprintf('  訓練データ: %d サンプル (%.1f%%)\n', length(trainIdx), (length(trainIdx)/numTrials)*100);
                fprintf('  検証データ: %d サンプル (%.1f%%)\n', length(valIdx), (length(valIdx)/numTrials)*100);
                fprintf('  テストデータ: %d サンプル (%.1f%%)\n', length(testIdx), (length(testIdx)/numTrials)*100);

            catch ME
                error('データ分割に失敗: %s', ME.message);
            end
        end

        %% LSTM用のデータ前処理（セル配列へ変換）
        function preparedData = prepareDataForLSTM(obj, data)
            try
                [channels, timepoints, trials] = size(data);
                fprintf('Input data dimensions:\n');
                fprintf('  Channels: %d\n', channels);
                fprintf('  Timepoints: %d\n', timepoints);
                fprintf('  Trials: %d\n', trials);

                preparedData = cell(trials, 1);
                for i = 1:trials
                    % ※ 転置を削除します
                    % もともと data は [channels, timepoints, trials] の形式であるため、
                    % 各シーケンスは [channels, timepoints] ＝ [numFeatures, sequenceLength] となる
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
                    if obj.useGPU
                        currentData = gpuArray(currentData);
                    end
                    preparedData{i} = currentData;
                end

                sampleSize = size(preparedData{1});
                fprintf('Prepared data format:\n');
                fprintf('  Number of sequences (trials): %d\n', length(preparedData));
                fprintf('  Number of features: %d\n', sampleSize(1));
                fprintf('  Sequence length: %d\n', sampleSize(2));

                % 各シーケンスの形状チェック
                for i = 2:trials
                    if ~isequal(size(preparedData{i}), sampleSize)
                        error('Trial %d has inconsistent dimensions: expected [%d, %d] but got [%d, %d]', ...
                            i, sampleSize(1), sampleSize(2), size(preparedData{i},1), size(preparedData{i},2));
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
            % 初期化（エラーハンドリング用に初期値を設定）
            sampleSize = [];
            numFeatures = [];
            try
                % 訓練データの形式確認
                fprintf('Training data format:\n');
                fprintf('  Number of sequences: %d\n', length(trainData));
                sampleSize = size(trainData{1});
                % 修正：入力シーケンスは [numFeatures, sequenceLength] の形式とする
                fprintf('  Number of features: %d\n', sampleSize(1));
                fprintf('  Sequence length: %d\n', sampleSize(2));
                numFeatures = sampleSize(1);

                % 層構築
                layers = obj.buildLSTMLayers(numFeatures);

                % ラベルのカテゴリカル変換
                trainLabels = categorical(trainLabels);
                if ~isempty(valData) && ~isempty(valLabels)
                    valLabels = categorical(valLabels);
                end

                % トレーニングオプションの取得（検証データがある場合は自動設定）
                options = obj.getTrainingOptions(valData, valLabels);

                % モデル学習
                [lstmModel, trainInfo] = trainNetwork(trainData, trainLabels, layers, options);
                fprintf('Model training completed successfully\n');

            catch ME
                errorInfo = struct(...
                    'message', ME.message, ...
                    'stack', ME.stack, ...
                    'dataInfo', struct(...
                        'trainDataLength', length(trainData), ...
                        'sampleSize', sampleSize, ...
                        'numFeatures', numFeatures));
                error('LSTMモデルの学習に失敗: %s\nデータ形式: %d sequences\nエラー情報: %s', ...
                    ME.message, length(trainData), jsonencode(errorInfo));
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
                k = obj.params.classifier.lstm.training.validation.kfold;
                fprintf('\nStarting %d-fold Cross-validation\n', k);

                % 結果保存用構造体の初期化
                results = struct();
                results.folds = struct(...
                    'accuracy', zeros(1, k), ...
                    'confusionMat', cell(1, k), ...
                    'classwise', cell(1, k));

                cv = cvpartition(length(labels), 'KFold', k);

                for i = 1:k
                    fprintf('\nProcessing fold %d/%d\n', i, k);

                    % データの分割
                    trainIdx = training(cv, i);
                    testIdx  = test(cv, i);

                    foldTrainData   = data(:, :, trainIdx);
                    foldTrainLabels = labels(trainIdx);
                    foldTestData    = data(:, :, testIdx);
                    foldTestLabels  = labels(testIdx);

                    % 前処理
                    prepTrainData = obj.prepareDataForLSTM(foldTrainData);
                    prepTestData  = obj.prepareDataForLSTM(foldTestData);

                    try
                        % 交差検証用の学習（検証データは使用しない）
                        [model, ~] = obj.trainLSTMModel(prepTrainData, foldTrainLabels, [], []);
                        metrics = obj.evaluateModel(model, prepTestData, foldTestLabels);

                        results.folds.accuracy(i) = metrics.accuracy;
                        results.folds.confusionMat{i} = metrics.confusionMat;
                        results.folds.classwise{i} = metrics.classwise;
                    catch ME
                        warning('Error in fold %d: %s', i, ME.message);
                        results.folds.accuracy(i) = 0;
                    end

                    % GPUメモリリセット
                    if obj.useGPU
                        reset(gpuDevice);
                    end
                end

                validFolds = results.folds.accuracy > 0;
                results.meanAccuracy = mean(results.folds.accuracy(validFolds));
                results.stdAccuracy  = std(results.folds.accuracy(validFolds));
                results.minAccuracy  = min(results.folds.accuracy(validFolds));
                results.maxAccuracy  = max(results.folds.accuracy(validFolds));

                fprintf('\nCross-validation Results:\n');
                fprintf('Mean Accuracy: %.2f%% (±%.2f%%)\n', results.meanAccuracy * 100, results.stdAccuracy * 100);

            catch ME
                error('Cross-validation failed: %s', ME.message);
            end
        end

        %% 過学習検証（学習曲線とテスト精度の比較）
        function [isOverfit, metrics] = validateOverfitting(obj, trainInfo, testMetrics)
            try
                trainAcc = trainInfo.TrainingAccuracy;
                valAcc   = trainInfo.ValidationAccuracy;
                testAcc  = testMetrics.accuracy;

                % Generalization Gap と Performance Gap の算出
                genGap = abs(trainAcc(end) - valAcc(end));
                perfGap = abs(trainAcc(end) - testAcc);

                % 検証データのトレンド分析
                trainDiff = diff(double(trainAcc));
                valDiff   = diff(valAcc);

                metrics = struct(...
                    'generalizationGap', genGap, ...
                    'performanceGap', perfGap, ...
                    'validationTrend', struct(...
                        'mean_change', mean(valDiff), ...
                        'volatility', std(valDiff)), ...
                    'severity', obj.determineOverfittingSeverity(genGap, perfGap));

                isOverfit = ismember(metrics.severity, {'critical', 'severe', 'moderate'});

            catch ME
                error('過学習検証に失敗: %s', ME.message);
            end
        end

        %% 過学習の重症度判定
        function severity = determineOverfittingSeverity(~, genGap, perfGap)
            if genGap > 0.2 || perfGap > 0.2
                severity = 'critical';
            elseif genGap > 0.15 || perfGap > 0.15
                severity = 'severe';
            elseif genGap > 0.1 || perfGap > 0.1
                severity = 'moderate';
            elseif genGap > 0.05 || perfGap > 0.05
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
                    fprintf('Generalization Gap: %.2f%%\n', obj.overfitMetrics.generalizationGap * 100);
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
