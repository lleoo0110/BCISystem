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
                    'overfitting', obj.overfitMetrics);

                obj.displayResults();

                % GPU使用時はGPUメモリをリセット
                if obj.useGPU
                    gpuDevice([]);
                end

            catch ME
                fprintf('\n=== Error in LSTM Training ===\n');
                fprintf('Error message: %s\n', ME.message);
                disp(getReport(ME, 'extended'));

                if obj.useGPU
                    gpuDevice([]);
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
                'Plots', 'none', ...
                'ExecutionEnvironment', execEnv, ...
                'Verbose', true, ...
                'OutputFcn', @(info)obj.trainingOutputFcn(info));

            % 検証データの設定
            options.ValidationData = {valData, categorical(valLabels)};
            options.ValidationFrequency = training.validation.frequency;
            options.ValidationPatience = training.validation.patience;
        end

        %% トレーニング進捗のコールバック関数
        function stop = trainingOutputFcn(obj, info)
            stop = false;
            
            if info.State == "start"
                obj.currentEpoch = 0;
                return;
            end
            
            % 学習情報の更新
            obj.currentEpoch = obj.currentEpoch + 1;
            
            % 学習履歴の更新
            if isfield(info, 'TrainingLoss')
                obj.trainingHistory.loss(end+1) = info.TrainingLoss;
            end
            if isfield(info, 'TrainingAccuracy')
                obj.trainingHistory.accuracy(end+1) = info.TrainingAccuracy;
            end
            
            % 検証データがある場合の処理
            if ~isempty(info.ValidationLoss)
                currentAccuracy = info.ValidationAccuracy;
                
                % 検証履歴の更新
                obj.validationHistory.loss(end+1) = info.ValidationLoss;
                obj.validationHistory.accuracy(end+1) = info.ValidationAccuracy;
                
                % Early Stopping判定
                if currentAccuracy > obj.bestValAccuracy
                    obj.bestValAccuracy = currentAccuracy;
                    obj.patienceCounter = 0;
                else
                    obj.patienceCounter = obj.patienceCounter + 1;
                    if obj.patienceCounter >= obj.params.classifier.lstm.training.validation.patience
                        fprintf('\nEarly stopping: エポック %d で学習を終了\n', obj.currentEpoch);
                        fprintf('最良検証精度 %.2f%% を %d エポック更新できず\n', ...
                            obj.bestValAccuracy * 100, obj.patienceCounter);
                        stop = true;
                    end
                end
            end
        end

        %% データセットの分割（訓練/検証/テスト）
        function [trainData, trainLabels, valData, valLabels, testData, testLabels] = splitDataset(obj, data, labels)
            try
                % 分割数の取得
                k = obj.params.classifier.evaluation.kfold;
                
                % データサイズの取得
                [~, ~, numTrials] = size(data);
                fprintf('\n=== Starting LSTM Training ===\n');
                fprintf('Total epochs: %d\n', numTrials);
        
                % インデックスのシャッフル（再現性のため固定シード）
                rng('default');
                shuffledIdx = randperm(numTrials);
        
                % 分割比率の設定
                trainRatio = (k-1)/k;  % 訓練データ
                valRatio = 1/(2*k);    % 検証データ
                testRatio = 1/(2*k);   % テストデータ
        
                % データ数の計算
                numTrain = floor(numTrials * trainRatio);
                numVal = floor(numTrials * valRatio);
        
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
        
                % データ分割情報の表示
                fprintf('データ分割 (k=%d):\n', k);
                fprintf('  訓練データ: %d サンプル (%.1f%%)\n', ...
                    length(trainIdx), (length(trainIdx)/numTrials));
                fprintf('  検証データ: %d サンプル (%.1f%%)\n', ...
                    length(valIdx), (length(valIdx)/numTrials));
                fprintf('  テストデータ: %d サンプル (%.1f%%)\n', ...
                    length(testIdx), (length(testIdx)/numTrials)*100);
        
                % クラスの分布を確認
                obj.displayClassDistribution('訓練データ', trainLabels);
                obj.displayClassDistribution('検証データ', valLabels);
                obj.displayClassDistribution('テストデータ', testLabels);
        
                % データの検証
                if isempty(trainData) || isempty(valData) || isempty(testData)
                    error('一つ以上のデータセットが空です');
                end
        
            catch ME
                error('データ分割に失敗: %s', ME.message);
            end
        end

        % クラスの分布を表示するヘルパーメソッド
        function displayClassDistribution(~, setName, labels)
            uniqueLabels = unique(labels);
            fprintf('\n%sのクラス分布:\n', setName);
            for i = 1:length(uniqueLabels)
                count = sum(labels == uniqueLabels(i));
                fprintf('  クラス %d: %d サンプル (%.1f%%)\n', ...
                    uniqueLabels(i), count, (count/length(labels))*100);
            end
        end

        %% LSTM用のデータ前処理（セル配列へ変換）
        function preparedData = prepareDataForLSTM(obj, data)
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
                        if obj.useGPU
                            currentData = gpuArray(currentData);
                        end
                        preparedData{i} = currentData;
                    end
                else
                    % 入力が数値配列の場合（3次元: channels x timepoints x trials）
                    [channels, timepoints, trials] = size(data);
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
                        if obj.useGPU
                            currentData = gpuArray(currentData);
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
                    trainData = cellfun(@(x) gpuArray(x), trainData, 'UniformOutput', false);
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
                    
                    if obj.useGPU
                        valData = cellfun(@(x) gpuArray(x), valData, 'UniformOutput', false);
                    end
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
                
                fprintf('訓練精度: %.2f%%\n', trainAcc(end));
                fprintf('検証精度: %.2f%%\n', valAcc(end));
                fprintf('テスト精度: %.2f%%\n', testAcc);
        
                % Generalization GapとPerformance Gapの計算
                genGap = abs(trainAcc(end) - valAcc(end));
                perfGap = abs(trainAcc(end) - testAcc);
                
                fprintf('\nGeneralization Gap: %.2f%%\n', genGap);
                fprintf('Performance Gap: %.2f%%\n', perfGap);
        
                % トレンド分析
                [trainTrend, valTrend] = obj.analyzeLearningCurves(trainAcc, valAcc);
                
                % 最適エポックの検出
                [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(valAcc);
                fprintf('Optimal Epoch: %d/%d\n', optimalEpoch, totalEpochs);
        
                % 過学習の重大度判定
                severity = obj.determineOverfittingSeverity(genGap, perfGap);
        
                % メトリクスの構築
                metrics = struct(...
                    'generalizationGap', genGap, ...
                    'performanceGap', perfGap, ...
                    'validationTrend', valTrend, ...
                    'trainingTrend', trainTrend, ...
                    'severity', severity, ...
                    'optimalEpoch', optimalEpoch, ...
                    'totalEpochs', totalEpochs);
                
                isOverfit = ismember(severity, {'critical', 'severe', 'moderate'});
                fprintf('過学習判定: %s (重大度: %s)\n', mat2str(isOverfit), severity);
        
            catch ME
                fprintf('\n=== 過学習検証中にエラーが発生 ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラースタック:\n');
                disp(getReport(ME, 'extended'));
                
                metrics = struct(...
                    'generalizationGap', Inf, ...
                    'performanceGap', Inf, ...
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
        function severity = determineOverfittingSeverity(~, genGap, perfGap)
            if genGap > 10 || perfGap > 10
                severity = 'severe';
            elseif genGap > 5 || perfGap > 5
                severity = 'moderate';
            elseif genGap > 3 || perfGap > 3
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