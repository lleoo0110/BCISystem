classdef CNNClassifier < handle
    properties (Access = private)
        params              % パラメータ設定
        net                 % CNNネットワーク
        isEnabled          % 有効/無効フラグ
        isInitialized      % 初期化フラグ
        
        % 学習進捗の追跡用
        trainingHistory    % 学習履歴
        validationHistory  % 検証履歴
        bestValAccuracy    % 最良の検証精度
        patienceCounter    % Early Stopping用カウンター
        currentEpoch
        
        % 過学習監視用
        overfitMetrics     % 過学習メトリクス
        
        % GPUを使用するかどうか
        useGPU = false;
    end
    
    properties (Access = public)
        performance        % 性能評価指標
    end
    
    methods (Access = public)
        function obj = CNNClassifier(params)
            obj.params = params;
            obj.isEnabled = params.classifier.cnn.enable;
            obj.isInitialized = false;
            obj.useGPU = params.classifier.cnn.gpu;
            obj.initializeProperties();
        end
        
        function results = trainCNN(obj, processedData, processedLabel)
            if ~obj.isEnabled
                error('CNN is disabled in configuration');
            end

            try                
                fprintf('\n=== Starting CNN Training ===\n');
                % データの形状を確認して必要に応じて変換
                if ndims(processedData) ~= 3
                    [channels, samples] = size(processedData);
                    processedData = reshape(processedData, [channels, samples, 1]);
                end

                % データ分割（学習用/検証用/テスト用）
                [trainData, trainLabels, testData, testLabels, validData, validLabels] = obj.splitDataset(processedData, processedLabel);
                prepTrainData = obj.prepareDataForCNN(trainData);
                prepTestData = obj.prepareDataForCNN(testData);
                prepValidData = obj.prepareDataForCNN(validData);

                % CNNモデルの構築と学習
                [cnnModel, trainInfo] = obj.trainCNNModel(prepTrainData, trainLabels, prepValidData, validLabels); % 検証データで学習
                obj.net = cnnModel;
                testMetrics = obj.evaluateModel(cnnModel, prepTestData, testLabels);

                % 過学習の検出と評価
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);
                if isOverfit
                    warning('Overfitting detected: %s severity', obj.overfitMetrics.severity);
                end

                % 性能指標の更新
                obj.updatePerformanceMetrics(testMetrics);
                % 交差検証の実行
                crossValidationResults = [];
                if obj.params.classifier.cnn.training.validation.enable
                    crossValidationResults = obj.performCrossValidation(processedData, processedLabel);
                    fprintf('Cross-validation mean accuracy: %.2f%% (\u00b1%.2f%%)\n',...
                    crossValidationResults.meanAccuracy * 100,...
                    crossValidationResults.stdAccuracy * 100);
                end

                % 結果の構築
                results = struct(...
                    'model', cnnModel,...
                    'performance', obj.performance,...
                    'trainInfo', trainInfo,...
                    'crossValidation', crossValidationResults,...
                    'overfitting', obj.overfitMetrics...
                );

                % 結果の表示
                obj.displayResults();

                % すべてのウィンドウを閉じる
                close all;  % 追加

            catch ME
                fprintf('\n=== Error in CNN Training ===\n');
                fprintf('Error message: %s\n', ME.message);
                fprintf('Error stack trace:\n');
                for i = 1:length(ME.stack)
                    fprintf('  File: %s\n  Line: %d\n  Function: %s\n\n',...
                    ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end
                rethrow(ME);
            end
        end
    
        function [label, score] = predictOnline(obj, data, cnnModel)
            if ~obj.isEnabled
                error('CNN is disabled');
            end

            try
                % データの整形
                prepData = obj.prepareDataForCNN(data);
                
                % 予測の実行
                [label, scores] = classify(cnnModel, prepData);
                
                % クラス1（安静状態）の確率を取得
                score = scores(:,1);

            catch ME
                fprintf('Error in online prediction: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        function [isOverfit, metrics] = validateOverfitting(obj, trainInfo, testMetrics)
            try
                disp('=== validateOverfitting ===');
                
                % データ検証
                if ~isfield(trainInfo, 'History') || ...
                   ~isfield(trainInfo.History, 'TrainingAccuracy') || ...
                   ~isfield(trainInfo.History, 'ValidationAccuracy')
                    error('学習履歴データが不完全です');
                end

                trainAcc = trainInfo.History.TrainingAccuracy;
                valAcc = trainInfo.History.ValidationAccuracy;
                
                % 計算前の値確認
                fprintf('Final TrainingAccuracy: %.4f\n', trainAcc(end));
                fprintf('Final ValidationAccuracy: %.4f\n', valAcc(end));

                genGap = trainAcc(end) - valAcc(end);
                fprintf('Generalization Gap: %.4f\n', genGap);

                % トレンド分析
                [trainTrend, valTrend] = obj.analyzeLearningCurves(trainAcc, valAcc);
                disp('トレンド分析結果:');
                disp(trainTrend);
                disp(valTrend);

                % 性能ギャップ
                if ~isfield(testMetrics, 'accuracy')
                    error('testMetricsにaccuracyフィールドがありません');
                end
                fprintf('TestMetrics accuracy: %.4f\n', testMetrics.accuracy);

                perfGap = abs(trainAcc(end) - testMetrics.accuracy);
                fprintf('Performance Gap: %.4f\n', perfGap);

                % Early Stopping分析
                [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(valAcc);
                fprintf('Optimal Epoch: %d/%d\n', optimalEpoch, totalEpochs);

                % 重大度判定
                severity = obj.determineOverfittingSeverity(genGap, perfGap, valTrend);

                % メトリクスの構築と確認
                metrics = struct(...
                    'generalizationGap', genGap, ...
                    'performanceGap', perfGap, ...
                    'earlyStoppingEffect', struct(...
                        'optimal_epoch', optimalEpoch, ...
                        'total_epochs', totalEpochs, ...
                        'stopping_efficiency', optimalEpoch/totalEpochs), ...
                    'validationTrend', valTrend, ...
                    'trainingTrend', trainTrend, ...
                    'severity', severity);

                disp('=== 生成されたメトリクス ===');
                disp(metrics);

                isOverfit = strcmp(severity, 'severe') || strcmp(severity, 'moderate');
                fprintf('過学習判定: %s\n', mat2str(isOverfit));

            catch ME
                fprintf('エラー発生in validateOverfitting: %s\n', ME.message);
                fprintf('エラー位置:\n');
                for i = 1:length(ME.stack)
                    fprintf('  File: %s, Line: %d, Function: %s\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end
                metrics = obj.createDefaultMetrics();
                isOverfit = false;
            end
        end
    end
    
    methods (Access = private)
        function initializeProperties(obj)
            % プロパティの初期化
            obj.trainingHistory = struct('loss', [], 'accuracy', []);
            obj.validationHistory = struct('loss', [], 'accuracy', []);
            obj.bestValAccuracy = 0;
            obj.patienceCounter = 0;
            obj.overfitMetrics = struct();
        end
        
        function [trainData, trainLabels, testData, testLabels, validData, validLabels] = splitDataset(~, data, labels)
            try
                % データサイズの取得と表示
                [~, ~, numEpochs] = size(data);
                fprintf('Total epochs: %d\n', numEpochs);

                % データとラベルのインデックスをシャッフル
                rng('default'); % 再現性のため
                shuffledIdx = randperm(numEpochs);

                % データを8:1:1に分割
                trainRatio = 0.8;
                testRatio = 0.1;
                validRatio = 0.1;

                trainSize = round(numEpochs * trainRatio);
                testSize = round(numEpochs * testRatio);
                validSize = numEpochs - trainSize - testSize;

                % データを分割して保存
                trainData = data(:,:,shuffledIdx(1:trainSize));
                testData = data(:,:,shuffledIdx(trainSize+1:trainSize+testSize));
                validData = data(:,:,shuffledIdx(trainSize+testSize+1:end));
                trainLabels = labels(shuffledIdx(1:trainSize));
                testLabels = labels(shuffledIdx(trainSize+1:trainSize+testSize));
                validLabels = labels(shuffledIdx(trainSize+testSize+1:end));

                % 分割状況の出力
                fprintf('Train: %d samples, Test: %d samples, Validation: %d samples\n',...
                    trainSize, testSize, validSize);

            catch ME
                error('Data splitting failed: %s', ME.message);
            end
        end
        
        function [cnnModel, trainInfo] = trainCNNModel(obj, trainData, trainLabels, testData, testLabels)
            try
                % データの準備
                if ndims(trainData) ~= 4
                    trainData = obj.prepareDataForCNN(trainData);
                end

                % ラベルのカテゴリカル変換
                uniqueLabels = unique(trainLabels);
                trainLabels = categorical(trainLabels, uniqueLabels);

                % テストデータの処理
                if ~isempty(testData)
                    if ndims(testData) ~= 4
                        testData = obj.prepareDataForCNN(testData);
                    end
                    testLabels = categorical(testLabels, uniqueLabels);
                end

                % トレーニング情報の初期化
                trainInfo = struct(...
                    'TrainingLoss', [], ...
                    'ValidationLoss', [], ...
                    'TrainingAccuracy', [], ...
                    'ValidationAccuracy', [], ...
                    'FinalEpoch', 0 ...
                );

                % 実行環境の選択
                executionEnvironment = 'cpu';
                if obj.useGPU && canUseGPU()
                    executionEnvironment = 'gpu';
                end

                % トレーニングオプションの設定
                options = trainingOptions(obj.params.classifier.cnn.training.optimizer.type, ...
                    'InitialLearnRate', obj.params.classifier.cnn.training.optimizer.learningRate, ...
                    'MaxEpochs', obj.params.classifier.cnn.training.maxEpochs, ...
                    'MiniBatchSize', obj.params.classifier.cnn.training.miniBatchSize, ...
                    'Plots', 'training-progress', ...
                    'OutputFcn', @(info)obj.trainingOutputFcn(info), ...
                    'ExecutionEnvironment', executionEnvironment, ...
                    'Verbose', true);

                % 検証データの設定
                if ~isempty(testData) && ~isempty(testLabels)
                    options.ValidationData = {testData, testLabels};
                    options.ValidationFrequency = obj.params.classifier.cnn.training.validation.frequency;
                    options.ValidationPatience = obj.params.classifier.cnn.training.validation.patience;
                    fprintf('検証データを使用して学習を開始します\n');
                else
                    fprintf('検証データなしで学習を開始します\n');
                end

                % レイヤーの構築とモデルの学習
                layers = obj.buildCNNLayers(trainData);
                [cnnModel, trainHistory] = trainNetwork(trainData, trainLabels, layers, options);

                % 学習履歴の保存
                trainInfo.History = trainHistory;
                trainInfo.FinalEpoch = length(trainHistory.TrainingLoss);

                fprintf('学習完了: 最終エポック %d\n', trainInfo.FinalEpoch);

            catch ME
                fprintf('trainCNNModelでエラーが発生: %s\n', ME.message);
                rethrow(ME);
            end
        end

        function layers = buildCNNLayers(obj, data)
            % 入力サイズの取得
            inputSize = size(data);

            % CNNの入力サイズを設定 [height width channels]
            layerInputSize = [inputSize(1), inputSize(2), 1];

            % アーキテクチャパラメータの取得
            arch = obj.params.classifier.cnn.architecture;

            % レイヤー数の計算
            numConvLayers = 3;
            numFCLayers = length(arch.fullyConnected);
            totalLayers = numConvLayers * 5 + numFCLayers * 2 + 4; % 各畳み込み層に5つ、FC層に2つ、その他4つ

            % レイヤー配列の事前割り当て
            layers = cell(totalLayers, 1);
            layerIdx = 1;

            % 入力層
            layers{layerIdx} = imageInputLayer(layerInputSize, 'Normalization', 'none', 'Name', 'input');
            layerIdx = layerIdx + 1;

            % 畳み込み層、プーリング層、ドロップアウト層の追加
            convLayers = {'conv1', 'conv2', 'conv3'};
            for i = 1:length(convLayers)
                convName = convLayers{i};
                convParams = arch.convLayers.(convName);
                poolParams = arch.poolLayers.(['pool' num2str(i)]);
                dropoutRate = arch.dropoutLayers.(['dropout' num2str(i)]);

                layers{layerIdx} = convolution2dLayer(convParams.size, convParams.filters, ...
                    'Stride', convParams.stride, 'Padding', convParams.padding, ...
                    'Name', ['conv' num2str(i)]);
                layerIdx = layerIdx + 1;

                if arch.batchNorm
                    layers{layerIdx} = batchNormalizationLayer('Name', ['bn' num2str(i)]);
                    layerIdx = layerIdx + 1;
                end

                layers{layerIdx} = reluLayer('Name', ['relu' num2str(i)]);
                layerIdx = layerIdx + 1;
                layers{layerIdx} = maxPooling2dLayer(poolParams.size, 'Stride', poolParams.stride, ...
                    'Name', ['pool' num2str(i)]);
                layerIdx = layerIdx + 1;
                layers{layerIdx} = dropoutLayer(dropoutRate, 'Name', ['dropout' num2str(i)]);
                layerIdx = layerIdx + 1;
            end

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

        function metrics = evaluateModel(~, model, testData, testLabels)
            metrics = struct(...
                'accuracy', [], ...
                'confusionMat', [], ...
                'classwise', [], ...
                'roc', [], ...
                'auc', [] ...
            );
    
           % テストラベルをカテゴリカル型に変換
           uniqueLabels = unique(testLabels);
           testLabels = categorical(testLabels, uniqueLabels);

           % モデルの評価
           [pred, scores] = classify(model, testData);

           % 基本的な指標の計算
           metrics.accuracy = mean(pred == testLabels);
           metrics.confusionMat = confusionmat(testLabels, pred);

           % クラスごとの性能評価
           classes = unique(testLabels);
           metrics.classwise = struct('precision', zeros(1,length(classes)), ...
                                       'recall', zeros(1,length(classes)), ...
                                       'f1score', zeros(1,length(classes)));

           for i = 1:length(classes)
               className = classes(i);
               classIdx = (testLabels == className);

               % 各クラスの指標計算
               TP = sum(pred(classIdx) == className);
               FP = sum(pred == className) - TP;
               FN = sum(classIdx) - TP;

               precision = TP / (TP + FP);
               recall = TP / (TP + FN);
               f1 = 2 * (precision * recall) / (precision + recall);

               metrics.classwise(i).precision = precision;
               metrics.classwise(i).recall = recall;
               metrics.classwise(i).f1score = f1;
           end

           % ROC曲線とAUC（2クラス分類の場合）
           if length(classes) == 2
               [X, Y, T, AUC] = perfcurve(testLabels, scores(:,2), classes(2));
               metrics.roc = struct('X', X, 'Y', Y, 'T', T);
               metrics.auc = AUC;
           end
        end
        
        function [trainTrend, valTrend] = analyzeLearningCurves(~, trainAcc, valAcc)
            disp('=== analyzeLearningCurves ===');
            
            try
                % データの移動平均計算
                windowSize = min(5, floor(length(trainAcc)/3));
                trainSmooth = movmean(trainAcc, windowSize);
                valSmooth = movmean(valAcc, windowSize);

                % 変化率計算
                trainDiff = diff(trainSmooth);
                valDiff = diff(valSmooth);

                % トレンド指標の計算
                trainTrend = struct(...
                    'mean_change', mean(trainDiff), ...
                    'volatility', std(trainDiff), ...
                    'increasing_ratio', sum(trainDiff > 0) / length(trainDiff));

                valTrend = struct(...
                    'mean_change', mean(valDiff), ...
                    'volatility', std(valDiff), ...
                    'increasing_ratio', sum(valDiff > 0) / length(valDiff));

            catch ME
                fprintf('エラー発生in analyzeLearningCurves: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        function effect = analyzeEarlyStoppingEffect(~, trainInfo)
            % Early Stoppingの効果分析
            valLoss = trainInfo.ValidationLoss;
            minLossEpoch = find(valLoss == min(valLoss), 1);
            totalEpochs = length(valLoss);
            
            effect = struct(...
                'optimal_epoch', minLossEpoch, ...
                'total_epochs', totalEpochs, ...
                'stopping_efficiency', minLossEpoch / totalEpochs);
        end
        
        function severity = determineOverfittingSeverity(~, genGap, perfGap, valTrend)
            disp('=== determineOverfittingSeverity ===');
            fprintf('generalizationGap: %.4f\n', genGap);
            fprintf('performanceGap: %.4f\n', perfGap);

            try
                if genGap > 0.2 && (valTrend.mean_change < -0.01 || perfGap > 0.2)
                    severity = 'severe';
                elseif genGap > 0.1 && (valTrend.mean_change < -0.005 || perfGap > 0.1)
                    severity = 'moderate';
                elseif genGap > 0.05 || perfGap > 0.05
                    severity = 'mild';
                else
                    severity = 'none';
                end

                fprintf('重大度: %s\n', severity);

            catch ME
                fprintf('エラー発生in determineOverfittingSeverity: %s\n', ME.message);
                severity = 'error';
            end
        end
        
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(~, valAcc)
            disp('=== findOptimalEpoch: 入力データ確認 ===');
            try
                totalEpochs = length(valAcc);
                [~, optimalEpoch] = max(valAcc);

                fprintf('最適エポック: %d/%d\n', optimalEpoch, totalEpochs);

            catch ME
                fprintf('エラー発生in findOptimalEpoch: %s\n', ME.message);
                optimalEpoch = 0;
                totalEpochs = 0;
            end
        end

        function metrics = createDefaultMetrics(~)
            metrics = struct(...
                'generalizationGap', 0, ...
                'performanceGap', 0, ...
                'earlyStoppingEffect', struct(...
                    'optimal_epoch', 0, ...
                    'total_epochs', 0, ...
                    'stopping_efficiency', 0), ...
                'validationTrend', struct(...
                    'mean_change', 0, ...
                    'volatility', 0, ...
                    'increasing_ratio', 0), ...
                'trainingTrend', struct(...
                    'mean_change', 0, ...
                    'volatility', 0, ...
                    'increasing_ratio', 0), ...
                'severity', 'unknown');
        end
        
        function preparedData = prepareDataForCNN(~, data)
            try
                if ndims(data) == 3
                    preparedData = permute(data, [2 1 4 3]);
                elseif ismatrix(data)
                    [~, samples] = size(data);
                    preparedData = permute(data, [2 1 3]);
                    preparedData = reshape(preparedData, [samples, size(data,1), 1, 1]);
                else
                    error('Unsupported data dimension: %d', ndims(data));
                end
            catch ME
                fprintf('\nError in prepareDataForCNN: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        function stop = trainingOutputFcn(obj, info)
            % 初期化
            stop = false;

            if info.State == "start"
                obj.currentEpoch = 0;
                return;
            end

            % 学習情報の更新
            obj.currentEpoch = obj.currentEpoch + 1;

            % 検証データがある場合のEarly Stopping
            if ~isempty(info.ValidationLoss)
                currentAccuracy = info.ValidationAccuracy;

                if currentAccuracy > obj.bestValAccuracy
                    obj.bestValAccuracy = currentAccuracy;
                    obj.patienceCounter = 0;
                else
                    obj.patienceCounter = obj.patienceCounter + 1;
                    if obj.patienceCounter >= obj.params.classifier.cnn.training.validation.patience
                        fprintf('\nEarly stopping triggered at epoch %d\n', obj.currentEpoch);
                        stop = true;
                    end
                end
            end
        end
        
        function cvResults = performCrossValidation(obj, data, labels)
            try
                % クロスバリデーションのパラメータ取得
                k = round(1 / obj.params.classifier.cnn.training.validation.holdout);

                % cvResultsの初期化を修正
                cvResults = struct();
                cvResults.meanAccuracy = 0;
                cvResults.stdAccuracy = 0;
                cvResults.folds = struct();
                cvResults.folds.accuracy = zeros(1, k);
                cvResults.folds.confusionMat = cell(1, k);
                cvResults.folds.classwise = cell(1, k);

                % データの分割（学習・テスト用）
                [trainData, trainLabels, testData, testLabels] = obj.splitDataset(data, labels);

                for i = 1:k
                    fprintf('Fold %d/%d...\n', i, k);

                    % データの準備
                    prepTrainData = obj.prepareDataForCNN(trainData{i});
                    prepTestData = obj.prepareDataForCNN(testData{i});

                    % モデルの学習
                    [model, ~] = obj.trainCNNModel(prepTrainData, trainLabels{i}, prepTestData, testLabels{i});

                    % テストデータでの評価
                    metrics = obj.evaluateModel(model, prepTestData, testLabels{i});

                    cvResults.folds.accuracy(i) = metrics.accuracy;
                    cvResults.folds.confusionMat{i} = metrics.confusionMat;
                    cvResults.folds.classwise{i} = metrics.classwise;

                    fprintf('Fold %d accuracy: %.2f%%\n', i, cvResults.folds.accuracy(i) * 100);
                end

                % 統計の計算
                cvResults.meanAccuracy = mean(cvResults.folds.accuracy);
                cvResults.stdAccuracy = std(cvResults.folds.accuracy);

                fprintf('\nCross-validation Results:\n');
                fprintf('Mean Accuracy: %.2f%% (±%.2f%%)\n', ...
                    cvResults.meanAccuracy * 100, cvResults.stdAccuracy * 100);

            catch ME
                fprintf('Cross-validation failed: %s\n', ME.message);
                fprintf('Error details:\n');
                disp(ME.stack);
                rethrow(ME);
            end
        end
        
        function updatePerformanceMetrics(obj, testMetrics)
            % 性能指標の更新
            obj.performance = testMetrics;
        end
        
        function displayResults(obj)
            try
                fprintf('\n=== CNN Classification Results ===\n');
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

                % 過学習分析の表示
                if ~isempty(obj.overfitMetrics) && isstruct(obj.overfitMetrics)
                    fprintf('\nOverfitting Analysis:\n');
                    fprintf('Generalization Gap: %.4f\n', obj.overfitMetrics.generalizationGap);
                    fprintf('Performance Gap: %.4f\n', obj.overfitMetrics.performanceGap);
                    fprintf('Severity: %s\n', obj.overfitMetrics.severity);

                    if isfield(obj.overfitMetrics, 'earlyStoppingEffect')
                        fprintf('\nEarly Stopping Effect:\n');
                        effect = obj.overfitMetrics.earlyStoppingEffect;
                        fprintf('Optimal epoch: %d/%d\n', effect.optimal_epoch, effect.total_epochs);
                        fprintf('Stopping efficiency: %.2f%%\n', effect.stopping_efficiency * 100);
                    end

                    if isfield(obj.overfitMetrics, 'validationTrend')
                        fprintf('\nValidation Trend:\n');
                        trend = obj.overfitMetrics.validationTrend;
                        fprintf('Mean Change: %.4f\n', trend.mean_change);
                        fprintf('Volatility: %.4f\n', trend.volatility);
                        fprintf('Increasing Ratio: %.2f%%\n', trend.increasing_ratio * 100);
                    end
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
    end
end