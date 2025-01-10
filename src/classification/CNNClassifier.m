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
    end
    
    properties (Access = public)
        performance        % 性能評価指標
    end
    
    methods (Access = public)
        function obj = CNNClassifier(params)
            obj.params = params;
            obj.isEnabled = params.classifier.cnn.enable;
            obj.isInitialized = false;
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
                
                % データ分割（学習用/訓練用）
                [trainData, trainLabels, testData, testLabels] = obj.splitDataset(processedData, processedLabel);
                
                prepTrainData = obj.prepareDataForCNN(trainData{1});
                prepTestData = obj.prepareDataForCNN(testData{1});

                % CNNモデルの構築と学習
                fprintf('\nTraining final model...\n');
                [testCNNModel, trainInfo] = obj.trainCNNModel(prepTrainData, trainLabels{1}, prepTestData, testLabels{1});
                testMetrics = obj.evaluateModel(testCNNModel, prepTestData, testLabels{1});

                % 過学習の検出と評価
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);
                if isOverfit
                    warning('Overfitting detected: %s severity', obj.overfitMetrics.severity);
                end

                % 性能指標の更新
                obj.updatePerformanceMetrics(testMetrics);
                
                % 交差検証の実行    
                crossValidationResults = [];        % クロスバリデーション結果の初期化
                if obj.params.classifier.cnn.training.validation.enable
                    crossValidationResults = obj.performCrossValidation(processedData, processedLabel);
                    fprintf('Cross-validation mean accuracy: %.2f%% (±%.2f%%)\n', ...
                        crossValidationResults.meanAccuracy * 100, ...
                        crossValidationResults.stdAccuracy * 100);
                end
                
                [cnnModel, ~] = obj.trainCNNModel(processedData, processedLabel, [], []);      %   検証なしで 全データを学習
                obj.net = cnnModel;

                % 結果の構築
                results = struct(...
                    'model', cnnModel, ...
                    'performance', obj.performance, ...
                    'trainingInfo', trainInfo, ...
                    'crossValidation', crossValidationResults, ...
                    'overfitting', obj.overfitMetrics ...
                );

                % 結果の表示
                obj.displayResults();

            catch ME
                fprintf('\n=== Error in CNN Training ===\n');
                fprintf('Error message: %s\n', ME.message);
                fprintf('Error stack trace:\n');
                for i = 1:length(ME.stack)
                    fprintf('  File: %s\n  Line: %d\n  Function: %s\n\n', ...
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
        
        function [trainData, trainLabels, testData, testLabels] = splitDataset(obj, data, labels)
            try
                % データサイズの取得と表示
                [~, ~, numEpochs] = size(data);
                fprintf('Total epochs: %d\n', numEpochs);

                % データとラベルのインデックスをシャッフル
                rng('default'); % 再現性のため
                shuffledIdx = randperm(numEpochs);

                % k値の計算
                k = round(1 / obj.params.classifier.cnn.training.validation.holdout);
                fprintf('Number of folds (k): %d\n', k);

                % 各フォールドのサイズを計算
                foldSize = floor(numEpochs / k);

                % データの分割
                trainData = cell(k, 1);
                testData = cell(k, 1);
                trainLabels = cell(k, 1);
                testLabels = cell(k, 1);

                for i = 1:k
                    % テストデータのインデックス範囲を計算
                    testStartIdx = (i-1) * foldSize + 1;
                    testEndIdx = min(i * foldSize, numEpochs);
                    testIdx = shuffledIdx(testStartIdx:testEndIdx);

                    % 訓練データのインデックスを取得
                    trainIdx = shuffledIdx(setdiff(1:numEpochs, testStartIdx:testEndIdx));

                    % データを分割して保存
                    testData{i} = data(:,:,testIdx);
                    trainData{i} = data(:,:,trainIdx);
                    testLabels{i} = labels(testIdx);
                    trainLabels{i} = labels(trainIdx);

                    % 分割状況の出力
                    fprintf('Fold %d - Train: %d samples, Test: %d samples\n', ...
                        i, length(trainIdx), length(testIdx));
                end

            catch ME
                error('Data splitting failed: %s', ME.message);
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

        function [cnnModel, trainInfo] = trainCNNModel(obj, trainData, trainLabels, testData, testLabels)
            try
                % トレーニング情報の保存用構造体を初期化
                trainInfo = struct(...
                    'TrainingLoss', [], ...
                    'ValidationLoss', [], ...
                    'TrainingAccuracy', [], ...
                    'ValidationAccuracy', [], ...
                    'FinalEpoch', 0 ...
                );

                % データの準備
                if ndims(trainData) ~= 4
                    trainData = obj.prepareDataForCNN(trainData);
                    if ~isempty(testData)
                        testData = obj.prepareDataForCNN(testData);
                    end
                end

                % ラベルの整形
                trainLabels = categorical(trainLabels(:));
                if ~isempty(testLabels)
                    testLabels = categorical(testLabels(:));
                end
                
                % トレーニングオプションの設定
                options = trainingOptions(obj.params.classifier.cnn.training.optimizer.type, ...
                    'InitialLearnRate', obj.params.classifier.cnn.training.optimizer.learningRate, ...
                    'MaxEpochs', obj.params.classifier.cnn.training.maxEpochs, ...
                    'MiniBatchSize', obj.params.classifier.cnn.training.miniBatchSize, ...
                    'Plots', 'training-progress', ...
                    'OutputFcn', @(info)obj.trainingOutputFcn(info, trainInfo), ...
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
           % 入力サイズの取得と表示
            inputSize = size(data);
            fprintf('Building network with input size: [%s]\n', mat2str(inputSize));

            % CNNの入力サイズを設定 [height width channels]
            layerInputSize = [inputSize(1), inputSize(2), 1];

            % アーキテクチャパラメータの取得
            arch = obj.params.classifier.cnn.architecture;

            % 入力層の設定
            layers = [
                imageInputLayer(layerInputSize, 'Normalization', 'none', 'Name', 'input')
            ];

            % 畳み込み層、プーリング層、ドロップアウト層の追加
            convLayers = {'conv1', 'conv2', 'conv3'};
            for i = 1:length(convLayers)
                convName = convLayers{i};
                convParams = arch.convLayers.(convName);
                poolParams = arch.poolLayers.(['pool' num2str(i)]);
                dropoutRate = arch.dropoutLayers.(['dropout' num2str(i)]);
                
                layers = [layers
                    convolution2dLayer(convParams.size, convParams.filters, ...
                        'Stride', convParams.stride, ...
                        'Padding', convParams.padding, ...
                        'Name', ['conv' num2str(i)])
                ];
                
                if arch.batchNorm
                    layers = [layers
                        batchNormalizationLayer('Name', ['bn' num2str(i)])
                    ];
                end
                
                layers = [layers
                    reluLayer('Name', ['relu' num2str(i)])
                    maxPooling2dLayer(poolParams.size, ...
                        'Stride', poolParams.stride, ...
                        'Name', ['pool' num2str(i)])
                    dropoutLayer(dropoutRate, 'Name', ['dropout' num2str(i)])
                ];
            end

            % 全結合層の追加
            for i = 1:length(arch.fullyConnected)
                layers = [layers
                    fullyConnectedLayer(arch.fullyConnected(i), ...
                        'Name', ['fc' num2str(i)])
                    reluLayer('Name', ['relu_fc' num2str(i)])
                ];
            end

            % 出力層の追加
            layers = [layers
                fullyConnectedLayer(arch.numClasses, 'Name', 'fc_output')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];
            
            % レイヤー構成の表示
            fprintf('\nNetwork Architecture:\n');
            for i = 1:length(layers)
                fprintf('%s\n', layers(i).Name);
            end
            fprintf('\n');
        end

        function stop = trainingOutputFcn(obj, info, savedInfo)
            % 初期化
            if info.State == "start"
                obj.trainingHistory = struct('loss', [], 'accuracy', []);
                obj.validationHistory = struct('loss', [], 'accuracy', []);
                obj.bestValAccuracy = 0;
                obj.patienceCounter = 0;
                obj.currentEpoch = 0;
            end

            stop = false;
            obj.currentEpoch = obj.currentEpoch + 1;

            % 学習情報の保存
            if ~isempty(info.TrainingLoss)
                savedInfo.TrainingLoss(obj.currentEpoch) = info.TrainingLoss;
                savedInfo.TrainingAccuracy(obj.currentEpoch) = info.TrainingAccuracy;
            end

            if ~isempty(info.ValidationLoss)
                savedInfo.ValidationLoss(obj.currentEpoch) = info.ValidationLoss;
                savedInfo.ValidationAccuracy(obj.currentEpoch) = info.ValidationAccuracy;
                currentAccuracy = info.ValidationAccuracy;

                % Early Stopping判定
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
            elseif mod(obj.currentEpoch, 10) == 0
                fprintf('\n');
            end
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
        
        function [isOverfit, metrics] = validateOverfitting(obj, trainInfo, testMetrics)
            try
                % trainInfoの内容確認
                disp('=== validateOverfitting: 入力データ確認 ===');
                disp('trainInfo構造体のフィールド:');
                disp(fieldnames(trainInfo));

                if isfield(trainInfo, 'History')
                    disp('History構造体のフィールド:');
                    disp(fieldnames(trainInfo.History));
                end

                % データ検証
                if ~isfield(trainInfo, 'History') || ...
                   ~isfield(trainInfo.History, 'TrainingAccuracy') || ...
                   ~isfield(trainInfo.History, 'ValidationAccuracy')
                    error('学習履歴データが不完全です');
                end

                trainAcc = trainInfo.History.TrainingAccuracy;
                valAcc = trainInfo.History.ValidationAccuracy;

                % データ形状の確認
                fprintf('TrainingAccuracy: サイズ=%s, 型=%s\n', mat2str(size(trainAcc)), class(trainAcc));
                fprintf('ValidationAccuracy: サイズ=%s, 型=%s\n', mat2str(size(valAcc)), class(valAcc));

                % 計算前の値確認
                fprintf('最終TrainingAccuracy: %.4f\n', trainAcc(end));
                fprintf('最終ValidationAccuracy: %.4f\n', valAcc(end));

                genGap = trainAcc(end) - valAcc(end);
                fprintf('計算されたGeneralization Gap: %.4f\n', genGap);

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
                fprintf('計算されたPerformance Gap: %.4f\n', perfGap);

                % Early Stopping分析
                [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(valAcc);
                fprintf('Optimal Epoch: %d/%d\n', optimalEpoch, totalEpochs);

                % 重大度判定
                severity = obj.determineOverfittingSeverity(genGap, perfGap, valTrend);
                fprintf('判定された重大度: %s\n', severity);

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
        
        function [trainTrend, valTrend] = analyzeLearningCurves(~, trainAcc, valAcc)
            disp('=== analyzeLearningCurves: 入力データ確認 ===');
            fprintf('trainAcc: サイズ=%s\n', mat2str(size(trainAcc)));
            fprintf('valAcc: サイズ=%s\n', mat2str(size(valAcc)));

            try
                % データの移動平均計算
                windowSize = min(5, floor(length(trainAcc)/3));
                fprintf('使用する窓サイズ: %d\n', windowSize);

                trainSmooth = movmean(trainAcc, windowSize);
                valSmooth = movmean(valAcc, windowSize);

                % 変化率計算
                trainDiff = diff(trainSmooth);
                valDiff = diff(valSmooth);

                fprintf('差分データのサイズ - Train: %s, Val: %s\n', ...
                    mat2str(size(trainDiff)), mat2str(size(valDiff)));

                % トレンド指標の計算
                trainTrend = struct(...
                    'mean_change', mean(trainDiff), ...
                    'volatility', std(trainDiff), ...
                    'increasing_ratio', sum(trainDiff > 0) / length(trainDiff));

                valTrend = struct(...
                    'mean_change', mean(valDiff), ...
                    'volatility', std(valDiff), ...
                    'increasing_ratio', sum(valDiff > 0) / length(valDiff));

                disp('計算されたトレンド指標:');
                disp(trainTrend);
                disp(valTrend);

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
            disp('=== determineOverfittingSeverity: 入力値確認 ===');
            fprintf('generalizationGap: %.4f\n', genGap);
            fprintf('performanceGap: %.4f\n', perfGap);
            disp('validationTrend:');
            disp(valTrend);

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

                fprintf('判定された重大度: %s\n', severity);

            catch ME
                fprintf('エラー発生in determineOverfittingSeverity: %s\n', ME.message);
                severity = 'error';
            end
        end
        
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(~, valAcc)
            disp('=== findOptimalEpoch: 入力データ確認 ===');
            fprintf('valAcc: サイズ=%s\n', mat2str(size(valAcc)));

            try
                totalEpochs = length(valAcc);
                [~, optimalEpoch] = max(valAcc);

                fprintf('検出された最適エポック: %d/%d\n', optimalEpoch, totalEpochs);

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
                % データ変換の過程を追跡
                if ndims(data) == 3
                    [channels, samples, epochs] = size(data);
                    preparedData = permute(data, [2 1 4 3]);

                elseif ismatrix(data)
                    [channels, samples] = size(data);
                    preparedData = permute(data, [2 1 3]);
                    preparedData = reshape(preparedData, [samples, channels, 1, 1]);

                else
                    error('Unsupported data dimension: %d', ndims(data));
                end

            catch ME
                fprintf('\nError in prepareDataForCNN: %s\n', ME.message);
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