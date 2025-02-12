classdef HybridClassifier < handle
    properties (Access = private)
        params              % パラメータ設定
        net                 % ハイブリッドネットワーク
        isEnabled          % 有効/無効フラグ
        isInitialized      % 初期化フラグ
        useGPU             % GPUを使用するかどうか
        
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
        function obj = HybridClassifier(params)
            obj.params = params;
            obj.isEnabled = params.classifier.hybrid.enable;
            obj.isInitialized = false;
            obj.initializeProperties();
            obj.useGPU = params.classifier.hybrid.gpu;
        end

        function results = trainHybrid(obj, processedData, processedLabel)
            if ~obj.isEnabled
                error('Hybrid model is disabled in configuration');
            end

            try                
                fprintf('\n=== Starting Hybrid Model Training ===\n');

                % データの形状を確認して必要に応じて変換
                if ndims(processedData) ~= 3
                    [channels, samples] = size(processedData);
                    processedData = reshape(processedData, [channels, samples, 1]);
                end
                
                % データを3セットに分割
                [trainData, trainLabels, valData, valLabels, testData, testLabels] = ...
                    obj.splitDataset(processedData, processedLabel);
                
                % データの準備
                prepTrainData = obj.prepareDataForHybrid(trainData);
                prepValData = obj.prepareDataForHybrid(valData);
                prepTestData = obj.prepareDataForHybrid(testData);
                
                % モデルの学習（検証データを使用）
                [hybridModel, trainInfo] = obj.trainHybridModel(prepTrainData, trainLabels, prepValData, valLabels);
                
                % テストデータでの最終評価
                testMetrics = obj.evaluateModel(hybridModel, prepTestData, testLabels);
                
                % 過学習の検証
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);
                if isOverfit
                    warning('Overfitting detected: %s severity', obj.overfitMetrics.severity);
                end

                % 性能指標の更新
                obj.updatePerformanceMetrics(testMetrics);
                
                % 交差検証の実行
                crossValidationResults = struct('meanAccuracy', [], 'stdAccuracy', []);
                if obj.params.classifier.hybrid.training.validation.enable
                    crossValidationResults = obj.performCrossValidation(processedData, processedLabel);
                    fprintf('Cross-validation mean accuracy: %.2f%% (±%.2f%%)\n', ...
                        crossValidationResults.meanAccuracy * 100, ...
                        crossValidationResults.stdAccuracy * 100);
                end

                % 結果構造体の構築
                results = struct(...
                    'model', hybridModel, ...
                    'performance', struct(...
                        'overallAccuracy', testMetrics.accuracy, ...
                        'crossValidation', struct(...
                            'accuracy', crossValidationResults.meanAccuracy, ...
                            'std', crossValidationResults.stdAccuracy ...
                        ), ...
                        'precision', testMetrics.classwise(1).precision, ...
                        'recall', testMetrics.classwise(1).recall, ...
                        'f1score', testMetrics.classwise(1).f1score, ...
                        'auc', testMetrics.auc, ...
                        'confusionMatrix', testMetrics.confusionMat ...
                    ), ...
                    'trainInfo', trainInfo, ...
                    'overfitting', obj.overfitMetrics ...
                );

                % 結果の表示
                obj.displayResults();

                % GPUメモリを解放
                if obj.useGPU
                    gpuDevice([]);
                end

            catch ME
                fprintf('\n=== Error in Hybrid Model Training ===\n');
                fprintf('Error message: %s\n', ME.message);
                fprintf('Error stack trace:\n');
                for i = 1:length(ME.stack)
                    fprintf('  File: %s\n  Line: %d\n  Function: %s\n\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end

                if obj.useGPU
                    gpuDevice([]);
                end

                rethrow(ME);
            end
        end

        function [label, score] = predictOnline(obj, data, hybridModel)
            if ~obj.isEnabled
                error('Hybrid model is disabled');
            end

            try
                % データの整形
                prepData = obj.prepareDataForHybrid(data);
                
                % 予測の実行
                [label, scores] = classify(hybridModel, prepData);
                
                % クラス1の確率を取得
                score = scores(:,1);
                
            catch ME
                fprintf('Error in hybrid model online prediction: %s\n', ME.message);
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

        function [trainData, trainLabels, valData, valLabels, testData, testLabels] = splitDataset(obj, data, labels)
            try
                % 分割数の取得
                k = obj.params.classifier.evaluation.kfold;
                
                % データサイズの取得
                [~, ~, numTrials] = size(data);
                fprintf('Total epochs: %d\n', numTrials);
        
                % インデックスのシャッフル
                rng('default'); % 再現性のため
                shuffledIdx = randperm(numTrials);
        
                % 分割比率の計算
                trainRatio = (k-1)/k;  % 1-k/k
                valRatio = 1/(2*k);    % k/2k
                testRatio = 1/(2*k);   % k/2k
        
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
        
                % 分割情報の表示
                fprintf('データ分割 (k=%d):\n', k);
                fprintf('  訓練データ: %d サンプル (%.1f%%)\n', ...
                    length(trainIdx), (length(trainIdx)/numTrials)*100);
                fprintf('  検証データ: %d サンプル (%.1f%%)\n', ...
                    length(valIdx), (length(valIdx)/numTrials)*100);
                fprintf('  テストデータ: %d サンプル (%.1f%%)\n', ...
                    length(testIdx), (length(testIdx)/numTrials)*100);
        
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

        function checkClassDistribution(~, setName, labels)
            uniqueLabels = unique(labels);
            fprintf('\n%sデータのクラス分布:\n', setName);
            for i = 1:length(uniqueLabels)
                count = sum(labels == uniqueLabels(i));
                fprintf('  クラス %d: %d サンプル (%.1f%%)\n', ...
                    uniqueLabels(i), count, (count/length(labels))*100);
            end
        end

        function preparedData = prepareDataForHybrid(obj, data)
            try
                % CNN部分のデータ準備
                [channels, timepoints, trials] = size(data);
                cnnInput = zeros(obj.params.classifier.hybrid.cnn.inputSize);
                
                % 時系列データをCNN入力形式に変換
                for i = 1:trials
                    cnnInput(:,:,:,i) = reshape(data(:,:,i), ...
                        obj.params.classifier.hybrid.cnn.inputSize);
                end
                
                % LSTM部分のデータ準備
                lstmInput = permute(data, [2,1,3]); % 時系列を最初の次元に
                
                % データの結合
                preparedData = struct('cnn', cnnInput, 'lstm', lstmInput);
                
                % GPU転送
                if obj.useGPU
                    preparedData.cnn = gpuArray(preparedData.cnn);
                    preparedData.lstm = gpuArray(preparedData.lstm);
                end
                
            catch ME
                error('Hybrid data preparation failed: %s', ME.message);
            end
        end

        function layers = buildHybridLayers(obj, inputSize)
            % CNNパート
            cnnParams = obj.params.classifier.hybrid.cnn;
            cnnLayers = [
                imageInputLayer(inputSize, 'Name', 'input')
                
                convolution2dLayer(cnnParams.filterSize, cnnParams.numFilters, ...
                    'Padding', 'same', 'Name', 'conv1')
                batchNormalizationLayer('Name', 'bn1')
                reluLayer('Name', 'relu1')
                maxPooling2dLayer(cnnParams.poolSize, 'Stride', cnnParams.poolStride, ...
                    'Name', 'pool1')
                dropoutLayer(0.5, 'Name', 'drop1')
            ];

            % LSTM パート
            lstmParams = obj.params.classifier.hybrid.lstm.architecture;
            lstmLayers = [];
            for i = 1:numel(fieldnames(lstmParams.lstmLayers))
                layerName = sprintf('lstm%d', i);
                layer = lstmParams.lstmLayers.(layerName);
                
                lstmLayers = [lstmLayers
                    lstmLayer(layer.numHiddenUnits, 'Name', layerName)
                    dropoutLayer(lstmParams.dropoutLayers.(['dropout' num2str(i)]), ...
                        'Name', ['dropout_' layerName])
                ];
            end

            % 全結合層パート
            fcParams = obj.params.classifier.hybrid.fc;
            fcLayers = [
                fullyConnectedLayer(fcParams.numUnits, 'Name', 'fc1')
                reluLayer('Name', 'relu_fc1')
                dropoutLayer(0.5, 'Name', 'drop_fc1')
                fullyConnectedLayer(cnnParams.numClasses, 'Name', 'fc_final')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];

            % レイヤーの結合
            layers = [cnnLayers; lstmLayers; fcLayers];
        end

        function [hybridModel, trainInfo] = trainHybridModel(obj, trainData, trainLabels, valData, valLabels)
            try
                % レイヤーの構築
                layers = obj.buildHybridLayers(size(trainData.cnn, [1,2,3]));

                % 実行環境の選択
               executionEnvironment = 'cpu';
               if obj.useGPU
                   executionEnvironment = 'gpu';
               end
                
                % トレーニングオプションの設定
                training = obj.params.classifier.hybrid.training;
                options = trainingOptions(training.optimizerType, ...
                    'InitialLearnRate', training.learningRate, ...
                    'MaxEpochs', training.maxEpochs, ...
                    'MiniBatchSize', training.miniBatchSize, ...
                    'ValidationFrequency', training.validationFrequency, ...
                    'ValidationPatience', training.validationPatience, ...
                    'Shuffle', 'every-epoch', ...
                    'Verbose', true, ...
                    'Plots', 'training-progress', ...
                    'ExecutionEnvironment', executionEnvironment );
                
                % 検証データの設定
                if ~isempty(valData)
                    options.ValidationData = {valData, categorical(valLabels)};
                end
                
                % モデルの学習
                [hybridModel, trainInfo] = trainNetwork(trainData, categorical(trainLabels), ...
                    layers, options);
                
                % GPU メモリの解放
                if obj.useGPU
                    gpuDevice([]);
                end
                
            catch ME
                if obj.useGPU
                    gpuDevice([]);
                end
                rethrow(ME);
            end
        end

        function metrics = evaluateModel(~, model, testData, testLabels)
            metrics = struct(...
                'accuracy', [], ...
                'confusionMat', [], ...
                'classwise', [], ...
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
                fprintf('\n=== 過学習の検証 ===\n');
                
                % trainInfoの検証
                if ~isstruct(trainInfo) || ~isfield(trainInfo, 'TrainingAccuracy')
                    error('学習履歴情報が不正です');
                end
                
                % 精度データの取得と表示
                trainAcc = trainInfo.TrainingAccuracy;
                valAcc = trainInfo.ValidationAccuracy;
                testAcc = testMetrics.accuracy * 100;
                
                fprintf('訓練精度: %.2f%%\n', trainAcc(end));
                fprintf('検証精度: %.2f%%\n', valAcc(end));
                fprintf('テスト精度: %.2f%%\n', testAcc);
        
                % Generalization GapとPerformance Gapの計算
                genGap = abs(trainAcc(end) - valAcc(end));
                perfGap = abs(trainAcc(end) - testAcc);
                
                fprintf('\nGeneralization Gap: %.2f%%\n', genGap);
                fprintf('Performance Gap: %.2f%%\n', perfGap);
        
                % 完全な偏りの検出
                if isfield(testMetrics, 'confusionMat')
                    cm = testMetrics.confusionMat;
                    missingActual = any(sum(cm, 2) == 0);
                    missingPredicted = any(sum(cm, 1) == 0);
                    isCompletelyBiased = missingActual || missingPredicted;
                else
                    isCompletelyBiased = false;
                end
        
                % 学習進行の確認
                trainDiff = diff(trainAcc);
                isLearningProgressing = std(trainDiff) > 0.01;
        
                % 最適エポックの検出
                [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(valAcc);
                fprintf('Optimal Epoch: %d/%d\n', optimalEpoch, totalEpochs);
        
                % 過学習の重大度判定
                severity = obj.determineOverfittingSeverity(genGap, perfGap, isCompletelyBiased, isLearningProgressing);
        
                % メトリクスの構築
                metrics = struct(...
                    'generalizationGap', genGap, ...
                    'performanceGap', perfGap, ...
                    'isCompletelyBiased', isCompletelyBiased, ...
                    'isLearningProgressing', isLearningProgressing, ...
                    'severity', severity, ...
                    'optimalEpoch', optimalEpoch, ...
                    'totalEpochs', totalEpochs);
                
                % 過学習判定
                isOverfit = ismember(severity, {'critical', 'severe', 'moderate'});
                fprintf('過学習判定: %s (重大度: %s)\n', mat2str(isOverfit), severity);
        
            catch ME
                fprintf('エラー発生in validateOverfitting: %s\n', ME.message);
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

        function severity = determineOverfittingSeverity(~, genGap, perfGap, isCompletelyBiased, isLearningProgressing)
            if isCompletelyBiased
                severity = 'critical';
            elseif ~isLearningProgressing
                severity = 'failed';
            elseif genGap > 10 || perfGap > 10
                severity = 'severe';
            elseif genGap > 5 || perfGap > 5
                severity = 'moderate';
            elseif genGap > 3 || perfGap > 3
                severity = 'mild';
            else
                severity = 'none';
            end
        end
        
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(~, valAcc)
            totalEpochs = length(valAcc);
            [~, optimalEpoch] = max(valAcc);
            
            % 最適エポックが最後のエポックの場合の警告
            if optimalEpoch == totalEpochs
                fprintf('Warning: 最適エポックが最終エポックと一致。より長い学習が必要かもしれません。\n');
            end
        end

        function results = performCrossValidation(obj, data, labels)
            try
                % k-fold cross validationのパラメータ取得
                k = obj.params.classifier.hybrid.evaluation.kfold;
                fprintf('\n=== %d分割交差検証開始 ===\n', k);
        
                % 結果保存用構造体の初期化
                results = struct();
                results.folds = struct(...
                    'accuracy', zeros(1, k), ...
                    'confusionMat', cell(1, k), ...
                    'classwise', cell(1, k), ...
                    'validation_curve', cell(1, k) ...
                );
        
                for i = 1:k
                    fprintf('\nFold %d/%d の処理開始\n', i, k);
                    
                    try
                        % データの分割と前処理
                        [trainData, trainLabels, valData, valLabels, ~, ~] = ...
                            obj.splitDataset(data, labels);
                        
                        prepTrainData = obj.prepareDataForHybrid(trainData);
                        prepValData = obj.prepareDataForHybrid(valData);
                        
                        % モデルの学習
                        [model, trainInfo] = obj.trainHybridModel(...
                            prepTrainData, trainLabels, prepValData, valLabels);
                        
                        % 性能評価
                        metrics = obj.evaluateModel(model, prepValData, valLabels);
                        
                        % 結果の保存
                        results.folds.accuracy(i) = metrics.accuracy;
                        results.folds.confusionMat{i} = metrics.confusionMat;
                        results.folds.classwise{i} = metrics.classwise;
                        
                        if ~isempty(trainInfo)
                            results.folds.validation_curve{i} = struct(...
                                'train_accuracy', trainInfo.TrainingAccuracy, ...
                                'val_accuracy', trainInfo.ValidationAccuracy, ...
                                'train_loss', trainInfo.TrainingLoss, ...
                                'val_loss', trainInfo.ValidationLoss);
                        end
                        
                        fprintf('Fold %d accuracy: %.2f%%\n', i, results.folds.accuracy(i) * 100);
                        
                    catch ME
                        warning('Fold %d でエラー発生: %s', i, ME.message);
                        results.folds.accuracy(i) = 0;
                    end
                end
        
                % 統計量の計算
                validFolds = results.folds.accuracy > 0;
                if any(validFolds)
                    results.meanAccuracy = mean(results.folds.accuracy(validFolds));
                    results.stdAccuracy = std(results.folds.accuracy(validFolds));
                else
                    results.meanAccuracy = 0;
                    results.stdAccuracy = 0;
                end
        
                fprintf('\n=== 交差検証結果 ===\n');
                fprintf('平均精度: %.2f%% (±%.2f%%)\n', ...
                    results.meanAccuracy * 100, results.stdAccuracy * 100);
                fprintf('有効フォールド数: %d/%d\n', sum(validFolds), k);
        
            catch ME
                error('Cross-validation failed: %s', ME.message);
            end
        end

        function updatePerformanceMetrics(obj, testMetrics)
            obj.performance = testMetrics;
        end

        function displayResults(obj)
            try
                fprintf('\n=== Hybrid Model Classification Results ===\n');
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
                        fprintf('  Precision: %.2f%%\n', ...
                            obj.performance.classwise(i).precision * 100);
                        fprintf('  Recall: %.2f%%\n', ...
                            obj.performance.classwise(i).recall * 100);
                        fprintf('  F1-Score: %.2f%%\n', ...
                            obj.performance.classwise(i).f1score * 100);
                    end
                end

                if ~isempty(obj.overfitMetrics)
                    fprintf('\nOverfitting Analysis:\n');
                    fprintf('Generalization Gap: %.2f%%\n', ...
                        obj.overfitMetrics.generalizationGap);
                    fprintf('Performance Gap: %.2f%%\n', ...
                        obj.overfitMetrics.performanceGap);
                    fprintf('Severity: %s\n', obj.overfitMetrics.severity);
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
    end
end