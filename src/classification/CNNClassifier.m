classdef CNNClassifier < handle
    properties (Access = private)
        params              % パラメータ設定
        net                 % CNNネットワーク
        isEnabled          % 有効/無効フラグ
        isInitialized      % 初期化フラグ
        useGPU            % GPUを使用するかどうか
        
        % 学習進捗の追跡用
        trainingHistory    % 学習履歴
        validationHistory  % 検証履歴
        bestValAccuracy    % 最良の検証精度
        patienceCounter    % Early Stopping用カウンター
        currentEpoch
        
        % 過学習監視用
        overfitMetrics     % 過学習メトリクス

        % コンポーネント
        dataAugmenter
        normalizer
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
            obj.useGPU = params.classifier.cnn.gpu;
            obj.dataAugmenter = DataAugmenter(params);
            obj.normalizer = EEGNormalizer(params);
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
                
                % データを3セットに分割
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
                    [trainData, normParams] = obl.normalizer.normalize(trainData);
                end

                % 検証データと評価データにも同じ正規化パラメータで正規化
                valData = obj.normalizer.normalizeOnline(valData, normParams);
                testData = obj.normalizer.normalizeOnline(testData, normParams);
                
                % 各データの準備
                prepTrainData = obj.prepareDataForCNN(trainData);
                prepValData = obj.prepareDataForCNN(valData);
                prepTestData = obj.prepareDataForCNN(testData);
                
                % モデルの学習（検証データを使用）
                [cnnModel, trainInfo] = obj.trainCNNModel(prepTrainData, trainLabels, prepValData, valLabels);
                
                % テストデータでの最終評価
                testMetrics = obj.evaluateModel(cnnModel, prepTestData, testLabels);
                
                % 過学習の検証（訓練・検証・テストの3つのデータセットを使用）
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);
    
                if isOverfit
                    warning('Overfitting detected: %s severity', obj.overfitMetrics.severity);
                end

                % 性能指標の更新
                obj.updatePerformanceMetrics(testMetrics);
                
                crossValidationResults = struct('meanAccuracy', [], 'stdAccuracy', []);
                % 交差検証の実行     
                if obj.params.classifier.cnn.training.validation.enable
                    crossValidationResults = obj.performCrossValidation(processedData, processedLabel);
                    fprintf('Cross-validation mean accuracy: %.2f%% (±%.2f%%)\n', ...
                        crossValidationResults.meanAccuracy * 100, ...
                        crossValidationResults.stdAccuracy * 100);
                end

                % 結果構造体の構築
                results = struct(...
                    'model', cnnModel, ...
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
                    'overfitting', obj.overfitMetrics, ...
                    'normParams', normParams ...
                );

                % 結果の表示
                obj.displayResults();

                % GPUメモリを解放
               if obj.useGPU
                   reset(gpuDevice);
               end

            catch ME
                fprintf('\n=== Error in CNN Training ===\n');
                fprintf('Error message: %s\n', ME.message);
                fprintf('Error stack trace:\n');
                for i = 1:length(ME.stack)
                    fprintf('  File: %s\n  Line: %d\n  Function: %s\n\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end

                % GPUメモリを解放
               if obj.useGPU
                   reset(gpuDevice);
               end

                rethrow(ME);
            end
        end

        function [isOverfit, metrics] = validateOverfitting(obj, trainInfo, testMetrics)
            disp('=== validateOverfitting ===');
            
            % 初期化
            isOverfit = false;
            metrics = struct();
            
            try
                % trainInfoの構造を検証
                if ~isstruct(trainInfo) || ~isfield(trainInfo, 'History')
                    error('trainInfo must be a struct with History field');
                end
                
                history = trainInfo.History;
                if ~isfield(history, 'TrainingAccuracy') || ~isfield(history, 'ValidationAccuracy')
                    error('History must contain TrainingAccuracy and ValidationAccuracy');
                end
        
                % 精度データの取得
                trainAcc = history.TrainingAccuracy;
                valAcc = history.ValidationAccuracy;
                
                % testMetricsからテスト精度を取得（0-100のスケールに変換）
                if isstruct(testMetrics) && isfield(testMetrics, 'accuracy')
                    testAcc = testMetrics.accuracy * 100;
                else
                    error('Invalid testMetrics structure');
                end
                
                fprintf('Validation Accuracy: %.2f%%\n', max(valAcc));
                fprintf('Test Accuracy: %.2f%%\n', testAcc);
        
                % Performance Gapの計算（検証結果とテスト結果の差）
                perfGap = abs(max(valAcc) - testAcc);
                fprintf('Performance Gap: %.2f%%\n', perfGap);
        
                % トレンド分析
                [trainTrend, valTrend] = obj.analyzeLearningCurves(trainAcc, valAcc);
                
                % 完全な偏りの検出
                if isfield(testMetrics, 'confusionMat')
                    cm = testMetrics.confusionMat;
                    
                    % 各実際のクラス（行）のサンプル数を確認
                    missingActual = any(sum(cm, 2) == 0);
                    % 各予測クラス（列）の予測件数を確認
                    missingPredicted = any(sum(cm, 1) == 0);
                    
                    % いずれかが true ならば、全く現れないクラスがあると判断
                    isCompletelyBiased = missingActual || missingPredicted;
                else
                    isCompletelyBiased = false;
                end
                
                % 学習進行の確認
                isLearningProgressing = std(diff(trainAcc)) > 0.01;
                
                % 過学習の重大度判定
                severity = obj.determineOverfittingSeverity(perfGap, isCompletelyBiased, isLearningProgressing);
                
                % 最適エポックの検出
                [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(valAcc);
                fprintf('Optimal Epoch: %d/%d\n', optimalEpoch, totalEpochs);
                
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
                
            catch ME
                fprintf('エラー発生in validateOverfitting: %s\n', ME.message);
                fprintf('Error stack:\n');
                disp(getReport(ME, 'extended'));
                
                % エラー時のフォールバック値を設定
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
                testRatio = 1/(2*k);   % k/2k
        
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

        function [cnnModel, trainInfo] = trainCNNModel(obj, trainData, trainLabels, valData, valLabels)
           try        
               % データの形状変換
               if ndims(trainData) ~= 4
                   trainData = obj.prepareDataForCNN(trainData);
               end
        
               % ラベルのカテゴリカル変換
               uniqueLabels = unique(trainLabels);
               trainLabels = categorical(trainLabels, uniqueLabels);

               % テストデータの処理
               if ~isempty(valData)
                   if ndims(valData) ~= 4
                       valData = obj.prepareDataForCNN(valData);
                   end
                   valLabels = categorical(valLabels, uniqueLabels);
               end

               % 検証データ
               valDS = {valData, valLabels};
        
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
                   'Verbose', true, ...
                   'ValidationData', valDS, ...
                   'ValidationFrequency', obj.params.classifier.cnn.training.frequency, ...
                   'ValidationPatience', obj.params.classifier.cnn.training.patience, ...
                   'GradientThreshold', 1);
       
               % レイヤーの構築とモデルの学習
               layers = obj.buildCNNLayers(trainData);
               [cnnModel, trainHistory] = trainNetwork(trainData, trainLabels, layers, options);
        
               % 学習履歴の保存
               trainInfo.History = trainHistory;
               trainInfo.FinalEpoch = length(trainHistory.TrainingLoss);
        
               fprintf('\n学習完了: %d反復\n', trainInfo.FinalEpoch);

               % GPUメモリを解放
               if obj.useGPU
                   reset(gpuDevice);
               end
        
           catch ME
               fprintf('trainCNNModelでエラーが発生: %s\n', ME.message);
        
               % GPUメモリを解放
               if obj.useGPU
                   reset(gpuDevice);
               end
        
               rethrow(ME);
           end
        end

        function layers = buildCNNLayers(obj, data)
            try
                % 入力サイズの取得
                inputSize = size(data);
                
                % 入力次元に基づいてレイヤー設定を決定
                if ndims(data) == 4
                    % 4次元データ [サンプル x チャンネル x 1 x エポック] の場合
                    layerInputSize = [inputSize(1), inputSize(2), 1];
                elseif ndims(data) == 3
                    % 3次元データ [サンプル x チャンネル x エポック] の場合
                    layerInputSize = [inputSize(1), inputSize(2), 1];
                elseif ismatrix(data)
                    % 2次元データ [チャンネル x サンプル] の場合
                    layerInputSize = [inputSize(2), inputSize(1), 1];
                else
                    error('対応していないデータ次元数: %d', ndims(data));
                end
        
                % アーキテクチャパラメータの取得
                arch = obj.params.classifier.cnn.architecture;
                
                % チャンネル数が少ない場合に特化したアーキテクチャを使用
                if layerInputSize(2) <= 10
                    fprintf('チャンネル数が少ないため(%d)、特化したCNNアーキテクチャを使用します\n', layerInputSize(2));
                    
                    % 簡略化されたCNNアーキテクチャを構築
                    layers = [
                        % 入力層
                        imageInputLayer(layerInputSize, 'Normalization', 'none', 'Name', 'input')
                        
                        % 1つ目の畳み込みブロック - 小さいフィルタサイズを使用
                        convolution2dLayer([3 3], 16, 'Padding', 'same', 'Name', 'conv1')
                        batchNormalizationLayer('Name', 'bn1')
                        reluLayer('Name', 'relu1')
                        % プーリングサイズを小さく (2,1) - チャンネル次元はプーリングしない
                        maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')
                        dropoutLayer(0.3, 'Name', 'dropout1')
                        
                        % 2つ目の畳み込みブロック
                        convolution2dLayer([3 3], 32, 'Padding', 'same', 'Name', 'conv2')
                        batchNormalizationLayer('Name', 'bn2')
                        reluLayer('Name', 'relu2')
                        maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')
                        dropoutLayer(0.4, 'Name', 'dropout2')
                        
                        % グローバルプーリング（チャンネル次元を保持）
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
                    
                else
                    % 標準的なCNNアーキテクチャ（チャンネル数が十分な場合）             
                    % 初期フィルタ数の決定 (チャンネル数によって調整)
                    numChannels = obj.params.device.channelCount;
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
                            warning('プーリング層設定が見つかりません。デフォルト値を使用: size=2, stride=2');
                        end
                        
                        % ドロップアウト率の取得
                        if isfield(arch.dropoutLayers, ['dropout' num2str(i)])
                            dropoutRate = arch.dropoutLayers.(['dropout' num2str(i)]);
                        else
                            % デフォルト値
                            dropoutRate = 0.5;
                            warning('ドロップアウト設定が見つかりません。デフォルト値を使用: 0.5');
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
                
                fprintf('CNNレイヤー構築完了: 合計%dレイヤー\n', length(layers));
                
            catch ME
                fprintf('\nbuildCNNLayersでエラー発生: %s\n', ME.message);
                fprintf('エラースタック:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
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
        
        function [trainTrend, valTrend] = analyzeLearningCurves(~, trainAcc, valAcc)
            disp('=== analyzeLearningCurves ===');
            
            try
                % 入力データの検証
                if isempty(trainAcc) || isempty(valAcc)
                    error('学習曲線データが空です');
                end
                
                % NaNチェック
                trainAcc = trainAcc(~isnan(trainAcc));
                valAcc = valAcc(~isnan(valAcc));
                
                if isempty(trainAcc) || isempty(valAcc)
                    error('有効なデータがありません（NaN除去後）');
                end
        
                % 移動平均の計算
                windowSize = min(5, floor(length(trainAcc)/3));
                if windowSize < 1
                    windowSize = 1;
                end
                
                trainSmooth = movmean(trainAcc, windowSize);
                valSmooth = movmean(valAcc, windowSize);
                
                % 変化率の計算
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
        
                % 結果の検証
                if isnan(valTrend.mean_change) || isnan(valTrend.volatility)
                    warning('計算結果にNaNが含まれています');
                    fprintf('Training data stats: min=%f, max=%f, length=%d\n', ...
                        min(trainAcc), max(trainAcc), length(trainAcc));
                    fprintf('Validation data stats: min=%f, max=%f, length=%d\n', ...
                        min(valAcc), max(valAcc), length(valAcc));
                end
        
            catch ME
                fprintf('エラー発生in analyzeLearningCurves: %s\n', ME.message);
                % エラー時のフォールバック値を設定
                trainTrend = struct(...
                    'mean_change', 0, ...
                    'volatility', 0, ...
                    'increasing_ratio', 0);
                valTrend = struct(...
                    'mean_change', 0, ...
                    'volatility', 0, ...
                    'increasing_ratio', 0);
                rethrow(ME);
            end
        end
        
        function effect = analyzeEarlyStoppingEffect(~, trainInfo)
            % Early Stoppingの効果分析
            valLoss = trainInfo.History.ValidationLoss;
            [~, minLossEpoch] = min(valLoss);
            totalEpochs = length(valLoss);
            
            effect = struct(...
                'optimal_epoch', minLossEpoch, ...
                'total_epochs', totalEpochs, ...
                'stopping_efficiency', minLossEpoch / totalEpochs);
        end

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
        
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(~, valAcc)
            try
                totalEpochs = length(valAcc);
                [~, optimalEpoch] = max(valAcc);
                
                % 最適エポックが最後のエポックの場合、まだ改善の余地があるかもしれない
                if optimalEpoch == totalEpochs
                    fprintf('Warning: 最適エポックが最終エポックと一致。より長い学習が必要かもしれません。\n');
                end
        
            catch ME
                fprintf('エラー発生in findOptimalEpoch: %s\n', ME.message);
                optimalEpoch = 0;
                totalEpochs = 0;
            end
        end

        function preparedData = prepareDataForCNN(obj, data)
            try                
                % データの次元数に基づいて処理を分岐
                if ndims(data) == 3
                    % 3次元データ（チャンネル x サンプル x エポック）の場合
                    preparedData = permute(data, [2, 1, 4, 3]);
                        
                elseif ismatrix(data)
                    % 2次元データ（チャンネル x サンプル）の場合
                    [channels, samples] = size(data);
                    preparedData = permute(data, [2, 1, 3]);
                    preparedData = reshape(preparedData, [samples, channels, 1, 1]);
                        
                else
                    % その他の次元数は対応外
                    error('対応していないデータ次元数: %d', ndims(data));
                end
                
                % Nan/Infチェック
                if any(isnan(preparedData(:))) || any(isinf(preparedData(:)))
                    warning('prepareDataForCNN: 処理後のデータにNaNまたはInfが含まれています');
                end
                
            catch ME
                fprintf('\nprepareDataForCNNでエラー発生: %s\n', ME.message);
                fprintf('データサイズ: [%s]\n', num2str(size(data)));
                fprintf('エラースタック:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end
        
        % function stop = trainingOutputFcn(obj, info)
        %     % 初期化
        %     stop = false;
        % 
        %     if info.State == "start"
        %         obj.currentEpoch = 0;
        %         return;
        %     end
        % 
        %     % 学習情報の更新
        %     obj.currentEpoch = obj.currentEpoch + 1;
        % 
        %     % 検証データがある場合のEarly Stopping
        %     if ~isempty(info.ValidationLoss)
        %         currentAccuracy = info.ValidationAccuracy;
        % 
        %         if currentAccuracy > obj.bestValAccuracy
        %             obj.bestValAccuracy = currentAccuracy;
        %             obj.patienceCounter = 0;
        %         else
        %             obj.patienceCounter = obj.patienceCounter + 1;
        %             if obj.patienceCounter >= obj.params.classifier.cnn.training.patience
        %                 fprintf('\nEarly stopping triggered at epoch %d\n', obj.currentEpoch);
        %                 stop = true;
        %             end
        %         end
        %     end
        % end
        
        function cvResults = performCrossValidation(obj, data, labels)
            try
                % k-fold cross validationのパラメータ取得
                k = obj.params.classifier.cnn.training.validation.kfold;
                fprintf('\n=== %d分割交差検証開始 ===\n', k);
        
                % cvResultsの初期化
                cvResults = struct();
                cvResults.folds = struct(...
                    'accuracy', zeros(1, k), ...
                    'confusionMat', cell(1, k), ...
                    'classwise', cell(1, k), ...
                    'validation_curve', cell(1, k) ...
                );
        
                % データの分割（学習・テスト・検証用）
                [trainData, trainLabels, valData, valLabels, testData, testLabels] = ...
                    obj.splitDataset(data, labels);
        
                for i = 1:k
                    fprintf('\nフォールド %d/%d の処理開始\n', i, k);
        
                    % データの準備
                    prepTrainData = obj.prepareDataForCNN(trainData{i});
                    prepValData = obj.prepareDataForCNN(valData{i});
                    prepTestData = obj.prepareDataForCNN(testData{i});
        
                    try
                        % モデルの学習（検証データを使用）
                        [model, trainInfo] = obj.trainCNNModel(...
                            prepTrainData, trainLabels{i}, prepValData, valLabels{i});
        
                        % テストデータでの評価
                        metrics = obj.evaluateModel(model, prepTestData, testLabels{i});
        
                        % 結果の保存
                        cvResults.folds.accuracy(i) = metrics.accuracy;
                        cvResults.folds.confusionMat{i} = metrics.confusionMat;
                        cvResults.folds.classwise{i} = metrics.classwise;
                        
                        % 学習曲線の保存
                        cvResults.folds.validation_curve{i} = struct(...
                            'train_accuracy', trainInfo.History.TrainingAccuracy, ...
                            'val_accuracy', trainInfo.History.ValidationAccuracy, ...
                            'train_loss', trainInfo.History.TrainingLoss, ...
                            'val_loss', trainInfo.History.ValidationLoss ...
                        );
        
                        fprintf('フォールド %d の精度: %.2f%%\n', i, results.folds.accuracy(i) * 100);
        
                    catch ME
                        warning('フォールド %d でエラーが発生: %s', i, ME.message);
                        % エラーが発生したフォールドは精度0として記録
                        cvResults.folds.accuracy(i) = 0;
                        cvResults.folds.confusionMat{i} = [];
                        cvResults.folds.classwise{i} = [];
                        cvResults.folds.validation_curve{i} = [];
                    end
                end
        
                % 統計量の計算
                validFolds = cvResults.folds.accuracy > 0;
                if any(validFolds)
                    cvResults.meanAccuracy = mean(cvResults.folds.accuracy(validFolds));
                    cvResults.stdAccuracy = std(cvResults.folds.accuracy(validFolds));
                    cvResults.minAccuracy = min(cvResults.folds.accuracy(validFolds));
                    cvResults.maxAccuracy = max(cvResults.folds.accuracy(validFolds));
                else
                    cvResults.meanAccuracy = 0;
                    cvResults.stdAccuracy = 0;
                    cvResults.minAccuracy = 0;
                    cvResults.maxAccuracy = 0;
                end
        
                % クラスごとの平均性能
                if ~isempty(cvResults.folds.classwise{1})
                    numClasses = length(cvResults.folds.classwise{1});
                    cvResults.classwise_mean = struct(...
                        'precision', zeros(1, numClasses), ...
                        'recall', zeros(1, numClasses), ...
                        'f1score', zeros(1, numClasses) ...
                    );
        
                    for class = 1:numClasses
                        validClassMetrics = cellfun(@(x) ~isempty(x), cvResults.folds.classwise);
                        precision_values = cellfun(@(x) x(class).precision, ...
                            cvResults.folds.classwise(validClassMetrics));
                        recall_values = cellfun(@(x) x(class).recall, ...
                            cvResults.folds.classwise(validClassMetrics));
                        f1_values = cellfun(@(x) x(class).f1score, ...
                            cvResults.folds.classwise(validClassMetrics));
        
                        cvResults.classwise_mean.precision(class) = mean(precision_values);
                        cvResults.classwise_mean.recall(class) = mean(recall_values);
                        cvResults.classwise_mean.f1score(class) = mean(f1_values);
                    end
                end
        
                % 結果の表示
                fprintf('\nCross-validation Results:\n');
                fprintf('Mean Accuracy: %.2f%% (±%.2f%%)\n', ...
                    cvResults.meanAccuracy * 100, cvResults.stdAccuracy * 100);
                fprintf('Min Accuracy: %.2f%%\n', cvResults.minAccuracy * 100);
                fprintf('Max Accuracy: %.2f%%\n', cvResults.maxAccuracy * 100);
                fprintf('Successfully completed folds: %d/%d\n', ...
                    sum(validFolds), k);
        
                if isfield(cvResults, 'classwise_mean')
                    fprintf('\nClass-wise Average Performance:\n');
                    for class = 1:numClasses
                        fprintf('Class %d:\n', class);
                        fprintf('  Precision: %.2f%%\n', ...
                            cvResults.classwise_mean.precision(class) * 100);
                        fprintf('  Recall: %.2f%%\n', ...
                            cvResults.classwise_mean.recall(class) * 100);
                        fprintf('  F1-Score: %.2f%%\n', ...
                            cvResults.classwise_mean.f1score(class) * 100);
                    end
                end
        
            catch ME
                fprintf('Cross-validation failed: %s\n', ME.message);
                fprintf('Error details:\n');
                disp(getReport(ME, 'extended'));
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