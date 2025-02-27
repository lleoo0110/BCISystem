classdef HybridClassifier < handle
    properties (Access = private)
        params              % パラメータ設定
        net                 % ハイブリッドネットワーク（特徴抽出器）
        adaBoostModel       % AdaBoost分類器
        isEnabled           % 有効/無効フラグ
        isInitialized       % 初期化フラグ
        useGPU              % GPU使用フラグ
        
        % 学習進捗の追跡用
        trainingHistory     % 学習履歴
        validationHistory   % 検証履歴
        bestValAccuracy     % 最良の検証精度
        patienceCounter     % Early Stopping用カウンター
        currentEpoch        % 現在のエポック
        lastEpoch
        
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
        % コンストラクタ
        function obj = HybridClassifier(params)
            obj.params = params;
            obj.isEnabled = params.classifier.hybrid.enable;
            obj.isInitialized = false;
            obj.initializeProperties();
            obj.useGPU = params.classifier.hybrid.gpu;
            obj.dataAugmenter = DataAugmenter(params);
            obj.normalizer = EEGNormalizer(params);
        end
        
        % ハイブリッドモデル学習メソッド
        function results = trainHybrid(obj, processedData, processedLabel)            
            try
                fprintf('\n=== Starting Hybrid Training with AdaBoost ===\n');

                % データ分割
                [trainData, trainLabels, valData, valLabels, testData, testLabels] = ...
                    obj.splitDataset(processedData, processedLabel);

                % 学習データのみ拡張
                if obj.params.signal.preprocessing.augmentation.enable
                    [trainData, trainLabels, augInfo] = obj.dataAugmenter.augmentData(trainData, trainLabels);
                    fprintf('訓練データを拡張しました:\n');
                    fprintf('  訓練データ: %d サンプル\n', length(trainData));
                end

                % 正規化
                if obj.params.signal.preprocessing.normalize.enable
                    [trainData, normParams] = obj.normalizer.normalize(trainData);

                    % 検証データと評価データにも同じ正規化パラメータで正規化
                    valData = obj.normalizer.normalizeOnline(valData, normParams);
                    testData = obj.normalizer.normalizeOnline(testData, normParams);
                end

                % データ前処理
                prepTrainCNN = obj.prepareDataForCNN(trainData);
                prepTrainLSTM = obj.prepareDataForLSTM(trainData);
                prepValCNN = obj.prepareDataForCNN(valData);
                prepValLSTM = obj.prepareDataForLSTM(valData);
                prepTestCNN = obj.prepareDataForCNN(testData);
                prepTestLSTM = obj.prepareDataForLSTM(testData);
                
                % ラベルを列ベクトルに変換
                trainLabels = trainLabels(:);
                valLabels = valLabels(:);
                testLabels = testLabels(:);
                
                % 入力サイズの決定
                cnnInputSize = size(prepTrainCNN(:,:,:,1));
                lstmInputSize = size(prepTrainLSTM{1},1);
                
                % ステップ1: 特徴抽出器としてのハイブリッドモデルの学習
                fprintf('\n--- Step 1: Training Hybrid Feature Extractor ---\n');
                [featureExtractor, trainInfo] = obj.trainFeatureExtractor(...
                    prepTrainCNN, prepTrainLSTM, trainLabels, ...
                    prepValCNN, prepValLSTM, valLabels, cnnInputSize, lstmInputSize);
                
                % 特徴抽出
                fprintf('\n--- Extracting Features for AdaBoost ---\n');
                trainFeatures = obj.extractFeatures(featureExtractor, prepTrainCNN, prepTrainLSTM);
                valFeatures = obj.extractFeatures(featureExtractor, prepValCNN, prepValLSTM);
                testFeatures = obj.extractFeatures(featureExtractor, prepTestCNN, prepTestLSTM);
                
                % ステップ2: AdaBoostモデルの学習
                fprintf('\n--- Step 2: Training AdaBoost Classifier ---\n');
                adaBoostModel = obj.trainAdaBoost(trainFeatures, trainLabels);
                
                % AdaBoostモデルの評価
                testPredictions = predict(adaBoostModel, testFeatures);
                testAccuracy = sum(testPredictions == testLabels) / length(testLabels);
                fprintf('AdaBoost Test Accuracy: %.2f%%\n', testAccuracy * 100);
                
                % 混同行列の計算
                confMat = confusionmat(testLabels, testPredictions);
                
                % クラスごとの性能評価
                uniqueClasses = unique(testLabels);
                classwiseMetrics = struct('precision', zeros(1, length(uniqueClasses)), ...
                                        'recall', zeros(1, length(uniqueClasses)), ...
                                        'f1score', zeros(1, length(uniqueClasses)));
                                    
                for i = 1:length(uniqueClasses)
                    classIdx = (testLabels == uniqueClasses(i));
                    TP = sum(testPredictions(classIdx) == uniqueClasses(i));
                    FP = sum(testPredictions == uniqueClasses(i)) - TP;
                    FN = sum(classIdx) - TP;
                    
                    precision = TP / (TP + FP);
                    recall = TP / (TP + FN);
                    f1score = 2 * (precision * recall) / (precision + recall);
                    
                    classwiseMetrics(i).precision = precision;
                    classwiseMetrics(i).recall = recall;
                    classwiseMetrics(i).f1score = f1score;
                end
                
                % 過学習の検証
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, struct('accuracy', testAccuracy));
                
                % モデルの保存
                obj.net = featureExtractor;
                obj.adaBoostModel = adaBoostModel;
                
                % 結果の構築
                results = struct(...
                    'model', struct('featureExtractor', featureExtractor, 'adaBoostModel', adaBoostModel), ...
                    'performance', struct(...
                        'overallAccuracy', testAccuracy, ...
                        'crossValidation', struct('accuracy', [], 'std', []), ...
                        'precision', classwiseMetrics(1).precision, ...
                        'recall', classwiseMetrics(1).recall, ...
                        'f1score', classwiseMetrics(1).f1score, ...
                        'confusionMatrix', confMat ...
                    ), ...
                    'trainInfo', trainInfo, ...
                    'overfitting', obj.overfitMetrics, ...
                    'normParams', normParams ...
                );
                
                % 性能指標の更新
                obj.performance = results.performance;
                
                % 結果の表示
                obj.displayResults();
                
                % GPUメモリの解放
                if obj.useGPU
                    reset(gpuDevice);
                end
                
            catch ME
                fprintf('\n=== Error in Hybrid Training ===\n');
                fprintf('Error message: %s\n', ME.message);
                fprintf('Error stack:\n');
                disp(getReport(ME, 'extended'));
                
                if obj.useGPU
                    reset(gpuDevice);
                end
                
                rethrow(ME);
            end
        end
        
        % オンライン予測メソッド
        function [label, score] = predictOnline(obj, data, model)
            if ~obj.isEnabled
                error('Hybrid classifier is disabled');
            end

            try
                % モデルを解凍（featureExtractorとadaBoostModelを取得）
                if isstruct(model)
                    featureExtractor = model.featureExtractor;
                    adaBoostModel = model.adaBoostModel;
                else
                    error('Invalid model structure. Expected struct with featureExtractor and adaBoostModel fields');
                end
                
                % 入力が1つの場合は複製
                if ~iscell(data)
                    data = {data, data};
                elseif isscalar(data)
                    data = {data{1}, data{1}};
                end

                % もしdata{2}がcellであれば、数値配列を取り出す
                if iscell(data{2}) && length(data{2}) == 1
                    data{2} = data{2}{1};
                end

                % 入力が2次元の場合は、サンプル数1として3次元に変換
                if ismatrix(data{1})
                    data{1} = reshape(data{1}, size(data{1},1), size(data{1},2), 1);
                end

                if ismatrix(data{2})
                    data{2} = reshape(data{2}, size(data{2},1), size(data{2},2), 1);
                end

                % データの前処理
                prepCNN = obj.prepareDataForCNN(data{1});
                prepLSTM = obj.prepareDataForLSTM(data{2});

                % ステップ1: 特徴を抽出
                features = obj.extractFeatures(featureExtractor, prepCNN, prepLSTM);
                
                % ステップ2: AdaBoostで予測
                [label, score] = predict(adaBoostModel, features);

            catch ME
                fprintf('Error in online prediction: %s\n', ME.message);
                rethrow(ME);
            end
        end
    end
    
    methods (Access = private)
        % プロパティの初期化
        function initializeProperties(obj)
            obj.trainingHistory = struct('loss', [], 'accuracy', []);
            obj.validationHistory = struct('loss', [], 'accuracy', []);
            obj.bestValAccuracy = 0;
            obj.patienceCounter = 0;
            obj.overfitMetrics = struct();
        end
        
        % ハイブリッドレイヤーの構築
        function lgraph = buildHybridLayers(obj, cnnInputSize, lstmInputSize)
            try
                arch = obj.params.classifier.hybrid.architecture;
                numClasses = arch.numClasses;
                
                %% === CNNブランチの構築 ===
                cnnLayers = [];
                
                % 入力層
                cnnInput = imageInputLayer([cnnInputSize(1) cnnInputSize(2) 1], ...
                    'Name', 'input_cnn', ...
                    'Normalization', 'none');
                cnnLayers = [cnnLayers; cnnInput];
                
                % 畳み込み層群
                convFields = fieldnames(arch.cnn.convLayers);
                for i = 1:numel(convFields)
                    convParam = arch.cnn.convLayers.(convFields{i});
                    convName = sprintf('conv_%d', i);
                    convLayerObj = convolution2dLayer(convParam.size, convParam.filters, ...
                        'Stride', convParam.stride, ...
                        'Padding', convParam.padding, ...
                        'Name', convName);
                    cnnLayers = [cnnLayers; convLayerObj];
                    
                    % バッチ正規化（オプション）
                    if arch.batchNorm
                        bnName = sprintf('bn_%s', convName);
                        bnLayerObj = batchNormalizationLayer('Name', bnName);
                        cnnLayers = [cnnLayers; bnLayerObj];
                    end
                    
                    % ReLU層
                    reluName = sprintf('relu_%d', i);
                    cnnLayers = [cnnLayers; reluLayer('Name', reluName)];
                    
                    % プーリング層
                    poolField = sprintf('pool%d', i);
                    if isfield(arch.cnn.poolLayers, poolField)
                        poolParam = arch.cnn.poolLayers.(poolField);
                        poolName = sprintf('pool_%d', i);
                        poolLayerObj = maxPooling2dLayer(poolParam.size, ...
                            'Stride', poolParam.stride, ...
                            'Name', poolName);
                        cnnLayers = [cnnLayers; poolLayerObj];
                    end
                    
                    % ドロップアウト層
                    dropoutField = sprintf('dropout%d', i);
                    if isfield(arch.cnn.dropoutLayers, dropoutField)
                        dropoutRate = arch.cnn.dropoutLayers.(dropoutField);
                        dropoutName = sprintf('dropout_%d', i);
                        dropoutLayerObj = dropoutLayer(dropoutRate, 'Name', dropoutName);
                        cnnLayers = [cnnLayers; dropoutLayerObj];
                    end
                end
                
                % 全結合層
                if isfield(arch.cnn, 'fullyConnected')
                    fcSizes = arch.cnn.fullyConnected;
                else
                    fcSizes = 128;
                end
                
                if ~isempty(fcSizes)
                    for j = 1:length(fcSizes)
                        fcName = sprintf('fc_cnn_%d', j);
                        fcLayerObj = fullyConnectedLayer(fcSizes(j), 'Name', fcName);
                        cnnLayers = [cnnLayers; fcLayerObj];
                        reluName = sprintf('relu_fc_cnn_%d', j);
                        cnnLayers = [cnnLayers; reluLayer('Name', reluName)];
                    end
                    cnnFeatureSize = fcSizes(end);
                else
                    error('CNN fully connected layers not defined.');
                end

                % リシェイプ層
                reshapeCNN = ReshapeCNNLayer('reshape_cnn', cnnFeatureSize);
                reshapeCNN = reshapeCNN.setOutputFormat('SSCB');
                cnnLayers = [cnnLayers; reshapeCNN];
                
                %% === LSTMブランチの構築 ===
                lstmLayers = [];
                
                % シーケンス入力層
                lstmInputLayerObj = sequenceInputLayer(lstmInputSize, ...
                    'Name', 'input_lstm', ...
                    'Normalization', arch.lstm.sequenceInputLayer.normalization);
                lstmLayers = [lstmLayers; lstmInputLayerObj];
                
                % 複数のLSTM層の追加
                lstmFields = fieldnames(arch.lstm.lstmLayers);
                for i = 1:numel(lstmFields)
                    lstmParam = arch.lstm.lstmLayers.(lstmFields{i});
                    lstmName = sprintf('lstm_%d', i);
                    lstmLayerObj = lstmLayer(lstmParam.numHiddenUnits, ...
                        'OutputMode', lstmParam.OutputMode, ...
                        'Name', lstmName);
                    lstmLayers = [lstmLayers; lstmLayerObj];
                    
                    % バッチ正規化
                    if arch.batchNorm
                        bnName = sprintf('bn_%s', lstmName);
                        bnLayerObj = batchNormalizationLayer('Name', bnName);
                        lstmLayers = [lstmLayers; bnLayerObj];
                    end
                    
                    % ドロップアウト層
                    dropoutField = sprintf('dropout%d', i);
                    if isfield(arch.lstm.dropoutLayers, dropoutField)
                        dropoutRate = arch.lstm.dropoutLayers.(dropoutField);
                        dropoutName = sprintf('dropout_%s', lstmName);
                        dropoutLayerObj = dropoutLayer(dropoutRate, 'Name', dropoutName);
                        lstmLayers = [lstmLayers; dropoutLayerObj];
                    end
                end
                
                % LSTM全結合層
                if isfield(arch.lstm, 'fullyConnected')
                    lstmFCSize = arch.lstm.fullyConnected;
                else
                    lstmFCSize = 128;
                end

                fcLSTM = fullyConnectedLayer(lstmFCSize, 'Name', 'fc_lstm');
                reluFCL = reluLayer('Name', 'relu_fc_lstm');
                lstmLayers = [lstmLayers; fcLSTM; reluFCL];
                
                % リシェイプ層
                reshapeLSTM = ReshapeLSTMLayer('reshape_lstm', lstmFCSize);
                reshapeLSTM = reshapeLSTM.setOutputFormat('SSCB');
                lstmLayers = [lstmLayers; reshapeLSTM];
                
                %% === マージ層 ===
                mergeLayers = [];
                
                % 特徴量の結合層
                mergeParams = arch.merge;
                concatLayer = concatenationLayer(...
                    mergeParams.concat.dimension, ...
                    mergeParams.concat.numInputs, ...
                    'Name', mergeParams.concat.name);
                mergeLayers = [mergeLayers; concatLayer];
                
                % Global Average Pooling
                if mergeParams.globalPooling.enable
                    gavgLayer = globalAveragePooling2dLayer('Name', mergeParams.globalPooling.name);
                    mergeLayers = [mergeLayers; gavgLayer];
                end
                
                % 結合後の全結合層
                if isfield(mergeParams.fullyConnected, 'layers')
                    for i = 1:length(mergeParams.fullyConnected.layers)
                        fcSize = mergeParams.fullyConnected.layers(i).units;
                        fcName = mergeParams.fullyConnected.layers(i).name;
                        fcLayer = fullyConnectedLayer(fcSize, 'Name', fcName);
                        mergeLayers = [mergeLayers; fcLayer];
                        
                        if strcmpi(mergeParams.fullyConnected.activation, 'relu')
                            reluName = sprintf('relu_%s', fcName);
                            mergeLayers = [mergeLayers; reluLayer('Name', reluName)];
                        end
                    end
                end
                
                % ドロップアウト層
                if mergeParams.dropout.rate > 0
                    dropoutLayerObj = dropoutLayer(mergeParams.dropout.rate, ...
                        'Name', mergeParams.dropout.name);
                    mergeLayers = [mergeLayers; dropoutLayerObj];
                end
                
                % 特徴抽出のための全結合層を最後に追加
                fcFinal = fullyConnectedLayer(64, 'Name', 'feature_output');
                mergeLayers = [mergeLayers; fcFinal];
                    
                %% === レイヤーグラフの構築と接続 ===
                lgraph = layerGraph();
        
                % 各ブランチのレイヤーを追加
                lgraph = addLayers(lgraph, cnnLayers);
                lgraph = addLayers(lgraph, lstmLayers);
                lgraph = addLayers(lgraph, mergeLayers);
                
                % ブランチの接続
                lgraph = connectLayers(lgraph, 'reshape_cnn', 'concat/in1');
                lgraph = connectLayers(lgraph, 'reshape_lstm', 'concat/in2');
                
                % レイヤーグラフの作成に関するデバッグ情報
                fprintf('Created layerGraph with %d layers\n', numel(lgraph.Layers));
                
                % デバッグ情報の出力
                fprintf('\nNetwork architecture summary (Feature Extractor):\n');
                fprintf('CNN branch: %d convolutional layers, final features: %d\n', ...
                    numel(convFields), cnnFeatureSize);
                fprintf('LSTM branch: %d LSTM layers, final features: %d\n', ...
                    numel(lstmFields), lstmFCSize);
                
            catch ME
                fprintf('\nError in buildHybridLayers:\n');
                fprintf('Error message: %s\n', ME.message);
                if ~isempty(ME.stack)
                    fprintf('Error in: %s, Line: %d\n', ME.stack(1).name, ME.stack(1).line);
                end
                rethrow(ME);
            end
        end
        
        % 特徴抽出器学習メソッド
        function [featureExtractor, trainInfo] = trainFeatureExtractor(obj, trainCNN, trainLSTM, trainLabels, valCNN, valLSTM, valLabels, cnnInputSize, lstmInputSize)
            try
                % レイヤーの構築
                layers = obj.buildHybridLayers(cnnInputSize, lstmInputSize);
                
                % データストアの作成
                trainDS = obj.createHybridDatastore(trainCNN, trainLSTM, categorical(trainLabels));
                valDS = obj.createHybridDatastore(valCNN, valLSTM, categorical(valLabels));
                
                % 学習環境設定
                executionEnvironment = 'cpu';
                if obj.useGPU
                    executionEnvironment = 'gpu';
                end
                
                % トレーニングオプションの設定
                options = trainingOptions(obj.params.classifier.hybrid.training.optimizer.type, ...
                    'InitialLearnRate', obj.params.classifier.hybrid.training.optimizer.learningRate, ...
                    'MaxEpochs', obj.params.classifier.hybrid.training.maxEpochs, ...
                    'MiniBatchSize', obj.params.classifier.hybrid.training.miniBatchSize, ...
                    'Shuffle', obj.params.classifier.hybrid.training.shuffle, ...
                    'Plots', 'none', ...
                    'Verbose', true, ...
                    'ExecutionEnvironment', executionEnvironment, ...
                    'ValidationData', valDS, ...
                    'ValidationFrequency', obj.params.classifier.hybrid.training.frequency, ...
                    'ValidationPatience', obj.params.classifier.hybrid.training.patience, ...
                    'GradientThreshold', 1);
                
                % レイヤーグラフにトレーニング用の分類層を追加
                tempLayers = layers;
                
                % 一時的に分類用の層を追加
                fcFinalClass = fullyConnectedLayer(length(unique(trainLabels)), 'Name', 'fc_final_class');
                softmaxLayer = softmaxLayer('Name', 'softmax_temp');
                classLayer = classificationLayer('Name', 'output_temp');
                
                tempLayers = addLayers(tempLayers, [fcFinalClass; softmaxLayer; classLayer]);
                tempLayers = connectLayers(tempLayers, 'feature_output', 'fc_final_class');
                
                % モデルの学習
                fprintf('\nTraining feature extractor network...\n');
                [tempModel, trainHistory] = trainNetwork(trainDS, tempLayers, options);
                
                % 特徴抽出器の取得（最後の分類層を除去）
                lgraph = layerGraph(tempModel.Layers);
                
                % 分類層を削除
                layersToRemove = {'fc_final_class', 'softmax_temp', 'output_temp'};
                for i = 1:length(layersToRemove)
                    if any(strcmp({lgraph.Layers.Name}, layersToRemove{i}))
                        lgraph = removeLayers(lgraph, layersToRemove{i});
                    end
                end
                
                % 特徴抽出器モデルを構築
                featureExtractor = assembleNetwork(lgraph);
                
                % トレーニング情報を保存
                trainInfo = struct('History', trainHistory);
                
            catch ME
                fprintf('\nError in trainFeatureExtractor: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        % 特徴抽出メソッド
        function features = extractFeatures(obj, featureExtractor, cnnData, lstmData)
            try
                % 入力データのバッチサイズを取得
                if isnumeric(cnnData)
                    batchSize = size(cnnData, 4);
                else
                    batchSize = numel(lstmData);
                end
                
                % 特徴を抽出
                features = zeros(batchSize, 64);  % feature_output層の出力サイズに合わせる
                
                % バッチ処理（メモリ使用量削減のため）
                batchSize = min(32, batchSize);  % バッチサイズを制限
                numBatches = ceil(size(features, 1) / batchSize);
                
                for i = 1:numBatches
                    startIdx = (i-1) * batchSize + 1;
                    endIdx = min(i * batchSize, size(features, 1));
                    
                    % バッチデータの準備
                    if isnumeric(cnnData)
                        batchCNN = cnnData(:,:,:,startIdx:endIdx);
                    else
                        batchCNN = cnnData(startIdx:endIdx);
                    end
                    
                    if iscell(lstmData)
                        batchLSTM = lstmData(startIdx:endIdx);
                    else
                        batchLSTM = lstmData(:,:,startIdx:endIdx);
                    end
                    
                    % 活性化を抽出
                    batchFeatures = activations(featureExtractor, {batchCNN, batchLSTM}, 'feature_output');
                    features(startIdx:endIdx, :) = batchFeatures;
                end
                
            catch ME
                fprintf('Error in extractFeatures: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        % AdaBoost分類器学習メソッド
        function adaBoostModel = trainAdaBoost(obj, features, labels)
            try
                % AdaBoostのパラメータを取得
                adaParams = obj.params.classifier.hybrid.adaBoost;
                numLearners = adaParams.numLearners;
                maxSplits = adaParams.maxSplits;
                learnRate = adaParams.learnRate;
                
                % AdaBoostモデルを構築
                fprintf('Training AdaBoost classifier with %d learners...\n', numLearners);
                adaBoostModel = fitcensemble(features, labels, ...
                    'Method', 'AdaBoostM1', ...  % AdaBoostアルゴリズム
                    'Learners', templateTree('MaxNumSplits', maxSplits), ...  % 弱い決定木
                    'NumLearningCycles', numLearners, ...  % 弱分類器の数
                    'LearnRate', learnRate);  % 学習率
                
                % クロスバリデーションでモデルの評価
                cvModel = crossval(adaBoostModel);
                cvError = kfoldLoss(cvModel);
                fprintf('Cross-validation error: %.4f\n', cvError);
                fprintf('Cross-validation accuracy: %.2f%%\n', (1 - cvError) * 100);
                
            catch ME
                fprintf('Error in trainAdaBoost: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        % データセット分割メソッド
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
        
        % クラス分布確認メソッド
        function checkClassDistribution(~, setName, labels)
            uniqueLabels = unique(labels);
            fprintf('\n%sデータのクラス分布:\n', setName);
            for i = 1:length(uniqueLabels)
                count = sum(labels == uniqueLabels(i));
                fprintf('  クラス %d: %d サンプル (%.1f%%)\n', ...
                    uniqueLabels(i), count, (count/length(labels))*100);
            end
        end
        
        % CNN用データ前処理メソッド
        function preparedData = prepareDataForCNN(~, data)
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
        
        % LSTM用データ前処理メソッド
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
        
        % ハイブリッドデータストアの作成
        function ds = createHybridDatastore(~, cnnData, lstmData, labels)
            try
                numSamples = size(cnnData, 4);

                % CNNデータの準備
                cnnInputs = cell(numSamples, 1);
                for i = 1:numSamples
                    cnnInputs{i} = double(cnnData(:,:,:,i));
                end

                % LSTMデータの準備
                if iscell(lstmData)
                    if iscell(lstmData{1})
                        fprintf('lstmData{1} is a cell. Unwrapping one level.\n');
                        lstmInputs = cellfun(@(x) x{1}, lstmData, 'UniformOutput', false);
                    else
                        lstmInputs = lstmData;
                    end
                else
                    lstmInputs = cell(numSamples, 1);
                    for i = 1:numSamples
                        lstmInputs{i} = double(squeeze(lstmData(:,:,i)));
                    end
                end

                % CNN+LSTMデータセット作成
                combinedData = [cnnInputs, lstmInputs];

                % ラベルの準備
                if ~iscategorical(labels)
                    labelOutputs = categorical(labels);
                else
                    labelOutputs = labels;
                end

                % データストアを作成
                combineDS = arrayDatastore(combinedData, 'OutputType', 'same');
                labelDS = arrayDatastore(labelOutputs);

                % データストアを結合
                ds = combine(combineDS, labelDS);

            catch ME
                error('Error creating hybrid datastore: %s', ME.message);
            end
        end
        
        % 過学習検証メソッド
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
        
                % トレンド分析
                [trainTrend, valTrend] = obj.analyzeLearningCurves(trainAcc, valAcc);

                % 完全な偏りの検出
                isCompletelyBiased = false;
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
                end
        
                % 学習進行の確認
                trainDiff = diff(trainAcc);
                isLearningProgressing = std(trainDiff) > 0.01;
                if ~isLearningProgressing
                    fprintf('警告: 学習が十分に進行していない可能性があります\n');
                    fprintf('  - 訓練精度の変化の標準偏差: %.4f\n', std(trainDiff));
                end
                
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
        
        % 学習曲線分析メソッド
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
        
        % 最適エポック特定メソッド
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(~, valAcc)
            % 検証精度が最大となるエポックを最適エポックとして返す
            totalEpochs = length(valAcc);
            [~, optimalEpoch] = max(valAcc);
        end
        
        % 過学習重症度判定メソッド
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
        
        % 性能表示メソッド
        function displayResults(obj)
            try
                fprintf('\n=== Hybrid Classification Results with AdaBoost ===\n');
                if ~isempty(obj.performance)
                    fprintf('Overall Accuracy: %.2f%%\n', obj.performance.overallAccuracy * 100);
                    
                    if ~isempty(obj.performance.confusionMatrix)
                        fprintf('\nConfusion Matrix:\n');
                        disp(obj.performance.confusionMatrix);
                    end
                    
                    fprintf('\nClass-wise Performance:\n');
                    fprintf('  Precision: %.2f%%\n', obj.performance.precision * 100);
                    fprintf('  Recall: %.2f%%\n', obj.performance.recall * 100);
                    fprintf('  F1-Score: %.2f%%\n', obj.performance.f1score * 100);
                end

                if ~isempty(obj.overfitMetrics)
                    fprintf('\nOverfitting Analysis:\n');
                    fprintf('Performance Gap: %.2f%%\n', obj.overfitMetrics.performanceGap);
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