classdef HybridClassifier < handle
    properties (Access = private)
        params              % パラメータ設定
        net                 % ハイブリッドネットワーク
        isEnabled           % 有効/無効フラグ
        isInitialized       % 初期化フラグ
        useGPU              % GPU使用フラグ
        
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
        function obj = HybridClassifier(params)
            obj.params = params;
            obj.isEnabled = params.classifier.hybrid.enable;
            obj.isInitialized = false;
            obj.initializeProperties();
            obj.useGPU = params.classifier.hybrid.gpu;
        end
        
        function results = trainHybrid(obj, processedData, processedLabel)
            % 入力が1つの場合は、同じデータを両ブランチに供給
            if ~iscell(processedData)
                processedData = {processedData, processedData};
            elseif isscalar(processedData)
                processedData = {processedData{1}, processedData{1}};
            elseif numel(processedData) ~= 2
                error('ProcessedData must be a single array or a cell array with one or two elements.');
            end
            
            try
                fprintf('\n=== Starting Hybrid Training ===\n');
                
                cnnData = processedData{1};
                lstmData = processedData{2};
                
                % サンプル数の一致を確認
                if size(cnnData,3) ~= size(lstmData,3)
                    error('The number of samples in cnnData and lstmData must be equal');
                end
                
                % データ分割
                numSamples = size(cnnData, 3);
                [trainIdx, valIdx, testIdx] = obj.splitDatasetIndices(numSamples);
                
                % ラベルの分割
                trainLabels = processedLabel(trainIdx);
                valLabels = processedLabel(valIdx);
                testLabels = processedLabel(testIdx);
                
                % ラベルを列ベクトルに変換し、categorical にする
                trainLabels = categorical(trainLabels(:));
                valLabels   = categorical(valLabels(:));
                testLabels  = categorical(testLabels(:));
                
                % データの前処理（CNN）
                prepTrainCNN = obj.prepareDataForCNN(cnnData(:,:,trainIdx));
                prepValCNN   = obj.prepareDataForCNN(cnnData(:,:,valIdx));
                prepTestCNN  = obj.prepareDataForCNN(cnnData(:,:,testIdx));
                
                % データの前処理（LSTM）
                prepTrainLSTM = obj.prepareDataForLSTM(lstmData(:,:,trainIdx));
                prepValLSTM   = obj.prepareDataForLSTM(lstmData(:,:,valIdx));
                prepTestLSTM  = obj.prepareDataForLSTM(lstmData(:,:,testIdx));
                
                % LSTM 用のセル配列を明示的に縦ベクトルに変換
                prepTrainLSTM = prepTrainLSTM(:);
                prepValLSTM   = prepValLSTM(:);
                prepTestLSTM  = prepTestLSTM(:);
                
                % 入力サイズの決定
                cnnInputSize = size(prepTrainCNN(:,:,:,1));
                lstmInputSize = size(prepTrainLSTM{1},1);
                
                % モデルの学習
                [hybridModel, trainInfo] = obj.trainHybridModel(...
                    prepTrainCNN, prepTrainLSTM, trainLabels, ...
                    prepValCNN, prepValLSTM, valLabels, cnnInputSize, lstmInputSize);
                
                % テストデータでの評価
                testData = {prepTestCNN, prepTestLSTM};
                testMetrics = obj.evaluateModel(hybridModel, testData, testLabels);
                
                % 過学習の検証
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);
                if isOverfit
                    warning('Overfitting detected: %s severity', obj.overfitMetrics.severity);
                end
                
                % 性能指標の更新
                obj.updatePerformanceMetrics(testMetrics);
                
                % 交差検証の実行（有効な場合）
                crossValidationResults = struct('meanAccuracy', [], 'stdAccuracy', []);
                if obj.params.classifier.hybrid.training.validation.enable
                    crossValidationResults = obj.performCrossValidation(processedData, processedLabel);
                end
                
                % 結果の構築
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
        
        function [label, score] = predictOnline(obj, data, hybridModel)
            if ~obj.isEnabled
                error('Hybrid classifier is disabled');
            end
        
            try
                % 入力が1つの場合は複製
                if ~iscell(data)
                    data = {data, data};
                elseif isscalar(data)
                    data = {data{1}, data{1}};
                end
        
                % もしdata{2}がcellであれば、数値配列を取り出す
                if iscell(data{2})
                    data{2} = data{2}{1};
                end
        
                % 入力が2次元の場合は、サンプル数1として3次元に変換
                if ndims(data{1}) == 2
                    data{1} = reshape(data{1}, size(data{1},1), size(data{1},2), 1);
                end
                if ndims(data{2}) == 2
                    data{2} = reshape(data{2}, size(data{2},1), size(data{2},2), 1);
                end
        
                % データの前処理
                prepCNN = obj.prepareDataForCNN(data{1});
                prepLSTM = obj.prepareDataForLSTM(data{2});
        
                % 予測の実行
                [label, scores] = classify(hybridModel, prepCNN, prepLSTM);
                score = scores(:,1);  % クラス1の確率
        
            catch ME
                fprintf('Error in online prediction: %s\n', ME.message);
                rethrow(ME);
            end
        end
    end
    
    methods (Access = private)
        function initializeProperties(obj)
            obj.trainingHistory = struct('loss', [], 'accuracy', []);
            obj.validationHistory = struct('loss', [], 'accuracy', []);
            obj.bestValAccuracy = 0;
            obj.patienceCounter = 0;
            obj.overfitMetrics = struct();
        end
        
        function [trainIdx, valIdx, testIdx] = splitDatasetIndices(obj, numSamples)
            k = obj.params.classifier.hybrid.training.validation.kfold;
            rng('default');  % 再現性のため
            
            shuffledIdx = randperm(numSamples);
            trainRatio = (k-1)/k;
            valRatio = 1/(2*k);
            
            numTrain = floor(numSamples * trainRatio);
            numVal = floor(numSamples * valRatio);
            
            trainIdx = shuffledIdx(1:numTrain);
            valIdx = shuffledIdx(numTrain+1:numTrain+numVal);
            testIdx = shuffledIdx(numTrain+numVal+1:end);
            
            fprintf('Data split (k=%d):\n', k);
            fprintf('  Training: %d samples (%.1f%%)\n', length(trainIdx), (length(trainIdx)/numSamples)*100);
            fprintf('  Validation: %d samples (%.1f%%)\n', length(valIdx), (length(valIdx)/numSamples)*100);
            fprintf('  Test: %d samples (%.1f%%)\n', length(testIdx), (length(testIdx)/numSamples)*100);
        end
        
        function preparedData = prepareDataForCNN(~, data)
            % 入力が2次元の場合は、サンプル数1として3次元に変換
            if ismatrix(data)
                data = reshape(data, size(data,1), size(data,2), 1);
            elseif ndims(data) ~= 3
                error('Input data must be 3-dimensional [channels x timepoints x samples]');
            end
        
            [channels, timepoints, samples] = size(data);
        
            % 入力データを4次元テンソルに変換: [timepoints x channels x 1 x samples]
            preparedData = zeros(timepoints, channels, 1, samples);
        
            for i = 1:samples
                preparedData(:,:,1,i) = data(:,:,i)';
            end
        end
        
        function preparedData = prepareDataForLSTM(~, data)
            % 入力が2次元の場合は、サンプル数1として3次元に変換
            if isnumeric(data) && ismatrix(data)
                data = reshape(data, size(data,1), size(data,2), 1);
            elseif ~isnumeric(data) || ndims(data) ~= 3
                error('Input data must be 3-dimensional [channels x timepoints x samples] or [timepoints x channels x samples]');
            end
        
            [dim1, dim2, samples] = size(data);
            preparedData = cell(samples, 1);
        
            for i = 1:samples
                % 入力が [timepoints x channels] の場合は転置して [channels x timepoints] にする
                if dim1 > dim2
                    preparedData{i} = data(:,:,i)';
                else
                    preparedData{i} = data(:,:,i);
                end
            end
        end
        
        function layers = buildHybridLayers(obj, cnnInputSize, lstmInputSize)
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
                
                % 畳み込み層群（各層ごとにプーリング・ドロップアウトを設定可能）
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
                    
                    % プーリング層（対応するフィールドが存在する場合）
                    poolField = sprintf('pool%d', i);
                    if isfield(arch.cnn.poolLayers, poolField)
                        poolParam = arch.cnn.poolLayers.(poolField);
                        poolName = sprintf('pool_%d', i);
                        poolLayerObj = maxPooling2dLayer(poolParam.size, ...
                            'Stride', poolParam.stride, ...
                            'Name', poolName);
                        cnnLayers = [cnnLayers; poolLayerObj];
                    end
                    
                    % ドロップアウト層（対応するフィールドが存在する場合）
                    dropoutField = sprintf('dropout%d', i);
                    if isfield(arch.cnn.dropoutLayers, dropoutField)
                        dropoutRate = arch.cnn.dropoutLayers.(dropoutField);
                        dropoutName = sprintf('dropout_%d', i);
                        dropoutLayerObj = dropoutLayer(dropoutRate, 'Name', dropoutName);
                        cnnLayers = [cnnLayers; dropoutLayerObj];
                    end
                end
                
                % 全結合層群（パラメータ arch.cnn.fullyConnected を使用、存在しなければデフォルト値を使用）
                if isfield(arch.cnn, 'fullyConnected')
                    fcSizes = arch.cnn.fullyConnected;
                else
                    fcSizes = 128; % またはデフォルト例: [128]
                end
                if ~isempty(fcSizes)
                    for j = 1:length(fcSizes)
                        fcName = sprintf('fc_cnn_%d', j);
                        fcLayerObj = fullyConnectedLayer(fcSizes(j), 'Name', fcName);
                        cnnLayers = [cnnLayers; fcLayerObj];
                        reluName = sprintf('relu_fc_cnn_%d', j);
                        cnnLayers = [cnnLayers; reluLayer('Name', reluName)];
                    end
                    cnnFeatureSize = fcSizes(end);  % 最終全結合層のユニット数を特徴数とする
                else
                    error('CNN fully connected layers not defined.');
                end

                % リシェイプ層（CNNブランチの出力を [1 1 cnnFeatureSize N] に変換）
                reshapeCNN = ReshapeCNNLayer('reshape_cnn', cnnFeatureSize);
                reshapeCNN = reshapeCNN.setOutputFormat('SSCB');
                cnnLayers = [cnnLayers; reshapeCNN];
                
                %% === LSTMブランチの構築 ===
                lstmLayers = [];
                
                % シーケンス入力層（パラメータより）
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
                    
                    % バッチ正規化（オプション）
                    if arch.batchNorm
                        bnName = sprintf('bn_%s', lstmName);
                        bnLayerObj = batchNormalizationLayer('Name', bnName);
                        lstmLayers = [lstmLayers; bnLayerObj];
                    end
                    
                    % ドロップアウト層（対応するフィールドが存在する場合）
                    dropoutField = sprintf('dropout%d', i);
                    if isfield(arch.lstm.dropoutLayers, dropoutField)
                        dropoutRate = arch.lstm.dropoutLayers.(dropoutField);
                        dropoutName = sprintf('dropout_%s', lstmName);
                        dropoutLayerObj = dropoutLayer(dropoutRate, 'Name', dropoutName);
                        lstmLayers = [lstmLayers; dropoutLayerObj];
                    end
                end
                
                % LSTMブランチ用の全結合層（arch.lstm.fullyConnected が存在すればそちらを、なければデフォルト128を使用）
                if isfield(arch.lstm, 'fullyConnected')
                    lstmFCSize = arch.lstm.fullyConnected;
                else
                    lstmFCSize = 128;
                end

                fcLSTM = fullyConnectedLayer(lstmFCSize, 'Name', 'fc_lstm');
                reluFCL = reluLayer('Name', 'relu_fc_lstm');
                lstmLayers = [lstmLayers; fcLSTM; reluFCL];
                
                % リシェイプ層（LSTMブランチの出力を [1 1 lstmFCSize N] に変換）
                reshapeLSTM = ReshapeLSTMLayer('reshape_lstm', lstmFCSize);
                reshapeLSTM = reshapeLSTM.setOutputFormat('SSCB');
                lstmLayers = [lstmLayers; reshapeLSTM];
                
                %% === マージ層と出力層 ===
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
                
                % 結合後の全結合層（パラメータ merge.fullyConnected.layers を使用）
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
                
                % 最終出力層
                fcFinal = fullyConnectedLayer(numClasses, 'Name', 'fc_final');
                mergeLayers = [mergeLayers; fcFinal];
                
                if strcmpi(mergeParams.output.activation, 'softmax')
                    softmaxL = softmaxLayer('Name', strcat(mergeParams.output.name, '_softmax'));
                    mergeLayers = [mergeLayers; softmaxL];
                end
                
                classOutput = classificationLayer('Name', mergeParams.output.name);
                mergeLayers = [mergeLayers; classOutput];
                    
                %% === レイヤーグラフの構築と接続 ===
                layers = layerGraph();
                layers = addLayers(layers, cnnLayers);
                layers = addLayers(layers, lstmLayers);
                layers = addLayers(layers, mergeLayers);
                
                % ブランチの接続（各ブランチの最終リシェイプ層からマージ層へ）
                layers = connectLayers(layers, 'reshape_cnn', 'concat/in1');
                layers = connectLayers(layers, 'reshape_lstm', 'concat/in2');
                
                % デバッグ情報の出力
                fprintf('\nNetwork architecture summary:\n');
                fprintf('CNN branch:\n');
                fprintf('  - Input size: [%d %d 1]\n', cnnInputSize(1), cnnInputSize(2));
                fprintf('  - Total Conv layers: %d\n', numel(convFields));
                if exist('fcSizes', 'var') && ~isempty(fcSizes)
                    fprintf('  - FC layers: [%s]\n', num2str(fcSizes));
                else
                    fprintf('  - FC layers: None\n');
                end
                fprintf('\nLSTM branch:\n');
                fprintf('  - Input size: %d features\n', lstmInputSize);
                fprintf('  - Total LSTM layers: %d\n', numel(lstmFields));
                fprintf('  - LSTM FC size: %d\n', lstmFCSize);
                fprintf('\nMerged output classes: %d\n', numClasses);
                
                % analyzeNetwork(layers);
                
            catch ME
                fprintf('\nError in buildHybridLayers:\n');
                fprintf('Error message: %s\n', ME.message);
                if ~isempty(ME.stack)
                    fprintf('Error in: %s, Line: %d\n', ME.stack(1).name, ME.stack(1).line);
                end
                rethrow(ME);
            end
        end

        function [hybridModel, trainInfo] = trainHybridModel(obj, trainCNNData, trainLSTMData, trainLabels, valCNNData, valLSTMData, valLabels, cnnInputSize, lstmInputSize)
            try
                % ラベルの前処理
                uniqueLabels = unique(trainLabels);
                trainLabels = categorical(trainLabels, uniqueLabels);
                valLabels = categorical(valLabels, uniqueLabels);

                % データストアの作成
                trainDS = obj.createHybridDatastore(trainCNNData, trainLSTMData, trainLabels);
                valDS = obj.createHybridDatastore(valCNNData, valLSTMData, valLabels);

                % 訓練設定
                trainInfo = struct('History', []);
                executionEnvironment = 'cpu';
                if obj.useGPU
                    executionEnvironment = 'gpu';
                end

                % トレーニングオプションの設定
                optimizerType = obj.params.classifier.hybrid.training.optimizer.type;
                options = trainingOptions(optimizerType, ...
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
                    'GradientThreshold', obj.params.classifier.hybrid.training.optimizer.gradientThreshold, ...
                    'OutputFcn', @(info)obj.trainingOutputFcn(info));

                % レイヤーグラフの構築とトレーニング実行
                layers = obj.buildHybridLayers(cnnInputSize, lstmInputSize);
                fprintf('\nStarting model training...\n');
                [hybridModel, trainHistory] = trainNetwork(trainDS, layers, options);

                trainInfo.History = trainHistory;
                trainInfo.FinalEpoch = length(trainHistory.TrainingLoss);
                fprintf('\nTraining completed: %d epochs\n', trainInfo.FinalEpoch);

            catch ME
                fprintf('\n=== Error in trainHybridModel: %s\n', ME.message);
                rethrow(ME);
            end
        end

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

                 % 個別のデータストアを作成
                combineDS = arrayDatastore(combinedData, 'OutputType', 'same');
                labelDS = arrayDatastore(labelOutputs);

                 % データストアを結合
                ds = combine(combineDS, labelDS);

            catch ME
                error('Error creating hybrid datastore: %s', ME.message);
            end
        end

        function stop = trainingOutputFcn(obj, info)
            stop = false;
            if info.State == "start"
                obj.currentEpoch = 0;
                return;
            end
            obj.currentEpoch = obj.currentEpoch + 1;
            if ~isempty(info.ValidationLoss)
                currentAccuracy = info.ValidationAccuracy;
                if currentAccuracy > obj.bestValAccuracy
                    obj.bestValAccuracy = currentAccuracy;
                    obj.patienceCounter = 0;
                else
                    obj.patienceCounter = obj.patienceCounter + 1;
                    if obj.patienceCounter >= obj.params.classifier.hybrid.training.patience
                        fprintf('\nEarly stopping triggered at epoch %d\n', obj.currentEpoch);
                        stop = true;
                    end
                end
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

            try
                uniqueLabels = unique(testLabels);
                testLabels = categorical(testLabels, uniqueLabels);
                
                % 予測実行
                [pred, scores] = classify(model, testData{1}, testData{2});

                metrics.accuracy = mean(pred == testLabels);
                metrics.confusionMat = confusionmat(testLabels, pred);
                numClasses = length(uniqueLabels);
                metrics.classwise = struct('precision', zeros(1, numClasses), ...
                                             'recall', zeros(1, numClasses), ...
                                             'f1score', zeros(1, numClasses));
                for i = 1:numClasses
                    className = uniqueLabels(i);
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
                        f1score = 0;
                    else
                        f1score = 2 * (precision * recall) / (precision + recall);
                    end
                    metrics.classwise(i).precision = precision;
                    metrics.classwise(i).recall = recall;
                    metrics.classwise(i).f1score = f1score;
                end
                if numClasses == 2
                    [~,~,~,AUC] = perfcurve(testLabels, scores(:,2), uniqueLabels(2));
                    metrics.auc = AUC;
                end
            catch ME
                fprintf('Error in evaluateModel: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        function [isOverfit, metrics] = validateOverfitting(obj, trainInfo, testMetrics)
            try
                fprintf('\n=== Validating Overfitting ===\n');
                history = trainInfo.History;
                trainAcc = history.TrainingAccuracy;
                valAcc = history.ValidationAccuracy;
                testAcc = testMetrics.accuracy * 100;
                fprintf('Final Training Accuracy: %.2f%%\n', trainAcc(end));
                fprintf('Final Validation Accuracy: %.2f%%\n', valAcc(end));
                fprintf('Test Accuracy: %.2f%%\n', testAcc);
                genGap = abs(trainAcc(end) - valAcc(end));
                perfGap = abs(trainAcc(end) - testAcc);
                fprintf('Generalization Gap: %.2f%%\n', genGap);
                fprintf('Performance Gap: %.2f%%\n', perfGap);
                [trainTrend, valTrend] = obj.analyzeLearningCurves(trainAcc, valAcc);
                isCompletelyBiased = false;
                if isfield(testMetrics, 'confusionMat')
                    cm = testMetrics.confusionMat;
                    missingActual = any(sum(cm, 2) == 0);
                    missingPredicted = any(sum(cm, 1) == 0);
                    isCompletelyBiased = missingActual || missingPredicted;
                end
                isLearningProgressing = std(diff(trainAcc)) > 0.01;
                [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(valAcc);
                severity = obj.determineOverfittingSeverity(genGap, perfGap, isCompletelyBiased, isLearningProgressing);
                metrics = struct('generalizationGap', genGap, ...
                                 'performanceGap', perfGap, ...
                                 'isCompletelyBiased', isCompletelyBiased, ...
                                 'isLearningProgressing', isLearningProgressing, ...
                                 'validationTrend', valTrend, ...
                                 'trainingTrend', trainTrend, ...
                                 'severity', severity, ...
                                 'optimalEpoch', optimalEpoch, ...
                                 'totalEpochs', totalEpochs);
                isOverfit = ismember(severity, {'critical', 'severe', 'moderate', 'mild'});
                fprintf('Overfitting Status: %s (Severity: %s)\n', mat2str(isOverfit), severity);
            catch ME
                fprintf('Error in validateOverfitting: %s\n', ME.message);
                metrics = struct('generalizationGap', Inf, 'performanceGap', Inf, ...
                                 'isCompletelyBiased', true, 'isLearningProgressing', false, ...
                                 'severity', 'error', 'optimalEpoch', 0, 'totalEpochs', 0);
                isOverfit = true;
            end
        end
        
        function displayResults(obj)
            try
                fprintf('\n=== Hybrid Classification Results ===\n');
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
                    fprintf('Generalization Gap: %.2f%%\n', obj.overfitMetrics.generalizationGap);
                    fprintf('Performance Gap: %.2f%%\n', obj.overfitMetrics.performanceGap);
                    fprintf('Severity: %s\n', obj.overfitMetrics.severity);
                    if isfield(obj.overfitMetrics, 'validationTrend')
                        trend = obj.overfitMetrics.validationTrend;
                        fprintf('\nValidation Trend:\n');
                        fprintf('  Mean Change: %.4f\n', trend.mean_change);
                        fprintf('  Volatility: %.4f\n', trend.volatility);
                        fprintf('  Increasing Ratio: %.2f%%\n', trend.increasing_ratio * 100);
                    end
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function [trainTrend, valTrend] = analyzeLearningCurves(~, trainAcc, valAcc)
            if isempty(trainAcc) || isempty(valAcc)
                error('Learning curve data is empty');
            end
            trainAcc = trainAcc(~isnan(trainAcc));
            valAcc = valAcc(~isnan(valAcc));
            if isempty(trainAcc) || isempty(valAcc)
                error('No valid data after removing NaNs');
            end
            windowSize = min(5, floor(length(trainAcc)/3));
            windowSize = max(windowSize, 1);
            trainSmooth = movmean(trainAcc, windowSize);
            valSmooth = movmean(valAcc, windowSize);
            trainDiff = diff(trainSmooth);
            valDiff = diff(valSmooth);
            trainTrend = struct('mean_change', mean(trainDiff), 'volatility', std(trainDiff), ...
                                'increasing_ratio', sum(trainDiff > 0) / length(trainDiff));
            valTrend = struct('mean_change', mean(valDiff), 'volatility', std(valDiff), ...
                              'increasing_ratio', sum(valDiff > 0) / length(valDiff));
        end
        
        function severity = determineOverfittingSeverity(~, genGap, perfGap, isCompletelyBiased, isLearningProgressing)
            if isCompletelyBiased
                severity = 'critical';
            elseif ~isLearningProgressing
                severity = 'failed';
            elseif genGap > 10 || perfGap > 15
                severity = 'severe';
            elseif genGap > 5 || perfGap > 8
                severity = 'moderate';
            elseif genGap > 3 || perfGap > 5
                severity = 'mild';
            else
                severity = 'none';
            end
        end
        
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(~, valAcc)
            totalEpochs = length(valAcc);
            [~, optimalEpoch] = max(valAcc);
            if optimalEpoch == totalEpochs
                fprintf('Warning: Optimal epoch is the last epoch. More training might be beneficial.\n');
            end
        end
        
        function cvResults = performCrossValidation(obj, data, labels)
            k = obj.params.classifier.hybrid.training.validation.kfold;
            fprintf('\n=== Starting %d-fold cross-validation ===\n', k);
            cvResults = struct();
            cvResults.folds = struct('accuracy', zeros(1, k), 'confusionMat', cell(1, k), ...
                                     'classwise', cell(1, k), 'validation_curve', cell(1, k));
            numSamples = size(data{1}, 3);
            for i = 1:k
                fprintf('\nProcessing fold %d/%d\n', i, k);
                try
                    [trainIdx, valIdx, testIdx] = obj.splitDatasetIndices(numSamples);
                    prepTrainCNN = obj.prepareDataForCNN(data{1}(:,:,trainIdx));
                    prepValCNN = obj.prepareDataForCNN(data{1}(:,:,valIdx));
                    prepTestCNN = obj.prepareDataForCNN(data{1}(:,:,testIdx));
                    prepTrainLSTM = obj.prepareDataForLSTM(data{2}(:,:,trainIdx));
                    prepValLSTM = obj.prepareDataForLSTM(data{2}(:,:,valIdx));
                    prepTestLSTM = obj.prepareDataForLSTM(data{2}(:,:,testIdx));
                    [model, trainInfo] = obj.trainHybridModel(prepTrainCNN, prepTrainLSTM, labels(trainIdx), ...
                                                              prepValCNN, prepValLSTM, labels(valIdx), ...
                                                              size(prepTrainCNN(:,:,:,1)), size(prepTrainLSTM{1},1));
                    metrics = obj.evaluateModel(model, {prepTestCNN, prepTestLSTM}, labels(testIdx));
                    cvResults.folds.accuracy(i) = metrics.accuracy;
                    cvResults.folds.confusionMat{i} = metrics.confusionMat;
                    cvResults.folds.classwise{i} = metrics.classwise;
                    if isfield(trainInfo, 'History')
                        cvResults.folds.validation_curve{i} = struct(...
                            'train_accuracy', trainInfo.History.TrainingAccuracy, ...
                            'val_accuracy', trainInfo.History.ValidationAccuracy, ...
                            'train_loss', trainInfo.History.TrainingLoss, ...
                            'val_loss', trainInfo.History.ValidationLoss);
                    end
                    fprintf('Fold %d accuracy: %.2f%%\n', i, cvResults.folds.accuracy(i) * 100);
                catch ME
                    warning('Error in fold %d: %s', i, ME.message);
                    cvResults.folds.accuracy(i) = 0;
                end
            end
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
            fprintf('\nCross-validation Results:\n');
            fprintf('Mean Accuracy: %.2f%% (±%.2f%%)\n', cvResults.meanAccuracy * 100, cvResults.stdAccuracy * 100);
            fprintf('Min Accuracy: %.2f%%\n', cvResults.minAccuracy * 100);
            fprintf('Max Accuracy: %.2f%%\n', cvResults.maxAccuracy * 100);
            fprintf('Successfully completed folds: %d/%d\n', sum(validFolds), k);
        end
        
        function updatePerformanceMetrics(obj, testMetrics)
            obj.performance = testMetrics;
        end
    end
end
