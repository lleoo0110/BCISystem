classdef CNNClassifier < handle
    properties (Access = private)
        params
        net
        trainInfo
        isEnabled
        inputSize
    end
    
    properties (Access = public)
        performance
    end
    
    methods (Access = public)
        function obj = CNNClassifier(params)
            obj.params = params;
            obj.isEnabled = params.classifier.cnn.enable;
            obj.inputSize = params.classifier.cnn.architecture.inputSize;
        end
        
        function results = trainCNN(obj, data, labels)
            if ~obj.isEnabled
                error('CNN is disabled in configuration');
            end

            try
                % EEGデータをCNN入力形式に変換
                features = obj.convertEEGtoCNNInput(data);
                
                obj.buildNetwork();
                obj.trainModel(features, labels);
                obj.evaluatePerformance(features, labels);

                results = struct('performance', obj.performance, 'model', obj.net);
                obj.displayResults();

            catch ME
                error('CNN training failed: %s', ME.message);
            end
        end
        
        function [label, score] = predictOnline(obj, features, model)
            if ~obj.isEnabled
                error('CNN is disabled');
            end

            try
                % CNNの入力形式に変換
                cnnInput = obj.convertEEGtoCNNInput(features);

                % 予測実行
                [label, scores] = classify(model, cnnInput);
                score = max(scores, [], 2);

            catch ME
                error('CNN prediction failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function cnnInput = convertEEGtoCNNInput(obj, data)
            try
                if ndims(data) == 3  % エポック化されたデータ [channels × samples × epochs]
                    numEpochs = size(data, 3);
                    cnnInput = zeros(numEpochs, obj.inputSize(1), obj.inputSize(2), obj.inputSize(3));
                    
                    for i = 1:numEpochs
                        epoch_data = data(:,:,i);
                        resized_data = imresize(epoch_data, [obj.inputSize(1) obj.inputSize(2)]);
                        cnnInput(i,:,:,1) = resized_data;
                    end
                    
                else  % 連続データ [channels × samples]
                    cnnInput = zeros(1, obj.inputSize(1), obj.inputSize(2), obj.inputSize(3));
                    resized_data = imresize(data, [obj.inputSize(1) obj.inputSize(2)]);
                    cnnInput(1,:,:,1) = resized_data;
                end
                
            catch ME
                error('EEG to CNN conversion failed: %s', ME.message);
            end
        end

        function buildNetwork(obj)
            layers = [
                imageInputLayer(obj.inputSize)
            ];
            
            % 畳み込み層の追加
            convLayers = obj.params.classifier.cnn.architecture.layers;
            fields = fieldnames(convLayers);
            
            for i = 1:length(fields)
                layerConfig = convLayers.(fields{i});
                layers = [layers
                    convolution2dLayer(layerConfig.size, layerConfig.filters, ...
                    'Stride', layerConfig.stride, ...
                    'Padding', layerConfig.padding)
                    batchNormalizationLayer
                    reluLayer
                    maxPooling2dLayer(obj.params.classifier.cnn.architecture.poolSize, ...
                    'Stride', 2)
                    dropoutLayer(obj.params.classifier.cnn.architecture.dropoutRate)];
            end
            
            % 全結合層の追加
            fcSizes = obj.params.classifier.cnn.architecture.fullyConnected;
            for i = 1:length(fcSizes)
                layers = [layers
                    fullyConnectedLayer(fcSizes(i))
                    batchNormalizationLayer
                    reluLayer
                    dropoutLayer(obj.params.classifier.cnn.architecture.dropoutRate)];
            end
            
            % 出力層
            layers = [layers
                fullyConnectedLayer(obj.params.classifier.cnn.architecture.numClasses)
                softmaxLayer
                classificationLayer];
            
            obj.net = layerGraph(layers);
        end
        
        function trainModel(obj, features, labels)
            try
                % バリデーションデータの設定
                if obj.params.classifier.cnn.training.validation.ratio > 0
                    cv = cvpartition(labels, 'HoldOut', ...
                        obj.params.classifier.cnn.training.validation.ratio);
                    trainIdx = cv.training;
                    valIdx = cv.test;
                    
                    valData = features(valIdx,:,:,:);
                    valLabels = categorical(labels(valIdx));
                    features = features(trainIdx,:,:,:);
                    labels = labels(trainIdx);
                end
                
                % 訓練オプションの設定
                options = trainingOptions(obj.params.classifier.cnn.training.optimizer, ...
                    'InitialLearnRate', obj.params.classifier.cnn.training.initialLearnRate, ...
                    'MaxEpochs', obj.params.classifier.cnn.training.maxEpochs, ...
                    'MiniBatchSize', obj.params.classifier.cnn.training.miniBatchSize, ...
                    'ValidationFrequency', obj.params.classifier.cnn.training.validation.frequency, ...
                    'ValidationPatience', obj.params.classifier.cnn.training.validation.patience, ...
                    'Shuffle', 'every-epoch', ...
                    'Verbose', true, ...
                    'Plots', 'training-progress');
                
                if exist('valData', 'var')
                    options.ValidationData = {valData, valLabels};
                end
                
                if obj.params.classifier.cnn.inference.useGPU
                    options.ExecutionEnvironment = 'gpu';
                end
                
                [obj.net, obj.trainInfo] = trainNetwork(features, categorical(labels), ...
                    obj.net, options);
                
            catch ME
                error('Model training failed: %s', ME.message);
            end
        end
        
        function evaluatePerformance(obj, features, labels)
            [pred, ~] = classify(obj.net, features);
            pred = double(pred);
            
            obj.performance = struct();
            obj.performance.accuracy = mean(pred == labels);
            obj.performance.confusionMat = confusionmat(labels, pred);
            obj.performance.classLabels = unique(labels);
            
            % 学習曲線の保存
            obj.performance.training.loss = obj.trainInfo.TrainingLoss;
            obj.performance.training.accuracy = obj.trainInfo.TrainingAccuracy;
            
            if isfield(obj.trainInfo, 'ValidationLoss')
                obj.performance.validation.loss = obj.trainInfo.ValidationLoss;
                obj.performance.validation.accuracy = obj.trainInfo.ValidationAccuracy;
            end
            
            % クラスごとの性能指標
            for i = 1:length(obj.performance.classLabels)
                className = obj.performance.classLabels(i);
                classIdx = labels == className;
                obj.performance.classwise(i).precision = ...
                    sum(pred(classIdx) == className) / sum(pred == className);
                obj.performance.classwise(i).recall = ...
                    sum(pred(classIdx) == className) / sum(classIdx);
                obj.performance.classwise(i).f1score = ...
                    2 * (obj.performance.classwise(i).precision * obj.performance.classwise(i).recall) / ...
                    (obj.performance.classwise(i).precision + obj.performance.classwise(i).recall);
            end
        end

        function displayResults(obj)
            fprintf('\n=== CNN Classification Results ===\n');
            fprintf('Overall Accuracy: %.2f%%\n', obj.performance.accuracy * 100);
            
            fprintf('\nTraining Information:\n');
            fprintf('Final Training Loss: %.4f\n', obj.performance.training.loss(end));
            fprintf('Final Training Accuracy: %.2f%%\n', ...
                obj.performance.training.accuracy(end) * 100);
            
            if isfield(obj.performance, 'validation')
                fprintf('Final Validation Loss: %.4f\n', ...
                    obj.performance.validation.loss(end));
                fprintf('Final Validation Accuracy: %.2f%%\n', ...
                    obj.performance.validation.accuracy(end) * 100);
            end
            
            fprintf('\nConfusion Matrix:\n');
            disp(obj.performance.confusionMat);
            
            fprintf('\nClass-wise Performance:\n');
            for i = 1:length(obj.performance.classLabels)
                fprintf('Class %d:\n', obj.performance.classLabels(i));
                fprintf('  Precision: %.2f%%\n', obj.performance.classwise(i).precision * 100);
                fprintf('  Recall: %.2f%%\n', obj.performance.classwise(i).recall * 100);
                fprintf('  F1-Score: %.2f%%\n', obj.performance.classwise(i).f1score * 100);
            end
        end
    end
end