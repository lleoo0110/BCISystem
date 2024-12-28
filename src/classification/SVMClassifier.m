classdef SVMClassifier < handle
    properties (Access = private)
        params          % パラメータ設定
        isOptimized     % 最適化の有無
        isEnabled       % SVMの有効/無効
        trainResults    % 学習結果の保存
        crossValModel   % 交差検証モデル
    end
    
    properties (Access = public)
        svmModel        % 学習済みSVMモデル
        performance     % 性能評価指標
    end
    
    methods (Access = public)
        function obj = SVMClassifier(params)
            % コンストラクタ
            obj.params = params;
            obj.isOptimized = params.classifier.svm.optimize;
            obj.isEnabled = params.classifier.svm.enable;
            obj.performance = struct();
        end
        
        function results = trainSVM(obj, features, labels)
            if ~obj.isEnabled
                error('SVM is disabled in configuration');
            end

            try
                % モデルの学習
                obj.trainModel(features, labels);

                % 交差検証の実行
                if obj.params.classifier.evaluation.enable
                    obj.performCrossValidation();
                end

                % 性能評価
                obj.evaluatePerformance(labels);

                % 最適閾値の探索
                if obj.params.classifier.evaluation.enable
                    bestThreshold = obj.findOptimalThreshold(features, labels);
                    obj.performance.optimalThreshold = bestThreshold;
                end

                results = struct('performance', obj.performance, 'model', obj.svmModel);
                obj.displayResults();

            catch ME
                error('SVM training failed: %s', ME.message);
            end
        end
        
        function [label, score] = predictOnline(obj, features, model, threshold)
            if ~obj.isEnabled
                error('SVM is disabled');
            end

            try
                [label, scores] = predict(model, features);
                score = scores(:,1);  % クラス1（安静）の確率を返す

                if obj.params.classifier.svm.probability
                    % 閾値判定による2値分類
                    if score >= threshold
                        label = 1;  % 安静状態
                    else
                        label = 2;  % タスク状態
                    end
                end
                
            catch ME
                error('SVM prediction failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function trainModel(obj, features, labels)
            if obj.isOptimized
                obj.svmModel = fitcsvm(features, labels, ...
                    'OptimizeHyperparameters', 'auto', ...
                    'KernelFunction', obj.params.classifier.svm.kernel);
            else
                obj.svmModel = fitcsvm(features, labels, ...
                    'KernelFunction', obj.params.classifier.svm.kernel);
            end

            if obj.params.classifier.svm.probability
                obj.svmModel = fitPosterior(obj.svmModel);
            end
        end
        
        function performCrossValidation(obj)
            if ~obj.params.classifier.evaluation.enable
                return;
            end
            
            obj.crossValModel = crossval(obj.svmModel, ...
                'KFold', obj.params.classifier.evaluation.kfold);
            
            obj.performance.cvAccuracies = zeros(obj.params.classifier.evaluation.kfold, 1);
            for i = 1:obj.params.classifier.evaluation.kfold
                [~, ~] = kfoldPredict(obj.crossValModel);
                obj.performance.cvAccuracies(i) = 1 - kfoldLoss(obj.crossValModel, 'Folds', i);
            end
            
            obj.performance.cvMeanAccuracy = mean(obj.performance.cvAccuracies);
            obj.performance.cvStdAccuracy = std(obj.performance.cvAccuracies);
        end
        
        function evaluatePerformance(obj, labels)
            [pred, score] = resubPredict(obj.svmModel);
            
            obj.performance.accuracy = mean(pred == labels);
            obj.performance.confusionMat = confusionmat(labels, pred);
            obj.performance.classLabels = unique(labels);
            
            if length(obj.performance.classLabels) == 2
                [obj.performance.roc.X, obj.performance.roc.Y, ~, obj.performance.auc] = ...
                    perfcurve(labels, score(:,2), obj.performance.classLabels(2));
            end
            
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
        
        function bestThreshold = findOptimalThreshold(obj, features, labels)
            thresholds = obj.params.classifier.svm.threshold.range;
            cv = cvpartition(labels, 'KFold', obj.params.classifier.evaluation.kfold);
            bestAccuracy = 0;
            bestThreshold = obj.params.classifier.svm.threshold.rest;

            for t = thresholds
                accuracies = zeros(cv.NumTestSets, 1);
                for i = 1:cv.NumTestSets
                    [~, scores] = predict(obj.svmModel, features(cv.test(i),:));
                    predictions = (scores(:,2) >= t) + 1;
                    accuracies(i) = sum(predictions == labels(cv.test(i))) / length(cv.test(i));
                end

                meanAccuracy = mean(accuracies);
                if meanAccuracy > bestAccuracy
                    bestAccuracy = meanAccuracy;
                    bestThreshold = t;
                end
            end
        end

        function displayResults(obj)
            fprintf('\n=== SVM Classification Results ===\n');
            fprintf('Overall Accuracy: %.2f%%\n', obj.performance.accuracy * 100);
            
            if obj.params.classifier.evaluation.enable
                fprintf('Cross-validation Accuracy: %.2f%% (±%.2f%%)\n', ...
                    obj.performance.cvMeanAccuracy * 100, ...
                    obj.performance.cvStdAccuracy * 100);
            end
            
            if length(obj.performance.classLabels) == 2
                fprintf('AUC: %.3f\n', obj.performance.auc);
            end
            
            fprintf('\nConfusion Matrix:\n');
            disp(obj.performance.confusionMat);
        end
    end
end