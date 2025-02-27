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

                results = struct( ...
                    'performance', obj.performance, ...
                    'model', obj.svmModel ...
                );
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
                % 入力データの検証
                validateattributes(features, {'numeric'}, {'2d'}, 'predictOnline', 'features');
                if isempty(features)
                    error('Features cannot be empty');
                end

                % モデルの検証
                if isempty(model)
                    error('Model cannot be empty');
                end

                % モデルのクラスラベルを取得
                classLabels = model.ClassNames;
                if length(classLabels) ~= 2
                    error('SVMClassifier only supports binary classification');
                end

                % 予測の実行
                [predicted_label, scores] = predict(model, features);

                % 確率推定が有効な場合の処理
                if obj.params.classifier.svm.probability
                    % 第1クラスの確率を取得
                    score = scores(:,1);

                    % 閾値による2値分類
                    if ~isempty(threshold)
                        % 閾値に基づいてラベルを割り当て
                        label = classLabels(2) * ones(size(score));  % デフォルトで第2クラス
                        label(score >= threshold) = classLabels(1);  % 閾値以上なら第1クラス
                    else
                        % 閾値がない場合はモデルの予測をそのまま使用
                        label = predicted_label;
                    end
                else
                    % 確率推定が無効な場合
                    label = predicted_label;
                    score = scores(:,1);
                end

                % 出力の検証
                if any(isnan(label)) || any(isnan(score))
                    error('NaN values detected in prediction output');
                end

                % デバッグ情報の出力（詳細モード時のみ）
                if obj.params.classifier.svm.probability
                    fprintf('予測詳細:\n');
                    fprintf('  クラス: [%d, %d]\n', classLabels(1), classLabels(2));
                    fprintf('  確率: [%.3f, %.3f]\n', scores(1,1), scores(1,2));
                    fprintf('  予測クラス: %d\n', label(1));
                    if ~isempty(threshold)
                        fprintf('  使用閾値: %.3f\n', threshold);
                    end
                end

            catch ME
                % エラースタックの表示（デバッグ用）
                fprintf('Error details:\n');
                fprintf('  Message: %s\n', ME.message);
                fprintf('  Input features size: [%s]\n', mat2str(size(features)));
                if exist('scores', 'var')
                    fprintf('  Scores size: [%s]\n', mat2str(size(scores)));
                    if exist('classLabels', 'var')
                        fprintf('  Class labels: [%d, %d]\n', classLabels(1), classLabels(2));
                    end
                    fprintf('  Score values: [%.3f, %.3f]\n', scores(1,1), scores(1,2));
                end
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