classdef SVMClassifier < handle
    properties (Access = private)
        params          % getConfig.mから取得した設定パラメータ
        modelType       % 'svm' または 'ecoc' 
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
            obj.modelType = lower(obj.params.classifier.svm.type);
            obj.isOptimized = obj.params.classifier.svm.optimize;
            obj.isEnabled = obj.params.classifier.svm.enable;
            obj.performance = struct();
            obj.validateParams();
        end
        
        function results = trainOffline(obj, features, labels)
            % オフラインモードでの学習と評価
            if ~obj.isEnabled
                error('SVM is disabled in configuration');
            end

            try
                % モデルの学習
                obj.trainModel(features, labels);

                % 交差検証の実行
                obj.performCrossValidation();

                % 性能評価
                obj.evaluatePerformance(labels);

                % 評価が有効な場合は最適閾値を探索
                if obj.params.classifier.evaluation.enable
                    bestThreshold = obj.findOptimalThreshold(features, labels);
                    obj.performance.optimalThreshold = bestThreshold;  % performanceに保存
                    fprintf('Optimal threshold found: %.3f\n', bestThreshold);
                end

                % 結果の格納
                results = struct();
                results.performance = obj.performance;  % 性能評価結果
                results.classifier = obj.svmModel;      % 学習済みモデル    

                % 結果の表示
                obj.displayResults();

            catch ME
                warning(ME.identifier, '%s', ME.message);
                results = [];
            end
        end
        
        function [label, score] = predictOnline(obj, features, svmclassifier, threshold)
            % オンラインモードでの予測
            if ~obj.isEnabled
                error('SVMClassifier:Disabled', 'SVM is disabled in configuration');
            end

            try
                validateattributes(features, {'numeric'}, {'2d'}, 'predictOnline', 'features');
                validateattributes(threshold, {'numeric'}, {'scalar', '>=', 0, '<=', 1}, ...
                    'predictOnline', 'threshold');

                if isempty(svmclassifier)
                    error('SVMClassifier:NoModel', 'No classifier model provided');
                end
                obj.svmModel = svmclassifier;

                % 予測の実行
                if obj.params.classifier.svm.probability
                    [~, scores] = predict(obj.svmModel, features);

                    % ECOCの場合
                    if strcmp(obj.params.classifier.svm.type, 'ecoc')
                        if scores(1) >= threshold
                            label = 1;  % 安静状態
                        else
                            % 安静状態以外で最も確信度が高いクラスを選択
                            [~, maxIdx] = max(scores(2:end));
                            label = maxIdx + 1;
                        end
                    else  % 通常のSVMの場合
                        if scores(1) >= threshold  % 安静状態のスコア
                            label = 1;
                        else
                            label = 2;
                        end
                    end
                    
                    % scoresをそのまま出力用のscoreとして保持
                    score = scores(1);  % 安静状態のスコア
                    
                else
                    % 通常の予測
                    [label, score] = predict(obj.svmModel, features);
                end

            catch ME
                error('SVMClassifier:PredictionFailed', 'Online prediction failed: %s', ME.message);
            end
        end

        function visualizeResults(obj)
            % 結果の可視化
            if isempty(obj.performance)
                error('No results available for visualization');
            end
            
            % 混同行列の表示
            figure('Name', 'Classification Results');
            
            subplot(2,2,1);
            confusionchart(obj.performance.confusionMat, ...
                obj.performance.classLabels);
            title('Confusion Matrix');
            
            % ROC曲線の表示（2クラス分類の場合）
            if length(obj.performance.classLabels) == 2
                subplot(2,2,2);
                plot(obj.performance.roc.X, obj.performance.roc.Y);
                xlabel('False Positive Rate');
                ylabel('True Positive Rate');
                title('ROC Curve');
                grid on;
            end
            
            % 分類精度の時系列表示
            if isfield(obj.performance, 'cvAccuracies')
                subplot(2,2,3);
                plot(obj.performance.cvAccuracies);
                xlabel('Fold');
                ylabel('Accuracy');
                title('Cross-validation Accuracies');
                grid on;
            end
            
            % 特徴重要度の表示
            subplot(2,2,4);
            bar(obj.performance.featureImportance);
            xlabel('Feature Index');
            ylabel('Importance');
            title('Feature Importance');
            grid on;
        end
    end
    
    methods (Access = private)
        function validateParams(obj)
            % パラメータの検証
            assert(ismember(obj.modelType, {'svm', 'ecoc'}), ...
                'Invalid model type. Must be either "svm" or "ecoc"');
            
            if obj.isOptimized
                assert(isfield(obj.params.classifier.svm, 'kernel'), ...
                    'Kernel function must be specified for optimization');
            end
        end
        
        function trainModel(obj, features, labels)
            if strcmp(obj.modelType, 'svm')
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
            else
                if obj.isOptimized
                    % パラメータの取得と検証
                    boxConstraints = obj.params.classifier.svm.hyperparameters.boxConstraint;
                    kernelScales = obj.params.classifier.svm.hyperparameters.kernelScale;


                    % 並列処理のための準備
                    numC = length(boxConstraints);
                    numK = length(kernelScales);
                    results = cell(numC * numK, 1);

                    % parfor用のインデックス作成
                    params_combinations = cell(numC * numK, 1);
                    idx = 1;
                    for i = 1:numC
                        for j = 1:numK
                            params_combinations{idx} = struct(...
                                'c', boxConstraints(i), ...
                                'k', kernelScales(j));
                            idx = idx + 1;
                        end
                    end

                    % 並列処理の実行
                    parfor i = 1:length(params_combinations)
                        param = params_combinations{i};

                        % SVMテンプレートの作成
                        t = templateSVM(...
                            'KernelFunction', obj.params.classifier.svm.kernel, ...
                            'BoxConstraint', param.c, ...
                            'KernelScale', param.k);

                        % モデルの学習
                        tempModel = fitcecoc(features, labels, 'Learners', t);

                        % 交差検証で性能評価
                        cvmodel = crossval(tempModel, 'KFold', 5);
                        accuracy = 1 - kfoldLoss(cvmodel);

                        % 結果の保存
                        results{i} = struct(...
                            'model', tempModel, ...
                            'accuracy', accuracy, ...
                            'boxConstraint', param.c, ...
                            'kernelScale', param.k);
                    end

                    % 最良のモデルを選択
                    accuracies = cellfun(@(x) x.accuracy, results);
                    [bestAccuracy, bestIdx] = max(accuracies);
                    bestResult = results{bestIdx};

                    % 結果の保存
                    obj.svmModel = bestResult.model;
                    obj.performance.grid_search.best_params.BoxConstraint = bestResult.boxConstraint;
                    obj.performance.grid_search.best_params.KernelScale = bestResult.kernelScale;
                    obj.performance.grid_search.best_accuracy = bestAccuracy;

                else
                    % 通常のECOCモデル学習
                    t = templateSVM('KernelFunction', obj.params.classifier.svm.kernel);
                    obj.svmModel = fitcecoc(features, labels, 'Learners', t);
                end
            end
        end
        
        function performCrossValidation(obj)
            % 交差検証の実行
            if ~obj.params.classifier.evaluation.enable
                return;
            end
            
            % crossvalModelは既にtrainModel内で作成されたsvmModelを使用する
            obj.crossValModel = crossval(obj.svmModel, ...
                'KFold', obj.params.classifier.evaluation.kfold);
            
            % 各フォールドの精度を計算
            obj.performance.cvAccuracies = zeros(obj.params.classifier.evaluation.kfold, 1);
            for i = 1:obj.params.classifier.evaluation.kfold
                [~, ~] = kfoldPredict(obj.crossValModel);
                obj.performance.cvAccuracies(i) = 1 - kfoldLoss(obj.crossValModel, 'Folds', i);
            end
            
            % 平均精度と標準偏差の計算
            obj.performance.cvMeanAccuracy = mean(obj.performance.cvAccuracies);
            obj.performance.cvStdAccuracy = std(obj.performance.cvAccuracies);
        end
        
        function evaluatePerformance(obj, labels)
            % 性能評価の実行
            [pred, score] = resubPredict(obj.svmModel);
            
            % 基本的な性能指標の計算
            obj.performance.accuracy = mean(pred == labels);
            obj.performance.confusionMat = confusionmat(labels, pred);
            obj.performance.classLabels = unique(labels);
            
            % 詳細な性能指標の計算
            if length(obj.performance.classLabels) == 2
                [obj.performance.roc.X, obj.performance.roc.Y, ~, obj.performance.auc] = ...
                    perfcurve(labels, score(:,2), obj.performance.classLabels(2));
            end
            
            % 各クラスの精度指標
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
            
            % 特徴重要度の計算
            if strcmp(obj.modelType, 'svm')
                obj.performance.featureImportance = abs(obj.svmModel.Beta);
            else
                obj.performance.featureImportance = mean(abs(obj.svmModel.BinaryLearners{1}.Beta));
            end
        end
        
        function [label, score] = predictSingle(obj, feature)
            % 単一の特徴量に対する予測
            if strcmp(obj.modelType, 'svm')
                % SVMモデルの場合
                [label, score_mat] = predict(obj.svmModel, feature);
                if obj.params.classifier.svm.probability
                    % 確率値の計算が有効な場合
                    score = max(score_mat, [], 2);
                else
                    score = [];
                end
            else
                % ECOCモデルの場合
                [label, neg_loss] = predict(obj.svmModel, feature);
                if obj.params.classifier.svm.probability
                    % 確率値の計算が有効な場合
                    score = max(neg_loss, [], 2);
                else
                    score = [];
                end
            end
        end
        
        function bestThreshold = findOptimalThreshold(obj, features, labels)
            % 閾値探索範囲をパラメータから取得
            thresholds = obj.params.classifier.svm.threshold.range;

            cv = cvpartition(labels, 'KFold', obj.params.classifier.evaluation.kfold);
            bestAccuracy = 0;
            bestThreshold = obj.params.classifier.svm.threshold.rest;  % デフォルト値

            for t = thresholds
                accuracies = zeros(cv.NumTestSets, 1);

                for i = 1:cv.NumTestSets
                    trainIdx = cv.training(i);
                    testIdx = cv.test(i);

                    % 現在のモデルを使用して予測
                    [~, scores] = predict(obj.svmModel, features(testIdx,:));

                    % 閾値を適用した予測
                    if strcmp(obj.params.classifier.svm.type, 'ecoc')
                        predictions = (scores(:,1) >= t);  % 安静状態の判定
                        for j = 1:length(predictions)
                            if ~predictions(j)  % 安静状態でない場合
                                [~, maxIdx] = max(scores(j,2:end));
                                predictions(j) = maxIdx + 1;
                            end
                        end
                    else
                        predictions = (scores(:,2) >= t);  % 2クラスSVMの場合
                    end

                    % 精度の計算
                    accuracies(i) = sum(predictions == (labels(testIdx) == 1)) / length(testIdx);
                end

                % 平均精度が改善された場合に更新
                meanAccuracy = mean(accuracies);
                if meanAccuracy > bestAccuracy
                    bestAccuracy = meanAccuracy;
                    bestThreshold = t;
                end
            end

            fprintf('Best threshold: %.3f (Accuracy: %.2f%%)\n', bestThreshold, bestAccuracy * 100);
        end
        
        function metrics = evaluateThreshold(obj, features, labels, threshold)
            % k分割交差検証による評価
            cv = cvpartition(labels, 'KFold', obj.params.classifier.evaluation.kfold);

            accuracies = zeros(cv.NumTestSets, 1);
            precisions = zeros(cv.NumTestSets, 1);
            recalls = zeros(cv.NumTestSets, 1);

            for i = 1:cv.NumTestSets
                trainIdx = cv.training(i);
                testIdx = cv.test(i);

                % モデルの学習
                if strcmp(obj.params.classifier.svm.type, 'ecoc')
                    template = templateSVM('KernelFunction', obj.params.classifier.svm.kernel);
                    model = fitcecoc(features(trainIdx,:), labels(trainIdx), 'Learners', template);
                else
                    model = fitcsvm(features(trainIdx,:), labels(trainIdx), ...
                        'KernelFunction', obj.params.classifier.svm.kernel, ...
                        'Standardize', true);
                end

                % 予測とメトリクスの計算
                [~, scores] = predict(model, features(testIdx,:));
                predictedLabels = obj.applyThreshold(scores, threshold);

                % 性能指標の計算
                accuracies(i) = sum(predictedLabels == labels(testIdx)) / length(labels(testIdx));
                precisions(i) = sum(predictedLabels == 1 & labels(testIdx) == 1) / sum(predictedLabels == 1);
                recalls(i) = sum(predictedLabels == 1 & labels(testIdx) == 1) / sum(labels(testIdx) == 1);
            end

            % 平均性能指標を返す
            metrics = struct(...
                'accuracy', mean(accuracies), ...
                'precision', mean(precisions), ...
                'recall', mean(recalls), ...
                'f1score', 2 * mean(precisions) * mean(recalls) / (mean(precisions) + mean(recalls)));
        end

        function predictedLabels = applyThreshold(obj, scores, threshold)
            if strcmp(obj.params.classifier.svm.type, 'ecoc')
                % ECOCの場合
                predictedLabels = zeros(size(scores,1), 1);
                for i = 1:size(scores,1)
                    if scores(i,1) >= threshold
                        predictedLabels(i) = 1;  % 安静状態
                    else
                        [~, maxIdx] = max(scores(i,2:end));
                        predictedLabels(i) = maxIdx + 1;
                    end
                end
            else
                % 2クラスSVMの場合
                predictedLabels = (scores(:,2) >= threshold) + 1;
            end
        end

        function displayResults(obj)
            % 結果の表示
            fprintf('\n=== Classification Results ===\n');
            fprintf('Model Type: %s\n', obj.modelType);
            fprintf('Overall Accuracy: %.2f%%\n', obj.performance.accuracy * 100);
            
            if obj.params.classifier.evaluation.enable
                fprintf('Cross-validation Accuracy: %.2f%% (±%.2f%%)\n', ...
                    obj.performance.cvMeanAccuracy * 100, ...
                    obj.performance.cvStdAccuracy * 100);
            end
            
            if length(obj.performance.classLabels) == 2
                fprintf('AUC: %.3f\n', obj.performance.auc);
            end
            
            fprintf('\nClass-wise Performance:\n');
            for i = 1:length(obj.performance.classLabels)
                fprintf('Class %d:\n', obj.performance.classLabels(i));
                fprintf('  Precision: %.2f%%\n', obj.performance.classwise(i).precision * 100);
                fprintf('  Recall: %.2f%%\n', obj.performance.classwise(i).recall * 100);
                fprintf('  F1-Score: %.2f%%\n', obj.performance.classwise(i).f1score * 100);
            end
            
            fprintf('\nConfusion Matrix:\n');
            disp(obj.performance.confusionMat);
        end
    end
end