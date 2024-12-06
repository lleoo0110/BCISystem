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
                
                % 結果の格納
                results = struct();
                results.performance = obj.performance;  % 性能評価結果
                results.classifier = obj.svmModel;          % 学習済みSVMモデル
                
                % 結果の表示
                obj.displayResults();
                
            catch ME
                error('Offline training failed: %s', ME.message);
            end
        end
        
        function [label, score] = predictOnline(obj, features, svmclassifier)
            % オンラインモードでの予測
            if ~obj.isEnabled
                error('SVMClassifier:Disabled', 'SVM is disabled in configuration');
            end

            try
                % 入力の検証
                validateattributes(features, {'numeric'}, {'2d'}, 'predictOnline', 'features');

                % 分類器の検証と設定
                if isempty(svmclassifier)
                    error('SVMClassifier:NoModel', 'No classifier model provided');
                end
                obj.svmModel = svmclassifier;

                % 予測の実行
                if obj.params.classifier.svm.probability
                    % 確率付きの予測
                    [label, scores] = predict(obj.svmModel, features);
                    score = max(scores, [], 2);  % 最大確率を信頼度として使用
                else
                    % 通常の予測
                    [label, ~] = predict(obj.svmModel, features);
                    score = [];
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