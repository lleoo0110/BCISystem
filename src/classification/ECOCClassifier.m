classdef ECOCClassifier < handle
    properties (Access = private)
        params
        isOptimized
        isEnabled
        trainResults
    end
    
    properties (Access = public)
        ecocModel
        performance
    end
    
    methods (Access = public)
        function obj = ECOCClassifier(params)
            obj.params = params;
            obj.isOptimized = params.classifier.svm.optimize;
            obj.isEnabled = params.classifier.ecoc.enable;
            obj.performance = struct();
        end
        
        function results = trainECOC(obj, features, labels)
            if ~obj.isEnabled
                error('ECOC is disabled in configuration');
            end

            try
                obj.trainModel(features, labels);
                if obj.params.classifier.evaluation.enable
                    obj.evaluatePerformance(features, labels);
                end

                results = struct('performance', obj.performance, 'model', obj.ecocModel);
                obj.displayResults();

            catch ME
                error('ECOC training failed: %s', ME.message);
            end
        end

        function [label, score] = predictOnline(obj, features, model)
            if ~obj.isEnabled
                error('ECOC is disabled');
            end

            try
                [label, scores] = predict(model, features);
                score = scores(:,1);  % クラス1（安静）の確率を返す

            catch ME
                error('ECOC prediction failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function trainModel(obj, features, labels)
            template = templateSVM(...
                'KernelFunction', obj.params.classifier.svm.kernel);

            if obj.isOptimized
                % ハイパーパラメータの最適化
                boxConstraints = obj.params.classifier.svm.hyperparameters.boxConstraint;
                kernelScales = obj.params.classifier.svm.hyperparameters.kernelScale;
                
                bestScore = -inf;
                bestParams = struct();
                
                for c = boxConstraints
                    for k = kernelScales
                        template = templateSVM(...
                            'KernelFunction', obj.params.classifier.svm.kernel, ...
                            'BoxConstraint', c, ...
                            'KernelScale', k);
                        
                        mdl = fitcecoc(features, labels, 'Learners', template);
                        cv = crossval(mdl, 'KFold', obj.params.classifier.evaluation.kfold);
                        score = 1 - kfoldLoss(cv);
                        
                        if score > bestScore
                            bestScore = score;
                            bestParams.BoxConstraint = c;
                            bestParams.KernelScale = k;
                        end
                    end
                end
                
                template = templateSVM(...
                    'KernelFunction', obj.params.classifier.svm.kernel, ...
                    'BoxConstraint', bestParams.BoxConstraint, ...
                    'KernelScale', bestParams.KernelScale);
            end
            
            obj.ecocModel = fitcecoc(features, labels, 'Learners', template);
        end
        
        function evaluatePerformance(obj, features, labels)
            cv = cvpartition(labels, 'KFold', obj.params.classifier.evaluation.kfold);
            cvAcc = zeros(cv.NumTestSets, 1);

            for i = 1:cv.NumTestSets
                trainIdx = cv.training(i);
                testIdx = cv.test(i);
                
                mdl = fitcecoc(features(trainIdx,:), labels(trainIdx), ...
                    'Learners', templateSVM('KernelFunction', obj.params.classifier.svm.kernel));
                
                [pred, ~] = predict(mdl, features(testIdx,:));
                cvAcc(i) = mean(pred == labels(testIdx));
            end

            obj.performance.cvMeanAccuracy = mean(cvAcc);
            obj.performance.cvStdAccuracy = std(cvAcc);
            
            [pred, score] = predict(obj.ecocModel, features);
            obj.performance.accuracy = mean(pred == labels);
            obj.performance.confusionMat = confusionmat(labels, pred);
            obj.performance.classLabels = unique(labels);
            
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
            fprintf('\n=== ECOC Classification Results ===\n');
            fprintf('Overall Accuracy: %.2f%%\n', obj.performance.accuracy * 100);
            
            if obj.params.classifier.evaluation.enable
                fprintf('Cross-validation Accuracy: %.2f%% (±%.2f%%)\n', ...
                    obj.performance.cvMeanAccuracy * 100, ...
                    obj.performance.cvStdAccuracy * 100);
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