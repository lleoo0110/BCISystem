classdef CNNOptimizer < handle
    properties (Access = private)
        params             % パラメータ設定
        searchSpace        % パラメータ探索空間
        bestParams        % 最良パラメータ
        bestPerformance   % 最良性能値
        optimizationLogs  % 最適化ログ
        useGPU           % GPU使用フラグ
    end
    
    methods (Access = public)
        function obj = CNNOptimizer(params)
            obj.params = params;
            obj.useGPU = params.classifier.cnn.gpu;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationLogs = struct('iteration', [], 'params', [], 'performance', []);
        end
        
                function [optimizedParams, bestPerformance, finalModel] = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.cnn.optimize
                    fprintf('CNN最適化は無効です。デフォルトパラメータを使用します。\n');
                    optimizedParams = [];
                    bestPerformance = [];
                    finalModel = [];
                    return;
                end

                fprintf('=== CNN最適化開始 ===\n');
                
                % 最適化問題の定義
                optimVars = obj.defineOptimizationVariables();
                objFcn = @(x) obj.evaluateParameters(x, data, labels);

                % ベイジアン最適化の実行
                options = obj.configureOptimization();
                
                % bayesoptの実行
                results = bayesopt(objFcn, optimVars, ...
                    'MaxObjectiveEvaluations', options.MaxObjectiveEvaluations, ...
                    'UseParallel', options.UseParallel, ...
                    'AcquisitionFunctionName', options.AcquisitionFunctionName, ...
                    'Verbose', 1);

                % 結果を構造体として取得
                optimizedParams = struct();
                bestPoint = results.XAtMinObjective;
                bestPoint = table2struct(bestPoint);
                
                % 構造体フィールドの設定
                optimizedParams.learningRate = bestPoint.learningRate;
                optimizedParams.miniBatchSize = bestPoint.miniBatchSize;
                optimizedParams.kernelSize = bestPoint.kernelSize;
                optimizedParams.numFilters = bestPoint.numFilters;
                optimizedParams.dropoutRate = bestPoint.dropoutRate;
                optimizedParams.fcUnits = bestPoint.fcUnits;
                
                bestPerformance = -results.MinObjective;

                % 最終モデルの学習
                finalParams = obj.updateParameters(optimizedParams);
                cnn = CNNClassifier(finalParams);
                finalResults = cnn.trainCNN(data, labels);
                finalModel = finalResults.model;

                % 結果の表示
                obj.displayResults(optimizedParams, bestPerformance);

                % GPUメモリのクリーンアップ
                if obj.useGPU
                    gpuDevice(1);
                end

            catch ME
                obj.handleOptimizationError(ME);
                rethrow(ME);
            end
        end

        function loss = evaluateParameters(obj, x, data, labels)
            try
                % パラメータの更新と検証
                currentParams = obj.updateParameters(x);
                
                % K分割交差検証
                numFolds = currentParams.classifier.cnn.training.validation.kfold;
                cv = cvpartition(labels, 'KFold', numFolds);
                
                performances = zeros(numFolds, 1);
                
                % 各フォールドでの評価
                parfor i = 1:numFolds
                    [trainIdx, testIdx] = obj.getValidationIndices(cv, i);
                    performance = obj.evaluateFold(currentParams, data, labels, trainIdx, testIdx);
                    performances(i) = performance;
                end
                
                % 平均性能の計算（負の値として返す - 最小化問題として解く）
                loss = -mean(performances);
                
                % 過学習チェック
                if std(performances) > 0.1
                    warning('高い性能変動を検出: std=%.3f', std(performances));
                end
                
                % ログの更新
                obj.updateOptimizationLog(x, -loss);
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
                loss = inf;
            end
        end
    end
    
    methods (Access = private)
        function initializeSearchSpace(obj)
            % テンプレートパラメータから探索空間を設定
            optParams = obj.params.classifier.cnn.optimization.searchSpace;
            
            obj.searchSpace = struct(...
                'learningRate', optParams.learningRate, ...
                'miniBatchSize', optParams.miniBatchSize, ...
                'kernelSize', optParams.kernelSize, ...
                'numFilters', optParams.numFilters, ...
                'dropoutRate', optParams.dropoutRate, ...
                'fcUnits', optParams.fcUnits);
            
            % パラメータの検証
            obj.validateSearchSpace();
        end

        function validateSearchSpace(obj)
            % 探索空間の値を検証
            fields = fieldnames(obj.searchSpace);
            for i = 1:length(fields)
                field = fields{i};
                value = obj.searchSpace.(field);
                
                if ~isempty(value) && (numel(value) ~= 2 || ~isnumeric(value))
                    error('Invalid search space for %s: Must be a numeric vector of length 2', field);
                end
                
                if value(1) >= value(2)
                    error('Invalid range for %s: First value must be less than second value', field);
                end
            end
        end

        function optimVars = defineOptimizationVariables(obj)
            % 最適化変数の定義
            optimVars = [
                optimizableVariable('learningRate', [obj.searchSpace.learningRate(1), obj.searchSpace.learningRate(2)], 'Transform', 'log')
                optimizableVariable('miniBatchSize', [obj.searchSpace.miniBatchSize(1), obj.searchSpace.miniBatchSize(2)], 'Type', 'integer')
                optimizableVariable('kernelSize', [obj.searchSpace.kernelSize(1), obj.searchSpace.kernelSize(2)], 'Type', 'integer')
                optimizableVariable('numFilters', [obj.searchSpace.numFilters(1), obj.searchSpace.numFilters(2)], 'Type', 'integer')
                optimizableVariable('dropoutRate', [obj.searchSpace.dropoutRate(1), obj.searchSpace.dropoutRate(2)])
                optimizableVariable('fcUnits', [obj.searchSpace.fcUnits(1), obj.searchSpace.fcUnits(2)], 'Type', 'integer')
            ];
        end

        function performance = evaluateFold(obj, params, data, labels, trainIdx, testIdx)
            % 単一フォールドの評価
            try
                cnn = CNNClassifier(params);
                results = cnn.trainCNN(data(:,:,trainIdx), labels(trainIdx));
                [~, performance] = cnn.predictOnline(data(:,:,testIdx), results.model);
                
                if obj.useGPU
                    gpuDevice(1);
                end
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
                performance = 0;
            end
        end

        function options = configureOptimization(obj)
            % ベイジアン最適化の設定
            options = struct();
            options.MaxObjectiveEvaluations = obj.params.classifier.cnn.optimization.searchStrategy.numIterations;
            options.UseParallel = true;
            options.AcquisitionFunctionName = 'expected-improvement-plus';
            options.GPActiveSetSize = 300;
            options.ExplorationRatio = 0.5;
            options.NumSeedPoints = 4;
        end

        function [optimizedParams, bestPerformance, finalModel] = runOptimization(obj, objFcn, optimVars, options, data, labels)
            % 最適化の実行
            results = bayesopt(objFcn, optimVars, ...
                'MaxObjectiveEvaluations', options.MaxObjectiveEvaluations, ...
                'UseParallel', options.UseParallel, ...
                'AcquisitionFunctionName', options.AcquisitionFunctionName);
            
            optimizedParams = results.XAtMinObjective;
            bestPerformance = -results.MinObjective;
            
            % 最終モデルの学習
            finalParams = obj.updateParameters(optimizedParams);
            cnn = CNNClassifier(finalParams);
            finalResults = cnn.trainCNN(data, labels);
            finalModel = finalResults.model;
        end

        function params = updateParameters(obj, x)
            % パラメータの更新をシンプル化
            params = obj.params;
            
            % 基本パラメータの更新（構造体としてアクセス）
            params.classifier.cnn.training.optimizer.learningRate = x.learningRate;
            params.classifier.cnn.training.miniBatchSize = x.miniBatchSize;
            
            % アーキテクチャパラメータの更新
            for i = 1:3
                layerName = sprintf('conv%d', i);
                params.classifier.cnn.architecture.convLayers.(layerName).filters = ...
                    x.numFilters * 2^(i-1);
                params.classifier.cnn.architecture.dropoutLayers.(['dropout' num2str(i)]) = ...
                    min(x.dropoutRate + 0.1 * (i-1), 0.7);
            end
            
            % 全結合層の更新
            for i = 1:length(params.classifier.cnn.architecture.fullyConnected)
                params.classifier.cnn.architecture.fullyConnected(i) = ...
                    max(round(x.fcUnits / 2^(i-1)), 32);
            end
        end

        % getValidationIndicesメソッドの追加
        function [trainIdx, testIdx] = getValidationIndices(~, cv, fold)
            trainIdx = cv.training(fold);
            testIdx = cv.test(fold);
        end

        function updateOptimizationLog(obj, params, performance)
            % 最適化ログの更新
            iteration = length(obj.optimizationLogs.iteration) + 1;
            obj.optimizationLogs.iteration(iteration) = iteration;
            obj.optimizationLogs.params{iteration} = params;
            obj.optimizationLogs.performance(iteration) = performance;
            
            % 最良性能の更新
            if performance > obj.bestPerformance
                obj.bestPerformance = performance;
                obj.bestParams = params;
            end
        end

        function displayResults(~, params, performance)
            % 結果の表示
            fprintf('\n=== 最適化結果 ===\n');
            fprintf('最良性能: %.4f\n\n', performance);
            
            fprintf('最適パラメータ:\n');
            paramFields = fields(params);
            for i = 1:length(paramFields)
                fprintf('%s: %g\n', paramFields{i}, params.(paramFields{i}));
            end
        end

        function handleOptimizationError(obj, ME)
            % エラーハンドリング
            fprintf('\n=== 最適化エラー ===\n');
            fprintf('エラーメッセージ: %s\n', ME.message);
            fprintf('エラー発生箇所:\n');
            for i = 1:length(ME.stack)
                fprintf('  File: %s\n  Line: %d\n  Function: %s\n\n', ...
                    ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
            end
            
            if obj.useGPU
                gpuDevice(1);
            end
        end
    end
end