classdef CNNOptimizer < handle
    properties (Access = private)
        params              % パラメータ設定
        optimizedModel      % 最適化されたモデル
        bestParams         % 最良パラメータ
        bestPerformance    % 最良性能値
        searchSpace        % パラメータ探索空間
        optimizationHistory % 最適化履歴
        useGPU            % GPUを使用するかどうか
    end
    
    methods (Access = public)
        function obj = CNNOptimizer(params)
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            obj.useGPU = params.classifier.cnn.gpu;
        end
        
        function [optimizedParams, performance, model] = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.cnn.optimize
                    fprintf('CNN最適化は無効です。デフォルトパラメータを使用します。\n');
                    optimizedParams = []; performance = []; model = [];
                    return;
                end

                % パラメータセットの生成
                numSamples = 30;
                paramSets = obj.generateParameterSets(numSamples);
                fprintf('パラメータ%dセットで最適化を開始します...\n', size(paramSets, 1));

                results = cell(size(paramSets, 1), 1);
                baseParams = obj.params;

                % 検索空間のローカルコピーを作成
                kernelSizeLocal = obj.searchSpace.kernelSize;

                parfor i = 1:size(paramSets, 1)
                    try
                        % パラメータの更新
                        localParams = baseParams;
                        localParams = obj.updateCNNParameters(localParams, paramSets(i,:), kernelSizeLocal);

                        % CNNの学習と評価
                        cnn = CNNClassifier(localParams);
                        trainResults = cnn.trainCNN(data, labels);

                        % 結果の保存
                        results{i} = struct(...
                            'params', paramSets(i,:), ...
                            'performance', trainResults.performance.accuracy, ...
                            'model', trainResults.model ...
                        );

                        fprintf('組み合わせ %d/%d: 精度 = %.4f\n', i, size(paramSets, 1), trainResults.performance.accuracy);
                    catch ME
                        warning('組み合わせ%dでエラー発生: %s', i, ME.message);
                        results{i} = struct('params', paramSets(i,:), 'performance', -inf, 'model', []);

                        % GPUメモリを解放
                        if obj.useGPU
                            reset(gpuDevice()); % GPUメモリのリセット
                        end
                    end
                end

                [optimizedParams, performance, model] = obj.processFinalResults(results);
                obj.updateOptimizationHistory(results);
                obj.displayOptimizationResults();

            catch ME
                error('CNN最適化に失敗: %s\n%s', ME.message, getReport(ME, 'extended'));
            end
        end
    end
    
    methods (Access = private)
        function initializeSearchSpace(obj)
            obj.searchSpace = obj.params.classifier.cnn.optimization.searchSpace;
        end
        
        function paramSets = generateParameterSets(obj, numSamples)
            % Latin Hypercube Sampling
            lhsPoints = lhsdesign(numSamples, 6);

            % パラメータ空間の初期化
            paramSets = zeros(numSamples, 6);

            % 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + (log10(lr_range(2)) - log10(lr_range(1))) * lhsPoints(:,1));

            % バッチサイズ
            bs_range = obj.searchSpace.miniBatchSize;
            paramSets(:,2) = round(bs_range(1) + (bs_range(2) - bs_range(1)) * lhsPoints(:,2));

            % カーネルサイズインデックス（kernelSizeはセル配列）
            num_kernel_sizes = numel(obj.searchSpace.kernelSize);
            paramSets(:,3) = ones(numSamples, 1);  % デフォルト値として1を設定
            if num_kernel_sizes > 1
                paramSets(:,3) = round(1 + (num_kernel_sizes - 1) * lhsPoints(:,3));
            end

            % 残りのパラメータ
            nf_range = obj.searchSpace.numFilters;
            paramSets(:,4) = round(nf_range(1) + (nf_range(2) - nf_range(1)) * lhsPoints(:,4));

            do_range = obj.searchSpace.dropoutRate;
            paramSets(:,5) = do_range(1) + (do_range(2) - do_range(1)) * lhsPoints(:,5);

            fc_range = obj.searchSpace.fcUnits;
            paramSets(:,6) = round(fc_range(1) + (fc_range(2) - fc_range(1)) * lhsPoints(:,6));
        end

        function params = updateCNNParameters(~, params, paramSet, ~)
            % カーネルサイズを固定値に設定
            kernelSize = [3 3]; % デフォルトのカーネルサイズを使用

            params.classifier.cnn.training.optimizer.learningRate = paramSet(1);
            params.classifier.cnn.training.miniBatchSize = paramSet(2);
            params.classifier.cnn.architecture.convLayers.conv1.size = kernelSize;
            params.classifier.cnn.architecture.convLayers.conv1.filters = paramSet(4);
            params.classifier.cnn.architecture.dropoutLayers.dropout1 = paramSet(5);
            params.classifier.cnn.architecture.fullyConnected = [paramSet(6)];
        end

        function metrics = calculatePerformanceMetrics(~, results)
            metrics = struct();
            
            if isfield(results, 'performance') && isfield(results.performance, 'cvAccuracies')
                metrics.avgAccuracy = mean(results.performance.cvAccuracies);
                metrics.stdAccuracy = std(results.performance.cvAccuracies);
            else
                metrics.avgAccuracy = results.performance.accuracy;
                metrics.stdAccuracy = 0;
            end
            
            if isfield(results.performance, 'auc')
                metrics.auc = results.performance.auc;
            end
            
            if isfield(results.performance, 'f1score')
                metrics.f1score = results.performance.f1score;
            end
        end

        function [optimizedParams, performance, model] = processFinalResults(obj, results)
            validResults = ~cellfun(@isempty, results);
            performances = cellfun(@(x) x.performance, results(validResults));
            [obj.bestPerformance, bestIdx] = max(performances);
            
            validIndices = find(validResults);
            bestResult = results{validIndices(bestIdx)};
            
            optimizedParams = bestResult.params;
            performance = bestResult.performance;
            model = bestResult.model;
            obj.bestParams = optimizedParams;
            obj.optimizedModel = model;
        end

        function updateOptimizationHistory(obj, results)
            for i = 1:length(results)
                if ~isempty(results{i})
                    obj.optimizationHistory(end+1) = struct(...
                        'params', results{i}.params, ...
                        'performance', results{i}.performance, ...
                        'model', results{i}.model);
                end
            end
        end

        function displayOptimizationResults(obj)
            fprintf('\n=== CNN最適化結果 ===\n');
            fprintf('最良性能: %.4f\n\n', obj.bestPerformance);

            fprintf('最適パラメータ:\n');
            fprintf('学習率: %.6f\n', obj.bestParams(1));
            fprintf('ミニバッチサイズ: %d\n', obj.bestParams(2));
            fprintf('フィルタ数: %d\n', obj.bestParams(4));
            fprintf('ドロップアウト率: %.2f\n', obj.bestParams(5));
            fprintf('全結合層ユニット数: %d\n', obj.bestParams(6));

            % 性能統計の表示
            performances = [obj.optimizationHistory.performance];
            validPerfs = performances(isfinite(performances));

            fprintf('\n性能統計:\n');
            fprintf('平均: %.4f\n', mean(validPerfs));
            fprintf('標準偏差: %.4f\n', std(validPerfs));
            fprintf('最小値: %.4f\n', min(validPerfs));
            fprintf('最大値: %.4f\n', max(validPerfs));
        end
    end
end