classdef CNNOptimizer < handle
    properties (Access = private)
        params              % パラメータ設定
        optimizedModel     % 最適化されたCNNモデル
        cnnClassifier      % CNNClassifier インスタンス
        bestParams         % 最適パラメータ
        bestPerformance    % 最良の性能
        searchSpace        % パラメータ探索空間
        parallelPool       % 並列処理プール
    end
    
    methods (Access = public)
        function obj = CNNOptimizer(params)
            obj.params = params;
            obj.cnnClassifier = CNNClassifier(params);
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
        end
        
        function [optimizedParams, performance, model] = optimize(obj, data, labels)
            try
                % 並列処理プールの初期化
                if isempty(gcp('nocreate'))
                    obj.parallelPool = parpool('local');
                end
                
                % グリッドサーチのパラメータ組み合わせを生成
                paramCombinations = obj.generateParamCombinations();
                numCombinations = size(paramCombinations, 1);
                
                % 結果保存用の配列
                results = cell(numCombinations, 1);
                
                % 並列処理でパラメータ探索を実行
                parfor i = 1:numCombinations
                    results{i} = obj.evaluateParameters(data, labels, paramCombinations(i,:));
                end
                
                % 最適パラメータの特定
                [obj.bestPerformance, bestIdx] = max(cellfun(@(x) x.performance, results));
                obj.bestParams = results{bestIdx}.params;
                
                % 結果の整形
                optimizedParams = obj.bestParams;
                performance = obj.bestPerformance;
                
                % 結果の表示
                obj.displayOptimizationResults(results);
                
            catch ME
                error('CNN Parameter optimization failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function initializeSearchSpace(obj)
            % パラメータ探索空間の定義
            obj.searchSpace = struct(...
                'learningRate', [0.0001, 0.001, 0.01], ...
                'miniBatchSize', [32, 64, 128], ...
                'kernelSize', {[3,3], [5,5], [7,7]}, ...
                'numFilters', [16, 32, 64], ...
                'dropoutRate', [0.3, 0.4, 0.5], ...
                'fcUnits', [64, 128, 256]);
        end
        
        function combinations = generateParamCombinations(obj)
            % グリッドサーチ用のパラメータ組み合わせを生成
            fields = fieldnames(obj.searchSpace);
            values = struct2cell(obj.searchSpace);
            
            % 全組み合わせの生成
            [A, B, C, D, E, F] = ndgrid(1:length(values{1}), ...
                                      1:length(values{2}), ...
                                      1:length(values{3}), ...
                                      1:length(values{4}), ...
                                      1:length(values{5}), ...
                                      1:length(values{6}));
            
            combinations = [A(:), B(:), C(:), D(:), E(:), F(:)];
            
            % インデックスを実際の値に変換
            numCombinations = size(combinations, 1);
            paramMatrix = zeros(numCombinations, length(fields));
            
            for i = 1:length(fields)
                if iscell(values{i})
                    paramMatrix(:,i) = combinations(:,i);  % kernel_sizeは特別処理
                else
                    paramMatrix(:,i) = values{i}(combinations(:,i));
                end
            end
            
            combinations = paramMatrix;
        end
        
        function result = evaluateParameters(obj, data, labels, paramSet)
            try
                % パラメータの設定
                currentParams = obj.params;
                currentParams.classifier.cnn.training.optimizer.learningRate = paramSet(1);
                currentParams.classifier.cnn.training.miniBatchSize = paramSet(2);
                
                if iscell(obj.searchSpace.kernelSize)
                    kernelSize = obj.searchSpace.kernelSize{paramSet(3)};
                else
                    kernelSize = paramSet(3);
                end
                
                % CNNアーキテクチャの更新
                currentParams.classifier.cnn.architecture.convLayers.conv1.size = kernelSize;
                currentParams.classifier.cnn.architecture.convLayers.conv1.filters = paramSet(4);
                currentParams.classifier.cnn.architecture.dropoutLayers.dropout1 = paramSet(5);
                currentParams.classifier.cnn.architecture.fullyConnected = [paramSet(6)];
                
                % CNNの学習と評価
                cnn = CNNClassifier(currentParams);
                results = cnn.trainCNN(data, labels);
                
                % 結果の保存
                result = struct();
                result.params = paramSet;
                result.performance = results.performance.crossValidation.meanAccuracy;
                result.confusionMatrix = results.performance.confusionMat;
                result.detailedMetrics = struct(...
                    'precision', results.performance.precision, ...
                    'recall', results.performance.recall, ...
                    'f1score', results.performance.f1score);
                
            catch ME
                warning('Parameter evaluation failed: %s', ME.message);
                result = struct('params', paramSet, 'performance', -inf);
            end
        end
        
        function displayOptimizationResults(obj, results)
            fprintf('\n=== CNN Parameter Optimization Results ===\n');
            fprintf('Best Performance: %.4f\n\n', obj.bestPerformance);
            
            fprintf('Optimal Parameters:\n');
            fprintf('Learning Rate: %.6f\n', obj.bestParams(1));
            fprintf('Mini-batch Size: %d\n', obj.bestParams(2));
            fprintf('Kernel Size: [%d %d]\n', ...
                obj.searchSpace.kernelSize{obj.bestParams(3)}(1), ...
                obj.searchSpace.kernelSize{obj.bestParams(3)}(2));
            fprintf('Number of Filters: %d\n', obj.bestParams(4));
            fprintf('Dropout Rate: %.2f\n', obj.bestParams(5));
            fprintf('FC Units: %d\n', obj.bestParams(6));
            
            % パラメータの性能比較をプロット
            obj.plotPerformanceComparison(results);
        end
        
        function plotPerformanceComparison(obj, results)
            performances = cellfun(@(x) x.performance, results);
            
            % 性能分布のヒストグラム
            figure('Name', 'Performance Distribution');
            histogram(performances, 20);
            xlabel('Performance (Accuracy)');
            ylabel('Frequency');
            title('Distribution of Performance Across Parameter Combinations');
            
            % パラメータと性能の関係を可視化
            figure('Name', 'Parameter-Performance Relationships');
            paramNames = {'Learning Rate', 'Batch Size', 'Kernel Size', ...
                         'Num Filters', 'Dropout Rate', 'FC Units'};
            
            for i = 1:length(paramNames)
                subplot(2, 3, i);
                paramValues = cellfun(@(x) x.params(i), results);
                scatter(paramValues, performances, 'filled');
                xlabel(paramNames{i});
                ylabel('Performance');
                title(sprintf('%s vs Performance', paramNames{i}));
                grid on;
            end
        end
    end
end