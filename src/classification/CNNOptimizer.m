classdef CNNOptimizer < handle
    properties (Access = private)
        params              % パラメータ設定
        optimizedModel      % 最適化されたモデル
        bestParams         % 最良パラメータ
        bestPerformance    % 最良性能値
        searchSpace        % パラメータ探索空間
        optimizationHistory % 最適化履歴
        overfitState      % 過学習判定
        bayesopt          % ベイジアン最適化オブジェクト
        useGPU            % GPUを使用するかどうか
    end

    methods (Access = public)
        function obj = CNNOptimizer(params)
            obj.params = params;
            obj.useGPU = params.classifier.cnn.gpu;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
        end

        function [optimizedParams, performance, model] = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.cnn.optimize
                    fprintf('CNN最適化は無効です。デフォルトパラメータを使用します。\n');
                    optimizedParams = [];
                    performance = [];
                    model = [];
                    return;
                end

                % 最適化問題の定義
                optimVars = obj.defineOptimizationVariables();

                % 目的関数の設定
                objFcn = @(x) obj.evaluateParameters(x, data, labels);

                % ベイジアン最適化の実行
                numIterations = obj.params.classifier.cnn.optimization.searchStrategy.numIterations;
                obj.bayesopt = bayesopt(objFcn, optimVars, ...
                    'AcquisitionFunctionName', 'expected-improvement-plus', ...
                    'MaxObjectiveEvaluations', numIterations, ...
                    'UseParallel', true, ...
                    'PlotFcn', {@plotObjective, @plotMinObjective}, ...
                    'OutputFcn', @obj.optimizationOutputFcn);

                % 最適なパラメータの取得
                optimizedParams = obj.bayesopt.XAtMinObjective;
                performance = -obj.bayesopt.MinObjective;

                % 最適パラメータでの最終モデルの学習
                fprintf('\n最適パラメータで最終モデルを学習中...\n');
                model = obj.trainFinalModel(optimizedParams, data, labels);

                % 結果の表示と保存
                obj.updateOptimizationHistory(optimizedParams, performance, model);
                obj.displayOptimizationResults();
                obj.plotOptimizationResults();

                % 最終的なGPUメモリのクリーンアップ
                if obj.useGPU
                    gpuDevice(1);
                end

            catch ME
                % エラー発生時のクリーンアップ
                if obj.useGPU
                    gpuDevice(1);
                end
                error('CNN最適化に失敗: %s\n%s', ME.message, getReport(ME, 'extended'));
            end
        end
    end

    methods (Access = private)
        function optimVars = defineOptimizationVariables(obj)
            optimVars = [
                optimizableVariable('learningRate', ...
                obj.searchSpace.learningRate, 'Transform', 'log')
                optimizableVariable('miniBatchSize', ...
                obj.searchSpace.miniBatchSize, 'Type', 'integer')
                optimizableVariable('numFilters', ...
                obj.searchSpace.numFilters, 'Type', 'integer')
                optimizableVariable('dropoutRate', ...
                obj.searchSpace.dropoutRate)
                optimizableVariable('fcUnits', ...
                obj.searchSpace.fcUnits, 'Type', 'integer')
                ];
        end

        function loss = evaluateParameters(obj, x, data, labels)
            try
                % パラメータの更新
                currentParams = obj.updateCNNParameters(obj.params, x);

                % クロスバリデーションの実行
                numFolds = currentParams.classifier.cnn.training.validation.kfold;
                cv = cvpartition(labels, 'KFold', numFolds);

                accuracies = zeros(numFolds, 1);

                % 各フォールドで評価
                for i = 1:numFolds
                    trainIdx = cv.training(i);
                    testIdx = cv.test(i);

                    % CNNの学習
                    cnn = CNNClassifier(currentParams);
                    results = cnn.trainCNN(data(:,:,trainIdx), labels(trainIdx));

                    % テストデータでの評価
                    [~, accuracy] = cnn.predictOnline(data(:,:,testIdx), results.model);
                    accuracies(i) = accuracy;

                    % GPUメモリのクリーンアップ
                    if obj.useGPU
                        gpuDevice(1);
                    end
                end

                % 平均精度の計算（負の値として返す - ベイジアン最適化は最小化問題として解く）
                loss = -mean(accuracies);

                % 過学習の監視
                if std(accuracies) > 0.1
                    fprintf('警告: 高い性能変動が検出されました（標準偏差: %.3f）\n', std(accuracies));
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
                loss = inf;
            end
        end

        function model = trainFinalModel(obj, bestParams, data, labels)
            try
                % 最適パラメータでの最終モデルの学習
                finalParams = obj.updateCNNParameters(obj.params, bestParams);
                cnn = CNNClassifier(finalParams);
                results = cnn.trainCNN(data, labels);
                model = results.model;

                % GPUメモリのクリーンアップ
                if obj.useGPU
                    gpuDevice(1);
                end

            catch ME
                if obj.useGPU
                    gpuDevice(1);
                end
                error('最終モデルの学習に失敗: %s', ME.message);
            end
        end

        function params = updateCNNParameters(~, params, x)
            try
                % 学習率の更新
                params.classifier.cnn.training.optimizer.learningRate = x.learningRate;

                % バッチサイズの更新
                params.classifier.cnn.training.miniBatchSize = x.miniBatchSize;

                % フィルタ数の更新
                convLayers = fieldnames(params.classifier.cnn.architecture.convLayers);
                for i = 1:length(convLayers)
                    params.classifier.cnn.architecture.convLayers.(convLayers{i}).filters = ...
                        x.numFilters * 2^(i-1);
                end

                % ドロップアウト率の更新
                dropoutLayers = fieldnames(params.classifier.cnn.architecture.dropoutLayers);
                for i = 1:length(dropoutLayers)
                    params.classifier.cnn.architecture.dropoutLayers.(dropoutLayers{i}) = ...
                        min(x.dropoutRate + 0.1 * (i-1), 0.7);
                end

                % 全結合層ユニット数の更新
                numLayers = length(params.classifier.cnn.architecture.fullyConnected);
                for i = 1:numLayers
                    params.classifier.cnn.architecture.fullyConnected(i) = ...
                        max(round(x.fcUnits / 2^(i-1)), 32);
                end

            catch ME
                error('パラメータ更新エラー: %s', ME.message);
            end
        end

        function stop = optimizationOutputFcn(obj, results, state)
            stop = false;

            if strcmp(state, 'iteration')
                fprintf('反復 %d/%d: 現在の最良性能 = %.4f\n', ...
                    results.NumObjectiveEvaluations, ...
                    obj.params.classifier.cnn.optimization.searchStrategy.numIterations, ...
                    -results.MinObjective);

                if isfield(results, 'ValidationError')
                    trainError = -results.ObjectiveTrace(end);
                    valError = -results.ValidationError(end);
                    gap = abs(trainError - valError);

                    if gap > 0.2
                        fprintf('警告: 過学習の可能性があります（ギャップ: %.2f）\n', gap);
                    end
                end
            end
        end

        function updateOptimizationHistory(obj, params, performance, model)
            entry = struct(...
                'params', params, ...
                'performance', performance, ...
                'model', model);
            obj.optimizationHistory(end+1) = entry;

            if performance > obj.bestPerformance
                obj.bestPerformance = performance;
                obj.bestParams = params;
                obj.optimizedModel = model;
            end
        end

        function displayOptimizationResults(obj)
            fprintf('\n=== CNN最適化結果 ===\n');
            fprintf('最良性能: %.4f\n\n', obj.bestPerformance);

            fprintf('最適パラメータ:\n');
            paramNames = {'learningRate', 'miniBatchSize', 'numFilters', 'dropoutRate', 'fcUnits'};
            for i = 1:length(paramNames)
                fprintf('%s: %g\n', paramNames{i}, obj.bestParams.(paramNames{i}));
            end

            performances = [obj.optimizationHistory.performance];
            fprintf('\n性能統計:\n');
            fprintf('平均: %.4f\n', mean(performances));
            fprintf('標準偏差: %.4f\n', std(performances));
            fprintf('最小値: %.4f\n', min(performances));
            fprintf('最大値: %.4f\n', max(performances));
        end

        % 探索空間の初期化
        function initializeSearchSpace(obj)
            obj.searchSpace = struct(...
                'learningRate', [0.0001, 0.001], ...     % より小さな範囲に
                'miniBatchSize', [32, 64], ...           % より小さなバッチサイズ
                'kernelSize', {[3,3], [5,5]}, ...        % 7x7は除外
                'numFilters', [32, 64, 128], ...         % フィルタ数を増加
                'dropoutRate', [0.2, 0.5], ...           % より広い範囲に
                'fcUnits', [128, 256, 512] ...           % ユニット数を増加
                );
        end

        function plotOptimizationResults(obj)
            try
                % 結果の可視化
                figure('Name', 'CNN最適化結果', 'Position', [100, 100, 1200, 800]);

                % 1. 最適化の進捗
                subplot(2,2,1);
                performances = [obj.optimizationHistory.performance];
                plot(1:length(performances), performances, '-o', 'LineWidth', 2);
                xlabel('反復回数');
                ylabel('性能');
                title('最適化の進捗');
                grid on;

                % 2. パラメータと性能の相関
                subplot(2,2,2);
                params = [obj.optimizationHistory.params];
                paramValues = struct2cell(params);
                paramMatrix = cell2mat(paramValues);
                paramNames = {'学習率', 'バッチサイズ', 'フィルタ数', ...
                    'ドロップアウト', 'FC層ユニット'};

                correlations = zeros(size(paramMatrix, 1), 1);
                for i = 1:size(paramMatrix, 1)
                    correlations(i) = corr(paramMatrix(i,:)', performances');
                end

                barh(correlations);
                set(gca, 'YTick', 1:length(paramNames));
                set(gca, 'YTickLabel', paramNames);
                xlabel('相関係数');
                title('パラメータと性能の相関');
                grid on;

                % 3. 性能分布
                subplot(2,2,3);
                histogram(performances, min(20, round(length(performances)/2)), ...
                    'Normalization', 'probability', ...
                    'FaceColor', [0.2 0.6 0.8], ...
                    'EdgeColor', 'w');
                xlabel('性能');
                ylabel('頻度');
                title('性能分布');
                grid on;

                % 4. 学習率vs性能のスキャッタープロット
                subplot(2,2,4);
                learningRates = [params.learningRate];
                scatter(log10(learningRates), performances, 50, 'filled', ...
                    'MarkerFaceAlpha', 0.6);
                xlabel('学習率（log10スケール）');
                ylabel('性能');
                title('学習率と性能の関係');
                grid on;

                % 最適値の表示
                hold on;
                bestIdx = find(performances == max(performances), 1);
                scatter(log10(learningRates(bestIdx)), performances(bestIdx), 100, 'r', ...
                    'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
                legend('試行点', '最適値');

                % レイアウトの調整
                set(gcf, 'Color', 'w');
                
                sgtitle('CNN最適化の詳細分析', 'FontSize', 14, 'FontWeight', 'bold');

                % 画像の保存（オプション）
                if isfield(obj.params.classifier.cnn, 'savePlots') && ...
                        obj.params.classifier.cnn.savePlots
                    saveas(gcf, 'cnn_optimization_results.png');
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end

        function plotLearningCurves(obj)
            try
                figure('Name', 'Learning Curves', 'Position', [100, 100, 800, 400]);

                % 学習曲線の取得
                trainingLoss = [obj.optimizationHistory.trainingLoss];
                validationLoss = [obj.optimizationHistory.validationLoss];
                epochs = 1:length(trainingLoss);

                % プロット
                plot(epochs, trainingLoss, 'b-', 'LineWidth', 2, 'DisplayName', '訓練損失');
                hold on;
                plot(epochs, validationLoss, 'r--', 'LineWidth', 2, 'DisplayName', '検証損失');

                % グラフの装飾
                xlabel('エポック');
                ylabel('損失');
                title('学習曲線');
                grid on;
                legend('show');

                % Early Stoppingポイントの表示（存在する場合）
                if isfield(obj.optimizationHistory, 'earlyStopEpoch') && ...
                        ~isempty(obj.optimizationHistory.earlyStopEpoch)
                    stopEpoch = obj.optimizationHistory.earlyStopEpoch;
                    plot(stopEpoch, validationLoss(stopEpoch), 'go', ...
                        'MarkerSize', 10, 'LineWidth', 2, ...
                        'DisplayName', 'Early Stopping Point');
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end

        function displayDetailedResults(obj)
            fprintf('\n=== 詳細な最適化結果 ===\n\n');

            % 1. 性能サマリー
            fprintf('性能サマリー:\n');
            fprintf('  最良性能: %.4f\n', obj.bestPerformance);
            fprintf('  平均性能: %.4f (±%.4f)\n\n', ...
                mean([obj.optimizationHistory.performance]), ...
                std([obj.optimizationHistory.performance]));

            % 2. 最適パラメータ
            fprintf('最適パラメータ:\n');
            params = obj.bestParams;
            paramFields = fieldnames(params);
            for i = 1:length(paramFields)
                fprintf('  %s: %g\n', paramFields{i}, params.(paramFields{i}));
            end
            fprintf('\n');

            % 3. 過学習分析
            fprintf('過学習分析:\n');
            if obj.overfitState.isOverfit
                fprintf('  状態: 過学習検出\n');
                fprintf('  重大度: %s\n', obj.overfitState.metrics.severity);
                fprintf('  汎化ギャップ: %.4f\n\n', obj.overfitState.metrics.generalizationGap);

                fprintf('推奨される対策:\n');
                fprintf('  - ドロップアウト率の増加 (現在: %.2f)\n', params.dropoutRate);
                fprintf('  - バッチサイズの調整 (現在: %d)\n', params.miniBatchSize);
                fprintf('  - L2正則化の強化\n');
                fprintf('  - データ拡張の検討\n\n');
            else
                fprintf('  状態: 正常\n\n');
            end

            % 4. 計算効率
            fprintf('計算効率:\n');
            fprintf('  総反復回数: %d\n', length(obj.optimizationHistory));
            if isfield(obj.optimizationHistory, 'computationTime')
                totalTime = sum([obj.optimizationHistory.computationTime]);
                fprintf('  総計算時間: %.2f分\n', totalTime/60);
                fprintf('  1反復あたりの平均時間: %.2f秒\n\n', ...
                    mean([obj.optimizationHistory.computationTime]));
            end

            % 5. メモリ使用状況（GPU使用時）
            if obj.useGPU
                g = gpuDevice();
                fprintf('GPUメモリ状況:\n');
                fprintf('  総メモリ: %.2f GB\n', g.TotalMemory/1e9);
                fprintf('  使用可能メモリ: %.2f GB\n', g.AvailableMemory/1e9);
                fprintf('  メモリ使用率: %.1f%%\n\n', ...
                    (1 - g.AvailableMemory/g.TotalMemory) * 100);
            end
        end
    end
end