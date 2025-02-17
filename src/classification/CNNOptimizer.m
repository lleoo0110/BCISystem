classdef CNNOptimizer < handle
    properties (Access = private)
        params              % パラメータ設定
        optimizedModel      % 最適化されたモデル
        bestParams         % 最良パラメータ
        bestPerformance    % 最良性能値
        searchSpace        % パラメータ探索空間
        optimizationHistory % 最適化履歴
        useGPU             % GPU使用の有無
        maxTrials          % 最大試行回数
    end
    
    methods (Access = public)
        function obj = CNNOptimizer(params)
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            obj.useGPU = params.classifier.cnn.gpu;
            obj.maxTrials = 30;  % デフォルトの試行回数
        end
        
        function results = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.cnn.optimize
                    fprintf('CNN最適化は無効です。デフォルトパラメータを使用します。\n');
                    results = struct(...
                        'model', [], ...
                        'performance', struct(...
                            'overallAccuracy', 0, ...
                            'crossValidation', struct('accuracy', 0, 'std', 0), ...
                            'precision', 0, ...
                            'recall', 0, ...
                            'f1score', 0, ...
                            'auc', 0, ...
                            'confusionMatrix', [] ...
                        ), ...
                        'trainInfo', [], ...
                        'overfitting', struct() ...
                    );
                    return;
                end

                % パラメータセットの生成
                fprintf('パラメータ探索を開始します...\n');
                paramSets = obj.generateParameterSets(obj.maxTrials);
                fprintf('パラメータ%dセットで最適化を開始します...\n', size(paramSets, 1));

                trialResults = cell(size(paramSets, 1), 1);
                baseParams = obj.params;

                for i = 1:size(paramSets, 1)
                    try
                        % パラメータの更新
                        localParams = baseParams;
                        localParams = obj.updateCNNParameters(localParams, paramSets(i,:));

                        % CNNの学習と評価
                        cnn = CNNClassifier(localParams);
                        trainResults = cnn.trainCNN(data, labels);

                        % 結果の保存
                        trialResults{i} = struct(...
                            'params', paramSets(i,:), ...
                            'model', trainResults.model, ...
                            'performance', trainResults.performance, ...
                            'trainInfo', trainResults.trainInfo, ...
                            'overfitting', trainResults.overfitting);

                        fprintf('組み合わせ %d/%d: 精度 = %.4f\n', i, size(paramSets, 1), ...
                            trainResults.performance.overallAccuracy);

                        if obj.useGPU
                            gpuDevice([]);
                        end

                    catch ME
                        warning('組み合わせ%dでエラー発生: %s', i, ME.message);
                        trialResults{i} = [];

                        if obj.useGPU
                            gpuDevice([]);
                        end
                    end
                end

                % 最良の結果を選択
                [bestResults, summary] = obj.processFinalResults(trialResults);
                
                % 結果構造体の作成
                results = struct(...
                    'model', bestResults.model, ...
                    'performance', bestResults.performance, ...
                    'trainInfo', bestResults.trainInfo, ...
                    'overfitting', bestResults.overfitting ...
                );

                obj.updateOptimizationHistory(trialResults);
                obj.displayOptimizationSummary(summary);

            catch ME
                error('CNN最適化に失敗: %s\n%s', ME.message, getReport(ME, 'extended'));
            end
        end
    end
    
    methods (Access = private)
        function initializeSearchSpace(obj)
            obj.searchSpace = obj.params.classifier.cnn.optimization.searchSpace;
        end
        
        function paramSets = generateParameterSets(obj, numTrials)
            % 最適化するパラメータ数 (6個)
            lhsPoints = lhsdesign(numTrials, 6);
            
            % パラメータセットの初期化
            paramSets = zeros(numTrials, 6);
            
            % 1. 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + ...
                (log10(lr_range(2)) - log10(lr_range(1))) * lhsPoints(:,1));
            
            % 2. ミニバッチサイズ
            bs_range = obj.searchSpace.miniBatchSize;
            paramSets(:,2) = round(bs_range(1) + ...
                (bs_range(2) - bs_range(1)) * lhsPoints(:,2));
            
            % 3. フィルタサイズ
            fs_range = obj.searchSpace.filterSize;
            paramSets(:,3) = round(fs_range(1) + ...
                (fs_range(2) - fs_range(1)) * lhsPoints(:,3));
            
            % 4. フィルタ数
            nf_range = obj.searchSpace.numFilters;
            paramSets(:,4) = round(nf_range(1) + ...
                (nf_range(2) - nf_range(1)) * lhsPoints(:,4));
            
            % 5. ドロップアウト率
            do_range = obj.searchSpace.dropoutRate;
            paramSets(:,5) = do_range(1) + ...
                (do_range(2) - do_range(1)) * lhsPoints(:,5);
            
            % 6. 全結合層ユニット数
            fc_range = obj.searchSpace.fcUnits;
            paramSets(:,6) = round(fc_range(1) + ...
                (fc_range(2) - fc_range(1)) * lhsPoints(:,6));
        end
        
        function params = updateCNNParameters(~, params, paramSet)
            % パラメータの更新
            params.classifier.cnn.training.optimizer.learningRate = paramSet(1);
            params.classifier.cnn.training.miniBatchSize = paramSet(2);
            
            % 畳み込み層の設定
            kernelSize = round(paramSet(3));  % カーネルサイズを整数に丸める
            params.classifier.cnn.architecture.convLayers.conv1.size = [kernelSize kernelSize];
            params.classifier.cnn.architecture.convLayers.conv1.filters = round(paramSet(4));
            
            % ドロップアウト率の設定
            params.classifier.cnn.architecture.dropoutLayers.dropout1 = paramSet(5);
            
            % 全結合層のユニット数設定
            params.classifier.cnn.architecture.fullyConnected = [round(paramSet(6))];
        end

        function [bestResults, summary] = processFinalResults(obj, results)
            try
                fprintf('\n=== パラメータ最適化の結果処理 ===\n');
                fprintf('総試行回数: %d\n', length(results));
                
                % 有効な結果のみを抽出
                validResults = results(~cellfun(@isempty, results));
                numResults = length(validResults);
                validScores = zeros(1, numResults);
                isOverfitFlags = false(1, numResults);
                
                fprintf('有効なパラメータセット数: %d\n', numResults);
                fprintf('無効な試行数: %d\n', length(results) - numResults);
                
                % 結果サマリーの初期化
                summary = struct(...
                    'total_trials', length(results), ...
                    'valid_trials', numResults, ...
                    'overfit_models', 0, ...
                    'best_accuracy', 0, ...
                    'worst_accuracy', inf, ...
                    'mean_accuracy', 0, ...
                    'learning_rates', [], ...
                    'batch_sizes', [], ...
                    'filter_sizes', [], ...
                    'num_filters', [], ...
                    'dropout_rates', [], ...
                    'fc_units', []);
                
                fprintf('\n=== 各試行の詳細評価 ===\n');
                for i = 1:numResults
                    result = validResults{i};
                    if ~isempty(result.model) && ~isempty(result.performance)
                        % 過学習の重症度により判定
                        severity = result.overfitting.severity;
                        isOverfit = ismember(severity, {'critical', 'severe', 'moderate'});
                        
                        validScores(i) = result.performance.overallAccuracy;
                        isOverfitFlags(i) = isOverfit;
                        
                        % サマリー情報の更新
                        summary.learning_rates(end+1) = result.params(1);
                        summary.batch_sizes(end+1) = result.params(2);
                        summary.filter_sizes(end+1) = result.params(3);
                        summary.num_filters(end+1) = result.params(4);
                        summary.dropout_rates(end+1) = result.params(5);
                        summary.fc_units(end+1) = result.params(6);
                        
                        if isOverfit
                            summary.overfit_models = summary.overfit_models + 1;
                        end
                        
                        % 結果の詳細表示
                        fprintf('\n--- パラメータセット %d/%d ---\n', i, numResults);
                        fprintf('性能指標:\n');
                        fprintf('  精度: %.4f\n', result.performance.overallAccuracy);
                        fprintf('  過学習: %s\n', string(isOverfit));
                        fprintf('  重大度: %s\n', severity);
                        
                        if isfield(result.performance, 'crossValidation')
                            fprintf('  交差検証精度: %.4f (±%.4f)\n', ...
                                result.performance.crossValidation.accuracy, ...
                                result.performance.crossValidation.std);
                        end
                        
                        fprintf('\nハイパーパラメータ設定:\n');
                        fprintf('  学習率: %.6f\n', result.params(1));
                        fprintf('  バッチサイズ: %d\n', result.params(2));
                        fprintf('  フィルタサイズ: %d\n', result.params(3));
                        fprintf('  フィルタ数: %d\n', result.params(4));
                        fprintf('  ドロップアウト率: %.2f\n', result.params(5));
                        fprintf('  全結合層ユニット数: %d\n', result.params(6));
                        
                        fprintf('\n学習統計:\n');
                        if isfield(result, 'trainInfo') && isfield(result.trainInfo, 'FinalEpoch')
                            fprintf('  最終エポック: %d\n', result.trainInfo.FinalEpoch);
                        end
                        
                        if isOverfit
                            fprintf('\n警告: このモデルは過学習の兆候を示しています\n');
                            fprintf('  - Generalization Gap: %.2f%%\n', ...
                                result.overfitting.generalizationGap);
                            fprintf('  - Performance Gap: %.2f%%\n', ...
                                result.overfitting.performanceGap);
                        end
                    end
                end
                
                % 統計サマリーの計算
                summary.best_accuracy = max(validScores);
                summary.worst_accuracy = min(validScores);
                summary.mean_accuracy = mean(validScores);
                
                % 過学習していないモデルから最良のものを選択
                nonOverfitIndices = find(~isOverfitFlags);
                if isempty(nonOverfitIndices)
                    warning('過学習していないモデルが見つかりませんでした。最も高いスコアのモデルを選択します。');
                    [~, bestIdx] = max(validScores);
                    fprintf('\n注意: 選択されたモデルは過学習していますが、最も良いスコア（%.4f）を持っています。\n', ...
                        validScores(bestIdx));
                else
                    [~, localBestIdx] = max(validScores(nonOverfitIndices));
                    bestIdx = nonOverfitIndices(localBestIdx);
                    fprintf('\n過学習していないモデルから最良のものを選択しました（スコア: %.4f）\n', ...
                        validScores(bestIdx));
                end
                
                bestResults = validResults{bestIdx};
                obj.bestParams = bestResults.params;
                obj.bestPerformance = bestResults.performance.overallAccuracy;
                obj.optimizedModel = bestResults.model;

            catch ME
                error('結果処理中にエラーが発生: %s', ME.message);
            end
        end

        function updateOptimizationHistory(obj, results)
            try
                validResults = results(~cellfun(@isempty, results));
                for i = 1:length(validResults)
                    result = validResults{i};
                    if ~isempty(result.model) && ~isempty(result.performance)
                        newEntry = struct(...
                            'params', result.params, ...
                            'performance', result.performance.overallAccuracy, ...
                            'model', result.model);
                        if isempty(obj.optimizationHistory)
                            obj.optimizationHistory = newEntry;
                        else
                            obj.optimizationHistory(end+1) = newEntry;
                        end
                    end
                end
                
                % 性能値で降順にソート
                [~, sortIdx] = sort([obj.optimizationHistory.performance], 'descend');
                obj.optimizationHistory = obj.optimizationHistory(sortIdx);
                
                fprintf('\n最適化履歴を更新しました（計 %d 個のモデル）\n', length(obj.optimizationHistory));
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function displayOptimizationSummary(obj, summary)
            fprintf('\n=== 最適化プロセスのサマリー ===\n');
            fprintf('試行結果:\n');
            fprintf('  総試行回数: %d\n', summary.total_trials);
            fprintf('  有効な試行: %d\n', summary.valid_trials);
            fprintf('  過学習モデル数: %d\n', summary.overfit_models);
            
            fprintf('\n精度統計:\n');
            fprintf('  最高精度: %.4f\n', summary.best_accuracy);
            fprintf('  最低精度: %.4f\n', summary.worst_accuracy);
            fprintf('  平均精度: %.4f\n', summary.mean_accuracy);
            
            fprintf('\nパラメータ分布:\n');
            
            % 学習率の統計
            fprintf('\n学習率:\n');
            fprintf('  平均: %.6f\n', mean(summary.learning_rates));
            fprintf('  標準偏差: %.6f\n', std(summary.learning_rates));
            fprintf('  最小: %.6f\n', min(summary.learning_rates));
            fprintf('  最大: %.6f\n', max(summary.learning_rates));
            
            % バッチサイズの統計
            fprintf('\nバッチサイズ:\n');
            fprintf('  平均: %.1f\n', mean(summary.batch_sizes));
            fprintf('  標準偏差: %.1f\n', std(summary.batch_sizes));
            fprintf('  最小: %d\n', min(summary.batch_sizes));
            fprintf('  最大: %d\n', max(summary.batch_sizes));
            
            % フィルタサイズの統計
            fprintf('\nフィルタサイズ:\n');
            fprintf('  平均: %.1f\n', mean(summary.filter_sizes));
            fprintf('  標準偏差: %.1f\n', std(summary.filter_sizes));
            fprintf('  最小: %d\n', min(summary.filter_sizes));
            fprintf('  最大: %d\n', max(summary.filter_sizes));
            
            % フィルタ数の統計
            fprintf('\nフィルタ数:\n');
            fprintf('  平均: %.1f\n', mean(summary.num_filters));
            fprintf('  標準偏差: %.1f\n', std(summary.num_filters));
            fprintf('  最小: %d\n', min(summary.num_filters));
            fprintf('  最大: %d\n', max(summary.num_filters));
            
            % ドロップアウト率の統計
            fprintf('\nドロップアウト率:\n');
            fprintf('  平均: %.3f\n', mean(summary.dropout_rates));
            fprintf('  標準偏差: %.3f\n', std(summary.dropout_rates));
            fprintf('  最小: %.3f\n', min(summary.dropout_rates));
            fprintf('  最大: %.3f\n', max(summary.dropout_rates));
            
            % 全結合層ユニット数の統計
            fprintf('\n全結合層ユニット数:\n');
            fprintf('  平均: %.1f\n', mean(summary.fc_units));
            fprintf('  標準偏差: %.1f\n', std(summary.fc_units));
            fprintf('  最小: %d\n', min(summary.fc_units));
            fprintf('  最大: %d\n', max(summary.fc_units));
            
            % パラメータ間の相関分析
            fprintf('\nパラメータ間の相関分析:\n');
            paramMatrix = [summary.learning_rates', summary.batch_sizes', ...
                         summary.filter_sizes', summary.num_filters', ...
                         summary.dropout_rates', summary.fc_units'];
            paramNames = {'学習率', 'バッチサイズ', 'フィルタサイズ', ...
                         'フィルタ数', 'ドロップアウト率', '全結合層ユニット数'};
            corrMatrix = corr(paramMatrix);
            
            for i = 1:size(corrMatrix, 1)
                for j = i+1:size(corrMatrix, 2)
                    if abs(corrMatrix(i,j)) > 0.3
                        fprintf('  %s と %s: %.3f\n', ...
                            paramNames{i}, paramNames{j}, corrMatrix(i,j));
                    end
                end
            end
            
            % 性能との相関
            fprintf('\n各パラメータと性能の相関:\n');
            performance = [obj.optimizationHistory.performance]';
            for i = 1:length(paramNames)
                correlation = corr(paramMatrix(:,i), performance);
                if abs(correlation) > 0.2
                    fprintf('  %s: %.3f\n', paramNames{i}, correlation);
                end
            end
            
            % 最適化の収束性評価
            performances = [obj.optimizationHistory.performance];
            improvement = diff(performances);
            meanImprovement = mean(improvement(improvement > 0));
            fprintf('\n収束性評価:\n');
            fprintf('  平均改善率: %.6f\n', meanImprovement);
            fprintf('  改善回数: %d\n', sum(improvement > 0));
            if meanImprovement < 0.001
                fprintf('  状態: 収束\n');
            else
                fprintf('  状態: 未収束（さらなる最適化の余地あり）\n');
            end
            
            if obj.bestParams
                fprintf('\n=== 最適なパラメータ ===\n');
                fprintf('学習率: %.6f\n', obj.bestParams(1));
                fprintf('バッチサイズ: %d\n', obj.bestParams(2));
                fprintf('フィルタサイズ: %d\n', obj.bestParams(3));
                fprintf('フィルタ数: %d\n', obj.bestParams(4));
                fprintf('ドロップアウト率: %.2f\n', obj.bestParams(5));
                fprintf('全結合層ユニット数: %d\n', obj.bestParams(6));
                fprintf('達成精度: %.4f\n', obj.bestPerformance);
            end
        end
    end
end