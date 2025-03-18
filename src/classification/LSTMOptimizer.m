classdef LSTMOptimizer < handle
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
        function obj = LSTMOptimizer(params)
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            obj.useGPU = params.classifier.lstm.gpu;
            obj.maxTrials = params.classifier.lstm.optimization.maxTrials;
        end
        
        function results = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.lstm.optimize
                    fprintf('LSTM最適化は無効です。デフォルトパラメータを使用します。\n');
                    results = struct(...
                        'params', [], ...
                        'model', [], ...
                        'performance', [], ...
                        'trainInfo', [], ...
                        'overfitting', [], ...
                        'normParams', [] ...
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
                        localParams = obj.updateLSTMParameters(localParams, paramSets(i,:));

                        % LSTMの学習と評価
                        lstm = LSTMClassifier(localParams);
                        trainResults = lstm.trainLSTM(data, labels);

                        % 結果の保存
                        trialResults{i} = struct(...
                            'params', paramSets(i,:), ...
                            'model', trainResults.model, ...
                            'performance', trainResults.performance, ...
                            'trainInfo', trainResults.trainInfo, ...
                            'overfitting', trainResults.overfitting, ...
                            'normParams', trainResults.normParams ...
                        );

                        fprintf('組み合わせ %d/%d: 精度 = %.4f\n', i, size(paramSets, 1), ...
                            trainResults.performance.accuracy);

                        if obj.useGPU
                            reset(gpuDevice);
                        end

                    catch ME
                        warning('組み合わせ%dでエラー発生: %s', i, ME.message);
                        trialResults{i} = [];

                        if obj.useGPU
                            reset(gpuDevice);
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
                    'overfitting', bestResults.overfitting, ...
                    'normParams', bestResults.normParams ...
                );

                obj.updateOptimizationHistory(trialResults);
                obj.displayOptimizationSummary(summary);

            catch ME
                error('LSTM最適化に失敗: %s\n%s', ME.message, getReport(ME, 'extended'));
            end
        end
    end
    
    methods (Access = private)
        function initializeSearchSpace(obj)
            % LSTM の探索空間設定
            if isfield(obj.params.classifier.lstm.optimization, 'searchSpace')
                obj.searchSpace = obj.params.classifier.lstm.optimization.searchSpace;
            else
                obj.searchSpace = struct(...
                    'learningRate', [0.0001, 0.01], ...    % 学習率範囲
                    'miniBatchSize', [16, 128], ...         % バッチサイズ範囲
                    'lstmUnits', [32, 256], ...             % LSTM ユニット数範囲
                    'numLayers', [1, 3], ...                % LSTM 層数範囲
                    'dropoutRate', [0.2, 0.7], ...          % ドロップアウト率範囲
                    'fcUnits', [64, 256] ...                % 全結合層ユニット数範囲
                );
            end
            
            fprintf('\n探索空間の範囲 (LSTM):\n');
            fprintf('  学習率: [%.6f, %.6f]\n', obj.searchSpace.learningRate);
            fprintf('  バッチサイズ: [%d, %d]\n', obj.searchSpace.miniBatchSize);
            fprintf('  LSTMユニット数: [%d, %d]\n', obj.searchSpace.lstmUnits);
            fprintf('  LSTM層数: [%d, %d]\n', obj.searchSpace.numLayers);
            fprintf('  ドロップアウト率: [%.2f, %.2f]\n', obj.searchSpace.dropoutRate);
            fprintf('  全結合層ユニット数: [%d, %d]\n', obj.searchSpace.fcUnits);
        end
        
        function paramSets = generateParameterSets(obj, numTrials)
            % 6個のパラメータを Latin Hypercube Sampling で生成
            lhsPoints = lhsdesign(numTrials, 6);
            paramSets = zeros(numTrials, 6);
            
            % 1. 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + (log10(lr_range(2))-log10(lr_range(1))) * lhsPoints(:,1));
            
            % 2. ミニバッチサイズ
            bs_range = obj.searchSpace.miniBatchSize;
            paramSets(:,2) = round(bs_range(1) + (bs_range(2)-bs_range(1)) * lhsPoints(:,2));
            
            % 3. LSTMユニット数
            lu_range = obj.searchSpace.lstmUnits;
            paramSets(:,3) = round(lu_range(1) + (lu_range(2)-lu_range(1)) * lhsPoints(:,3));
            
            % 4. LSTM層数
            nl_range = obj.searchSpace.numLayers;
            paramSets(:,4) = round(nl_range(1) + (nl_range(2)-nl_range(1)) * lhsPoints(:,4));
            
            % 5. ドロップアウト率
            do_range = obj.searchSpace.dropoutRate;
            paramSets(:,5) = do_range(1) + (do_range(2)-do_range(1)) * lhsPoints(:,5);
            
            % 6. 全結合層ユニット数
            fc_range = obj.searchSpace.fcUnits;
            paramSets(:,6) = round(fc_range(1) + (fc_range(2)-fc_range(1)) * lhsPoints(:,6));
        end
        
        function params = updateLSTMParameters(~, params, paramSet)
            % 1. 学習率とミニバッチサイズの更新
            params.classifier.lstm.training.optimizer.learningRate = paramSet(1);
            params.classifier.lstm.training.miniBatchSize = paramSet(2);
            
            % 2. LSTM層の更新：最終層は 'last'、それ以外は 'sequence'
            lstmUnits = round(paramSet(3));
            numLayers = round(paramSet(4));
            lstmLayers = struct();
            for i = 1:numLayers
                if i == numLayers
                    lstmLayers.(sprintf('lstm%d', i)) = struct('numHiddenUnits', floor(lstmUnits/2), 'OutputMode', 'last');
                else
                    lstmLayers.(sprintf('lstm%d', i)) = struct('numHiddenUnits', lstmUnits, 'OutputMode', 'sequence');
                end
            end
            params.classifier.lstm.architecture.lstmLayers = lstmLayers;
            
            % 3. ドロップアウト層の更新：各層に同じ値を設定
            dropoutLayers = struct();
            for i = 1:numLayers
                dropoutLayers.(sprintf('dropout%d', i)) = paramSet(5);
            end
            params.classifier.lstm.architecture.dropoutLayers = dropoutLayers;
            
            % 4. 全結合層ユニット数の更新
            params.classifier.lstm.architecture.fullyConnected = [round(paramSet(6))];
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
                    'hidden_units', [], ...
                    'num_layers', [], ...
                    'dropout_rates', [], ...
                    'fc_units', [] ...
                );
                
                fprintf('\n=== 各試行の詳細評価 ===\n');
                for i = 1:numResults
                    result = validResults{i};
                    if ~isempty(result.model) && ~isempty(result.performance)
                        % 過学習の重症度により判定
                        severity = result.overfitting.severity;
                        isOverfit = ismember(severity, {'critical', 'severe', 'moderate'});
                        
                        validScores(i) = result.performance.accuracy;
                        isOverfitFlags(i) = isOverfit;
                        
                        % サマリー情報の更新
                        summary.learning_rates(end+1) = result.params(1);
                        summary.batch_sizes(end+1) = result.params(2);
                        summary.hidden_units(end+1) = result.params(3);
                        summary.num_layers(end+1) = result.params(4);
                        summary.dropout_rates(end+1) = result.params(5);
                        summary.fc_units(end+1) = result.params(6);
                        
                        if isOverfit
                            summary.overfit_models = summary.overfit_models + 1;
                        end
                        
                        % 結果の詳細表示
                        fprintf('\n--- パラメータセット %d/%d ---\n', i, numResults);
                        fprintf('性能指標:\n');
                        fprintf('  精度: %.4f\n', result.performance.accuracy);
                        fprintf('  過学習: %s\n', string(isOverfit));
                        fprintf('  重大度: %s\n', severity);
                        
                        if isfield(result, 'crossValidation')
                            fprintf('  交差検証精度: %.4f (±%.4f)\n', ...
                                result.crossValidation.accuracy, ...
                                result.crossValidation.std);
                        end
                        
                        fprintf('\nハイパーパラメータ設定:\n');
                        fprintf('  学習率: %.6f\n', result.params(1));
                        fprintf('  バッチサイズ: %d\n', result.params(2));
                        fprintf('  隠れ層ユニット数: %d\n', result.params(3));
                        fprintf('  LSTM層数: %d\n', result.params(4));
                        fprintf('  ドロップアウト率: %.2f\n', result.params(5));
                        fprintf('  全結合層ユニット数: %d\n', result.params(6));
                        
                        fprintf('\n学習統計:\n');
                        if isfield(result, 'trainInfo') && isfield(result.trainInfo, 'FinalEpoch')
                            fprintf('  最終エポック: %d\n', result.trainInfo.FinalEpoch);
                        end
                        
                        if isOverfit
                            fprintf('\n警告: このモデルは過学習の兆候を示しています\n');
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
                obj.bestPerformance = bestResults.performance.accuracy;
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
                            'performance', result.performance.accuracy, ...
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
            fprintf('\n=== 最適化プロセス ===\n');
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
            
            % 隠れ層ユニット数の統計
            fprintf('\n隠れ層ユニット数:\n');
            fprintf('  平均: %.1f\n', mean(summary.hidden_units));
            fprintf('  標準偏差: %.1f\n', std(summary.hidden_units));
            fprintf('  最小: %d\n', min(summary.hidden_units));
            fprintf('  最大: %d\n', max(summary.hidden_units));
            
            % LSTM層数の統計
            fprintf('\nLSTM層数:\n');
            fprintf('  平均: %.1f\n', mean(summary.num_layers));
            fprintf('  標準偏差: %.1f\n', std(summary.num_layers));
            fprintf('  最小: %d\n', min(summary.num_layers));
            fprintf('  最大: %d\n', max(summary.num_layers));
            
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
                         summary.hidden_units', summary.num_layers', ...
                         summary.dropout_rates', summary.fc_units'];
            paramNames = {'学習率', 'バッチサイズ', '隠れ層ユニット数', ...
                         'LSTM層数', 'ドロップアウト率', '全結合層ユニット数'};
            corrMatrix = corr(paramMatrix);
            
            % 強い相関（|r| > 0.3）のみを表示
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
            
            % LSTM特有の分析
            fprintf('\nLSTM特有の分析:\n');
            
            % 層数とユニット数の関係分析
            layerUnitCorr = corr(summary.num_layers', summary.hidden_units');
            fprintf('  層数とユニット数の相関: %.3f\n', layerUnitCorr);
            
            % 最適構成の提案
            fprintf('\n推奨構成:\n');
            % 性能上位20%のモデルを分析
            topPerformances = sort(performances, 'descend');
            topThreshold = topPerformances(ceil(length(topPerformances) * 0.2));
            topIndices = find(performances >= topThreshold);
            
            fprintf('  学習率範囲: %.6f - %.6f\n', ...
                min(summary.learning_rates(topIndices)), max(summary.learning_rates(topIndices)));
            fprintf('  推奨層数: %d - %d\n', ...
                min(summary.num_layers(topIndices)), max(summary.num_layers(topIndices)));
            fprintf('  推奨ユニット数: %d - %d\n', ...
                min(summary.hidden_units(topIndices)), max(summary.hidden_units(topIndices)));
            
            if obj.bestParams
                fprintf('\n=== 最適なパラメータ ===\n');
                fprintf('学習率: %.6f\n', obj.bestParams(1));
                fprintf('バッチサイズ: %d\n', obj.bestParams(2));
                fprintf('隠れ層ユニット数: %d\n', obj.bestParams(3));
                fprintf('LSTM層数: %d\n', obj.bestParams(4));
                fprintf('ドロップアウト率: %.2f\n', obj.bestParams(5));
                fprintf('全結合層ユニット数: %d\n', obj.bestParams(6));
                fprintf('達成精度: %.4f\n', obj.bestPerformance);
            end
        end
    end
end