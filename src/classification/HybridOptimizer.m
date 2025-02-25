classdef HybridOptimizer < handle
    properties (Access = private)
        params              % パラメータ設定
        optimizedModel      % 最適化されたモデル
        bestParams          % 最良パラメータ
        bestPerformance     % 最良性能値
        searchSpace         % パラメータ探索空間
        optimizationHistory % 最適化履歴
        useGPU              % GPU使用の有無
        maxTrials           % 最大試行回数
    end
    
    methods (Access = public)
        function obj = HybridOptimizer(params)
            % コンストラクタ - 初期化
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            obj.useGPU = params.classifier.hybrid.gpu;
            obj.maxTrials = 30;  % デフォルトの試行回数
        end
        
        function results = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.hybrid.optimize
                    fprintf('ハイブリッド最適化は無効です。デフォルトパラメータを使用します。\n');
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
                
                fprintf('パラメータ探索を開始します...\n');
                paramSets = obj.generateParameterSets(obj.maxTrials);
                fprintf('パラメータ%dセットで最適化を開始します...\n', size(paramSets, 1));
                
                trialResults = cell(size(paramSets, 1), 1);
                baseParams = obj.params;
                
                for i = 1:size(paramSets, 1)
                    try
                        % パラメータの更新
                        localParams = baseParams;
                        localParams = obj.updateHybridParameters(localParams, paramSets(i,:));
                        
                        % ハイブリッドモデルの学習と評価
                        hybridClassifier = HybridClassifier(localParams);
                        trainResults = hybridClassifier.trainHybrid(data, labels);
                        
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
                
                % 最良の結果の処理
                [bestResults, summary] = obj.processFinalResults(trialResults);
                obj.updateOptimizationHistory(trialResults);
                obj.displayOptimizationSummary(summary);
                
                % 出力構造体
                results = struct(...
                    'model', bestResults.model, ...
                    'performance', bestResults.performance, ...
                    'trainInfo', bestResults.trainInfo, ...
                    'overfitting', bestResults.overfitting);
                
            catch ME
                error('ハイブリッド最適化に失敗: %s\n%s', ME.message, getReport(ME, 'extended'));
            end
        end
    end
    
    methods (Access = private)
        function initializeSearchSpace(obj)
            if isfield(obj.params.classifier.hybrid.optimization, 'searchSpace')
                obj.searchSpace = obj.params.classifier.hybrid.optimization.searchSpace;
            else
                % デフォルトの探索空間設定（Hybridモデル用）
                obj.searchSpace = struct(...
                    'learningRate', [0.0001, 0.01], ...    % 学習率範囲
                    'miniBatchSize', [16, 128], ...         % バッチサイズ範囲
                    'numConvLayers', [1, 3], ...            % CNNの畳み込み層数範囲
                    'cnnFilters', [32, 128], ...            % CNNフィルタ数範囲
                    'filterSize', [3, 7], ...               % フィルタサイズ範囲
                    'lstmUnits', [32, 256], ...             % LSTMユニット数範囲
                    'numLstmLayers', [2, 4], ...            % LSTM層数範囲
                    'dropoutRate', [0.2, 0.7], ...          % ドロップアウト率範囲
                    'fcUnits', [64, 256] ...                % 全結合層ユニット数範囲
                );
            end
            
            fprintf('\n探索空間の範囲:\n');
            fprintf('  学習率: [%.6f, %.6f]\n', obj.searchSpace.learningRate);
            fprintf('  バッチサイズ: [%d, %d]\n', obj.searchSpace.miniBatchSize);
            fprintf('  CNN層数: [%d, %d]\n', obj.searchSpace.numConvLayers);
            fprintf('  CNNフィルタ数: [%d, %d]\n', obj.searchSpace.cnnFilters);
            fprintf('  フィルタサイズ: [%d, %d]\n', obj.searchSpace.filterSize);
            fprintf('  LSTMユニット数: [%d, %d]\n', obj.searchSpace.lstmUnits);
            fprintf('  LSTM層数: [%d, %d]\n', obj.searchSpace.numLstmLayers);
            fprintf('  ドロップアウト率: [%.2f, %.2f]\n', obj.searchSpace.dropoutRate);
            fprintf('  全結合層ユニット数: [%d, %d]\n', obj.searchSpace.fcUnits);
        end
        
        function paramSets = generateParameterSets(obj, numTrials)
            % 9個のパラメータを Latin Hypercube Sampling により生成
            lhsPoints = lhsdesign(numTrials, 9);
            paramSets = zeros(numTrials, 9);
            
            % 1. 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + (log10(lr_range(2))-log10(lr_range(1))) * lhsPoints(:,1));
            
            % 2. ミニバッチサイズ
            bs_range = obj.searchSpace.miniBatchSize;
            paramSets(:,2) = round(bs_range(1) + (bs_range(2)-bs_range(1)) * lhsPoints(:,2));
            
            % 3. CNN層数 (numConvLayers)
            ncl_range = obj.searchSpace.numConvLayers;
            paramSets(:,3) = round(ncl_range(1) + (ncl_range(2)-ncl_range(1)) * lhsPoints(:,3));
            
            % 4. CNNフィルタ数
            cf_range = obj.searchSpace.cnnFilters;
            paramSets(:,4) = round(cf_range(1) + (cf_range(2)-cf_range(1)) * lhsPoints(:,4));
            
            % 5. フィルタサイズ
            fs_range = obj.searchSpace.filterSize;
            paramSets(:,5) = round(fs_range(1) + (fs_range(2)-fs_range(1)) * lhsPoints(:,5));
            
            % 6. LSTMユニット数
            lu_range = obj.searchSpace.lstmUnits;
            paramSets(:,6) = round(lu_range(1) + (lu_range(2)-lu_range(1)) * lhsPoints(:,6));
            
            % 7. LSTM層数
            nl_range = obj.searchSpace.numLstmLayers;
            paramSets(:,7) = round(nl_range(1) + (nl_range(2)-nl_range(1)) * lhsPoints(:,7));
            
            % 8. ドロップアウト率
            do_range = obj.searchSpace.dropoutRate;
            paramSets(:,8) = do_range(1) + (do_range(2)-do_range(1)) * lhsPoints(:,8);
            
            % 9. 全結合層ユニット数
            fc_range = obj.searchSpace.fcUnits;
            paramSets(:,9) = round(fc_range(1) + (fc_range(2)-fc_range(1)) * lhsPoints(:,9));
        end
        
        function params = updateHybridParameters(~, params, paramSet)
            % 1. 学習率とミニバッチサイズ
            params.classifier.hybrid.training.optimizer.learningRate = paramSet(1);
            params.classifier.hybrid.training.miniBatchSize = paramSet(2);
            
            % 2. CNNパラメータの更新
            numConvLayers = round(paramSet(3));
            cnnFilters = round(paramSet(4));
            filterSize = round(paramSet(5));
            convLayers = struct();
            for j = 1:numConvLayers
                layerName = sprintf('conv%d', j);
                convLayers.(layerName) = struct('size', [filterSize, filterSize], 'filters', cnnFilters, 'stride', 1, 'padding', 'same');
            end
            params.classifier.hybrid.architecture.cnn.convLayers = convLayers;
            
            % 3. LSTMパラメータの更新
            lstmUnits = round(paramSet(6));
            numLstmLayers = round(paramSet(7));
            lstmLayers = struct();
            for i = 1:numLstmLayers
                if i == numLstmLayers
                    lstmLayers.(['lstm' num2str(i)]) = struct('numHiddenUnits', floor(lstmUnits/2), 'OutputMode', 'last');
                else
                    lstmLayers.(['lstm' num2str(i)]) = struct('numHiddenUnits', lstmUnits, 'OutputMode', 'sequence');
                end
            end
            params.classifier.hybrid.architecture.lstm.lstmLayers = lstmLayers;
            
            % 4. ドロップアウト率の更新（CNN, LSTM共通で設定）
            dropoutRate = paramSet(8);
            % LSTM側のドロップアウトレイヤー更新
            dropoutLayers = struct();
            for i = 1:numLstmLayers
                dropoutLayers.(['dropout' num2str(i)]) = dropoutRate;
            end
            params.classifier.hybrid.architecture.lstm.dropoutLayers = dropoutLayers;
            % CNN側のドロップアウト（存在する場合）
            if isfield(params.classifier.hybrid.architecture, 'cnnDropout')
                dropoutLayerNames = fieldnames(params.classifier.hybrid.architecture.cnnDropout);
                for i = 1:length(dropoutLayerNames)
                    layerName = dropoutLayerNames{i};
                    params.classifier.hybrid.architecture.cnnDropout.(layerName) = dropoutRate;
                end
            end
            
            % 5. 全結合層ユニット数の更新
            params.classifier.hybrid.architecture.fullyConnected = [round(paramSet(9))];
        end

        function [bestResults, summary] = processFinalResults(obj, results)
            try
                fprintf('\n=== パラメータ最適化の結果処理 ===\n');
                fprintf('総試行回数: %d\n', length(results));
                
                % 有効な結果のみ抽出
                validResults = results(~cellfun(@isempty, results));
                numResults = length(validResults);
                validScores = zeros(1, numResults);
                isOverfitFlags = false(1, numResults);
                
                fprintf('有効なパラメータセット数: %d\n', numResults);
                fprintf('無効な試行数: %d\n', length(results) - numResults);
                
                % サマリ構造体の初期化
                summary = struct(...
                    'total_trials', length(results), ...
                    'valid_trials', numResults, ...
                    'overfit_models', 0, ...
                    'best_accuracy', 0, ...
                    'worst_accuracy', inf, ...
                    'mean_accuracy', 0, ...
                    'learning_rates', [], ...
                    'batch_sizes', [], ...
                    'num_conv_layers', [], ...
                    'cnn_filters', [], ...
                    'filter_sizes', [], ...
                    'lstm_units', [], ...
                    'num_lstm_layers', [], ...
                    'dropout_rates', [], ...
                    'fc_units', []);
                
                fprintf('\n=== 各試行の詳細評価 ===\n');
                for i = 1:numResults
                    result = validResults{i};
                    if ~isempty(result.model) && ~isempty(result.performance)
                        if isfield(result.overfitting, 'severity')
                            severity = result.overfitting.severity;
                        else
                            severity = 'unknown';
                        end
                        isOverfit = ismember(severity, {'critical', 'severe', 'moderate', 'mild'});
                        validScores(i) = result.performance.overallAccuracy;
                        isOverfitFlags(i) = isOverfit;
                        
                        summary.learning_rates(end+1) = result.params(1);
                        summary.batch_sizes(end+1) = result.params(2);
                        summary.num_conv_layers(end+1) = result.params(3);
                        summary.cnn_filters(end+1) = result.params(4);
                        summary.filter_sizes(end+1) = result.params(5);
                        summary.lstm_units(end+1) = result.params(6);
                        summary.num_lstm_layers(end+1) = result.params(7);
                        summary.dropout_rates(end+1) = result.params(8);
                        summary.fc_units(end+1) = result.params(9);
                        
                        if isOverfit
                            summary.overfit_models = summary.overfit_models + 1;
                        end
                        
                        fprintf('\n--- パラメータセット %d/%d ---\n', i, numResults);
                        fprintf('性能指標:\n');
                        fprintf('  精度: %.4f\n', result.performance.overallAccuracy);
                        fprintf('  過学習: %s\n', string(isOverfit));
                        fprintf('  重大度: %s\n', severity);
                        fprintf('\nハイパーパラメータ設定:\n');
                        fprintf('  学習率: %.6f\n', result.params(1));
                        fprintf('  バッチサイズ: %d\n', result.params(2));
                        printf('  CNN層数: %d\n', result.params(3));
                        fprintf('  CNNフィルタ数: %d\n', result.params(4));
                        fprintf('  フィルタサイズ: %d\n', result.params(5));
                        fprintf('  LSTMユニット数: %d\n', result.params(6));
                        fprintf('  LSTM層数: %d\n', result.params(7));
                        fprintf('  ドロップアウト率: %.2f\n', result.params(8));
                        fprintf('  全結合層ユニット数: %d\n', result.params(9));
                        fprintf('\n学習統計:\n');
                        if isfield(result.trainInfo, 'FinalEpoch')
                            fprintf('  最終エポック: %d\n', result.trainInfo.FinalEpoch);
                        end
                        if isOverfit
                            fprintf('\n警告: このモデルは過学習の兆候を示しています\n');
                            if isfield(result.overfitting, 'generalizationGap')
                                fprintf('  - Generalization Gap: %.2f%%\n', result.overfitting.generalizationGap);
                            end
                            if isfield(result.overfitting, 'performanceGap')
                                fprintf('  - Performance Gap: %.2f%%\n', result.overfitting.performanceGap);
                            end
                        end
                    end
                end
                
                summary.best_accuracy = max(validScores);
                summary.worst_accuracy = min(validScores);
                summary.mean_accuracy = mean(validScores);
                
                nonOverfitIndices = find(~isOverfitFlags);
                if isempty(nonOverfitIndices)
                    warning('過学習していないモデルが見つかりませんでした。最も高いスコアのモデルを選択します。');
                    [~, bestIdx] = max(validScores);
                    fprintf('\n注意: 選択されたモデルは過学習していますが、最も良いスコア（%.4f）を持っています。\n', validScores(bestIdx));
                else
                    [~, localBestIdx] = max(validScores(nonOverfitIndices));
                    bestIdx = nonOverfitIndices(localBestIdx);
                    fprintf('\n過学習していないモデルから最良のものを選択しました（スコア: %.4f）\n', validScores(bestIdx));
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
                validResults = validResults(~cellfun(@(x) isempty(x.model), validResults));
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

            % CNNの層数の統計
            fprintf('\nCNN層数:\n');
            fprintf('  平均: %.1f\n', mean(summary.num_conv_layers));
            fprintf('  標準偏差: %.1f\n', std(summary.num_conv_layers));
            fprintf('  最小: %d\n', min(summary.num_conv_layers));
            fprintf('  最大: %d\n', max(summary.num_conv_layers));
            
            % CNNフィルタ数の統計
            fprintf('\nCNNフィルタ数:\n');
            fprintf('  平均: %.1f\n', mean(summary.cnn_filters));
            fprintf('  標準偏差: %.1f\n', std(summary.cnn_filters));
            fprintf('  最小: %d\n', min(summary.cnn_filters));
            fprintf('  最大: %d\n', max(summary.cnn_filters));
            
            % フィルタサイズの統計
            fprintf('\nフィルタサイズ:\n');
            fprintf('  平均: %.1f\n', mean(summary.filter_sizes));
            fprintf('  標準偏差: %.1f\n', std(summary.filter_sizes));
            fprintf('  最小: %d\n', min(summary.filter_sizes));
            fprintf('  最大: %d\n', max(summary.filter_sizes));
            
            % LSTMユニット数の統計
            fprintf('\nLSTMユニット数:\n');
            fprintf('  平均: %.1f\n', mean(summary.lstm_units));
            fprintf('  標準偏差: %.1f\n', std(summary.lstm_units));
            fprintf('  最小: %d\n', min(summary.lstm_units));
            fprintf('  最大: %d\n', max(summary.lstm_units));
            
            % LSTM層数の統計
            fprintf('\nLSTM層数:\n');
            fprintf('  平均: %.1f\n', mean(summary.num_lstm_layers));
            fprintf('  標準偏差: %.1f\n', std(summary.num_lstm_layers));
            fprintf('  最小: %d\n', min(summary.num_lstm_layers));
            fprintf('  最大: %d\n', max(summary.num_lstm_layers));
            
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
                           summary.cnn_filters', summary.filter_sizes', ...
                           summary.lstm_units', summary.num_lstm_layers', ...
                           summary.dropout_rates', summary.fc_units'];
            paramNames = {'学習率', 'バッチサイズ', 'CNNフィルタ数', 'フィルタサイズ', 'LSTMユニット数', 'LSTM層数', 'ドロップアウト率', '全結合層ユニット数'};
            corrMatrix = corr(paramMatrix);
            for i = 1:size(corrMatrix, 1)
                for j = i+1:size(corrMatrix, 2)
                    if abs(corrMatrix(i,j)) > 0.3
                        fprintf('  %s と %s: %.3f\n', paramNames{i}, paramNames{j}, corrMatrix(i,j));
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
                fprintf('CNN層数: %d\n', obj.bestParams(3));
                fprintf('CNNフィルタ数: %d\n', obj.bestParams(4));
                fprintf('フィルタサイズ: %d\n', obj.bestParams(5));
                fprintf('LSTMユニット数: %d\n', obj.bestParams(6));
                fprintf('LSTM層数: %d\n', obj.bestParams(7));
                fprintf('ドロップアウト率: %.2f\n', obj.bestParams(8));
                fprintf('全結合層ユニット数: %d\n', obj.bestParams(9));
                fprintf('達成精度: %.4f\n', obj.bestPerformance);
            end
        end
    end
end