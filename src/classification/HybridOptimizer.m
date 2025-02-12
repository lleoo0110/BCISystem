classdef HybridOptimizer < handle
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
        function obj = HybridOptimizer(params)
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            obj.useGPU = params.classifier.hybrid.gpu;
            obj.maxTrials = 30;  % デフォルトの試行回数
        end
        
        function [optimizedParams, performance, model] = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.hybrid.optimize
                    fprintf('ハイブリッドモデル最適化は無効です。デフォルトパラメータを使用します。\n');
                    optimizedParams = [];
                    performance = [];
                    model = [];
                    return;
                end
                
                % Latin Hypercube Sampling によりパラメータセットを生成
                paramSets = obj.generateParameterSets(obj.maxTrials);
                numParamSets = size(paramSets, 1);
                fprintf('パラメータ %d セットで最適化を開始します...\n', numParamSets);
                
                % 試行結果保存用の配列
                results = cell(numParamSets, 1);
                baseParams = obj.params;
                
                % 各パラメータセットごとに学習・評価を実施
                for i = 1:numParamSets
                    try
                        % パラメータ更新（ローカルコピー）
                        localParams = baseParams;
                        localParams = obj.updateHybridParameters(localParams, paramSets(i,:));
                        
                        % ハイブリッドモデルの学習と評価
                        hybridClassifier = HybridClassifier(localParams);
                        trainResults = hybridClassifier.trainHybrid(data, labels);
                        
                        % 結果を記録
                        results{i} = struct(...
                            'params', paramSets(i,:), ...
                            'model', trainResults.model, ...
                            'performance', trainResults.performance, ...
                            'trainInfo', trainResults.trainInfo, ...
                            'overfitting', trainResults.overfitting);
                        
                        fprintf('試行 %d/%d: 精度 = %.4f\n', i, numParamSets, ...
                            trainResults.performance.overallAccuracy);
                        
                        % GPU使用時はメモリリセット
                        if obj.useGPU
                            gpuDevice([]);
                        end
                        
                    catch ME
                        warning('試行 %d でエラー発生: %s', i, ME.message);
                        results{i} = struct('params', paramSets(i,:), 'performance', -inf);
                        
                        if obj.useGPU
                            gpuDevice([]);
                        end
                    end
                end
                
                % 結果の処理と最良パラメータ・モデルの選択
                [optimizedParams, performance, model] = obj.processFinalResults(results);
                obj.updateOptimizationHistory(results);
                obj.displayOptimizationResults();
                
            catch ME
                error('ハイブリッドモデル最適化に失敗: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function initializeSearchSpace(obj)
            obj.searchSpace = obj.params.classifier.hybrid.optimization.searchSpace;
        end
        
        function paramSets = generateParameterSets(obj, numTrials)
            % 最適化するパラメータ数: CNNとLSTMの両方のパラメータ
            numParameters = 7;  % 学習率、バッチサイズ、CNNフィルタ数、フィルタサイズ、LSTM層数、LSTMユニット数、ドロップアウト率
            lhsPoints = lhsdesign(numTrials, numParameters);
            
            % パラメータセットの初期化
            paramSets = zeros(numTrials, numParameters);
            
            % 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + ...
                (log10(lr_range(2)) - log10(lr_range(1))) * lhsPoints(:,1));
            
            % バッチサイズ
            bs_range = obj.searchSpace.miniBatchSize;
            paramSets(:,2) = round(bs_range(1) + ...
                (bs_range(2) - bs_range(1)) * lhsPoints(:,2));
            
            % CNNフィルタ数
            cf_range = obj.searchSpace.cnnFilters;
            paramSets(:,3) = round(cf_range(1) + ...
                (cf_range(2) - cf_range(1)) * lhsPoints(:,3));
            
            % フィルタサイズ
            fs_range = obj.searchSpace.filterSize;
            paramSets(:,4) = round(fs_range(1) + ...
                (fs_range(2) - fs_range(1)) * lhsPoints(:,4));
            
            % LSTM層数
            nl_range = obj.searchSpace.numLstmLayers;
            paramSets(:,5) = round(nl_range(1) + ...
                (nl_range(2) - nl_range(1)) * lhsPoints(:,5));
            
            % LSTMユニット数
            lu_range = obj.searchSpace.lstmUnits;
            paramSets(:,6) = round(lu_range(1) + ...
                (lu_range(2) - lu_range(1)) * lhsPoints(:,6));
            
            % ドロップアウト率
            do_range = obj.searchSpace.dropoutRate;
            paramSets(:,7) = do_range(1) + ...
                (do_range(2) - do_range(1)) * lhsPoints(:,7);
        end
        
        function params = updateHybridParameters(~, params, paramSet)
            % ハイブリッドモデルのパラメータを更新
            params.classifier.hybrid.training.learningRate = paramSet(1);
            params.classifier.hybrid.training.miniBatchSize = paramSet(2);
            
            % CNNパラメータの更新
            params.classifier.hybrid.cnn.numFilters = paramSet(3);
            params.classifier.hybrid.cnn.filterSize = paramSet(4);
            
            % LSTMパラメータの更新
            numLayers = paramSet(5);
            lstmUnits = paramSet(6);
            
            % LSTM層の再構築
            lstmLayers = struct();
            for i = 1:numLayers
                if i == numLayers  % 最後の層
                    lstmLayers.(['lstm' num2str(i)]) = ...
                        struct('numHiddenUnits', floor(lstmUnits/4), 'OutputMode', 'last');
                else
                    lstmLayers.(['lstm' num2str(i)]) = ...
                        struct('numHiddenUnits', lstmUnits, 'OutputMode', 'last');
                end
            end
            params.classifier.hybrid.lstm.architecture.lstmLayers = lstmLayers;
            
            % ドロップアウトレイヤーの更新
            dropoutLayers = struct();
            for i = 1:numLayers
                dropoutLayers.(['dropout' num2str(i)]) = paramSet(7);
            end
            params.classifier.hybrid.lstm.architecture.dropoutLayers = dropoutLayers;
        end
        
        function [optimizedParams, performance, model] = processFinalResults(obj, results)
            try
                fprintf('\n=== パラメータ最適化の結果処理 ===\n');
                
                % 有効な結果のみを抽出
                validResults = results(~cellfun(@isempty, results));
                numResults = length(validResults);
                validScores = zeros(1, numResults);
                isOverfitFlags = false(1, numResults);
                
                fprintf('有効なパラメータセット数: %d\n', numResults);
                
                for i = 1:numResults
                    result = validResults{i};
                    if ~isempty(result.model) && ~isempty(result.performance)
                        % 過学習の重症度により判定
                        severity = result.overfitting.severity;
                        isOverfit = ismember(severity, {'critical', 'severe', 'moderate'});
                        
                        validScores(i) = result.performance.overallAccuracy;
                        isOverfitFlags(i) = isOverfit;
                        
                        % 各セットの結果表示
                        fprintf('\nパラメータセット %d の評価結果:\n', i);
                        fprintf('  精度: %.4f\n', result.performance.overallAccuracy);
                        fprintf('  過学習: %s\n', string(isOverfit));
                        if isOverfit
                            fprintf('  重大度: %s\n', severity);
                        end
                        fprintf('  パラメータ設定:\n');
                        fprintf('    学習率: %.6f\n', result.params(1));
                        fprintf('    バッチサイズ: %d\n', result.params(2));
                        fprintf('    CNNフィルタ数: %d\n', result.params(3));
                        fprintf('    フィルタサイズ: %d\n', result.params(4));
                        fprintf('    LSTM層数: %d\n', result.params(5));
                        fprintf('    LSTMユニット数: %d\n', result.params(6));
                        fprintf('    ドロップアウト率: %.2f\n', result.params(7));
                    end
                end
                
                % 過学習していないモデルから最良のものを選択
                nonOverfitIndices = find(~isOverfitFlags);
                if isempty(nonOverfitIndices)
                    warning('過学習していないモデルが見つかりませんでした。最も高いスコアのモデルを選択します。');
                    [bestScore, bestIdx] = max(validScores);
                    fprintf('\n注意: 選択されたモデルは過学習していますが、最も良いスコア（%.4f）を持っています。\n', bestScore);
                else
                    [bestScore, bestLocalIdx] = max(validScores(nonOverfitIndices));
                    bestIdx = nonOverfitIndices(bestLocalIdx);
                    fprintf('\n過学習していないモデルから最良のものを選択しました（スコア: %.4f）\n', bestScore);
                end
                
                bestResult = validResults{bestIdx};
                optimizedParams = bestResult.params;
                performance = bestResult.performance.overallAccuracy;
                model = bestResult.model;
                obj.bestParams = optimizedParams;
                obj.bestPerformance = performance;
                obj.optimizedModel = model;
                
                fprintf('\n=== 最終選択モデル ===\n');
                fprintf('精度: %.4f\n', performance);
                fprintf('最適化されたパラメータ:\n');
                fprintf('  学習率: %.6f\n', optimizedParams(1));
                fprintf('  バッチサイズ: %d\n', optimizedParams(2));
                fprintf('  CNNフィルタ数: %d\n', optimizedParams(3));
                fprintf('  フィルタサイズ: %d\n', optimizedParams(4));
                fprintf('  LSTM層数: %d\n', optimizedParams(5));
                fprintf('  LSTMユニット数: %d\n', optimizedParams(6));
                fprintf('  ドロップアウト率: %.2f\n', optimizedParams(7));
                
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
                
                fprintf('\n最適化履歴を更新しました（計 %d 個のモデル）\n', ...
                    length(obj.optimizationHistory));
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function displayOptimizationResults(obj)
            fprintf('\n=== ハイブリッドモデル最適化結果 ===\n');
            
            if ~isempty(obj.optimizationHistory)
                performances = [obj.optimizationHistory.performance];
                validPerfs = performances(isfinite(performances));
                
                fprintf('最適化試行の統計:\n');
                fprintf('  平均精度: %.4f\n', mean(validPerfs));
                fprintf('  標準偏差: %.4f\n', std(validPerfs));
                fprintf('  最小精度: %.4f\n', min(validPerfs));
                fprintf('  最大精度: %.4f\n', max(validPerfs));
                fprintf('  有効な試行数: %d\n', length(validPerfs));
                
                fprintf('\nパラメータ分布:\n');
                paramsMat = vertcat(obj.optimizationHistory.params);
                paramNames = {'学習率', 'バッチサイズ', 'CNNフィルタ数', ...
                            'フィルタサイズ', 'LSTM層数', 'LSTMユニット数', ...
                            'ドロップアウト率'};
                
                for i = 1:size(paramsMat, 2)
                    fprintf('%s:\n', paramNames{i});
                    fprintf('  平均: %.6f\n', mean(paramsMat(:,i)));
                    fprintf('  標準偏差: %.6f\n', std(paramsMat(:,i)));
                    fprintf('  最小値: %.6f\n', min(paramsMat(:,i)));
                    fprintf('  最大値: %.6f\n', max(paramsMat(:,i)));
                    fprintf('\n');
                end
                
                % 収束性評価
                improvement = diff(performances);
                meanImprovement = mean(improvement(improvement > 0));
                fprintf('収束性評価:\n');
                fprintf('  平均改善率: %.6f\n', meanImprovement);
                fprintf('  改善回数: %d\n', sum(improvement > 0));
                if meanImprovement < 0.001
                    fprintf('  状態: 収束\n');
                else
                    fprintf('  状態: 未収束（さらなる最適化の余地あり）\n');
                end

                % パラメータ相関分析
                if size(paramsMat, 1) > 1
                    fprintf('\nパラメータ間の相関分析:\n');
                    corrMat = corr(paramsMat, 'rows', 'complete');
                    for i = 1:length(paramNames)
                        for j = i+1:length(paramNames)
                            if abs(corrMat(i,j)) > 0.5
                                fprintf('  %s と %s: %.3f\n', ...
                                    paramNames{i}, paramNames{j}, corrMat(i,j));
                            end
                        end
                    end
                end

                % 主要な影響因子の特定
                if size(paramsMat, 1) > 1
                    fprintf('\n性能への影響が大きいパラメータ:\n');
                    for i = 1:size(paramsMat, 2)
                        correlation = corr(paramsMat(:,i), performances');
                        if abs(correlation) > 0.3
                            fprintf('  %s: %.3f\n', paramNames{i}, correlation);
                        end
                    end
                end
            else
                fprintf('最適化履歴が空です\n');
            end
        end
    end
end