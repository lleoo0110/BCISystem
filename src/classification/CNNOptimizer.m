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
                numSamples = 20;
                paramSets = obj.generateParameterSets(numSamples);
                fprintf('パラメータ%dセットで最適化を開始します...\n', size(paramSets, 1));

                results = cell(size(paramSets, 1), 1);
                baseParams = obj.params;

                % 検索空間のローカルコピーを作成
                kernelSizeLocal = obj.searchSpace.kernelSize;

                for i = 1:size(paramSets, 1)
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

                       % GPUメモリを解放
                       if obj.useGPU
                           gpuDevice([]);
                       end

                    catch ME
                        warning('組み合わせ%dでエラー発生: %s', i, ME.message);
                        results{i} = struct('params', paramSets(i,:), 'performance', -inf, 'model', []);

                        % GPUメモリを解放
                       if obj.useGPU
                           gpuDevice([]);
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
            try
                fprintf('\n=== パラメータ最適化の結果処理 ===\n');
                
                % 有効な結果のみを抽出
                validResults = ~cellfun(@isempty, results);
                validResults = results(validResults);
                
                % CNNClassifierのインスタンス作成（validateOverfitting用）
                cnnClassifier = CNNClassifier(obj.params);
                
                % 各結果の評価
                numResults = length(validResults);
                validScores = zeros(1, numResults);
                isOverfitFlags = false(1, numResults);
                
                fprintf('有効なパラメータセット数: %d\n', numResults);
                
                for i = 1:numResults
                    result = validResults{i};
                    
                    if ~isempty(result.model) && ~isempty(result.performance)
                        % CNNClassifierのvalidateOverfittingを使用して過学習を評価
                        [isOverfit, overfitMetrics] = cnnClassifier.validateOverfitting(...
                            result.trainInfo, result.performance);
                        
                        % スコアの計算
                        baseScore = result.performance;
                        
                        % 過学習の場合のスコア調整
                        if isOverfit
                            % 汎化ギャップ（Generalization Gap）に基づくスコア調整
                            genGap = overfitMetrics.generalizationGap;
                            perfGap = overfitMetrics.performanceGap;
                            
                            % 実際の性能低下を反映したスコア調整
                            baseScore = baseScore * (1 - max(genGap, perfGap));
                        end
                        
                        validScores(i) = baseScore;
                        isOverfitFlags(i) = isOverfit;
                        
                        % 結果の表示
                        fprintf('\nパラメータセット %d の評価結果:\n', i);
                        fprintf('  基本精度: %.4f\n', result.performance);
                        fprintf('  調整後スコア: %.4f\n', baseScore);
                        fprintf('  過学習: %s\n', string(isOverfit));
                        if isOverfit
                            fprintf('  Generalization Gap: %.4f\n', overfitMetrics.generalizationGap);
                            fprintf('  Performance Gap: %.4f\n', overfitMetrics.performanceGap);
                            fprintf('  重大度: %s\n', overfitMetrics.severity);
                        end
                        
                        % パラメータ値の表示
                        fprintf('  パラメータ設定:\n');
                        fprintf('    学習率: %.6f\n', result.params(1));
                        fprintf('    バッチサイズ: %d\n', result.params(2));
                        fprintf('    カーネルサイズインデックス: %d\n', result.params(3));
                        fprintf('    フィルタ数: %d\n', result.params(4));
                        fprintf('    ドロップアウト率: %.2f\n', result.params(5));
                        fprintf('    全結合層ユニット数: %d\n', result.params(6));
                    else
                        validScores(i) = -inf;
                        isOverfitFlags(i) = true;
                        fprintf('\nパラメータセット %d: 評価不能\n', i);
                    end
                end
                
                % 過学習していないモデルの中から最良のものを選択
                nonOverfitIndices = find(~isOverfitFlags);
                
                if isempty(nonOverfitIndices)
                    warning('過学習していないモデルが見つかりませんでした。最も高いスコアのモデルを選択します。');
                    [bestScore, bestIdx] = max(validScores);
                    fprintf('\n注意: 選択されたモデルは過学習していますが、最も良いスコア（%.4f）を持っています。\n', bestScore);
                else
                    % 過学習していないモデルから最良のものを選択
                    [bestScore, bestLocalIdx] = max(validScores(nonOverfitIndices));
                    bestIdx = nonOverfitIndices(bestLocalIdx);
                    fprintf('\n過学習していないモデルから最良のものを選択しました（スコア: %.4f）\n', bestScore);
                end
                
                bestResult = validResults{bestIdx};
                
                % 結果の設定
                optimizedParams = bestResult.params;
                performance = bestResult.performance;
                model = bestResult.model;
                obj.bestParams = optimizedParams;
                obj.optimizedModel = model;
                
                fprintf('\n=== 最終選択モデル（パラメータセット %d）===\n', bestIdx);
                fprintf('基本精度: %.4f\n', performance);
                fprintf('調整後スコア: %.4f\n', validScores(bestIdx));
                fprintf('パラメータ:\n');
                fprintf('  学習率: %.6f\n', optimizedParams(1));
                fprintf('  バッチサイズ: %d\n', optimizedParams(2));
                fprintf('  カーネルサイズインデックス: %d\n', optimizedParams(3));
                fprintf('  フィルタ数: %d\n', optimizedParams(4));
                fprintf('  ドロップアウト率: %.2f\n', optimizedParams(5));
                fprintf('  全結合層ユニット数: %d\n', optimizedParams(6));
                
            catch ME
                error('結果処理中にエラーが発生: %s', ME.message);
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

        function updateOptimizationHistory(obj, results)
            try
                % 有効な結果のみを取得
                validResults = results(~cellfun(@isempty, results));
                
                % 各結果を履歴に追加
                for i = 1:length(validResults)
                    result = validResults{i};
                    if ~isempty(result.model) && ~isempty(result.performance)
                        newEntry = struct(...
                            'params', result.params, ...
                            'performance', result.performance, ...
                            'model', result.model);
                        
                        % 履歴が空の場合は新しい構造体として、そうでない場合は追加
                        if isempty(obj.optimizationHistory)
                            obj.optimizationHistory = newEntry;
                        else
                            obj.optimizationHistory(end+1) = newEntry;
                        end
                    end
                end
                
                % 履歴を性能順にソート
                [~, sortIdx] = sort([obj.optimizationHistory.performance], 'descend');
                obj.optimizationHistory = obj.optimizationHistory(sortIdx);
                
                fprintf('\n最適化履歴を更新しました（計%d個のモデル）\n', ...
                    length(obj.optimizationHistory));
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
    end
end