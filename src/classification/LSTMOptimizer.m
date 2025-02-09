classdef LSTMOptimizer < handle
    properties (Access = private)
        params              % パラメータ設定（構成情報）
        optimizedModel      % 最適化されたモデル
        bestParams          % 最良パラメータ
        bestPerformance     % 最良性能値
        searchSpace         % パラメータ探索空間
        optimizationHistory % 最適化履歴（各試行結果の記録）
        useGPU              % GPU使用の有無
        maxTrials           % 最大試行回数
    end
    
    methods (Access = public)
        %% コンストラクタ
        function obj = LSTMOptimizer(params)
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            obj.useGPU = params.classifier.lstm.gpu;
            obj.maxTrials = 30;  % デフォルトの試行回数
        end
        
        %% パラメータ最適化実行
        function [optimizedParams, performance, model] = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.lstm.optimize
                    fprintf('LSTM最適化は無効です。デフォルトパラメータを使用します。\n');
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
                        localParams = obj.updateLSTMParameters(localParams, paramSets(i,:));
                        
                        % LSTMの学習と評価
                        lstm = LSTMClassifier(localParams);
                        trainResults = lstm.trainLSTM(data, labels);
                        
                        % 結果を記録
                        results{i} = struct(...
                            'params', paramSets(i,:), ...
                            'model', trainResults.model, ...
                            'performance', trainResults.performance, ...
                            'trainInfo', trainResults.trainInfo, ...
                            'overfitting', trainResults.overfitting);
                        
                        fprintf('試行 %d/%d: 精度 = %.4f\n', i, numParamSets, trainResults.performance.overallAccuracy);
                        
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
                error('LSTM最適化に失敗: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        %% 探索空間の初期化
        function initializeSearchSpace(obj)
            % 構成ファイルに定義された探索空間を取得
            obj.searchSpace = obj.params.classifier.lstm.optimization.searchSpace;
        end
        
        %% パラメータセットの生成（Latin Hypercube Sampling）
        function paramSets = generateParameterSets(obj, numTrials)
            % 最適化するパラメータは6個
            numParameters = 6;
            lhsPoints = lhsdesign(numTrials, numParameters);
            
            % 事前にパラメータセットを初期化
            paramSets = zeros(numTrials, numParameters);
            
            % 1. 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + (log10(lr_range(2)) - log10(lr_range(1))) * lhsPoints(:,1));
            
            % 2. ミニバッチサイズ
            bs_range = obj.searchSpace.miniBatchSize;
            paramSets(:,2) = round(bs_range(1) + (bs_range(2) - bs_range(1)) * lhsPoints(:,2));
            
            % 3. 隠れ層ユニット数
            hu_range = obj.searchSpace.numHiddenUnits;
            paramSets(:,3) = round(hu_range(1) + (hu_range(2) - hu_range(1)) * lhsPoints(:,3));
            
            % 4. レイヤー数
            nl_range = obj.searchSpace.numLayers;
            paramSets(:,4) = round(nl_range(1) + (nl_range(2) - nl_range(1)) * lhsPoints(:,4));
            
            % 5. ドロップアウト率
            do_range = obj.searchSpace.dropoutRate;
            paramSets(:,5) = do_range(1) + (do_range(2) - do_range(1)) * lhsPoints(:,5);
            
            % 6. 全結合層ユニット数
            fc_range = obj.searchSpace.fcUnits;
            paramSets(:,6) = round(fc_range(1) + (fc_range(2) - fc_range(1)) * lhsPoints(:,6));
        end
        
        %% LSTMパラメータの更新
        function params = updateLSTMParameters(~, params, paramSet)
            % トレーニングパラメータ更新
            params.classifier.lstm.training.optimizer.learningRate = paramSet(1);
            params.classifier.lstm.training.miniBatchSize = paramSet(2);
            
            % LSTM層の更新（レイヤー数 paramSet(4) に基づく）
            numLayers = paramSet(4);
            lstmLayers = struct();
            for i = 1:numLayers
                layerName = sprintf('lstm%d', i);
                lstmLayers.(layerName) = struct(...
                    'numHiddenUnits', paramSet(3), ...
                    'OutputMode', 'last');
            end
            params.classifier.lstm.architecture.lstmLayers = lstmLayers;
            
            % ドロップアウト層の更新
            dropoutLayers = struct();
            for i = 1:numLayers
                dropoutName = sprintf('dropout%d', i);
                dropoutLayers.(dropoutName) = paramSet(5);
            end
            params.classifier.lstm.architecture.dropoutLayers = dropoutLayers;
            
            % 全結合層の更新
            params.classifier.lstm.architecture.fullyConnected = [paramSet(6)];
        end
        
        %% 結果処理と最良パラメータ・モデルの選択
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
                        fprintf('    隠れ層ユニット数: %d\n', result.params(3));
                        fprintf('    レイヤー数: %d\n', result.params(4));
                        fprintf('    ドロップアウト率: %.2f\n', result.params(5));
                        fprintf('    全結合層ユニット数: %d\n', result.params(6));
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
                fprintf('  隠れ層ユニット数: %d\n', optimizedParams(3));
                fprintf('  レイヤー数: %d\n', optimizedParams(4));
                fprintf('  ドロップアウト率: %.2f\n', optimizedParams(5));
                fprintf('  全結合層ユニット数: %d\n', optimizedParams(6));
                
            catch ME
                error('結果処理中にエラーが発生: %s', ME.message);
            end
        end
        
        %% 最適化履歴の更新
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
                warning(ME.identifier, '%s', ME.messge);
            end
        end
        
        %% 最適化結果と統計の表示
        function displayOptimizationResults(obj)
            fprintf('\n=== LSTM最適化結果 ===\n');
            if ~isempty(obj.bestParams)
                fprintf('最良モデルの性能:\n');
                fprintf('  精度: %.4f\n\n', obj.bestPerformance);
                fprintf('最適化されたパラメータ:\n');
                fprintf('  学習率: %.6f\n', obj.bestParams(1));
                fprintf('  バッチサイズ: %d\n', obj.bestParams(2));
                fprintf('  隠れ層ユニット数: %d\n', obj.bestParams(3));
                fprintf('  レイヤー数: %d\n', obj.bestParams(4));
                fprintf('  ドロップアウト率: %.2f\n', obj.bestParams(5));
                fprintf('  全結合層ユニット数: %d\n\n', obj.bestParams(6));
            end
            
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
                paramNames = {'学習率', 'バッチサイズ', '隠れ層ユニット数', 'レイヤー数', 'ドロップアウト率', '全結合層ユニット数'};
                for i = 1:size(paramsMat, 2)
                    fprintf('%s:\n', paramNames{i});
                    fprintf('  平均: %.6f\n', mean(paramsMat(:,i)));
                    fprintf('  標準偏差: %.6f\n', std(paramsMat(:,i)));
                    fprintf('  最小値: %.6f\n', min(parmsMat(:,i)));
                    fprintf('  最大値: %.6f\n', max(paramsMat(:,i)));
                    fprintf('\n');
                end
                
                % 収束性評価（各試行間の改善率）
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
            end
        end
    end
end
