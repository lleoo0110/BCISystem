classdef CNNOptimizer < handle
    properties (Access = private)
        params              % パラメータ設定
        optimizedModel      % 最適化されたモデル
        bestParams         % 最良パラメータ
        bestPerformance    % 最良性能値
        searchSpace        % パラメータ探索空間
        optimizationHistory % 最適化履歴
        overfitState      % 過学習判定
    end
    
    methods (Access = public)
        function obj = CNNOptimizer(params)
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
        end
        
        function [optimizedParams, performance, model] = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.cnn.optimize
                    fprintf('CNN最適化は無効です。デフォルトパラメータを使用します。\n');
                    optimizedParams =[]; 
                    performance =[]; 
                    model =[];
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

                parfor i = 1:size(paramSets, 1)
                    try
                        % パラメータの更新（parfor内で安全に使用できる変数のみを使用）
                        localParams = baseParams;
                        localParams = obj.updateCNNParameters(localParams, paramSets(i,:), kernelSizeLocal);

                        % CNNの学習と評価
                        cnn = CNNClassifier(localParams);
                        trainResults = cnn.trainCNN(data, labels);

                        % 結果の保存
                        results{i} = struct(...
                            'params', paramSets(i,:),...
                            'performance', trainResults.performance.accuracy,... % 検証データに対するaccuracyを保存
                            'model', trainResults.model);

                        fprintf('組み合わせ %d/%d: 精度 = %.4f\n', i, size(paramSets, 1), trainResults.performance.accuracy);

                    catch ME
                        warning('組み合わせ%dでエラー発生: %s', i, ME.message);
                        results{i} = struct('params', paramSets(i,:), 'performance', -inf, 'model',[]);
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
            % CNN最適化のためのパラメータセットを生成
            %
            % 入力:
            %   numSamples - 生成するパラメータセットの数
            %
            % 出力:
            %   paramSets - 生成されたパラメータセット [numSamples × パラメータ数]

            try
                % Latin Hypercube Samplingの実行
                % 8パラメータ: 学習率、バッチサイズ、カーネルサイズ、フィルタ数、
                % ドロップアウト率、FC層ユニット数、L2正則化係数、Early Stopping閾値
                lhsPoints = lhsdesign(numSamples, 8);

                % パラメータセットの初期化
                paramSets = zeros(numSamples, 8);

                % 1. 学習率（対数スケール）
                lr_range = obj.searchSpace.learningRate;
                paramSets(:,1) = 10.^(log10(lr_range(1)) + ...
                    (log10(lr_range(2)) - log10(lr_range(1))) * lhsPoints(:,1));

                % 2. バッチサイズ（32以上に制限）
                bs_range = obj.searchSpace.miniBatchSize;
                bs_range(1) = max(bs_range(1), 32);  % 最小バッチサイズを32に設定
                paramSets(:,2) = round(bs_range(1) + ...
                    (bs_range(2) - bs_range(1)) * lhsPoints(:,2));

                % 3. カーネルサイズ
                kernel_sizes = obj.searchSpace.kernelSize;
                num_kernel_sizes = numel(kernel_sizes);
                if num_kernel_sizes > 1
                    paramSets(:,3) = round(1 + (num_kernel_sizes - 1) * lhsPoints(:,3));
                else
                    paramSets(:,3) = ones(numSamples, 1);
                end

                % 4. フィルタ数（2のべき乗に調整）
                nf_range = obj.searchSpace.numFilters;
                base_filters = round(nf_range(1) + ...
                    (nf_range(2) - nf_range(1)) * lhsPoints(:,4));
                paramSets(:,4) = 2.^round(log2(base_filters));  % 2のべき乗に調整

                % 5. ドロップアウト率（最小値を0.3に制限）
                do_range = obj.searchSpace.dropoutRate;
                do_range(1) = max(do_range(1), 0.3);  % 最小ドロップアウト率を0.3に設定
                paramSets(:,5) = do_range(1) + ...
                    (do_range(2) - do_range(1)) * lhsPoints(:,5);

                % 6. 全結合層ユニット数（2のべき乗に調整）
                fc_range = obj.searchSpace.fcUnits;
                base_units = round(fc_range(1) + ...
                    (fc_range(2) - fc_range(1)) * lhsPoints(:,6));
                paramSets(:,6) = 2.^round(log2(base_units));  % 2のべき乗に調整

                % 7. L2正則化係数（対数スケール）
                l2_range = [0.0001, 0.01];
                paramSets(:,7) = 10.^(log10(l2_range(1)) + ...
                    (log10(l2_range(2)) - log10(l2_range(1))) * lhsPoints(:,7));

                % 8. Early Stopping閾値
                es_range = [0.001, 0.01];  % 改善閾値の範囲
                paramSets(:,8) = es_range(1) + ...
                    (es_range(2) - es_range(1)) * lhsPoints(:,8);

                % パラメータの物理的制約の確認
                paramSets = obj.validateParameterSets(paramSets);

                % パラメータセットの多様性を確保
                paramSets = obj.ensureParameterDiversity(paramSets);

                % 生成されたパラメータセットの情報表示
                obj.displayParameterSetsInfo(paramSets);

            catch ME
                error('パラメータセット生成エラー: %s', ME.message);
            end
        end

        function paramSets = validateParameterSets(obj, paramSets)
            % パラメータセットの物理的制約をチェックし、必要に応じて調整

            % 学習率の制約
            paramSets(:,1) = max(min(paramSets(:,1), 0.1), 0.00001);

            % バッチサイズの制約（2のべき乗に調整）
            paramSets(:,2) = 2.^round(log2(paramSets(:,2)));

            % ドロップアウト率の制約
            paramSets(:,5) = max(min(paramSets(:,5), 0.7), 0.3);

            % メモリ制約を考慮したユニット数の調整
            maxUnits = obj.calculateMaxUnits(paramSets(:,2));  % バッチサイズに基づく最大ユニット数
            paramSets(:,6) = min(paramSets(:,6), maxUnits);
        end

        function paramSets = ensureParameterDiversity(obj, paramSets)
            % パラメータセット間の多様性を確保

            % パラメータ間の相関を計算
            correlation_matrix = corr(paramSets);

            % 強い相関がある場合、一方のパラメータをランダムに再生成
            threshold = 0.8;  % 相関閾値
            [i, j] = find(abs(correlation_matrix) > threshold & ...
                abs(correlation_matrix) < 1);

            for k = 1:length(i)
                if i(k) < j(k)  % 重複を避けるため
                    % パラメータの再生成
                    paramSets(:,j(k)) = obj.regenerateParameter(j(k), size(paramSets,1));
                end
            end
        end

        function maxUnits = calculateMaxUnits(~, batchSizes)
            % 利用可能なメモリに基づいて最大ユニット数を計算
            baseMemory = 8 * 1024 * 1024;  % 8GB
            bytesPerParameter = 4;  % float32

            maxUnits = floor(baseMemory ./ (batchSizes * bytesPerParameter));
            maxUnits = min(maxUnits, 2048);  % 上限を2048ユニットに制限
        end
        
        function newParam = regenerateParameter(obj, paramIndex, numSamples)
            % 特定のパラメータを再生成するメソッド
            %
            % 入力:
            %   paramIndex - 再生成するパラメータのインデックス
            %   numSamples - 生成するサンプル数

            try
                % 一様分布でランダムな値を生成
                randValues = rand(numSamples, 1);

                switch paramIndex
                    case 1  % 学習率
                        lr_range = obj.searchSpace.learningRate;
                        newParam = 10.^(log10(lr_range(1)) + ...
                            (log10(lr_range(2)) - log10(lr_range(1))) * randValues);

                    case 2  % バッチサイズ
                        bs_range = obj.searchSpace.miniBatchSize;
                        newParam = round(bs_range(1) + ...
                            (bs_range(2) - bs_range(1)) * randValues);
                        % 2のべき乗に調整
                        newParam = 2.^round(log2(newParam));

                    case 3  % カーネルサイズ
                        kernel_sizes = obj.searchSpace.kernelSize;
                        num_kernel_sizes = numel(kernel_sizes);
                        if num_kernel_sizes > 1
                            newParam = round(1 + (num_kernel_sizes - 1) * randValues);
                        else
                            newParam = ones(numSamples, 1);
                        end

                    case 4  % フィルタ数
                        nf_range = obj.searchSpace.numFilters;
                        base_filters = round(nf_range(1) + ...
                            (nf_range(2) - nf_range(1)) * randValues);
                        newParam = 2.^round(log2(base_filters));  % 2のべき乗に調整

                    case 5  % ドロップアウト率
                        do_range = obj.searchSpace.dropoutRate;
                        do_range(1) = max(do_range(1), 0.3);
                        newParam = do_range(1) + ...
                            (do_range(2) - do_range(1)) * randValues;

                    case 6  % FC層ユニット数
                        fc_range = obj.searchSpace.fcUnits;
                        base_units = round(fc_range(1) + ...
                            (fc_range(2) - fc_range(1)) * randValues);
                        newParam = 2.^round(log2(base_units));

                    case 7  % L2正則化係数
                        l2_range = [0.0001, 0.01];
                        newParam = 10.^(log10(l2_range(1)) + ...
                            (log10(l2_range(2)) - log10(l2_range(1))) * randValues);

                    case 8  % Early Stopping閾値
                        es_range = [0.001, 0.01];
                        newParam = es_range(1) + ...
                            (es_range(2) - es_range(1)) * randValues;

                    otherwise
                        error('無効なパラメータインデックス: %d', paramIndex);
                end

            catch ME
                error('パラメータ再生成エラー: %s', ME.message);
            end
        end

        function displayParameterSetsInfo(~, paramSets)
            % 生成されたパラメータセットの統計情報を表示

            fprintf('\n=== 生成されたパラメータセット情報 ===\n');
            fprintf('パラメータセット数: %d\n', size(paramSets,1));

            % 各パラメータの統計
            paramNames = {'学習率', 'バッチサイズ', 'カーネルサイズ', ...
                'フィルタ数', 'ドロップアウト率', 'FC層ユニット数', ...
                'L2正則化係数', 'Early Stopping閾値'};

            for i = 1:size(paramSets,2)
                fprintf('\n%s:\n', paramNames{i});
                fprintf('  最小値: %g\n', min(paramSets(:,i)));
                fprintf('  最大値: %g\n', max(paramSets(:,i)));
                fprintf('  平均値: %g\n', mean(paramSets(:,i)));
                fprintf('  標準偏差: %g\n', std(paramSets(:,i)));
            end

            fprintf('\n');
        end
        
        function [optimizedParams, performance, model] = processFinalResults(obj, results)
            try
                % 有効な結果のみを抽出
                validResults = ~cellfun(@isempty, results);
                performances = nan(size(results));

                % 性能値の抽出
                for i = 1:length(results)
                    if validResults(i)
                        performances(i) = results{i}.performance;
                    end
                end

                % 過学習チェックの初期化
                overfitStatus = cell(size(results));

                % CNNClassifierを使用して過学習チェック（可能な場合のみ）
                for i = 1:length(results)
                    if validResults(i) && isfield(results{i}, 'model')
                        % モデルの評価結果から過学習を判定
                        trainAcc = results{i}.performance;  % 学習データでの性能
                        valAcc = NaN;  % 検証データでの性能（利用可能な場合）

                        % 検証性能が利用可能な場合は使用
                        if isfield(results{i}, 'validation') && isfield(results{i}.validation, 'accuracy')
                            valAcc = results{i}.validation.accuracy;
                        end

                        % 過学習の判定基準（単純化）
                        isOverfit = false;
                        metrics = struct();

                        if ~isnan(valAcc)
                            % 学習性能と検証性能の差が大きい場合を過学習とみなす
                            genGap = trainAcc - valAcc;
                            isOverfit = genGap > 0.2;  % 20%以上の差を過学習とする

                            metrics = struct(...
                                'generalizationGap', genGap, ...
                                'severity', obj.determineOverfitSeverity(genGap));
                        else
                            % 検証データがない場合は、より保守的な判定
                            metrics = struct(...
                                'generalizationGap', NaN, ...
                                'severity', 'unknown');
                        end

                        overfitStatus{i} = struct('isOverfit', isOverfit, 'metrics', metrics);
                    end
                end

                % 最良のモデルの選択
                validIndices = find(validResults);
                [maxPerf, bestIdx] = max(performances(validResults));

                % 結果の設定
                bestResult = results{validIndices(bestIdx)};
                optimizedParams = bestResult.params;
                performance = bestResult.performance;
                model = bestResult.model;

                % クラスプロパティの更新
                obj.bestParams = optimizedParams;
                obj.bestPerformance = performance;
                obj.optimizedModel = model;

                % 過学習状態の記録
                if ~isempty(overfitStatus{validIndices(bestIdx)})
                    obj.overfitState = overfitStatus{validIndices(bestIdx)};
                else
                    obj.overfitState = struct('isOverfit', false, 'metrics', struct('severity', 'unknown'));
                end

            catch ME
                error('結果処理エラー: %s', ME.message);
            end
        end
    
        function params = updateCNNParameters(~, params, paramSet, kernelSizeLocal)
            try
                % 1. 学習率の更新
                params.classifier.cnn.training.optimizer.learningRate = paramSet(1);

                % 2. バッチサイズの更新
                params.classifier.cnn.training.miniBatchSize = round(paramSet(2));

                % 3. カーネルサイズの更新
                kernelIdx = round(paramSet(3));
                % kernelSizeLocalが配列の場合
                if isnumeric(kernelSizeLocal)
                    if kernelIdx <= size(kernelSizeLocal, 1)
                        newKernelSize = kernelSizeLocal(kernelIdx, :);
                    else
                        newKernelSize = [3 3]; % デフォルト値
                    end
                % kernelSizeLocalがセル配列の場合
                elseif iscell(kernelSizeLocal)
                    if kernelIdx <= numel(kernelSizeLocal)
                        newKernelSize = kernelSizeLocal{kernelIdx};
                    else
                        newKernelSize = [3 3]; % デフォルト値
                    end
                else
                    newKernelSize = [3 3]; % デフォルト値
                end

                % カーネルサイズの適用
                convLayers = fieldnames(params.classifier.cnn.architecture.convLayers);
                for i = 1:length(convLayers)
                    params.classifier.cnn.architecture.convLayers.(convLayers{i}).size = newKernelSize;
                end

                % 4. フィルタ数の更新
                numFilters = round(paramSet(4));
                convLayers = fieldnames(params.classifier.cnn.architecture.convLayers);
                for i = 1:length(convLayers)
                    params.classifier.cnn.architecture.convLayers.(convLayers{i}).filters = ...
                        numFilters * 2^(i-1);  % 各層でフィルタ数を2倍に
                end

                % 5. ドロップアウト率の更新
                dropoutRate = paramSet(5);
                dropoutLayers = fieldnames(params.classifier.cnn.architecture.dropoutLayers);
                baseRate = dropoutRate;
                for i = 1:length(dropoutLayers)
                    % 深い層ほどドロップアウト率を若干増加
                    params.classifier.cnn.architecture.dropoutLayers.(dropoutLayers{i}) = ...
                        min(baseRate + 0.1 * (i-1), 0.7);
                end

                % 6. 全結合層ユニット数の更新
                fcUnits = round(paramSet(6));
                numLayers = length(params.classifier.cnn.architecture.fullyConnected);
                for i = 1:numLayers
                    % 深い層ほどユニット数を減少
                    params.classifier.cnn.architecture.fullyConnected(i) = ...
                        max(round(fcUnits / 2^(i-1)), 32);
                end

                % 7. L2正則化係数の更新（オプション）
                if length(paramSet) >= 7
                    params.classifier.cnn.training.regularization = paramSet(7);
                end

                % 8. Early Stopping閾値の更新（オプション）
                if length(paramSet) >= 8
                    params.classifier.cnn.training.validation.threshold = paramSet(8);
                end

            catch ME
                error('パラメータ更新エラー: %s', ME.message);
            end
        end

        function severity = determineOverfitSeverity(~, genGap)
            % 過学習の重大度を判定
            if genGap > 0.3
                severity = 'severe';
            elseif genGap > 0.2
                severity = 'moderate';
            elseif genGap > 0.1
                severity = 'mild';
            else
                severity = 'none';
            end
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
            
            if obj.overfitState.isOverfit
                fprintf('\n過学習状態: 検出\n');
                fprintf('過学習の重大度: %s\n', obj.overfitState.metrics.severity);
                fprintf('推奨される対策:\n');
                fprintf('- ドロップアウト率の増加\n');
                fprintf('- 正則化の強化\n');
                fprintf('- バッチサイズの調整\n');
                fprintf('- 学習率の見直し\n');
            else
                fprintf('\n過学習状態: 未検出\n');
            end
        end

        function plotOptimizationResults(obj)
            try
                figure('Name', 'CNN最適化結果');
                
                subplot(2,2,1);
                performances = [obj.optimizationHistory.performance];
                plot(performances, '-o');
                xlabel('反復回数');
                ylabel('性能');
                title('最適化履歴');
                grid on;
                
                subplot(2,2,2);
                params = vertcat(obj.optimizationHistory.params);
                performances = [obj.optimizationHistory.performance];
                paramNames = {'学習率', 'バッチ', 'カーネル', 'フィルタ', 'ドロップアウト', '全結合'};
                
                correlations = zeros(6,1);
                for i = 1:6
                    correlations(i) = corr(params(:,i), performances');
                end
                
                bar(correlations);
                set(gca, 'XTickLabel', paramNames);
                title('パラメータと性能の相関');
                ylabel('相関係数');
                grid on;
                
                subplot(2,2,3);
                histogram(performances, 'Normalization', 'probability');
                xlabel('性能');
                ylabel('頻度');
                title('性能分布');
                grid on;
                
                subplot(2,2,4);
                scatter(log10(params(:,1)), performances, 'filled');
                xlabel('学習率（対数）');
                ylabel('性能');
                title('学習率の影響');
                grid on;
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
    end
end