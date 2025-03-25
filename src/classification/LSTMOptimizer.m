classdef LSTMOptimizer < handle
    properties (Access = private)
        params              % パラメータ設定
        optimizedModel      % 最適化されたモデル
        bestParams          % 最良パラメータ
        bestPerformance     % 最良性能値
        searchSpace         % パラメータ探索空間
        optimizationHistory % 最適化履歴
        useGPU              % GPU使用の有無
        maxTrials           % 最大試行回数
        earlyStopParams     % 早期停止パラメータ
        
        % 評価重み
        evaluationWeights   % 各評価指標の重みづけ
    end
    
    methods (Access = public)
        function obj = LSTMOptimizer(params)
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            obj.useGPU = params.classifier.lstm.gpu;
            obj.maxTrials = params.classifier.lstm.optimization.maxTrials;
            
            % 早期停止パラメータの初期化
            obj.earlyStopParams = struct(...
                'enable', true, ...     % 早期停止を有効化
                'patience', 5, ...      % 改善なしで待機する試行回数
                'min_delta', 0.005, ... % 有意な改善と見なす最小値
                'best_score', -inf, ... % 最良スコア初期値
                'counter', 0, ...       % 改善なしカウンター
                'history', [] ...       % スコア履歴
            );
            
            % 評価重みの初期化
            obj.evaluationWeights = struct(...
                'test', 0.6, ...        % テスト精度の重み
                'validation', 0.4, ...  % 検証精度の重み
                'crossValidation', 0.3, ... % 交差検証の重み（検証精度の代わりに使用する場合）
                'f1Score', 0.3, ...     % F1スコアの重み
                'complexity', 0.1, ...  % 複雑性のペナルティ最大値
                'overfitMax', 0.5 ...   % 過学習の最大ペナルティ値
            );
            
            % GPU利用可能性のチェック
            if obj.useGPU
                try
                    gpuInfo = gpuDevice();
                    fprintf('GPUが検出されました: %s (メモリ: %.2f GB)\n', ...
                        gpuInfo.Name, gpuInfo.TotalMemory/1e9);
                catch
                    warning('GPU使用が指定されていますが、GPUが利用できません。CPUで実行します。');
                    obj.useGPU = false;
                end
            end
        end
        
        function results = optimize(obj, data, labels)
            try
                if ~obj.params.classifier.lstm.optimize
                    fprintf('LSTM最適化は無効です。デフォルトパラメータを使用します。\n');
                    results = obj.createDefaultResults();
                    return;
                end

                fprintf('\n=== LSTMハイパーパラメータ最適化を開始 ===\n');
                
                % 探索アルゴリズムの選択
                searchAlgorithm = 'lhs';  % デフォルト: Latin Hypercube Sampling
                if isfield(obj.params.classifier.lstm.optimization, 'searchAlgorithm')
                    searchAlgorithm = obj.params.classifier.lstm.optimization.searchAlgorithm;
                end
                fprintf('探索アルゴリズム: %s\n', searchAlgorithm);
                
                % パラメータセットの生成
                fprintf('パラメータ探索空間を設定中...\n');
                paramSets = obj.generateParameterSets(obj.maxTrials, searchAlgorithm);
                fprintf('パラメータ%dセットで最適化を開始します\n', size(paramSets, 1));

                % 結果保存用配列
                trialResults = cell(size(paramSets, 1), 1);
                baseParams = obj.params;

                % 各パラメータセットで評価
                for i = 1:size(paramSets, 1)
                    try
                        fprintf('\n--- パラメータセット %d/%d の評価 ---\n', i, size(paramSets, 1));
                        
                        % 現在のパラメータ構成の表示
                        obj.displayCurrentParams(paramSets(i,:));
                        
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
                            'normParams', trainResults.normParams, ...
                            'crossValidation', trainResults.crossValidation ...
                        );

                        % モデル性能の総合スコアを計算（早期停止用）
                        performance = trainResults.performance.accuracy;
                        
                        % 検証精度データの取得
                        valAccuracy = 0;
                        if isfield(trainResults.trainInfo.History, 'ValidationAccuracy') && ...
                           ~isempty(trainResults.trainInfo.History.ValidationAccuracy)
                            valAcc = trainResults.trainInfo.History.ValidationAccuracy;
                            valAccuracy = mean(valAcc(max(1, end-5):end)); % 最後の5エポックの平均
                        end
                        
                        % 交差検証データの使用
                        cvAccuracy = 0;
                        if isfield(trainResults, 'crossValidation') && ...
                           isfield(trainResults.crossValidation, 'meanAccuracy')
                            cvAccuracy = trainResults.crossValidation.meanAccuracy;
                        end
                        
                        % F1スコア計算
                        f1Score = obj.calculateMeanF1Score(trainResults.performance);
                        
                        % 総合評価スコアの計算
                        evaluationScore = obj.calculateTrialScore(performance, valAccuracy, cvAccuracy, f1Score, trainResults);
                        
                        fprintf('組み合わせ %d/%d: テスト精度 = %.4f, 総合スコア = %.4f\n', ...
                            i, size(paramSets, 1), performance, evaluationScore);
                        
                        % GPUメモリの解放
                        if obj.useGPU
                            reset(gpuDevice);
                        end

                    catch ME
                        warning('組み合わせ%dでエラー発生: %s', i, ME.message);
                        fprintf('エラー詳細:\n');
                        disp(getReport(ME, 'extended'));
                        
                        % エラー発生時でも最低限のパラメータ情報は保存
                        trialResults{i} = struct(...
                            'params', paramSets(i,:), ...
                            'error', true, ...
                            'errorMessage', ME.message, ...
                            'performance', struct('accuracy', 0) ...
                        );

                        if obj.useGPU
                            reset(gpuDevice);
                        end
                    end
                end

                % 最良の結果を選択
                [bestResults, summary] = obj.processFinalResults(trialResults);
                
                % 最適化履歴の更新
                obj.updateOptimizationHistory(trialResults);
                
                % 最適化サマリーの表示
                obj.displayOptimizationSummary(summary);
                
                % 結果構造体の作成
                results = struct(...
                    'model', bestResults.model, ...
                    'performance', bestResults.performance, ...
                    'trainInfo', bestResults.trainInfo, ...
                    'overfitting', bestResults.overfitting, ...
                    'normParams', bestResults.normParams, ...
                    'crossValidation', bestResults.crossValidation ...
                );
                
                fprintf('\n=== LSTM最適化が完了しました ===\n');

            catch ME
                fprintf('\n=== LSTM最適化中にエラーが発生しました ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラースタック:\n');
                disp(getReport(ME, 'extended'));
                error('LSTM最適化に失敗: %s', ME.message);
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
            fprintf('  - 学習率: [%.6f, %.6f]\n', obj.searchSpace.learningRate);
            fprintf('  - バッチサイズ: [%d, %d]\n', obj.searchSpace.miniBatchSize);
            fprintf('  - LSTMユニット数: [%d, %d]\n', obj.searchSpace.lstmUnits);
            fprintf('  - LSTM層数: [%d, %d]\n', obj.searchSpace.numLayers);
            fprintf('  - ドロップアウト率: [%.2f, %.2f]\n', obj.searchSpace.dropoutRate);
            fprintf('  - 全結合層ユニット数: [%d, %d]\n', obj.searchSpace.fcUnits);
        end
        
        function paramSets = generateParameterSets(obj, numTrials, algorithm)
            % 指定されたアルゴリズムでパラメータセットを生成
            %
            % 入力:
            %   numTrials - 試行回数
            %   algorithm - 探索アルゴリズム
            %
            % 出力:
            %   paramSets - パラメータセット行列 [numTrials x 6]
            
            % 探索アルゴリズムの選択
            if nargin < 3
                algorithm = 'lhs';  % デフォルトはLatin Hypercube Sampling
            end
            
            fprintf('パラメータ探索アルゴリズム: %s\n', algorithm);
            
            switch lower(algorithm)
                case 'lhs'  % Latin Hypercube Sampling
                    paramSets = obj.generateLHSParams(numTrials);
                case 'random'  % ランダムサンプリング
                    paramSets = obj.generateRandomParams(numTrials);
                case 'grid'  % グリッドサーチ
                    paramSets = obj.generateGridParams();
                    numTrials = size(paramSets, 1);  % グリッドサイズに合わせる
                otherwise
                    warning('未知の探索アルゴリズム: %s。LHSにフォールバックします。', algorithm);
                    paramSets = obj.generateLHSParams(numTrials);
            end
            
            fprintf('探索パラメータセット数: %d\n', size(paramSets, 1));
        end
        
        function paramSets = generateLHSParams(obj, numTrials)
            % Latin Hypercube Samplingによるパラメータセット生成
            
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
        
        function paramSets = generateRandomParams(obj, numTrials)
            % ランダムサンプリングによるパラメータセット生成
            
            paramSets = zeros(numTrials, 6);
            
            % 1. 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + ...
                (log10(lr_range(2))-log10(lr_range(1))) * rand(numTrials, 1));
            
            % 2. ミニバッチサイズ
            bs_range = obj.searchSpace.miniBatchSize;
            paramSets(:,2) = round(bs_range(1) + (bs_range(2)-bs_range(1)) * rand(numTrials, 1));
            
            % 3. LSTMユニット数
            lu_range = obj.searchSpace.lstmUnits;
            paramSets(:,3) = round(lu_range(1) + (lu_range(2)-lu_range(1)) * rand(numTrials, 1));
            
            % 4. LSTM層数
            nl_range = obj.searchSpace.numLayers;
            paramSets(:,4) = round(nl_range(1) + (nl_range(2)-nl_range(1)) * rand(numTrials, 1));
            
            % 5. ドロップアウト率
            do_range = obj.searchSpace.dropoutRate;
            paramSets(:,5) = do_range(1) + (do_range(2)-do_range(1)) * rand(numTrials, 1);
            
            % 6. 全結合層ユニット数
            fc_range = obj.searchSpace.fcUnits;
            paramSets(:,6) = round(fc_range(1) + (fc_range(2)-fc_range(1)) * rand(numTrials, 1));
        end
        
        function paramSets = generateGridParams(obj)
            % グリッドサーチによるパラメータセット生成
            
            % グリッドポイント数の決定
            numGridPoints = 3;  % グリッドサイズの調整可能なパラメータ
            
            % 各パラメータの線形グリッド
            lr_range = obj.searchSpace.learningRate;
            lr_grid = logspace(log10(lr_range(1)), log10(lr_range(2)), numGridPoints);
            
            bs_range = obj.searchSpace.miniBatchSize;
            bs_grid = round(linspace(bs_range(1), bs_range(2), numGridPoints));
            
            lu_range = obj.searchSpace.lstmUnits;
            lu_grid = round(linspace(lu_range(1), lu_range(2), numGridPoints));
            
            nl_range = obj.searchSpace.numLayers;
            nl_grid = round(linspace(nl_range(1), nl_range(2), 2));  % 層数は少ないのでポイント数を減らす
            
            do_range = obj.searchSpace.dropoutRate;
            do_grid = linspace(do_range(1), do_range(2), numGridPoints);
            
            fc_range = obj.searchSpace.fcUnits;
            fc_grid = round(linspace(fc_range(1), fc_range(2), numGridPoints));
            
            % グリッドの全組み合わせを生成
            [LR, BS, LU, NL, DO, FC] = ndgrid(lr_grid, bs_grid, lu_grid, nl_grid, do_grid, fc_grid);
            
            % 行列に変換
            paramSets = [LR(:), BS(:), LU(:), NL(:), DO(:), FC(:)];
            
            % グリッド数が多すぎる場合は警告
            if size(paramSets, 1) > 200
                warning(['グリッドサーチのパラメータ数が非常に多いです (%d組)。' ...
                    '最適化には長時間かかる可能性があります。'], size(paramSets, 1));
            end
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

        function displayCurrentParams(~, paramSet)
            % 現在評価中のパラメータセットを表示
            
            fprintf('評価対象のパラメータ構成:\n');
            fprintf('  - 学習率: %.6f\n', paramSet(1));
            fprintf('  - バッチサイズ: %d\n', paramSet(2));
            fprintf('  - LSTMユニット数: %d\n', paramSet(3));
            fprintf('  - LSTM層数: %d\n', paramSet(4));
            fprintf('  - ドロップアウト率: %.2f\n', paramSet(5));
            fprintf('  - 全結合層ユニット数: %d\n', paramSet(6));
        end
        
        function f1Score = calculateMeanF1Score(~, performance)
            % 各クラスのF1スコアの平均を計算
            f1Score = 0;
            
            if isfield(performance, 'classwise') && ~isempty(performance.classwise)
                f1Scores = zeros(1, length(performance.classwise));
                for i = 1:length(performance.classwise)
                    f1Scores(i) = performance.classwise(i).f1score;
                end
                f1Score = mean(f1Scores);
            end
        end
        
        function score = calculateTrialScore(obj, testAccuracy, valAccuracy, cvAccuracy, f1Score, results)
            % モデルの総合評価スコアを計算
            %
            % 入力:
            %   testAccuracy - テスト精度
            %   valAccuracy - 検証精度
            %   cvAccuracy - 交差検証精度
            %   f1Score - F1スコア
            %   results - 評価結果全体（過学習情報含む）
            %
            % 出力:
            %   score - 総合評価スコア
            
            % 1. 基本精度スコアの計算
            testWeight = obj.evaluationWeights.test;
            valWeight = obj.evaluationWeights.validation;
            
            % 検証精度かクロスバリデーション精度のいずれかを選択
            validationScore = 0;
            if cvAccuracy > 0
                % 交差検証があれば優先的に使用
                validationScore = cvAccuracy;
                
                % 交差検証の標準偏差で重み付け
                if isfield(results, 'crossValidation') && ...
                   isfield(results.crossValidation, 'stdAccuracy')
                    cvStd = results.crossValidation.stdAccuracy;
                    if cvStd > 0
                        % 標準偏差の逆数で安定性を表現
                        cvStability = 1 / (1 + 10 * cvStd);  % 低い標準偏差 = 高い安定性
                        validationScore = validationScore * (0.7 + 0.3 * cvStability);
                    end
                end
            elseif valAccuracy > 0
                % 交差検証がなければ検証精度を使用
                validationScore = valAccuracy;
            else
                % どちらもなければテスト精度のみで評価
                validationScore = testAccuracy;
                valWeight = 0;
            end
            
            % 基本スコアの計算
            accuracyScore = (testWeight * testAccuracy + valWeight * validationScore) / (testWeight + valWeight);
            
            % 2. F1スコアの統合（クラス不均衡への対応）
            if f1Score > 0
                f1Weight = obj.evaluationWeights.f1Score;
                combinedScore = (1 - f1Weight) * accuracyScore + f1Weight * f1Score;
            else
                combinedScore = accuracyScore;
            end
            
            % 3. 過学習ペナルティの計算
            overfitPenalty = 0;
            if isfield(results, 'overfitting')
                % 動的な過学習ペナルティ計算
                if isfield(results.overfitting, 'performanceGap')
                    % 検証-テスト間のギャップに基づく動的ペナルティ
                    perfGap = results.overfitting.performanceGap / 100; % パーセントから小数に
                    overfitPenalty = min(obj.evaluationWeights.overfitMax, perfGap);
                else
                    % 既存の重大度ベースのペナルティをフォールバックとして使用
                    severity = results.overfitting.severity;
                    switch severity
                        case 'critical'
                            overfitPenalty = 0.5;  % 50%ペナルティ
                        case 'severe'
                            overfitPenalty = 0.3;  % 30%ペナルティ
                        case 'moderate'
                            overfitPenalty = 0.2;  % 20%ペナルティ
                        case 'mild'
                            overfitPenalty = 0.1;  % 10%ペナルティ
                        otherwise
                            overfitPenalty = 0;    % ペナルティなし
                    end
                end
            end
            
            % 4. モデル複雑性ペナルティの計算
            if isfield(results, 'params') && length(results.params) >= 6
                lstmUnits = results.params(3);
                numLayers = results.params(4);
                fcUnits = results.params(6);
                
                % 探索空間の最大値を参照して相対的な複雑さを計算
                maxUnits = obj.searchSpace.lstmUnits(2);
                maxLayers = obj.searchSpace.numLayers(2);
                maxFCUnits = obj.searchSpace.fcUnits(2);
                
                complexityScore = 0.4 * (lstmUnits / maxUnits) + ...
                                 0.4 * (numLayers / maxLayers) + ...
                                 0.2 * (fcUnits / maxFCUnits);
                complexityPenalty = obj.evaluationWeights.complexity * complexityScore;
            else
                % paramsフィールドがない場合はデフォルト値を使用
                fprintf('  注意: 結果の複雑性計算に必要なパラメータがありません。デフォルト値を使用します。\n');
                complexityPenalty = 0.05; % デフォルトの中程度の複雑性ペナルティ
            end
            
            % 5. 最終スコアの計算 (精度 - 過学習ペナルティ - 複雑性ペナルティ)
            score = combinedScore * (1 - overfitPenalty - complexityPenalty);
                
            return;
        end

        function [bestResults, summary] = processFinalResults(obj, results)
            % 全ての試行結果を処理し、最良のモデルを選択
            
            try
                fprintf('\n=== パラメータ最適化の結果処理 ===\n');
                fprintf('総試行回数: %d\n', length(results));
                
                % 有効な結果のみを抽出
                validResults = results(~cellfun(@isempty, results));
                numResults = length(validResults);
                
                fprintf('有効なパラメータセット数: %d\n', numResults);
                fprintf('無効な試行数: %d\n', length(results) - numResults);
                
                % 結果がない場合のエラー処理
                if numResults == 0
                    error('有効な結果がありません。全ての試行が失敗しました。');
                end
                
                % 評価結果保存用の変数
                modelScores = [];
                testAccuracies = [];
                valAccuracies = [];
                cvAccuracies = [];
                f1Scores = [];
                overfitPenalties = [];
                complexityPenalties = [];
                
                % サマリー情報の初期化
                summary = struct(...
                    'total_trials', length(results), ...
                    'valid_trials', numResults, ...
                    'overfit_models', 0, ...
                    'best_accuracy', 0, ...
                    'worst_accuracy', 1, ...
                    'mean_accuracy', 0, ...
                    'learning_rates', [], ...
                    'batch_sizes', [], ...
                    'hidden_units', [], ...
                    'num_layers', [], ...
                    'dropout_rates', [], ...
                    'fc_units', []);
                
                % 有効な結果のインデックス
                validIndices = [];
                
                fprintf('\n=== 各試行の詳細評価 ===\n');
                for i = 1:numResults
                    result = validResults{i};
                    try
                        if ~isempty(result) && ~isempty(result.model) && ~isempty(result.performance)
                            % 基本的な精度スコア
                            testAccuracy = result.performance.accuracy;
                            
                            % 検証精度の取得
                            valAccuracy = 0;
                            if isfield(result.trainInfo.History, 'ValidationAccuracy') && ...
                               ~isempty(result.trainInfo.History.ValidationAccuracy)
                                valAcc = result.trainInfo.History.ValidationAccuracy;
                                valAccuracy = mean(valAcc(max(1, end-5):end)); % 最後の5エポックの平均
                            end
                            
                            % 交差検証精度の取得
                            cvAccuracy = 0;
                            if isfield(result, 'crossValidation') && ...
                               isfield(result.crossValidation, 'meanAccuracy')
                                cvAccuracy = result.crossValidation.meanAccuracy;
                            end
                            
                            % F1スコアの計算
                            f1Score = obj.calculateMeanF1Score(result.performance);
                            
                            % 過学習ペナルティ計算
                            overfitPenalty = 0;
                            if isfield(result, 'overfitting')
                                % 動的な過学習ペナルティ計算
                                if isfield(result.overfitting, 'performanceGap')
                                    % 検証-テスト間のギャップに基づく動的ペナルティ
                                    perfGap = result.overfitting.performanceGap / 100; % パーセントから小数に
                                    overfitPenalty = min(obj.evaluationWeights.overfitMax, perfGap);
                                else
                                    % 既存の重大度ベースのペナルティをフォールバックとして使用
                                    if isfield(result.overfitting, 'severity')
                                        severity = result.overfitting.severity;
                                        switch severity
                                            case 'critical'
                                                overfitPenalty = 0.5;  % 50%ペナルティ
                                            case 'severe'
                                                overfitPenalty = 0.3;  % 30%ペナルティ
                                            case 'moderate'
                                                overfitPenalty = 0.2;  % 20%ペナルティ
                                            case 'mild'
                                                overfitPenalty = 0.1;  % 10%ペナルティ
                                            otherwise
                                                overfitPenalty = 0;    % ペナルティなし
                                        end
                                    end
                                end
                            end
                            
                            % モデル複雑性ペナルティ
                            complexityPenalty = 0.05; % デフォルト値
                            if isfield(result, 'params') && length(result.params) >= 6
                                lstmUnits = result.params(3);
                                numLayers = result.params(4);
                                fcUnits = result.params(6);
                                
                                % 探索空間の最大値を参照して相対的な複雑さを計算
                                maxUnits = obj.searchSpace.lstmUnits(2);
                                maxLayers = obj.searchSpace.numLayers(2);
                                maxFCUnits = obj.searchSpace.fcUnits(2);
                                
                                complexityScore = 0.4 * (lstmUnits / maxUnits) + ...
                                                 0.4 * (numLayers / maxLayers) + ...
                                                 0.2 * (fcUnits / maxFCUnits);
                                complexityPenalty = obj.evaluationWeights.complexity * complexityScore;
                            else
                                fprintf('  試行 %d: パラメータ情報がないため、デフォルトの複雑性ペナルティを適用します\n', i);
                            end
                            
                            % 総合スコアの計算
                            score = obj.calculateTrialScore(testAccuracy, valAccuracy, cvAccuracy, f1Score, result);
                            
                            % 配列に追加
                            testAccuracies = [testAccuracies; testAccuracy];
                            valAccuracies = [valAccuracies; valAccuracy];
                            cvAccuracies = [cvAccuracies; cvAccuracy];
                            f1Scores = [f1Scores; f1Score];
                            overfitPenalties = [overfitPenalties; overfitPenalty];
                            complexityPenalties = [complexityPenalties; complexityPenalty];
                            modelScores = [modelScores; score];
                            validIndices = [validIndices; i];
                            
                            % 過学習フラグの設定
                            severity = 'none';
                            if isfield(result, 'overfitting') && isfield(result.overfitting, 'severity')
                                severity = result.overfitting.severity;
                            end
                            isOverfit = ismember(severity, {'critical', 'severe', 'moderate'});
                            if isOverfit
                                summary.overfit_models = summary.overfit_models + 1;
                            end
                            
                            % サマリー情報の更新
                            if isfield(result, 'params') && length(result.params) >= 6
                                summary.learning_rates = [summary.learning_rates, result.params(1)];
                                summary.batch_sizes = [summary.batch_sizes, result.params(2)];
                                summary.hidden_units = [summary.hidden_units, result.params(3)];
                                summary.num_layers = [summary.num_layers, result.params(4)];
                                summary.dropout_rates = [summary.dropout_rates, result.params(5)];
                                summary.fc_units = [summary.fc_units, result.params(6)];
                            end
                            
                            % 結果の詳細表示
                            fprintf('\n--- パラメータセット %d/%d ---\n', i, numResults);
                            fprintf('性能指標:\n');
                            fprintf('  - テスト精度: %.4f\n', testAccuracy);
                            
                            if valAccuracy > 0
                                fprintf('  - 検証精度: %.4f\n', valAccuracy);
                            end
                            
                            if cvAccuracy > 0
                                fprintf('  - 交差検証精度: %.4f\n', cvAccuracy);
                                if isfield(result.crossValidation, 'stdAccuracy')
                                    fprintf('    - 標準偏差: %.4f\n', result.crossValidation.stdAccuracy);
                                end
                            end
                            
                            if f1Score > 0
                                fprintf('  - 平均F1スコア: %.4f\n', f1Score);
                            end
                            
                            fprintf('  - 過学習判定: %s\n', string(isOverfit));
                            fprintf('  - 重大度: %s\n', severity);
                            
                            fprintf('複合スコア:\n');
                            fprintf('  - 過学習ペナルティ: %.2f\n', overfitPenalty);
                            fprintf('  - 複雑性ペナルティ: %.2f\n', complexityPenalty);
                            fprintf('  - 最終スコア: %.4f\n', score);
                        else
                            fprintf('\n--- パラメータセット %d/%d: 有効なモデルがありません ---\n', i, numResults);
                        end
                    catch ME
                        fprintf('\n--- パラメータセット %d/%d の評価中にエラーが発生: %s ---\n', i, numResults, ME.message);
                    end
                end
                
                % 有効な結果がない場合はエラーを返す
                if isempty(modelScores)
                    error('有効な評価結果がありません。詳細な解析ができません。');
                end
                
                % 統計サマリーの計算
                if ~isempty(testAccuracies)
                    summary.best_accuracy = max(testAccuracies);
                    summary.worst_accuracy = min(testAccuracies);
                    summary.mean_accuracy = mean(testAccuracies);
                end
                
                % モデル選択の詳細情報
                fprintf('\n=== モデルスコアの分布 ===\n');
                if ~isempty(modelScores)
                    scorePercentiles = prctile(modelScores, [0, 25, 50, 75, 100]);
                    fprintf('  - 最小値: %.4f\n', scorePercentiles(1));
                    fprintf('  - 25パーセンタイル: %.4f\n', scorePercentiles(2));
                    fprintf('  - 中央値: %.4f\n', scorePercentiles(3));
                    fprintf('  - 75パーセンタイル: %.4f\n', scorePercentiles(4));
                    fprintf('  - 最大値: %.4f\n', scorePercentiles(5));
                end
                
                % 最良モデルの選択（最高スコア）
                [bestScore, bestLocalIdx] = max(modelScores);
                if ~isempty(bestLocalIdx) && bestLocalIdx <= length(validIndices)
                    bestIdx = validIndices(bestLocalIdx);
                    bestResults = validResults{bestIdx};
                    
                    fprintf('\n最良モデル選択 (インデックス: %d)\n', bestIdx);
                    fprintf('  - 最終スコア: %.4f\n', bestScore);
                    fprintf('  - テスト精度: %.4f\n', testAccuracies(bestLocalIdx));
                    
                    % 検証精度の表示部分
                    if bestLocalIdx <= length(valAccuracies)
                        fprintf('  - 検証精度: %.4f\n', valAccuracies(bestLocalIdx)/100);
                    end
                    
                    if bestLocalIdx <= length(cvAccuracies) && cvAccuracies(bestLocalIdx) > 0
                        fprintf('  - 交差検証精度: %.4f\n', cvAccuracies(bestLocalIdx));
                    end
                    
                    if bestLocalIdx <= length(f1Scores) && f1Scores(bestLocalIdx) > 0
                        fprintf('  - 平均F1スコア: %.4f\n', f1Scores(bestLocalIdx));
                    end
                    
                    if bestLocalIdx <= length(overfitPenalties)
                        fprintf('  - 過学習ペナルティ: %.2f\n', overfitPenalties(bestLocalIdx));
                    end
                    
                    if bestLocalIdx <= length(complexityPenalties)
                        fprintf('  - 複雑性ペナルティ: %.2f\n', complexityPenalties(bestLocalIdx));
                    end
                    
                    % 最良パラメータの保存
                    if isfield(bestResults, 'params')
                        obj.bestParams = bestResults.params;
                        obj.bestPerformance = bestScore;
                        obj.optimizedModel = bestResults.model;
        
                        % 上位モデルのパラメータ傾向分析
                        topN = min(5, length(validIndices));
                        if topN > 0
                            [~, topLocalIndices] = sort(modelScores, 'descend');
                            topLocalIndices = topLocalIndices(1:topN);
                            topIndices = validIndices(topLocalIndices);
                            
                            % パラメータ情報のあるモデルを集計
                            top_params = [];
                            valid_top_count = 0;
                            
                            for j = 1:length(topIndices)
                                if isfield(validResults{topIndices(j)}, 'params') && ...
                                   length(validResults{topIndices(j)}.params) >= 6
                                    valid_top_count = valid_top_count + 1;
                                    top_params(valid_top_count, :) = validResults{topIndices(j)}.params;
                                end
                            end
                            
                            if valid_top_count > 0
                                fprintf('\n上位%dモデルのパラメータ傾向:\n', valid_top_count);
                                fprintf('  - 平均学習率: %.6f\n', mean(top_params(:, 1)));
                                fprintf('  - 平均バッチサイズ: %.1f\n', mean(top_params(:, 2)));
                                fprintf('  - 平均LSTMユニット数: %.1f\n', mean(top_params(:, 3)));
                                fprintf('  - 平均LSTM層数: %.1f\n', mean(top_params(:, 4)));
                                fprintf('  - 平均ドロップアウト率: %.2f\n', mean(top_params(:, 5)));
                                fprintf('  - 平均FC層ユニット数: %.1f\n', mean(top_params(:, 6)));
                            else
                                fprintf('\n上位モデルに有効なパラメータ情報がありません\n');
                            end
                        else
                            fprintf('\n上位モデル分析に十分な有効結果がありません\n');
                        end
                    else
                        fprintf('\n最良モデルにパラメータ情報がありません\n');
                    end
                else
                    fprintf('\n有効な最良モデルが見つかりませんでした。最初の有効なモデルを使用します。\n');
                    bestResults = validResults{1};
                end
        
            catch ME
                fprintf('結果処理中にエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                
                % 最低限の結果を返す
                if ~isempty(validResults)
                    bestResults = validResults{1};
                else
                    bestResults = struct(...
                        'model', [], ...
                        'performance', struct('accuracy', 0), ...
                        'trainInfo', struct('History', struct('TrainingAccuracy', [], 'ValidationAccuracy', [])), ...
                        'crossValidation', struct('meanAccuracy', 0), ...
                        'overfitting', struct('severity', 'unknown'), ...
                        'normParams', [] ...
                    );
                end
                
                summary = struct(...
                    'total_trials', length(results), ...
                    'valid_trials', length(validResults), ...
                    'overfit_models', 0, ...
                    'best_accuracy', 0, ...
                    'worst_accuracy', 0, ...
                    'mean_accuracy', 0 ...
                );
                
                % エラーを再スローせず、可能な限り処理を続行
                fprintf('警告: エラーが発生しましたが、可能な限り処理を続行します。\n');
            end
        end
        
        function updateOptimizationHistory(obj, results)
            % 最適化の履歴を更新
            
            try
                % 有効な結果のみを抽出
                validResults = results(~cellfun(@isempty, results));
                
                % 履歴に追加
                for i = 1:length(validResults)
                    result = validResults{i};
                    if ~isempty(result.model) && ~isempty(result.performance)
                        % テスト精度と検証精度の取得
                        testAccuracy = result.performance.accuracy;
                        
                        valAccuracy = 0;
                        if isfield(result.trainInfo.History, 'ValidationAccuracy') && ...
                           ~isempty(result.trainInfo.History.ValidationAccuracy)
                            valAcc = result.trainInfo.History.ValidationAccuracy;
                            valAccuracy = mean(valAcc(max(1, end-5):end));
                        end
                        
                        cvAccuracy = 0;
                        if isfield(result, 'crossValidation') && ...
                           isfield(result.crossValidation, 'meanAccuracy')
                            cvAccuracy = result.crossValidation.meanAccuracy;
                        end
                        
                        f1Score = obj.calculateMeanF1Score(result.performance);
                        
                        % 総合スコアの計算
                        score = obj.calculateTrialScore(testAccuracy, valAccuracy, cvAccuracy, f1Score, result);
                        
                        newEntry = struct(...
                            'params', result.params, ...
                            'testAccuracy', testAccuracy, ...
                            'valAccuracy', valAccuracy, ...
                            'cvAccuracy', cvAccuracy, ...
                            'f1Score', f1Score, ...
                            'score', score, ...
                            'model', result.model);
                        
                        if isempty(obj.optimizationHistory)
                            obj.optimizationHistory = newEntry;
                        else
                            obj.optimizationHistory(end+1) = newEntry;
                        end
                    end
                end
                
                % スコアで降順にソート
                if ~isempty(obj.optimizationHistory)
                    [~, sortIdx] = sort([obj.optimizationHistory.score], 'descend');
                    obj.optimizationHistory = obj.optimizationHistory(sortIdx);
                    
                    fprintf('\n最適化履歴を更新しました（計 %d 個のモデル）\n', ...
                        length(obj.optimizationHistory));
                end
                
            catch ME
                fprintf('最適化履歴の更新に失敗: %s', ME.message);
            end
        end
        
        function displayOptimizationSummary(obj, summary)
            % 最適化プロセスの結果サマリーを表示
            
            fprintf('\n=== 最適化プロセスサマリー ===\n');
            fprintf('試行結果:\n');
            fprintf('  - 総試行回数: %d\n', summary.total_trials);
            fprintf('  - 有効な試行: %d\n', summary.valid_trials);
            fprintf('  - 過学習モデル数: %d (%.1f%%)\n', summary.overfit_models, ...
                (summary.overfit_models/max(summary.valid_trials,1))*100);
            
            fprintf('\n精度統計:\n');
            fprintf('  - 最高精度: %.4f\n', summary.best_accuracy);
            fprintf('  - 最低精度: %.4f\n', summary.worst_accuracy);
            fprintf('  - 平均精度: %.4f\n', summary.mean_accuracy);
            
            fprintf('\nパラメータ分布:\n');
            
            % 学習率の統計
            fprintf('\n学習率:\n');
            fprintf('  - 平均: %.6f\n', mean(summary.learning_rates));
            fprintf('  - 標準偏差: %.6f\n', std(summary.learning_rates));
            fprintf('  - 最小: %.6f\n', min(summary.learning_rates));
            fprintf('  - 最大: %.6f\n', max(summary.learning_rates));
            
            % バッチサイズの統計
            fprintf('\nバッチサイズ:\n');
            fprintf('  - 平均: %.1f\n', mean(summary.batch_sizes));
            fprintf('  - 標準偏差: %.1f\n', std(summary.batch_sizes));
            fprintf('  - 最小: %d\n', min(summary.batch_sizes));
            fprintf('  - 最大: %d\n', max(summary.batch_sizes));
            
            % 隠れ層ユニット数の統計
            fprintf('\n隠れ層ユニット数:\n');
            fprintf('  - 平均: %.1f\n', mean(summary.hidden_units));
            fprintf('  - 標準偏差: %.1f\n', std(summary.hidden_units));
            fprintf('  - 最小: %d\n', min(summary.hidden_units));
            fprintf('  - 最大: %d\n', max(summary.hidden_units));
            
            % LSTM層数の統計
            fprintf('\nLSTM層数:\n');
            fprintf('  - 平均: %.1f\n', mean(summary.num_layers));
            fprintf('  - 標準偏差: %.1f\n', std(summary.num_layers));
            fprintf('  - 最小: %d\n', min(summary.num_layers));
            fprintf('  - 最大: %d\n', max(summary.num_layers));
            
            % ドロップアウト率の統計
            fprintf('\nドロップアウト率:\n');
            fprintf('  - 平均: %.3f\n', mean(summary.dropout_rates));
            fprintf('  - 標準偏差: %.3f\n', std(summary.dropout_rates));
            fprintf('  - 最小: %.3f\n', min(summary.dropout_rates));
            fprintf('  - 最大: %.3f\n', max(summary.dropout_rates));
            
            % 全結合層ユニット数の統計
            fprintf('\n全結合層ユニット数:\n');
            fprintf('  - 平均: %.1f\n', mean(summary.fc_units));
            fprintf('  - 標準偏差: %.1f\n', std(summary.fc_units));
            fprintf('  - 最小: %d\n', min(summary.fc_units));
            fprintf('  - 最大: %d\n', max(summary.fc_units));
            
            % パラメータ間の相関分析
            fprintf('\nパラメータ間の相関分析:\n');
            paramMatrix = [summary.learning_rates', summary.batch_sizes', ...
                         summary.hidden_units', summary.num_layers', ...
                         summary.dropout_rates', summary.fc_units'];
                     
            paramNames = {'学習率', 'バッチサイズ', 'LSTMユニット数', ...
                         'LSTM層数', 'ドロップアウト率', '全結合層ユニット数'};
                     
            % 相関行列を計算
            if size(paramMatrix, 1) > 1
                corrMatrix = corr(paramMatrix);
                
                % 強い相関のみを表示
                for i = 1:size(corrMatrix, 1)
                    for j = i+1:size(corrMatrix, 2)
                        if abs(corrMatrix(i,j)) > 0.3
                            fprintf('  - %s と %s: %.3f\n', ...
                                paramNames{i}, paramNames{j}, corrMatrix(i,j));
                        end
                    end
                end
            else
                fprintf('  - 相関分析には複数の有効なモデルが必要です\n');
            end
            
            % 最適化の収束性評価
            if length(obj.optimizationHistory) > 1
                performances = [obj.optimizationHistory.score];
                improvement = diff(performances);
                
                if ~isempty(improvement)
                    positiveImprovement = improvement(improvement > 0);
                    if ~isempty(positiveImprovement)
                        meanImprovement = mean(positiveImprovement);
                    else
                        meanImprovement = 0;
                    end
                    
                    fprintf('\n収束性評価:\n');
                    fprintf('  - 平均改善率: %.6f\n', meanImprovement);
                    fprintf('  - 改善回数: %d/%d\n', sum(improvement > 0), length(improvement));
                    
                    if meanImprovement < 0.001
                        fprintf('  - 状態: 収束\n');
                    else
                        fprintf('  - 状態: 未収束（さらなる最適化の余地あり）\n');
                    end
                end
            end
            
            % LSTM特有の分析
            fprintf('\nLSTM特有の分析:\n');
            
            % 層数とユニット数の関係分析
            layerUnitCorr = corr(summary.num_layers', summary.hidden_units');
            fprintf('  - 層数とユニット数の相関: %.3f\n', layerUnitCorr);
            
            % 性能と各パラメータの相関
            if ~isempty(obj.optimizationHistory) && length(obj.optimizationHistory) > 1
                performances = [obj.optimizationHistory.score]';
                fprintf('\n各パラメータと性能の相関:\n');
                for i = 1:length(paramNames)
                    if length(paramMatrix(:,i)) == length(performances)
                        correlation = corr(paramMatrix(:,i), performances);
                        if abs(correlation) > 0.2
                            fprintf('  - %s: %.3f\n', paramNames{i}, correlation);
                        end
                    end
                end
            end
            
            % 最適パラメータの表示
            if ~isempty(obj.bestParams)
                fprintf('\n=== 最適なパラメータ ===\n');
                fprintf('  - 学習率: %.6f\n', obj.bestParams(1));
                fprintf('  - バッチサイズ: %d\n', obj.bestParams(2));
                fprintf('  - LSTMユニット数: %d\n', obj.bestParams(3));
                fprintf('  - LSTM層数: %d\n', obj.bestParams(4));
                fprintf('  - ドロップアウト率: %.2f\n', obj.bestParams(5));
                fprintf('  - 全結合層ユニット数: %d\n', obj.bestParams(6));
                fprintf('  - 達成スコア: %.4f\n', obj.bestPerformance);
            end
        end
        
        function results = createDefaultResults(obj)
            % 最適化無効時のデフォルト結果構造体を生成
            
            results = struct(...
                'model', [], ...
                'performance', [], ...
                'crossValidation', [], ...
                'trainInfo', [], ...
                'overfitting', [], ...
                'normParams', [] ...
            );
            
            fprintf('最適化スキップ: デフォルト結果を返します\n');
        end
    end
end