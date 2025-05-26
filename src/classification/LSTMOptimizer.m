classdef LSTMOptimizer < handle
    %% LSTMOptimizer - LSTMのハイパーパラメータ最適化クラス
    %
    % このクラスはLSTMの性能を向上させるためにハイパーパラメータの
    % 最適化を行います。複数の探索アルゴリズムをサポートし、バランスの
    % 取れた評価メトリクスに基づいて最適なモデルを選択します。
    %
    % 主な機能:
    %   - Latin Hypercube Sampling, ランダム探索, グリッド探索などの実装
    %   - 複数のパラメータ組み合わせの並列評価
    %   - 過学習と複雑性を考慮した包括的な評価メトリクス
    %   - 早期停止による効率的な探索
    %   - 詳細な最適化結果の分析と可視化
    %
    % 使用方法:
    %   optimizer = LSTMOptimizer(params, verbosity);
    %   results = optimizer.optimize(data, labels);
    %
    % 最適化されるパラメータ:
    %   1. 学習率 (Learning Rate)
    %   2. ミニバッチサイズ (Mini-batch Size)
    %   3. LSTMユニット数 (Number of LSTM Units)
    %   4. LSTM層数 (Number of LSTM Layers)
    %   5. ドロップアウト率 (Dropout Rate)
    %   6. 全結合層ユニット数 (Fully Connected Units)
    
    properties (Access = private)
        params              % システム設定パラメータ構造体
        optimizedModel      % 最適化されたLSTMモデル
        bestParams          % 最良パラメータセット [1x6]
        bestPerformance     % 最良性能値（スコア）
        searchSpace         % パラメータ探索空間の定義
        optimizationHistory % 最適化履歴（全試行の記録）
        useGPU              % GPU使用フラグ (true/false)
        maxTrials           % 最大試行回数
        verbosity           % 出力詳細度 (0:最小限, 1:通常, 2:詳細, 3:デバッグ)

        % 評価重み - モデル選択時の各指標の重要度
        evaluationWeights   % 各評価指標の重みづけ構造体
    end
    
    methods (Access = public)
        %% コンストラクタ - 初期化処理
        function obj = LSTMOptimizer(params, verbosity)
            % LSTMOptimizerのインスタンスを初期化
            %
            % 入力:
            %   params - 設定パラメータ（getConfig関数から取得）
            %   verbosity - 出力詳細度 (0:最小限, 1:通常, 2:詳細, 3:デバッグ)
            %               0: エラーメッセージのみ
            %               1: 基本的な進行状況と結果
            %               2: 詳細な統計情報と分析結果
            %               3: デバッグ情報を含む全ての出力
            
            % パラメータの設定
            obj.params = params;
            
            % 探索空間の初期化
            obj.initializeSearchSpace();
            
            % 最良性能の初期値（最小値）
            obj.bestPerformance = -inf;
            
            % 最適化履歴の初期化（空の構造体配列）
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            
            % GPU設定の読み込み
            obj.useGPU = params.classifier.lstm.gpu;
            
            % 最大試行回数の設定
            obj.maxTrials = params.classifier.lstm.optimization.maxTrials;
            
            % verbosityレベルの設定（デフォルトは1）
            if nargin < 2
                obj.verbosity = 1;  % デフォルトは通常出力
            else
                obj.verbosity = verbosity;
            end
            
            % 評価重みの初期化
            % これらの重みは、モデルの総合スコア計算時に使用される
            obj.evaluationWeights = struct(...
                'test', 0.6, ...        % テスト精度の重み（最も重要）
                'validation', 0.4, ...  % 検証精度の重み（過学習検出用）
                'f1Score', 0.3, ...     % F1スコアの重み（クラス不均衡対策）
                'complexity', 0.1, ...  % 複雑性のペナルティ最大値
                'overfitMax', 0.5 ...   % 過学習の最大ペナルティ値
            );
            
            % 初期化完了メッセージ
            obj.logMessage(3, 'LSTMOptimizer初期化完了\n');
            obj.logMessage(3, '  - 最大試行回数: %d\n', obj.maxTrials);
            obj.logMessage(3, '  - GPU使用: %s\n', mat2str(obj.useGPU));
            obj.logMessage(3, '  - Verbosityレベル: %d\n', obj.verbosity);
        end
        
        %% 最適化実行メソッド - LSTMハイパーパラメータの最適化を実行
        function results = optimize(obj, data, labels)
            % LSTMハイパーパラメータの最適化を実行
            %
            % 入力:
            %   data - 前処理済みEEGデータ [チャンネル x サンプル x エポック]
            %   labels - クラスラベル [エポック x 1]
            %
            % 出力:
            %   results - 最適化結果構造体
            %     .model - 最適化されたLSTMモデル
            %     .performance - 性能評価メトリクス
            %     .trainInfo - 学習情報
            %     .overfitting - 過学習分析結果
            %     .normParams - 正規化パラメータ
            %     .isValid - 有効な結果かどうかのフラグ
            
            try
                % 最適化が有効かチェック
                if ~obj.params.classifier.lstm.optimize
                    obj.logMessage(1, 'LSTM最適化は設定で無効化されています。デフォルトパラメータを使用します。\n');
                    results = obj.createDefaultResults();
                    return;
                end
        
                obj.logMessage(1, '\n=== LSTMハイパーパラメータ最適化を開始 ===\n');
                obj.logMessage(1, '入力データサイズ: [%s]\n', num2str(size(data)));
                obj.logMessage(1, 'クラス数: %d\n', length(unique(labels)));
                
                % 探索アルゴリズムの選択
                searchAlgorithm = 'lhs';  % デフォルト: Latin Hypercube Sampling
                if isfield(obj.params.classifier.lstm.optimization, 'searchAlgorithm')
                    searchAlgorithm = obj.params.classifier.lstm.optimization.searchAlgorithm;
                end
                obj.logMessage(1, '探索アルゴリズム: %s\n', searchAlgorithm);
                
                % パラメータセットの生成
                obj.logMessage(2, '\nパラメータ探索空間を設定中...\n');
                paramSets = obj.generateParameterSets(obj.maxTrials, searchAlgorithm);
                obj.logMessage(1, 'パラメータ%dセットで最適化を開始します\n', size(paramSets, 1));
        
                % ========== 最適化ループの準備 ==========
                % 結果保存用配列の初期化
                trialResults = cell(size(paramSets, 1), 1);
                baseParams = obj.params; % ベースパラメータの保存
                
                % 有効な評価結果カウンタ
                validResultsCount = 0;
                
                % 実行時間の記録開始
                startTime = tic;
        
                % 各パラメータセットで評価
                for i = 1:size(paramSets, 1)
                    try
                        obj.logMessage(1, '\n--- パラメータセット %d/%d の評価 ---\n', i, size(paramSets, 1));
                        
                        % 現在のパラメータ構成の表示
                        obj.displayCurrentParams(paramSets(i,:));
                        
                        % パラメータの更新（ベースパラメータをコピーして更新）
                        localParams = baseParams;
                        localParams = obj.updateLSTMParameters(localParams, paramSets(i,:));
        
                        % LSTMの学習と評価
                        obj.logMessage(2, 'LSTMモデルの学習を開始...\n');
                        lstm = LSTMClassifier(localParams, obj.verbosity);
                        trainResults = lstm.trainLSTM(data, labels);
        
                        % 結果の保存
                        trialResults{i} = struct(...
                            'params', paramSets(i,:), ...      % 使用したパラメータ
                            'model', trainResults.model, ...    % 学習済みモデル
                            'performance', trainResults.performance, ... % 性能評価
                            'trainInfo', trainResults.trainInfo, ...     % 学習情報
                            'overfitting', trainResults.overfitting, ... % 過学習分析
                            'normParams', trainResults.normParams, ...   % 正規化パラメータ
                            'error', false ...  % エラーフラグ：正常
                        );
                        validResultsCount = validResultsCount + 1;
        
                        % モデル性能の評価
                        % テスト精度の取得
                        performance = trainResults.performance.accuracy;
                        
                        % 検証精度データの取得（過学習の指標）
                        valAccuracy = 0;
                        if isfield(trainResults.trainInfo.History, 'ValidationAccuracy') && ...
                           ~isempty(trainResults.trainInfo.History.ValidationAccuracy)
                            valAcc = trainResults.trainInfo.History.ValidationAccuracy;
                            % 最後の30エポックの平均を使用（安定性のため）
                            valAccuracy = mean(valAcc(max(1, end-30):end));
                        end
                        
                        % F1スコア計算（クラス不均衡対策）
                        f1Score = obj.calculateMeanF1Score(trainResults.performance);
                        
                        % 総合評価スコアの計算
                        evaluationScore = obj.calculateTrialScore(performance, valAccuracy, f1Score, trainResults);
        
                        obj.logMessage(1, '組み合わせ %d/%d: テスト精度 = %.4f, 総合スコア = %.4f\n', ...
                            i, size(paramSets, 1), performance, evaluationScore);
                        
                        % 経過時間の表示（verbosity >= 2）
                        if obj.verbosity >= 2
                            elapsedTime = toc(startTime);
                            obj.logMessage(2, '経過時間: %.1f秒\n', elapsedTime);
                        end
                        
                        % GPUメモリの解放（GPU使用時）
                        if obj.useGPU
                            gpuDevice([]);
                            obj.logMessage(3, 'GPUメモリを解放しました\n');
                        end
        
                    catch ME
                        % エラー処理
                        obj.logMessage(0, '組み合わせ%dでエラー発生: %s\n', i, ME.message);
                        obj.logMessage(2, 'エラー詳細:\n');
                        
                        if obj.verbosity >= 2
                            disp(getReport(ME, 'extended'));
                        end
                        
                        % エラー発生時でも最低限のパラメータ情報は保存
                        trialResults{i} = struct(...
                            'params', paramSets(i,:), ...
                            'error', true, ...              % エラーフラグ
                            'errorMessage', ME.message, ... % エラーメッセージ
                            'performance', struct('accuracy', 0) ... % ダミー性能値
                        );
        
                        % GPUメモリの解放（エラー時も実行）
                        if obj.useGPU
                            try
                                gpuDevice([]);
                            catch
                                % GPU解放エラーは無視
                            end
                        end
                    end
                end
                
                % 結果の集計と最良モデルの選択
                % 有効な結果がない場合のチェック
                if validResultsCount == 0
                    obj.logMessage(0, '\n警告: 有効な最適化結果がありません。すべての試行が失敗しました。\n');
                    obj.logMessage(0, 'デフォルト設定を使用してLSTMモデルを生成します。\n');
                    
                    % デフォルト結果を生成して返す
                    results = obj.createDefaultResults();
                    results.isValid = false;  % これは有効な最適化結果ではないフラグ
                    
                    % 警告を出力して戻る
                    warning('LSTMOptimizer:NoValidResults', '有効な最適化結果がありません');
                    return;
                end
        
                % 最良の結果を選択
                obj.logMessage(1, '\n最適化結果を処理中...\n');
                [bestResults, summary] = obj.processFinalResults(trialResults);
                
                % 最適化履歴の更新
                obj.updateOptimizationHistory(trialResults);
                
                % 最適化サマリーの表示
                obj.displayOptimizationSummary(summary);
                
                % 最終結果の構築
                % 結果構造体の作成
                results = struct(...
                    'model', bestResults.model, ...         % 最適化されたモデル
                    'performance', bestResults.performance, ... % 性能評価
                    'trainInfo', bestResults.trainInfo, ...     % 学習情報
                    'overfitting', bestResults.overfitting, ... % 過学習分析
                    'normParams', bestResults.normParams, ...   % 正規化パラメータ
                    'isValid', true ...  % これは正常な最適化結果フラグ
                );
                
                % 最適化完了時間の表示
                totalTime = toc(startTime);
                obj.logMessage(1, '\n=== LSTM最適化が完了しました ===\n');
                obj.logMessage(1, '総実行時間: %.1f秒 (%.1f分)\n', totalTime, totalTime/60);
        
            catch ME
                % 致命的エラーのハンドリング
                obj.logMessage(0, '\n=== LSTM最適化中にエラーが発生しました ===\n');
                obj.logMessage(0, 'エラーメッセージ: %s\n', ME.message);
                obj.logMessage(2, 'エラースタック:\n');
                
                if obj.verbosity >= 2
                    disp(getReport(ME, 'extended'));
                end
                
                % 致命的なエラーでもデフォルト結果を返す
                obj.logMessage(0, '致命的なエラーが発生しました。デフォルト設定を使用します。\n');
                results = obj.createDefaultResults();
                results.isValid = false;  % 無効な結果フラグ
                results.error = true;     % エラーフラグ
                results.errorMessage = ME.message; % エラーメッセージ
                
                % エラーを再スローしない（処理を継続）
                warning('LSTMOptimizer:OptimizationFailed', 'LSTM最適化に失敗: %s', ME.message);
            end
        end
        
        %% ログ出力メソッド - verbosityレベルに応じたメッセージ出力
        function logMessage(obj, level, format, varargin)
            % 指定されたverbosityレベル以上の場合にメッセージを出力
            %
            % 入力:
            %   level - メッセージの重要度 
            %     0: エラー（常に表示）
            %     1: 警告/通常メッセージ
            %     2: 情報/詳細メッセージ
            %     3: デバッグメッセージ
            %   format - fprintf形式の文字列
            %   varargin - 追加パラメータ
            %
            % 使用例:
            %   obj.logMessage(1, '処理開始: %s\n', datestr(now));
            %   obj.logMessage(3, 'デバッグ: 変数値 = %d\n', value);
            
            if obj.verbosity >= level
                fprintf(format, varargin{:});
            end
        end
    end
    
    methods (Access = private)
        %% 探索空間初期化メソッド - パラメータ探索範囲の設定
        function initializeSearchSpace(obj)
            % パラメータ探索空間を設定
            % 各パラメータの最小値と最大値を定義
            
            % パラメータ設定から探索空間を取得
            if isfield(obj.params.classifier.lstm.optimization, 'searchSpace')
                obj.searchSpace = obj.params.classifier.lstm.optimization.searchSpace;
                obj.logMessage(3, '設定ファイルから探索空間を読み込みました\n');
            else
                % デフォルト探索空間の定義
                % これらの範囲は、LSTMモデルで一般的に有効とされる値に基づいている
                obj.searchSpace = struct(...
                    'learningRate', [0.0001, 0.01], ...    % 学習率範囲（対数スケール）
                    'miniBatchSize', [16, 128], ...        % バッチサイズ範囲（2の累乗）
                    'lstmUnits', [32, 256], ...            % LSTM ユニット数範囲
                    'numLayers', [1, 3], ...               % LSTM 層数範囲（深すぎると勾配消失）
                    'dropoutRate', [0.2, 0.7], ...         % ドロップアウト率範囲
                    'fcUnits', [32, 256] ...               % 全結合層ユニット数範囲
                );
                obj.logMessage(3, 'デフォルト探索空間を使用します\n');
            end
            
            % 探索空間の表示（デバッグレベル）
            if obj.verbosity >= 3
                obj.logMessage(3, '探索空間:\n');
                fields = fieldnames(obj.searchSpace);
                for i = 1:length(fields)
                    range = obj.searchSpace.(fields{i});
                    obj.logMessage(3, '  - %s: [%.4f, %.4f]\n', fields{i}, range(1), range(2));
                end
            end
        end
        
        %% パラメータセット生成メソッド - 探索アルゴリズムに基づくパラメータ生成
        function paramSets = generateParameterSets(obj, numTrials, algorithm)
            % 指定されたアルゴリズムでパラメータセットを生成
            %
            % 入力:
            %   numTrials - 試行回数
            %   algorithm - 探索アルゴリズム ('lhs', 'random', 'grid')
            %
            % 出力:
            %   paramSets - パラメータセット行列 [numTrials x 6]
            %     各行: [学習率, バッチサイズ, LSTMユニット数, 層数, ドロップアウト率, FC層ユニット数]
            
            % 探索アルゴリズムの選択
            if nargin < 3
                algorithm = 'lhs';  % デフォルトはLatin Hypercube Sampling
            end
            
            obj.logMessage(2, 'パラメータ探索アルゴリズム: %s\n', algorithm);
            
            % アルゴリズムに応じてパラメータセットを生成
            switch lower(algorithm)
                case 'lhs'  % Latin Hypercube Sampling（推奨）
                    % 効率的な空間探索が可能
                    paramSets = obj.generateLHSParams(numTrials);
                    
                case 'random'  % ランダムサンプリング
                    % 単純だが偏りの可能性あり
                    paramSets = obj.generateRandomParams(numTrials);
                    
                case 'grid'  % グリッドサーチ
                    % 網羅的だが計算コストが高い
                    paramSets = obj.generateGridParams();
                    
                otherwise
                    % 未知のアルゴリズム
                    obj.logMessage(1, '未知の探索アルゴリズム: %s。LHSにフォールバックします。\n', algorithm);
                    paramSets = obj.generateLHSParams(numTrials);
            end
            
            obj.logMessage(2, '探索パラメータセット数: %d\n', size(paramSets, 1));
            
            % パラメータセットの統計情報表示（デバッグレベル）
            if obj.verbosity >= 3
                obj.logMessage(3, '\nパラメータセットの統計:\n');
                paramNames = {'学習率', 'バッチサイズ', 'LSTMユニット数', ...
                            'LSTM層数', 'ドロップアウト率', '全結合層ユニット数'};
                for i = 1:size(paramSets, 2)
                    obj.logMessage(3, '  %s: 平均=%.3f, 標準偏差=%.3f\n', ...
                        paramNames{i}, mean(paramSets(:,i)), std(paramSets(:,i)));
                end
            end
        end
        
        %% LHS (Latin Hypercube Sampling) パラメータ生成
        function paramSets = generateLHSParams(obj, numTrials)
            % Latin Hypercube Samplingによるパラメータセット生成
            % 多次元空間を効率的に探索するためのサンプリング手法
            %
            % 入力:
            %   numTrials - 生成するパラメータセット数
            %
            % 出力:
            %   paramSets - パラメータセット行列 [numTrials x 6]
            
            obj.logMessage(3, 'Latin Hypercube Sampling開始 (試行数: %d)\n', numTrials);
            
            % 6個のパラメータを Latin Hypercube Sampling で生成
            % lhsdesignは[0,1]の一様分布から効率的にサンプリング
            lhsPoints = lhsdesign(numTrials, 6);
            paramSets = zeros(numTrials, 6);
            
            % 1. 学習率（対数スケールで探索）
            % 対数スケールを使用することで、小さい値も大きい値も均等に探索
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + ...
                (log10(lr_range(2))-log10(lr_range(1))) * lhsPoints(:,1));
            
            % 2. ミニバッチサイズ（整数値に丸める）
            bs_range = obj.searchSpace.miniBatchSize;
            paramSets(:,2) = round(bs_range(1) + (bs_range(2)-bs_range(1)) * lhsPoints(:,2));
            
            % 3. LSTMユニット数（整数値に丸める）
            lu_range = obj.searchSpace.lstmUnits;
            paramSets(:,3) = round(lu_range(1) + (lu_range(2)-lu_range(1)) * lhsPoints(:,3));
            
            % 4. LSTM層数（整数値に丸める）
            nl_range = obj.searchSpace.numLayers;
            paramSets(:,4) = round(nl_range(1) + (nl_range(2)-nl_range(1)) * lhsPoints(:,4));
            
            % 5. ドロップアウト率（0-1の実数値）
            do_range = obj.searchSpace.dropoutRate;
            paramSets(:,5) = do_range(1) + (do_range(2)-do_range(1)) * lhsPoints(:,5);
            
            % 6. 全結合層ユニット数（整数値に丸める）
            fc_range = obj.searchSpace.fcUnits;
            paramSets(:,6) = round(fc_range(1) + (fc_range(2)-fc_range(1)) * lhsPoints(:,6));
            
            obj.logMessage(3, 'LHS パラメータ生成完了\n');
        end
        
        %% ランダムサンプリングパラメータ生成
        function paramSets = generateRandomParams(obj, numTrials)
            % ランダムサンプリングによるパラメータセット生成
            % 単純なランダム探索（比較用）
            %
            % 入力:
            %   numTrials - 生成するパラメータセット数
            %
            % 出力:
            %   paramSets - パラメータセット行列 [numTrials x 6]
            
            obj.logMessage(3, 'ランダムサンプリング開始 (試行数: %d)\n', numTrials);
            
            paramSets = zeros(numTrials, 6);
            
            % 各パラメータを独立にランダムサンプリング
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
            
            obj.logMessage(3, 'ランダムパラメータ生成完了\n');
        end
        
        %% グリッドサーチパラメータ生成
        function paramSets = generateGridParams(obj)
            % グリッドサーチによるパラメータセット生成
            % 網羅的探索（計算コストが高い）
            %
            % 出力:
            %   paramSets - パラメータセット行列 [総組み合わせ数 x 6]
            
            obj.logMessage(3, 'グリッドサーチパラメータ生成開始\n');
            
            % グリッドポイント数の決定（計算量とのトレードオフ）
            numGridPoints = 3;  % 各パラメータのグリッド数
            
            % 各パラメータのグリッド生成
            % 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            lr_grid = logspace(log10(lr_range(1)), log10(lr_range(2)), numGridPoints);
            
            % バッチサイズ
            bs_range = obj.searchSpace.miniBatchSize;
            bs_grid = round(linspace(bs_range(1), bs_range(2), numGridPoints));
            
            % LSTMユニット数
            lu_range = obj.searchSpace.lstmUnits;
            lu_grid = round(linspace(lu_range(1), lu_range(2), numGridPoints));
            
            % LSTM層数（少ないので2点のみ）
            nl_range = obj.searchSpace.numLayers;
            nl_grid = round(linspace(nl_range(1), nl_range(2), 2));
            
            % ドロップアウト率
            do_range = obj.searchSpace.dropoutRate;
            do_grid = linspace(do_range(1), do_range(2), numGridPoints);
            
            % 全結合層ユニット数
            fc_range = obj.searchSpace.fcUnits;
            fc_grid = round(linspace(fc_range(1), fc_range(2), numGridPoints));
            
            % グリッドの全組み合わせを生成
            [LR, BS, LU, NL, DO, FC] = ndgrid(lr_grid, bs_grid, lu_grid, nl_grid, do_grid, fc_grid);
            
            % 行列に変換
            paramSets = [LR(:), BS(:), LU(:), NL(:), DO(:), FC(:)];
            
            % グリッド数が多すぎる場合は警告
            if size(paramSets, 1) > 200
                obj.logMessage(1, '警告: グリッドサーチのパラメータ数が非常に多いです (%d組)。', ...
                    size(paramSets, 1));
                obj.logMessage(1, '最適化には長時間かかる可能性があります。\n');
            end
            
            obj.logMessage(2, 'グリッドサーチパラメータ数: %d\n', size(paramSets, 1));
        end
        
        %% LSTMパラメータ更新メソッド
        function params = updateLSTMParameters(~, params, paramSet)
            % LSTMパラメータを更新
            % ベースパラメータ構造体に新しいパラメータ値を適用
            %
            % 入力:
            %   params - ベースパラメータ構造体
            %   paramSet - 新しいパラメータ値 [1x6]
            %     [学習率, バッチサイズ, LSTMユニット数, 層数, ドロップアウト率, FC層ユニット数]
            %
            % 出力:
            %   params - 更新されたパラメータ構造体
            
            % 1. 学習率とミニバッチサイズの更新
            params.classifier.lstm.training.optimizer.learningRate = paramSet(1);
            params.classifier.lstm.training.miniBatchSize = paramSet(2);
            
            % 2. LSTM層の構築
            % 動的に層数を変更し、各層のパラメータを設定
            lstmUnits = round(paramSet(3));
            numLayers = round(paramSet(4));
            lstmLayers = struct();
            
            for j = 1:numLayers
                if j == numLayers
                    % 最終層は 'last' モード（時系列の最後の出力のみ）
                    lstmLayers.(sprintf('lstm%d', j)) = struct(...
                        'numHiddenUnits', lstmUnits, ...
                        'OutputMode', 'last' ...
                    );
                else
                    % 中間層は 'sequence' モード（全時系列を出力）
                    lstmLayers.(sprintf('lstm%d', j)) = struct(...
                        'numHiddenUnits', lstmUnits, ...
                        'OutputMode', 'sequence' ...
                    );
                end
            end
            params.classifier.lstm.architecture.lstmLayers = lstmLayers;
            
            % 3. ドロップアウト層の更新
            % 各LSTM層の後にドロップアウトを配置
            dropoutLayers = struct();
            for j = 1:numLayers
                dropoutLayers.(sprintf('dropout%d', j)) = paramSet(5);
            end
            params.classifier.lstm.architecture.dropoutLayers = dropoutLayers;
            
            % 4. 全結合層ユニット数の更新
            params.classifier.lstm.architecture.fullyConnected = [round(paramSet(6))];
        end
        
        %% 現在のパラメータ表示メソッド
        function displayCurrentParams(obj, paramSet)
            % 現在評価中のパラメータセットを表示
            % verbosityレベル2以上で詳細情報を出力
            %
            % 入力:
            %   paramSet - パラメータ値 [1x6]
            
            obj.logMessage(2, '評価対象のパラメータ構成:\n');
            obj.logMessage(2, '  - 学習率: %.6f\n', paramSet(1));
            obj.logMessage(2, '  - バッチサイズ: %d\n', paramSet(2));
            obj.logMessage(2, '  - LSTMユニット数: %d\n', paramSet(3));
            obj.logMessage(2, '  - LSTM層数: %d\n', paramSet(4));
            obj.logMessage(2, '  - ドロップアウト率: %.2f\n', paramSet(5));
            obj.logMessage(2, '  - 全結合層ユニット数: %d\n', paramSet(6));
            
            % モデルの複雑性の推定（デバッグレベル）
            if obj.verbosity >= 3
                totalParams = paramSet(3) * paramSet(4) + paramSet(6);  % 概算
                obj.logMessage(3, '  - 推定パラメータ数: 約%d\n', totalParams);
            end
        end
        
        %% 平均F1スコア計算メソッド
        function f1Score = calculateMeanF1Score(obj, performance)
            % 各クラスのF1スコアの平均を計算
            % クラス不均衡データセットでの性能評価に有効
            %
            % 入力:
            %   performance - 性能評価構造体
            %
            % 出力:
            %   f1Score - 平均F1スコア（0-1の値）
            
            f1Score = 0;
            
            if isfield(performance, 'classwise') && ~isempty(performance.classwise)
                % 各クラスのF1スコアを取得
                f1Scores = zeros(1, length(performance.classwise));
                for i = 1:length(performance.classwise)
                    f1Scores(i) = performance.classwise(i).f1score;
                end
                
                % マクロ平均（各クラスを均等に重視）
                f1Score = mean(f1Scores);
                
                obj.logMessage(3, '平均F1スコア: %.4f (クラス数: %d)\n', ...
                    f1Score, length(performance.classwise));
            else
                obj.logMessage(3, 'F1スコア情報が利用できません\n');
            end
        end
        
        %% モデル評価スコア計算メソッド
        function score = calculateTrialScore(obj, testAccuracy, valAccuracy, f1Score, results)
            % モデルの総合評価スコアを計算
            % 複数の評価指標を統合し、過学習と複雑性を考慮
            %
            % 入力:
            %   testAccuracy - テスト精度（0-1）
            %   valAccuracy - 検証精度（0-100、履歴から計算）
            %   f1Score - F1スコア（0-1）
            %   results - 評価結果全体（過学習情報含む）
            %
            % 出力:
            %   score - 総合評価スコア（0-1、高いほど良い）
            
            obj.logMessage(3, '総合評価スコア計算開始\n');
            
            % 1. 基本精度スコアの計算
            testWeight = obj.evaluationWeights.test;
            valWeight = obj.evaluationWeights.validation;
            
            % 検証精度の正規化（0-100から0-1へ）
            validationScore = testAccuracy; % デフォルトはテスト精度
            if valAccuracy > 0
                validationScore = valAccuracy / 100;  % パーセントから小数へ
            else
                % 検証精度がない場合はテスト精度のみで評価
                valWeight = 0;
            end
            
            % 加重平均による基本スコア
            accuracyScore = (testWeight * testAccuracy + valWeight * validationScore) / ...
                          (testWeight + valWeight);
            
            obj.logMessage(3, '  基本精度スコア: %.4f\n', accuracyScore);
            
            % 2. F1スコアの統合
            % クラス不均衡への対応
            if f1Score > 0
                f1Weight = obj.evaluationWeights.f1Score;
                combinedScore = (1 - f1Weight) * accuracyScore + f1Weight * f1Score;
                obj.logMessage(3, '  F1統合後スコア: %.4f\n', combinedScore);
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
                    % 重大度ベースのペナルティ（フォールバック）
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
                obj.logMessage(3, '  過学習ペナルティ: %.4f\n', overfitPenalty);
            end
            
            % 4. モデル複雑性ペナルティの計算
            complexityPenalty = 0.05; % デフォルト値
            if isfield(results, 'params') && length(results.params) >= 6
                lstmUnits = results.params(3);
                numLayers = results.params(4);
                fcUnits = results.params(6);
                
                % 探索空間の最大値を参照して相対的な複雑さを計算
                maxUnits = obj.searchSpace.lstmUnits(2);
                maxLayers = obj.searchSpace.numLayers(2);
                maxFCUnits = obj.searchSpace.fcUnits(2);
                
                % 重み付き複雑性スコア
                complexityScore = 0.4 * (lstmUnits / maxUnits) + ...     % LSTMユニットの影響：40%
                                0.4 * (numLayers / maxLayers) + ...       % 層数の影響：40%
                                0.2 * (fcUnits / maxFCUnits);             % FC層の影響：20%
                                
                complexityPenalty = obj.evaluationWeights.complexity * complexityScore;
                obj.logMessage(3, '  複雑性ペナルティ: %.4f\n', complexityPenalty);
            else
                obj.logMessage(3, '  注意: パラメータ情報がないため、デフォルトの複雑性ペナルティを使用\n');
            end
            
            % 5. 最終スコアの計算
            % スコア = 基本スコア × (1 - 過学習ペナルティ - 複雑性ペナルティ)
            score = combinedScore * (1 - overfitPenalty - complexityPenalty);
            
            obj.logMessage(3, '  最終評価スコア: %.4f\n', score);
            
            % スコアの妥当性チェック
            if score < 0 || score > 1
                obj.logMessage(1, '警告: 評価スコアが範囲外です: %.4f\n', score);
                score = max(0, min(1, score));  % クリッピング
            end
                
            return;
        end

        %% 最終結果処理メソッド
        function [bestResults, summary] = processFinalResults(obj, results)
            % 全ての試行結果を処理し、最良のモデルを選択
            % 詳細な統計分析と最適モデルの選定を実行
            %
            % 入力:
            %   results - 全試行結果のセル配列
            %
            % 出力:
            %   bestResults - 最良モデルの結果構造体
            %   summary - 最適化プロセスのサマリー統計
            
            try
                obj.logMessage(1, '\n=== パラメータ最適化の結果処理 ===\n');
                obj.logMessage(1, '総試行回数: %d\n', length(results));
                
                % 有効結果の抽出
                % 空でない結果のみを抽出
                validResults = results(~cellfun(@isempty, results));
                numResults = length(validResults);
                
                obj.logMessage(1, '有効なパラメータセット数: %d\n', numResults);
                obj.logMessage(1, '無効な試行数: %d\n', length(results) - numResults);
                
                % モデルの有効性チェック
                validModelCount = 0;
                for i = 1:numResults
                    if ~isfield(validResults{i}, 'error') || ~validResults{i}.error
                        if isfield(validResults{i}, 'model') && ~isempty(validResults{i}.model)
                            validModelCount = validModelCount + 1;
                        end
                    end
                end
                
                obj.logMessage(1, '有効なモデル数: %d\n', validModelCount);
                
                % 結果がない場合のエラー処理
                if validModelCount == 0
                    obj.logMessage(0, '有効なモデル結果がありません。デフォルト値を使用します。\n');
                    
                    % デフォルト結果の構造体
                    bestResults = struct(...
                        'model', [], ...
                        'performance', struct('accuracy', 0), ...
                        'trainInfo', struct('History', struct(...
                            'TrainingAccuracy', [], ...
                            'ValidationAccuracy', [])), ...
                        'overfitting', struct('severity', 'unknown'), ...
                        'normParams', [], ...
                        'params', [], ...
                        'isValid', false ...  % 有効な結果ではない
                    );
                    
                    % 空のサマリー情報
                    summary = obj.createEmptySummary(length(results), numResults, validModelCount);
                    return;
                end
                
                % 評価結果の集計
                % 評価結果保存用の配列を事前割り当て（効率化）
                modelScores = zeros(numResults, 1);
                testAccuracies = zeros(numResults, 1);
                valAccuracies = zeros(numResults, 1);
                f1Scores = zeros(numResults, 1); 
                overfitPenalties = zeros(numResults, 1);
                complexityPenalties = zeros(numResults, 1);
                validIndices = zeros(numResults, 1);
                
                % サマリー情報の初期化
                summary = obj.initializeSummary(length(results), numResults, validModelCount);
                
                % 有効な結果数を追跡するカウンタ
                validCount = 0;
                
                obj.logMessage(2, '\n=== 各試行の詳細評価 ===\n');
                
                % 各結果の評価
                for i = 1:numResults
                    result = validResults{i};
                    try
                        % エラーチェック
                        if isfield(result, 'error') && result.error
                            obj.logMessage(2, '\n--- パラメータセット %d/%d: エラーあり (%s) ---\n', ...
                                i, numResults, result.errorMessage);
                            continue;
                        end
                        
                        % 必須フィールドの存在確認
                        if ~obj.validateResult(result, i, numResults)
                            continue;
                        end
                    
                        % 性能指標の抽出
                        % 基本的な精度スコア
                        testAccuracy = result.performance.accuracy;
                        
                        % 検証精度の取得
                        valAccuracy = obj.extractValidationAccuracy(result);
                        
                        % F1スコアの計算
                        f1Score = obj.calculateMeanF1Score(result.performance);
                        
                        % ペナルティの計算
                        [overfitPenalty, complexityPenalty] = obj.calculatePenalties(result);
                        
                        % 総合スコアの計算
                        score = obj.calculateTrialScore(testAccuracy, valAccuracy, f1Score, result);
                        
                        % 結果の保存
                        validCount = validCount + 1;
                        testAccuracies(validCount) = testAccuracy;
                        valAccuracies(validCount) = valAccuracy;
                        f1Scores(validCount) = f1Score;
                        overfitPenalties(validCount) = overfitPenalty;
                        complexityPenalties(validCount) = complexityPenalty;
                        modelScores(validCount) = score;
                        validIndices(validCount) = i;
                        
                        % 過学習統計の更新
                        if obj.isOverfitted(result)
                            summary.overfit_models = summary.overfit_models + 1;
                        end
                        
                        % パラメータ統計の更新
                        obj.updateSummaryParams(summary, result, validCount);
                        
                        % 詳細結果の表示（verbosity >= 2）
                        obj.displayTrialResults(i, numResults, testAccuracy, valAccuracy, ...
                            f1Score, overfitPenalty, complexityPenalty, score, result);
                        
                    catch ME
                        obj.logMessage(1, '\n--- パラメータセット %d/%d の評価中にエラーが発生: %s ---\n', ...
                            i, numResults, ME.message);
                    end
                end
                
                % 配列サイズの調整
                if validCount > 0
                    testAccuracies = testAccuracies(1:validCount);
                    valAccuracies = valAccuracies(1:validCount);
                    f1Scores = f1Scores(1:validCount);
                    overfitPenalties = overfitPenalties(1:validCount);
                    complexityPenalties = complexityPenalties(1:validCount);
                    modelScores = modelScores(1:validCount);
                    validIndices = validIndices(1:validCount);
                    
                    % サマリー配列も調整
                    obj.adjustSummaryArrays(summary, validCount);
                end
                
                % 有効な結果がない場合
                if validCount == 0
                    obj.logMessage(0, '有効な評価結果がありません。デフォルト設定を使用します。\n');
                    bestResults = obj.createDefaultBestResults();
                    return;
                end
                
                % 統計サマリーの計算
                obj.calculateSummaryStatistics(summary, testAccuracies);
                
                % モデル選択の詳細情報表示
                obj.displayScoreDistribution(modelScores);
                
                % 最良モデルの選択
                bestResults = obj.selectBestModel(modelScores, validIndices, ...
                    validResults, testAccuracies, valAccuracies, f1Scores, ...
                    overfitPenalties, complexityPenalties);
                
                % 上位モデルの分析
                if obj.verbosity >= 2
                    obj.analyzeTopModels(modelScores, validIndices, validResults);
                end
        
            catch ME
                obj.logMessage(0, '結果処理中にエラーが発生: %s\n', ME.message);
                
                if obj.verbosity >= 2
                    obj.logMessage(2, 'エラー詳細:\n');
                    disp(getReport(ME, 'extended'));
                end
                
                % エラー時のフォールバック
                bestResults = obj.createDefaultBestResults();
                summary = obj.createEmptySummary(length(results), 0, 0);
                
                obj.logMessage(1, '警告: エラーが発生しましたが、可能な限り処理を続行します。\n');
            end
        end
        
        %% 結果検証ヘルパーメソッド
        function isValid = validateResult(obj, result, idx, total)
            % 個別結果の妥当性を検証
            isValid = true;
            
            % モデルの存在確認
            if ~isfield(result, 'model') || isempty(result.model)
                obj.logMessage(2, '\n--- パラメータセット %d/%d: モデルが空または存在しません ---\n', idx, total);
                isValid = false;
                return;
            end
            
            % 性能データの存在確認
            if ~isfield(result, 'performance') || isempty(result.performance)
                obj.logMessage(2, '\n--- パラメータセット %d/%d: パフォーマンスデータが空または存在しません ---\n', idx, total);
                isValid = false;
                return;
            end
        end
        
        %% 検証精度抽出ヘルパーメソッド
        function valAccuracy = extractValidationAccuracy(obj, result)
            % 検証精度履歴から安定した値を抽出
            valAccuracy = 0;
            
            if isfield(result, 'trainInfo') && isfield(result.trainInfo, 'History')
                if isfield(result.trainInfo.History, 'ValidationAccuracy') && ...
                   ~isempty(result.trainInfo.History.ValidationAccuracy)
                    valAcc = result.trainInfo.History.ValidationAccuracy;
                    % 最後の30エポックの平均（安定性のため）
                    valAccuracy = mean(valAcc(max(1, end-30):end));
                    obj.logMessage(3, '  検証精度（安定区間平均）: %.4f\n', valAccuracy/100);
                end
            end
        end
        
        %% ペナルティ計算ヘルパーメソッド
        function [overfitPenalty, complexityPenalty] = calculatePenalties(obj, result)
            % 過学習と複雑性のペナルティを計算
            
            % 過学習ペナルティ
            overfitPenalty = 0;
            if isfield(result, 'overfitting')
                if isfield(result.overfitting, 'performanceGap')
                    perfGap = result.overfitting.performanceGap / 100;
                    overfitPenalty = min(obj.evaluationWeights.overfitMax, perfGap);
                elseif isfield(result.overfitting, 'severity')
                    severity = result.overfitting.severity;
                    switch severity
                        case 'critical'
                            overfitPenalty = 0.5;
                        case 'severe'
                            overfitPenalty = 0.3;
                        case 'moderate'
                            overfitPenalty = 0.2;
                        case 'mild'
                            overfitPenalty = 0.1;
                        otherwise
                            overfitPenalty = 0;
                    end
                end
            end
            
            % 複雑性ペナルティ
            complexityPenalty = 0.05; % デフォルト値
            if isfield(result, 'params') && length(result.params) >= 6
                lstmUnits = result.params(3);
                numLayers = result.params(4);
                fcUnits = result.params(6);
                
                maxUnits = obj.searchSpace.lstmUnits(2);
                maxLayers = obj.searchSpace.numLayers(2);
                maxFCUnits = obj.searchSpace.fcUnits(2);
                
                complexityScore = 0.4 * (lstmUnits / maxUnits) + ...
                                0.4 * (numLayers / maxLayers) + ...
                                0.2 * (fcUnits / maxFCUnits);
                complexityPenalty = obj.evaluationWeights.complexity * complexityScore;
            end
        end
        
        %% 過学習判定ヘルパーメソッド
        function isOverfit = isOverfitted(~, result)
            % モデルが過学習しているかを判定
            isOverfit = false;
            
            if isfield(result, 'overfitting') && isfield(result.overfitting, 'severity')
                severity = result.overfitting.severity;
                isOverfit = ismember(severity, {'critical', 'severe', 'moderate'});
            end
        end
        
        %% サマリー初期化メソッド
        function summary = initializeSummary(~, totalTrials, validTrials, validModels)
            % サマリー構造体の初期化
            summary = struct(...
                'total_trials', totalTrials, ...
                'valid_trials', validTrials, ...
                'valid_models', validModels, ...
                'overfit_models', 0, ...
                'best_accuracy', 0, ...
                'worst_accuracy', 1, ...
                'mean_accuracy', 0, ...
                'learning_rates', zeros(1, validTrials), ...
                'batch_sizes', zeros(1, validTrials), ...
                'hidden_units', zeros(1, validTrials), ...
                'num_layers', zeros(1, validTrials), ...
                'dropout_rates', zeros(1, validTrials), ...
                'fc_units', zeros(1, validTrials));
        end
        
        %% 空のサマリー作成メソッド
        function summary = createEmptySummary(~, totalTrials, validTrials, validModels)
            % 空のサマリー構造体を作成
            summary = struct(...
                'total_trials', totalTrials, ...
                'valid_trials', validTrials, ...
                'valid_models', validModels, ...
                'overfit_models', 0, ...
                'best_accuracy', 0, ...
                'worst_accuracy', 0, ...
                'mean_accuracy', 0, ...
                'learning_rates', [], ...
                'batch_sizes', [], ...
                'hidden_units', [], ...
                'num_layers', [], ...
                'dropout_rates', [], ...
                'fc_units', [] ...
            );
        end
        
        %% サマリーパラメータ更新メソッド
        function updateSummaryParams(~, summary, result, idx)
            % サマリーのパラメータ統計を更新
            if isfield(result, 'params') && length(result.params) >= 6
                summary.learning_rates(idx) = result.params(1);
                summary.batch_sizes(idx) = result.params(2);
                summary.hidden_units(idx) = result.params(3);
                summary.num_layers(idx) = result.params(4);
                summary.dropout_rates(idx) = result.params(5);
                summary.fc_units(idx) = result.params(6);
            end
        end
        
        %% サマリー配列調整メソッド
        function adjustSummaryArrays(~, summary, validCount)
            % サマリー配列を実際の有効数に調整
            summary.learning_rates = summary.learning_rates(1:validCount);
            summary.batch_sizes = summary.batch_sizes(1:validCount);
            summary.hidden_units = summary.hidden_units(1:validCount);
            summary.num_layers = summary.num_layers(1:validCount);
            summary.dropout_rates = summary.dropout_rates(1:validCount);
            summary.fc_units = summary.fc_units(1:validCount);
        end
        
        %% サマリー統計計算メソッド
        function calculateSummaryStatistics(~, summary, testAccuracies)
            % サマリーの統計値を計算
            if ~isempty(testAccuracies)
                summary.best_accuracy = max(testAccuracies);
                summary.worst_accuracy = min(testAccuracies);
                summary.mean_accuracy = mean(testAccuracies);
            end
        end
        
        %% スコア分布表示メソッド
        function displayScoreDistribution(obj, modelScores)
            % モデルスコアの分布統計を表示
            obj.logMessage(1, '\n=== モデルスコアの分布 ===\n');
            if ~isempty(modelScores)
                scorePercentiles = prctile(modelScores, [0, 25, 50, 75, 100]);
                obj.logMessage(1, '  - 最小値: %.4f\n', scorePercentiles(1));
                obj.logMessage(1, '  - 25パーセンタイル: %.4f\n', scorePercentiles(2));
                obj.logMessage(1, '  - 中央値: %.4f\n', scorePercentiles(3));
                obj.logMessage(1, '  - 75パーセンタイル: %.4f\n', scorePercentiles(4));
                obj.logMessage(1, '  - 最大値: %.4f\n', scorePercentiles(5));
            end
        end
        
        %% 試行結果表示メソッド
        function displayTrialResults(obj, idx, total, testAcc, valAcc, f1Score, ...
                overfitPenalty, complexityPenalty, score, result)
            % 個別試行の詳細結果を表示
            
            obj.logMessage(2, '\n--- パラメータセット %d/%d ---\n', idx, total);
            obj.logMessage(2, '性能指標:\n');
            obj.logMessage(2, '  - テスト精度: %.4f\n', testAcc);
            
            if valAcc > 0
                obj.logMessage(2, '  - 検証精度: %.4f\n', valAcc/100);
            end
            
            if f1Score > 0
                obj.logMessage(2, '  - 平均F1スコア: %.4f\n', f1Score);
            end
            
            % 過学習判定
            severity = 'none';
            if isfield(result, 'overfitting') && isfield(result.overfitting, 'severity')
                severity = result.overfitting.severity;
            end
            isOverfit = obj.isOverfitted(result);
            obj.logMessage(2, '  - 過学習判定: %s\n', mat2str(isOverfit));
            obj.logMessage(2, '  - 重大度: %s\n', severity);
            
            obj.logMessage(2, '複合スコア:\n');
            obj.logMessage(2, '  - 過学習ペナルティ: %.2f\n', overfitPenalty);
            obj.logMessage(2, '  - 複雑性ペナルティ: %.2f\n', complexityPenalty);
            obj.logMessage(2, '  - 最終スコア: %.4f\n', score);
        end
        
        %% 最良モデル選択メソッド
        function bestResults = selectBestModel(obj, modelScores, validIndices, ...
                validResults, testAccuracies, valAccuracies, f1Scores, ...
                overfitPenalties, complexityPenalties)
            % 最高スコアのモデルを選択
            
            [bestScore, bestLocalIdx] = max(modelScores);
            
            if ~isempty(bestLocalIdx) && bestLocalIdx <= length(validIndices)
                bestIdx = validIndices(bestLocalIdx);
                bestResults = validResults{bestIdx};
                bestResults.isValid = true;
                
                % 最良モデルの詳細表示
                obj.logMessage(1, '\n最良モデル選択 (インデックス: %d)\n', bestIdx);
                obj.logMessage(1, '  - 最終スコア: %.4f\n', bestScore);
                obj.logMessage(1, '  - テスト精度: %.4f\n', testAccuracies(bestLocalIdx));
                
                if bestLocalIdx <= length(valAccuracies)
                    obj.logMessage(1, '  - 検証精度: %.4f\n', valAccuracies(bestLocalIdx)/100);
                end
                
                if bestLocalIdx <= length(f1Scores) && f1Scores(bestLocalIdx) > 0
                    obj.logMessage(1, '  - 平均F1スコア: %.4f\n', f1Scores(bestLocalIdx));
                end
                
                if bestLocalIdx <= length(overfitPenalties)
                    obj.logMessage(1, '  - 過学習ペナルティ: %.2f\n', overfitPenalties(bestLocalIdx));
                end
                
                if bestLocalIdx <= length(complexityPenalties)
                    obj.logMessage(1, '  - 複雑性ペナルティ: %.2f\n', complexityPenalties(bestLocalIdx));
                end
                
                % 最良パラメータの保存
                if isfield(bestResults, 'params')
                    obj.bestParams = bestResults.params;
                    obj.bestPerformance = bestScore;
                    obj.optimizedModel = bestResults.model;
                end
            else
                % フォールバック処理
                obj.logMessage(1, '\n有効な最良モデルが見つかりませんでした。最初の有効なモデルを使用します。\n');
                bestResults = obj.findFirstValidModel(validResults);
            end
        end
        
        %% 最初の有効モデル検索メソッド
        function bestResults = findFirstValidModel(obj, validResults)
            % 最初の有効なモデルを検索
            bestResults = obj.createDefaultBestResults();
            
            for i = 1:length(validResults)
                if ~isfield(validResults{i}, 'error') || ~validResults{i}.error
                    if isfield(validResults{i}, 'model') && ~isempty(validResults{i}.model)
                        bestResults = validResults{i};
                        bestResults.isValid = true;
                        break;
                    end
                end
            end
        end
        
        %% デフォルト最良結果作成メソッド
        function bestResults = createDefaultBestResults(~)
            % デフォルトの最良結果構造体を作成
            bestResults = struct(...
                'model', [], ...
                'performance', struct('accuracy', 0), ...
                'trainInfo', struct('History', struct(...
                    'TrainingAccuracy', [], ...
                    'ValidationAccuracy', [])), ...
                'overfitting', struct('severity', 'unknown'), ...
                'normParams', [], ...
                'params', [], ...
                'isValid', false ...
            );
        end
        
        %% 上位モデル分析メソッド
        function analyzeTopModels(obj, modelScores, validIndices, validResults)
            % 上位モデルのパラメータ傾向を分析
            
            topN = min(5, length(validIndices));
            if topN > 0
                [~, topLocalIndices] = sort(modelScores, 'descend');
                topLocalIndices = topLocalIndices(1:topN);
                topIndices = validIndices(topLocalIndices);
                
                % パラメータ情報のあるモデルを集計
                top_params = zeros(topN, 6);
                valid_top_count = 0;
                
                for j = 1:length(topIndices)
                    if isfield(validResults{topIndices(j)}, 'params') && ...
                       length(validResults{topIndices(j)}.params) >= 6
                        valid_top_count = valid_top_count + 1;
                        if valid_top_count <= topN
                            top_params(valid_top_count, :) = validResults{topIndices(j)}.params;
                        end
                    end
                end
                
                % 有効なパラメータ数に配列を調整
                if valid_top_count > 0
                    top_params = top_params(1:valid_top_count, :);
                    
                    obj.logMessage(2, '\n上位%dモデルのパラメータ傾向:\n', valid_top_count);
                    obj.logMessage(2, '  - 平均学習率: %.6f\n', mean(top_params(:, 1)));
                    obj.logMessage(2, '  - 平均バッチサイズ: %.1f\n', mean(top_params(:, 2)));
                    obj.logMessage(2, '  - 平均LSTMユニット数: %.1f\n', mean(top_params(:, 3)));
                    obj.logMessage(2, '  - 平均LSTM層数: %.1f\n', mean(top_params(:, 4)));
                    obj.logMessage(2, '  - 平均ドロップアウト率: %.2f\n', mean(top_params(:, 5)));
                    obj.logMessage(2, '  - 平均FC層ユニット数: %.1f\n', mean(top_params(:, 6)));
                else
                    obj.logMessage(2, '\n上位モデルに有効なパラメータ情報がありません\n');
                end
            else
                obj.logMessage(3, '\n上位モデル分析に十分な有効結果がありません\n');
            end
        end
        
        %% 最適化履歴更新メソッド
        function updateOptimizationHistory(obj, results)
            % 最適化の履歴を更新
            % 後の分析や可視化のために全試行結果を保存
            
            try
                obj.logMessage(3, '\n最適化履歴を更新中...\n');
                
                % 有効な結果のみを抽出
                validResults = results(~cellfun(@isempty, results));
                validCount = 0;
                
                % 履歴に追加
                for i = 1:length(validResults)
                    result = validResults{i};
                    
                    % エラーフラグのチェック
                    if isfield(result, 'error') && result.error
                        continue;  % エラーのある結果はスキップ
                    end
                    
                    % モデルと性能データの存在確認
                    if ~isfield(result, 'model') || isempty(result.model) || ...
                       ~isfield(result, 'performance') || isempty(result.performance)
                        continue;  % 必要なフィールドがない結果はスキップ
                    end
                    
                    % 性能指標の抽出
                    testAccuracy = result.performance.accuracy;
                    
                    valAccuracy = 0;
                    if isfield(result, 'trainInfo') && isfield(result.trainInfo, 'History')
                        if isfield(result.trainInfo.History, 'ValidationAccuracy') && ...
                           ~isempty(result.trainInfo.History.ValidationAccuracy)
                            valAcc = result.trainInfo.History.ValidationAccuracy;
                            valAccuracy = mean(valAcc(max(1, end-5):end));
                        end
                    end
                    
                    f1Score = obj.calculateMeanF1Score(result.performance);
                    
                    % 総合スコアの計算
                    score = obj.calculateTrialScore(testAccuracy, valAccuracy, f1Score, result);
                    
                    % 履歴エントリの作成
                    newEntry = struct(...
                        'params', result.params, ...           % 使用パラメータ
                        'testAccuracy', testAccuracy, ...      % テスト精度
                        'valAccuracy', valAccuracy, ...        % 検証精度
                        'f1Score', f1Score, ...                % F1スコア
                        'score', score, ...                    % 総合スコア
                        'model', result.model);                % モデル本体
                    
                    % 履歴への追加
                    if isempty(obj.optimizationHistory)
                        obj.optimizationHistory = newEntry;
                    else
                        obj.optimizationHistory(end+1) = newEntry;
                    end
                    
                    validCount = validCount + 1;
                end
                
                % 有効な結果がなかった場合の処理
                if validCount == 0
                    obj.logMessage(1, '\n有効な最適化履歴データがありません\n');
                    return;
                end
                
                % スコアで降順にソート
                if ~isempty(obj.optimizationHistory)
                    [~, sortIdx] = sort([obj.optimizationHistory.score], 'descend');
                    obj.optimizationHistory = obj.optimizationHistory(sortIdx);
                    
                    obj.logMessage(2, '最適化履歴を更新しました（計 %d 個のモデル）\n', ...
                        length(obj.optimizationHistory));
                end
                
            catch ME
                obj.logMessage(1, '最適化履歴の更新に失敗: %s\n', ME.message);
                
                if obj.verbosity >= 2
                    obj.logMessage(2, 'エラー詳細:\n');
                    disp(getReport(ME, 'extended'));
                end
            end
        end
        
        %% 最適化サマリー表示メソッド
        function displayOptimizationSummary(obj, summary)
            % 最適化プロセスの結果サマリーを表示
            % 統計情報と分析結果の包括的な表示
            
            obj.logMessage(1, '\n=== 最適化プロセスサマリー ===\n');
            
            %  試行結果の概要
            obj.logMessage(1, '試行結果:\n');
            obj.logMessage(1, '  - 総試行回数: %d\n', summary.total_trials);
            obj.logMessage(1, '  - 有効な試行: %d\n', summary.valid_trials);
            
            if isfield(summary, 'valid_models')
                obj.logMessage(1, '  - 有効なモデル: %d\n', summary.valid_models);
            end
            
            obj.logMessage(1, '  - 過学習モデル数: %d (%.1f%%)\n', summary.overfit_models, ...
                (summary.overfit_models/max(summary.valid_trials,1))*100);
            
            % 精度統計
            obj.logMessage(1, '\n精度統計:\n');
            obj.logMessage(1, '  - 最高精度: %.4f\n', summary.best_accuracy);
            obj.logMessage(1, '  - 最低精度: %.4f\n', summary.worst_accuracy);
            obj.logMessage(1, '  - 平均精度: %.4f\n', summary.mean_accuracy);
            
            % 詳細統計情報（verbosityレベル2以上)
            if obj.verbosity >= 2 && ~isempty(summary.learning_rates)
                obj.logMessage(2, '\nパラメータ分布:\n');
                
                % 学習率の統計
                obj.displayParameterStats('学習率', summary.learning_rates, '%.6f');
                
                % バッチサイズの統計
                obj.displayParameterStats('バッチサイズ', summary.batch_sizes, '%.1f');
                
                % LSTMユニット数の統計
                obj.displayParameterStats('LSTMユニット数', summary.hidden_units, '%.1f');
                
                % LSTM層数の統計
                obj.displayParameterStats('LSTM層数', summary.num_layers, '%.1f');
                
                % ドロップアウト率の統計
                obj.displayParameterStats('ドロップアウト率', summary.dropout_rates, '%.3f');
                
                % 全結合層ユニット数の統計
                obj.displayParameterStats('全結合層ユニット数', summary.fc_units, '%.1f');
                
                % パラメータ間の相関分析
                obj.performCorrelationAnalysis(summary);
            end
            
            % 最適化の収束性評価
            if obj.verbosity >= 2
                obj.evaluateConvergence();
            end
            
            % 最適パラメータの表示
            obj.displayBestParameters();
        end
        
        %% パラメータ統計表示ヘルパーメソッド
        function displayParameterStats(obj, name, values, format)
            % 個別パラメータの統計情報を表示
            obj.logMessage(2, '\n%s:\n', name);
            obj.logMessage(2, ['  - 平均: ' format '\n'], mean(values));
            obj.logMessage(2, ['  - 標準偏差: ' format '\n'], std(values));
            obj.logMessage(2, ['  - 最小: ' format '\n'], min(values));
            obj.logMessage(2, ['  - 最大: ' format '\n'], max(values));
        end
        
        %% 相関分析メソッド
        function performCorrelationAnalysis(obj, summary)
            % パラメータ間の相関分析を実行
            obj.logMessage(2, '\nパラメータ間の相関分析:\n');
            
            % パラメータ行列の構築
            paramMatrix = [summary.learning_rates', summary.batch_sizes', ...
                         summary.hidden_units', summary.num_layers', ...
                         summary.dropout_rates', summary.fc_units'];
                     
            paramNames = {'学習率', 'バッチサイズ', 'LSTMユニット数', ...
                         'LSTM層数', 'ドロップアウト率', '全結合層ユニット数'};
                     
            % 相関行列を計算
            if size(paramMatrix, 1) > 1
                corrMatrix = corr(paramMatrix);
                
                % 強い相関のみを表示（閾値: 0.3）
                for i = 1:size(corrMatrix, 1)
                    for j = i+1:size(corrMatrix, 2)
                        if abs(corrMatrix(i,j)) > 0.3
                            obj.logMessage(2, '  - %s と %s: %.3f\n', ...
                                paramNames{i}, paramNames{j}, corrMatrix(i,j));
                        end
                    end
                end
            else
                obj.logMessage(2, '  - 相関分析には複数の有効なモデルが必要です\n');
            end
        end
        
        %% 収束性評価メソッド
        function evaluateConvergence(obj)
            % 最適化プロセスの収束性を評価
            if length(obj.optimizationHistory) > 1
                scores = [obj.optimizationHistory.score];
                improvement = diff(scores);
                
                if ~isempty(improvement)
                    positiveImprovement = improvement(improvement > 0);
                    if ~isempty(positiveImprovement)
                        meanImprovement = mean(positiveImprovement);
                    else
                        meanImprovement = 0;
                    end
                    
                    obj.logMessage(2, '\n収束性評価:\n');
                    obj.logMessage(2, '  - 平均改善率: %.6f\n', meanImprovement);
                    obj.logMessage(2, '  - 改善回数: %d/%d\n', sum(improvement > 0), length(improvement));
                    
                    if meanImprovement < 0.001
                        obj.logMessage(2, '  - 状態: 収束\n');
                    else
                        obj.logMessage(2, '  - 状態: 未収束（さらなる最適化の余地あり）\n');
                    end
                end
            end
        end
        
        %% 最適パラメータ表示メソッド
        function displayBestParameters(obj)
            % 最適なパラメータセットを表示
            if ~isempty(obj.bestParams)
                obj.logMessage(1, '\n=== 最適なパラメータ ===\n');
                obj.logMessage(1, '  - 学習率: %.6f\n', obj.bestParams(1));
                obj.logMessage(1, '  - バッチサイズ: %d\n', obj.bestParams(2));
                obj.logMessage(1, '  - LSTMユニット数: %d\n', obj.bestParams(3));
                obj.logMessage(1, '  - LSTM層数: %d\n', obj.bestParams(4));
                obj.logMessage(1, '  - ドロップアウト率: %.2f\n', obj.bestParams(5));
                obj.logMessage(1, '  - 全結合層ユニット数: %d\n', obj.bestParams(6));
                obj.logMessage(1, '  - 達成スコア: %.4f\n', obj.bestPerformance);
                
                % モデルの推定複雑性（デバッグレベル）
                if obj.verbosity >= 3
                    totalParams = obj.bestParams(3) * obj.bestParams(4) + obj.bestParams(6);
                    obj.logMessage(3, '  - 推定総パラメータ数: 約%d\n', totalParams);
                end
            end
        end
        
        %% デフォルト結果作成メソッド
        function results = createDefaultResults(obj)
            % 最適化無効時のデフォルト結果構造体を生成
            % CNNOptimizerと同じ構造を維持
            
            try
                % 基本的なLSTMモデルのデフォルトパラメータを使用
                obj.logMessage(1, 'デフォルトLSTMパラメータを使用します。\n');

                % デフォルトパラメータの設定
                % これらの値は、一般的なLSTMモデルで良好な結果を示す標準的な値
                defaultLR = 0.001;          % 標準的な学習率
                defaultBatchSize = 32;      % 標準的なバッチサイズ
                defaultLSTMUnits = 128;     % 標準的なLSTMユニット数
                defaultNumLayers = 2;       % 2層のLSTM（深すぎず浅すぎず）
                defaultDropoutRate = 0.3;   % 標準的なドロップアウト率
                defaultFCUnits = 64;        % 標準的な全結合層
                
                % デフォルトパラメータの格納
                defaultParams = [
                    defaultLR; 
                    defaultBatchSize; 
                    defaultLSTMUnits; 
                    defaultNumLayers; 
                    defaultDropoutRate; 
                    defaultFCUnits
                ];
                
                obj.logMessage(1, 'デフォルトパラメータ: \n');
                obj.logMessage(1, '  - 学習率: %.6f\n', defaultLR);
                obj.logMessage(1, '  - バッチサイズ: %d\n', defaultBatchSize);
                obj.logMessage(1, '  - LSTMユニット数: %d\n', defaultLSTMUnits);
                obj.logMessage(1, '  - LSTM層数: %d\n', defaultNumLayers);
                obj.logMessage(1, '  - ドロップアウト率: %.2f\n', defaultDropoutRate);
                obj.logMessage(1, '  - 全結合層ユニット数: %d\n', defaultFCUnits);
                
                % 結果構造体の作成
                results = struct(...
                    'model', [], ...  % モデル自体は空（後で学習）
                    'performance', struct('accuracy', 0), ...  % 初期パフォーマンスは0
                    'trainInfo', struct('History', struct(...
                        'TrainingAccuracy', [], ...
                        'ValidationAccuracy', [])), ...
                    'overfitting', struct('severity', 'unknown'), ...
                    'normParams', [], ...
                    'defaultParams', defaultParams, ... % デフォルトパラメータを保存
                    'isValid', false ...  % これはデフォルト値フラグ
                );
                
                obj.logMessage(1, '最適化スキップ: デフォルト結果を返します\n');
                
            catch ME
                obj.logMessage(0, 'デフォルト結果の作成中にエラーが発生: %s\n', ME.message);
                
                % 最低限の結果を返す
                results = struct(...
                    'model', [], ...
                    'performance', struct('accuracy', 0), ...
                    'trainInfo', struct(), ...
                    'defaultParams', [], ...
                    'isValid', false, ...
                    'error', true, ...
                    'errorMessage', ME.message ...
                );
            end
        end
    end
end