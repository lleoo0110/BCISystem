classdef HybridOptimizer < handle
    %% HybridOptimizer - ハイブリッドモデル（CNN+LSTM）のハイパーパラメータ最適化クラス
    %
    % このクラスはハイブリッドモデルの性能を向上させるためにハイパーパラメータの
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
    % 使用例:
    %   params = getConfig('epocx', 'preset', 'template');
    %   optimizer = HybridOptimizer(params, 1);  % verbosity=1
    %   results = optimizer.optimize(processedData, processedLabel);
    
    properties (Access = private)
        params              % システム設定パラメータ
        optimizedModel      % 最適化されたモデル
        bestParams          % 最良パラメータセット
        bestPerformance     % 最良性能値
        searchSpace         % パラメータ探索空間
        optimizationHistory % 最適化履歴
        useGPU              % GPU使用フラグ
        maxTrials           % 最大試行回数
        verbosity           % 出力詳細度 (0:最小限, 1:通常, 2:詳細, 3:デバッグ)

        % 評価重み
        evaluationWeights   % 各評価指標の重みづけ
        
        % パフォーマンス監視
        gpuMemory           % GPU使用メモリ監視
    end
    
    properties (Access = public)
        performance         % 評価メトリクス (精度・確率など)
    end
    
    methods (Access = public)
        %% コンストラクタ - 初期化処理
        function obj = HybridOptimizer(params, verbosity)
            % HybridOptimizerのインスタンスを初期化
            %
            % 入力:
            %   params - 設定パラメータ（getConfig関数から取得）
            %   verbosity - 出力詳細度 (0:最小限, 1:通常, 2:詳細, 3:デバッグ)
            
            % 基本パラメータの設定
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            obj.useGPU = params.classifier.hybrid.gpu;
            obj.maxTrials = params.classifier.hybrid.optimization.maxTrials;
            
            % verbosityレベルの設定（デフォルトは1）
            if nargin < 2
                obj.verbosity = 1;
            else
                obj.verbosity = verbosity;
            end
            
            % プロパティの初期化
            obj.initializeProperties();
            
            % 評価重みの初期化
            obj.evaluationWeights = struct(...
                'test', 0.6, ...        % テスト精度の重み
                'validation', 0.4, ...  % 検証精度の重み
                'f1Score', 0.3, ...     % F1スコアの重み
                'complexity', 0.1, ...  % 複雑性のペナルティ最大値
                'overfitMax', 0.5 ...   % 過学習の最大ペナルティ値
            );
        end
        
        %% 最適化実行メソッド
        function results = optimize(obj, data, processedLabel)
            % ハイブリッドモデルのハイパーパラメータの最適化を実行
            %
            % 入力:
            %   data - 前処理済みEEGデータ
            %   processedLabel - クラスラベル
            %
            % 出力:
            %   results - 最適化結果（最良モデル、性能評価など）
            
            try
                % 最適化が有効かチェック
                if ~obj.params.classifier.hybrid.optimize
                    obj.logMessage(1, 'ハイブリッドモデル最適化は設定で無効化されています。デフォルトパラメータを使用します。\n');
                    results = obj.createDefaultResults();
                    return;
                end
        
                obj.logMessage(1, '\n=== ハイブリッドモデルのハイパーパラメータ最適化を開始 ===\n');
                
                % 探索アルゴリズムの選択
                searchAlgorithm = 'lhs';  % デフォルト: Latin Hypercube Sampling
                if isfield(obj.params.classifier.hybrid.optimization, 'searchAlgorithm')
                    searchAlgorithm = obj.params.classifier.hybrid.optimization.searchAlgorithm;
                end
                obj.logMessage(1, '探索アルゴリズム: %s\n', searchAlgorithm);
                
                % パラメータセットの生成
                obj.logMessage(2, 'パラメータ探索空間を設定中...\n');
                paramSets = obj.generateParameterSets(obj.maxTrials, searchAlgorithm);
                obj.logMessage(1, 'パラメータ%dセットで最適化を開始します\n', size(paramSets, 1));
        
                % 結果保存用配列
                trialResults = cell(size(paramSets, 1), 1);
                baseParams = obj.params; % この変数は後で使用される
                
                % 有効な評価結果カウンタ
                validResultsCount = 0;
        
                % 各パラメータセットで評価
                for i = 1:size(paramSets, 1)
                    try
                        obj.logMessage(1, '\n--- パラメータセット %d/%d の評価 ---\n', i, size(paramSets, 1));
                        
                        % 現在のパラメータ構成の表示
                        obj.displayCurrentParams(paramSets(i,:));
                        
                        % パラメータの更新
                        localParams = baseParams;
                        localParams = obj.updateHybridParameters(localParams, paramSets(i,:));
        
                        % ハイブリッドモデルの学習と評価
                        hybridClassifier = HybridClassifier(localParams);
                        trainResults = hybridClassifier.trainHybrid(data, processedLabel);
        
                        % 結果の保存
                        trialResults{i} = struct(...
                            'params', paramSets(i,:), ...
                            'model', trainResults.model, ...
                            'performance', trainResults.performance, ...
                            'trainInfo', trainResults.trainInfo, ...
                            'overfitting', trainResults.overfitting, ...
                            'normParams', trainResults.normParams, ...
                            'error', false ...  % エラーフラグ：正常
                        );
                        validResultsCount = validResultsCount + 1;
        
                        % モデル性能の総合スコアを計算（早期停止用）
                        testAccuracy = trainResults.performance.accuracy;
                        
                        % 検証精度データの取得
                        valAccuracy = 0;
                        if isfield(trainResults.trainInfo, 'hybridValMetrics') && ...
                           isfield(trainResults.trainInfo.hybridValMetrics, 'accuracy')
                            valAccuracy = trainResults.trainInfo.hybridValMetrics.accuracy;
                        elseif isfield(trainResults.trainInfo, 'cnnHistory') && ...
                               isfield(trainResults.trainInfo.cnnHistory, 'ValidationAccuracy') && ...
                               ~isempty(trainResults.trainInfo.cnnHistory.ValidationAccuracy)
                            cnnValAcc = trainResults.trainInfo.cnnHistory.ValidationAccuracy;
                            lstmValAcc = trainResults.trainInfo.lstmHistory.ValidationAccuracy;
                            cnnValAcc = cnnValAcc(~isnan(cnnValAcc));
                            lstmValAcc = lstmValAcc(~isnan(lstmValAcc));
                            
                            % 両モデルの検証精度の平均を計算（利用可能な場合のみ）
                            if ~isempty(cnnValAcc) && ~isempty(lstmValAcc)
                                meanCnnValAcc = mean(cnnValAcc(max(1, end-30):end)); % 最後の30エポックの平均
                                meanLstmValAcc = mean(lstmValAcc(max(1, end-30):end));
                                valAccuracy = (meanCnnValAcc + meanLstmValAcc) / 2;
                            elseif ~isempty(cnnValAcc)
                                valAccuracy = mean(cnnValAcc(max(1, end-30):end));
                            elseif ~isempty(lstmValAcc)
                                valAccuracy = mean(lstmValAcc(max(1, end-30):end));
                            end
                        end
                        
                        % F1スコア計算
                        f1Score = obj.calculateMeanF1Score(trainResults.performance);
                        
                        % 総合評価スコアの計算
                        evaluationScore = obj.calculateTrialScore(testAccuracy, valAccuracy, f1Score, trainResults);
        
                        obj.logMessage(1, '組み合わせ %d/%d: テスト精度 = %.4f, 総合スコア = %.4f\n', ...
                            i, size(paramSets, 1), testAccuracy, evaluationScore);
                        
                        % GPUメモリの解放
                        if obj.useGPU
                            gpuDevice([]);
                        end
        
                    catch ME
                        obj.logMessage(0, '組み合わせ%dでエラー発生: %s\n', i, ME.message);
                        obj.logMessage(2, 'エラー詳細:\n');
                        
                        if obj.verbosity >= 2
                            disp(getReport(ME, 'extended'));
                        end
                        
                        % エラー発生時でも最低限のパラメータ情報は保存
                        trialResults{i} = struct(...
                            'params', paramSets(i,:), ...
                            'error', true, ...
                            'errorMessage', ME.message, ...
                            'performance', struct('accuracy', 0) ...
                        );
        
                        % GPUメモリの解放
                        if obj.useGPU
                            gpuDevice([]);
                        end
                    end
                end
                
                % 有効な結果がない場合のチェック
                if validResultsCount == 0
                    obj.logMessage(0, '\n警告: 有効な最適化結果がありません。すべての試行が失敗しました。\n');
                    obj.logMessage(0, 'デフォルト設定を使用してハイブリッドモデルを生成します。\n');
                    
                    % デフォルト結果を生成して返す
                    results = obj.createDefaultResults();
                    results.isValid = false;  % これは有効な最適化結果ではないフラグ
                    
                    % 警告を出力して戻る
                    warning('HybridOptimizer:NoValidResults', '有効な最適化結果がありません');
                    return;
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
                    'isValid', true ...  % これは正常な最適化結果フラグ
                );
                
                obj.logMessage(1, '\n=== ハイブリッドモデル最適化が完了しました ===\n');
        
            catch ME
                obj.logMessage(0, '\n=== ハイブリッドモデル最適化中にエラーが発生しました ===\n');
                obj.logMessage(0, 'エラーメッセージ: %s\n', ME.message);
                obj.logMessage(2, 'エラースタック:\n');
                
                if obj.verbosity >= 2
                    disp(getReport(ME, 'extended'));
                end
                
                % 致命的なエラーでもデフォルト結果を返す
                obj.logMessage(0, '致命的なエラーが発生しました。デフォルト設定を使用します。\n');
                results = obj.createDefaultResults();
                results.isValid = false;  % 無効な結果フラグ
                results.error = true;
                results.errorMessage = ME.message;
                
                % エラーを再スローしない
                warning('HybridOptimizer:OptimizationFailed', 'ハイブリッドモデル最適化に失敗: %s', ME.message);
            end
        end
        
        %% ログ出力メソッド
        function logMessage(obj, level, format, varargin)
            % 指定されたverbosityレベル以上の場合にメッセージを出力
            %
            % 入力:
            %   level - メッセージの重要度 (0:エラー, 1:警告/通常, 2:情報, 3:デバッグ)
            %   format - fprintf形式の文字列
            %   varargin - 追加パラメータ
            
            if obj.verbosity >= level
                fprintf(format, varargin{:});
            end
        end
    end
    
    methods (Access = private)
        %% プロパティ初期化メソッド
        function initializeProperties(obj)
            % クラスプロパティの初期化
            obj.gpuMemory = struct('total', 0, 'used', 0, 'peak', 0);
        end
        
        %% 探索空間初期化メソッド
        function initializeSearchSpace(obj)
            % パラメータ探索空間を設定
            
            % パラメータ設定から探索空間を取得
            if isfield(obj.params.classifier.hybrid.optimization, 'searchSpace')
                obj.searchSpace = obj.params.classifier.hybrid.optimization.searchSpace;
            else
                % デフォルト探索空間
                obj.searchSpace = struct(...
                    'learningRate', [0.0001, 0.01], ...    % 学習率範囲
                    'miniBatchSize', [16, 128], ...        % バッチサイズ範囲
                    'numConvLayers', [1, 3], ...           % 畳み込み層数範囲
                    'cnnFilters', [32, 128], ...           % CNNフィルタ数範囲
                    'filterSize', [3, 7], ...              % フィルタサイズ範囲
                    'lstmUnits', [32, 256], ...            % LSTMユニット数範囲
                    'numLstmLayers', [1, 3], ...           % LSTM層数範囲
                    'dropoutRate', [0.2, 0.7], ...         % ドロップアウト率範囲
                    'fcUnits', [64, 256] ...               % 全結合層ユニット数範囲
                );
            end
        end
        
        %% パラメータセット生成メソッド
        function paramSets = generateParameterSets(obj, numTrials, algorithm)
            % 指定されたアルゴリズムでパラメータセットを生成
            %
            % 入力:
            %   numTrials - 試行回数
            %   algorithm - 探索アルゴリズム
            %
            % 出力:
            %   paramSets - パラメータセット行列 [numTrials x 9]
            
            % 探索アルゴリズムの選択
            if nargin < 3
                algorithm = 'lhs';  % デフォルトはLatin Hypercube Sampling
            end
            
            obj.logMessage(2, 'パラメータ探索アルゴリズム: %s\n', algorithm);
            
            switch lower(algorithm)
                case 'lhs'  % Latin Hypercube Sampling
                    paramSets = obj.generateLHSParams(numTrials);
                case 'random'  % ランダムサンプリング
                    paramSets = obj.generateRandomParams(numTrials);
                case 'grid'  % グリッドサーチ
                    paramSets = obj.generateGridParams();
                otherwise
                    obj.logMessage(1, '未知の探索アルゴリズム: %s。LHSにフォールバックします。\n', algorithm);
                    paramSets = obj.generateLHSParams(numTrials);
            end
            
            obj.logMessage(2, '探索パラメータセット数: %d\n', size(paramSets, 1));
            
            % パラメータセット品質チェック
            obj.validateParameterSets(paramSets);
        end
        
        %% パラメータセット検証メソッド
        function validateParameterSets(obj, paramSets)
            % 生成されたパラメータセットの品質チェック
            
            % 無効な値のチェック
            if any(isnan(paramSets(:)))
                obj.logMessage(1, '警告: 生成されたパラメータセットにNaN値が含まれています\n');
            end
            
            if any(isinf(paramSets(:)))
                obj.logMessage(1, '警告: 生成されたパラメータセットにInf値が含まれています\n');
            end
            
            % 範囲外のパラメータをチェック
            if any(paramSets(:,1) < obj.searchSpace.learningRate(1)) || any(paramSets(:,1) > obj.searchSpace.learningRate(2))
                obj.logMessage(1, '警告: 学習率パラメータに範囲外の値があります\n');
            end
            
            % パラメータの多様性をチェック
            uniqueValues = zeros(1, size(paramSets, 2));
            for i = 1:size(paramSets, 2)
                uniqueValues(i) = length(unique(paramSets(:,i)));
            end
            
            if any(uniqueValues < 3) && size(paramSets, 1) >= 5
                obj.logMessage(1, '警告: 一部のパラメータ次元で多様性が低いです（%d次元目: %d個のユニーク値）\n', ...
                    find(uniqueValues < 3, 1), uniqueValues(find(uniqueValues < 3, 1)));
            end
            
            obj.logMessage(2, 'パラメータセット検証完了\n');
        end
        
        %% LHS (Latin Hypercube Sampling) パラメータ生成
        function paramSets = generateLHSParams(obj, numTrials)
            % Latin Hypercube Samplingによるパラメータセット生成
            
            % 9個のパラメータを Latin Hypercube Sampling で生成
            lhsPoints = lhsdesign(numTrials, 9);
            paramSets = zeros(numTrials, 9);
            
            % 1. 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + ...
                (log10(lr_range(2))-log10(lr_range(1))) * lhsPoints(:,1));
            
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
            
            obj.logMessage(3, 'Latin Hypercube Samplingによりパラメータセットを生成しました（%d組）\n', numTrials);
        end
        
        %% ランダムサンプリングパラメータ生成
        function paramSets = generateRandomParams(obj, numTrials)
            % ランダムサンプリングによるパラメータセット生成
            
            paramSets = zeros(numTrials, 9);
            
            % 1. 学習率（対数スケール）
            lr_range = obj.searchSpace.learningRate;
            paramSets(:,1) = 10.^(log10(lr_range(1)) + ...
                (log10(lr_range(2))-log10(lr_range(1))) * rand(numTrials, 1));
            
            % 2. ミニバッチサイズ
            bs_range = obj.searchSpace.miniBatchSize;
            paramSets(:,2) = round(bs_range(1) + (bs_range(2)-bs_range(1)) * rand(numTrials, 1));
            
            % 3. CNN層数 (numConvLayers)
            ncl_range = obj.searchSpace.numConvLayers;
            paramSets(:,3) = round(ncl_range(1) + (ncl_range(2)-ncl_range(1)) * rand(numTrials, 1));
            
            % 4. CNNフィルタ数
            cf_range = obj.searchSpace.cnnFilters;
            paramSets(:,4) = round(cf_range(1) + (cf_range(2)-cf_range(1)) * rand(numTrials, 1));
            
            % 5. フィルタサイズ
            fs_range = obj.searchSpace.filterSize;
            paramSets(:,5) = round(fs_range(1) + (fs_range(2)-fs_range(1)) * rand(numTrials, 1));
            
            % 6. LSTMユニット数
            lu_range = obj.searchSpace.lstmUnits;
            paramSets(:,6) = round(lu_range(1) + (lu_range(2)-lu_range(1)) * rand(numTrials, 1));
            
            % 7. LSTM層数
            nl_range = obj.searchSpace.numLstmLayers;
            paramSets(:,7) = round(nl_range(1) + (nl_range(2)-nl_range(1)) * rand(numTrials, 1));
            
            % 8. ドロップアウト率
            do_range = obj.searchSpace.dropoutRate;
            paramSets(:,8) = do_range(1) + (do_range(2)-do_range(1)) * rand(numTrials, 1);
            
            % 9. 全結合層ユニット数
            fc_range = obj.searchSpace.fcUnits;
            paramSets(:,9) = round(fc_range(1) + (fc_range(2)-fc_range(1)) * rand(numTrials, 1));
            
            obj.logMessage(3, 'ランダムサンプリングによりパラメータセットを生成しました（%d組）\n', numTrials);
        end
        
        %% グリッドサーチパラメータ生成
        function paramSets = generateGridParams(obj)
            % グリッドサーチによるパラメータセット生成
            
            % グリッドポイント数の決定（パラメータ空間が大きいので少なめに）
            numGridPoints = 2;  % 各次元2点でも2^9=512組になる
            
            % 各パラメータの線形グリッド
            lr_range = obj.searchSpace.learningRate;
            lr_grid = logspace(log10(lr_range(1)), log10(lr_range(2)), numGridPoints);
            
            bs_range = obj.searchSpace.miniBatchSize;
            bs_grid = round(linspace(bs_range(1), bs_range(2), numGridPoints));
            
            ncl_range = obj.searchSpace.numConvLayers;
            ncl_grid = round(linspace(ncl_range(1), ncl_range(2), numGridPoints));
            
            cf_range = obj.searchSpace.cnnFilters;
            cf_grid = round(linspace(cf_range(1), cf_range(2), numGridPoints));
            
            fs_range = obj.searchSpace.filterSize;
            fs_grid = round(linspace(fs_range(1), fs_range(2), numGridPoints));
            
            lu_range = obj.searchSpace.lstmUnits;
            lu_grid = round(linspace(lu_range(1), lu_range(2), numGridPoints));
            
            nl_range = obj.searchSpace.numLstmLayers;
            nl_grid = round(linspace(nl_range(1), nl_range(2), numGridPoints));
            
            do_range = obj.searchSpace.dropoutRate;
            do_grid = linspace(do_range(1), do_range(2), numGridPoints);
            
            fc_range = obj.searchSpace.fcUnits;
            fc_grid = round(linspace(fc_range(1), fc_range(2), numGridPoints));
            
            % グリッドの全組み合わせを生成
            [LR, BS, NCL, CF, FS, LU, NL, DO, FC] = ndgrid(lr_grid, bs_grid, ncl_grid, cf_grid, fs_grid, lu_grid, nl_grid, do_grid, fc_grid);
            
            % 行列に変換
            paramSets = [LR(:), BS(:), NCL(:), CF(:), FS(:), LU(:), NL(:), DO(:), FC(:)];
            
            % グリッド数が多すぎる場合は警告
            if size(paramSets, 1) > 200
                obj.logMessage(1, ['グリッドサーチのパラメータ数が非常に多いです (%d組)。' ...
                    '最適化には長時間かかる可能性があります。'], size(paramSets, 1));
            end
            
            obj.logMessage(3, 'グリッドサーチによりパラメータセットを生成しました（%d組）\n', size(paramSets, 1));
        end
        
        %% ハイブリッドパラメータ更新メソッド
        function params = updateHybridParameters(~, params, paramSet)
            % ハイブリッドモデルのパラメータを更新
            %
            % 入力:
            %   params - ベースパラメータ構造体
            %   paramSet - 新しいパラメータ値 [1x9]
            %
            % 出力:
            %   params - 更新されたパラメータ構造体
            
            try
                % 1. 学習率とミニバッチサイズの更新
                params.classifier.hybrid.training.optimizer.learningRate = paramSet(1);
                params.classifier.hybrid.training.miniBatchSize = paramSet(2);
                
                % 2. CNNパラメータの更新
                numConvLayers = round(paramSet(3));
                cnnFilters = round(paramSet(4));
                filterSize = round(paramSet(5));
                
                % 畳み込み層の構築
                convLayers = struct();
                for j = 1:numConvLayers
                    layerName = sprintf('conv%d', j);
                    convLayers.(layerName) = struct('size', [filterSize, filterSize], ...
                        'filters', cnnFilters, 'stride', 1, 'padding', 'same');
                end
                params.classifier.hybrid.architecture.cnn.convLayers = convLayers;
                
                % 3. LSTMパラメータの更新
                lstmUnits = round(paramSet(6));
                numLstmLayers = round(paramSet(7));
                
                % LSTM層の構築
                lstmLayers = struct();
                for i = 1:numLstmLayers
                    if i == numLstmLayers
                        % 最後の層は'last'出力モード
                        lstmLayers.(['lstm' num2str(i)]) = struct('numHiddenUnits', floor(lstmUnits/2), 'OutputMode', 'last');
                    else
                        % 中間層は'sequence'出力モード
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
                
                % CNN側のドロップアウトレイヤー更新
                cnnDropoutLayers = struct();
                for j = 1:numConvLayers
                    cnnDropoutLayers.(['dropout' num2str(j)]) = dropoutRate;
                end
                params.classifier.hybrid.architecture.cnn.dropoutLayers = cnnDropoutLayers;
                
                % 5. 全結合層ユニット数の更新
                fcUnits = round(paramSet(9));
                params.classifier.hybrid.architecture.cnn.fullyConnected = fcUnits;
                params.classifier.hybrid.architecture.lstm.fullyConnected = fcUnits;
                
                return;
                
            catch ME
                error('パラメータ更新中にエラーが発生: %s', ME.message);
            end
        end
        
        %% 現在のパラメータ表示メソッド
        function displayCurrentParams(obj, paramSet)
            % 現在評価中のパラメータセットを表示
            
            obj.logMessage(2, '評価対象のパラメータ構成:\n');
            obj.logMessage(2, '  - 学習率: %.6f\n', paramSet(1));
            obj.logMessage(2, '  - バッチサイズ: %d\n', paramSet(2));
            obj.logMessage(2, '  - CNN層数: %d\n', paramSet(3));
            obj.logMessage(2, '  - CNNフィルタ数: %d\n', paramSet(4));
            obj.logMessage(2, '  - フィルタサイズ: %d\n', paramSet(5));
            obj.logMessage(2, '  - LSTMユニット数: %d\n', paramSet(6));
            obj.logMessage(2, '  - LSTM層数: %d\n', paramSet(7));
            obj.logMessage(2, '  - ドロップアウト率: %.2f\n', paramSet(8));
            obj.logMessage(2, '  - 全結合層ユニット数: %d\n', paramSet(9));
        end
        
        %% 平均F1スコア計算メソッド
        function f1Score = calculateMeanF1Score(~, performance)
            % 各クラスのF1スコアの平均を計算
            f1Score = 0;
            
            try
                if isfield(performance, 'classwise') && ~isempty(performance.classwise)
                    f1Scores = zeros(1, length(performance.classwise));
                    for i = 1:length(performance.classwise)
                        f1Scores(i) = performance.classwise(i).f1score;
                    end
                    f1Score = mean(f1Scores);
                end
            catch ME
                % エラー時は0を返す
                f1Score = 0;
            end
        end
        
        %% モデル評価スコア計算メソッド
        function score = calculateTrialScore(obj, testAccuracy, valAccuracy, f1Score, results)
            % モデルの総合評価スコアを計算
            %
            % 入力:
            %   testAccuracy - テスト精度
            %   valAccuracy - 検証精度
            %   f1Score - F1スコア
            %   results - 評価結果全体（過学習情報含む）
            %
            % 出力:
            %   score - 総合評価スコア
            
            try
                % 1. 基本精度スコアの計算
                testWeight = obj.evaluationWeights.test;
                valWeight = obj.evaluationWeights.validation;
                
                % 検証精度のスコア計算
                validationScore = testAccuracy; % デフォルトはテスト精度と同じ
                if valAccuracy > 0
                    validationScore = valAccuracy;
                else
                    % なければテスト精度のみで評価
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
                        if isfield(results.overfitting, 'severity')
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
                end
                
                % 4. モデル複雑性ペナルティの計算
                if isfield(results, 'params') && length(results.params) >= 9
                    % CNN複雑性の計算
                    cnnLayers = results.params(3);
                    cnnFilters = results.params(4);
                    
                    % LSTM複雑性の計算
                    lstmLayers = results.params(7);
                    lstmUnits = results.params(6);
                    
                    % 全結合層複雑性
                    fcUnits = results.params(9);
                    
                    % 探索空間の最大値を参照して相対的な複雑さを計算
                    maxCnnLayers = obj.searchSpace.numConvLayers(2);
                    maxCnnFilters = obj.searchSpace.cnnFilters(2);
                    maxLstmLayers = obj.searchSpace.numLstmLayers(2);
                    maxLstmUnits = obj.searchSpace.lstmUnits(2);
                    maxFcUnits = obj.searchSpace.fcUnits(2);
                    
                    % CNN複雑性スコア
                    cnnComplexity = 0.5 * (cnnLayers / maxCnnLayers) + 0.5 * (cnnFilters / maxCnnFilters);
                    
                    % LSTM複雑性スコア
                    lstmComplexity = 0.5 * (lstmLayers / maxLstmLayers) + 0.5 * (lstmUnits / maxLstmUnits);
                    
                    % 全結合層複雑性スコア
                    fcComplexity = fcUnits / maxFcUnits;
                    
                    % 総合複雑性スコア
                    complexityScore = 0.4 * cnnComplexity + 0.4 * lstmComplexity + 0.2 * fcComplexity;
                    complexityPenalty = obj.evaluationWeights.complexity * complexityScore;
                else
                    % paramsフィールドがない場合はデフォルト値を使用
                    obj.logMessage(3, '  注意: 結果の複雑性計算に必要なパラメータがありません。デフォルト値を使用します。\n');
                    complexityPenalty = 0.05; % デフォルトの中程度の複雑性ペナルティ
                end
                
                % 5. 最終スコアの計算 (精度 - 過学習ペナルティ - 複雑性ペナルティ)
                score = combinedScore * (1 - overfitPenalty - complexityPenalty);
                
                % 詳細なスコア計算ログ出力（デバッグ用）
                obj.logMessage(3, 'スコア計算詳細:\n');
                obj.logMessage(3, '  - テスト精度: %.4f (重み: %.2f)\n', testAccuracy, testWeight);
                obj.logMessage(3, '  - 検証スコア: %.4f (重み: %.2f)\n', validationScore, valWeight);
                obj.logMessage(3, '  - F1スコア: %.4f (重み: %.2f)\n', f1Score, obj.evaluationWeights.f1Score);
                obj.logMessage(3, '  - 基本精度スコア: %.4f\n', accuracyScore);
                obj.logMessage(3, '  - 複合精度スコア: %.4f\n', combinedScore);
                obj.logMessage(3, '  - 過学習ペナルティ: %.4f\n', overfitPenalty);
                obj.logMessage(3, '  - 複雑性ペナルティ: %.4f\n', complexityPenalty);
                obj.logMessage(3, '  - 最終スコア: %.4f\n', score);
                
            catch ME
                obj.logMessage(1, 'スコア計算でエラー: %s\n', ME.message);
                score = 0;
            end
        end

        %% 最終結果処理メソッド
        function [bestResults, summary] = processFinalResults(obj, results)
            % 全ての試行結果を処理し、最良のモデルを選択
            
            try
                obj.logMessage(1, '\n=== パラメータ最適化の結果処理 ===\n');
                obj.logMessage(1, '総試行回数: %d\n', length(results));
                
                % 有効な結果のみを抽出
                validResults = results(~cellfun(@isempty, results));
                numResults = length(validResults);
                
                obj.logMessage(1, '有効なパラメータセット数: %d\n', numResults);
                obj.logMessage(1, '無効な試行数: %d\n', length(results) - numResults);
                
                % 有効結果チェック
                validModelCount = 0;
                for i = 1:numResults
                    if ~isfield(validResults{i}, 'error') || ~validResults{i}.error
                        if isfield(validResults{i}, 'model') && ~isempty(validResults{i}.model)
                            validModelCount = validModelCount + 1;
                        end
                    end
                end
                
                % 結果がない場合のエラー処理
                if validModelCount == 0
                    obj.logMessage(0, '有効なモデル結果がありません。デフォルト値を使用します。\n');
                    
                    % デフォルト結果の構造体
                    bestResults = struct(...
                        'model', [], ...
                        'performance', struct('accuracy', 0), ...
                        'trainInfo', struct('cnnHistory', struct(), 'lstmHistory', struct()), ...
                        'overfitting', struct('severity', 'unknown'), ...
                        'normParams', [], ...
                        'params', [], ...
                        'isValid', false ...  % 有効な結果ではない
                    );
                    
                    % サマリー情報の初期化
                    summary = struct(...
                        'total_trials', length(results), ...
                        'valid_trials', numResults, ...
                        'valid_models', validModelCount, ...
                        'overfit_models', 0, ...
                        'best_accuracy', 0, ...
                        'worst_accuracy', 0, ...
                        'mean_accuracy', 0, ...
                        'learning_rates', [], ...
                        'batch_sizes', [], ...
                        'num_conv_layers', [], ...
                        'cnn_filters', [], ...
                        'filter_sizes', [], ...
                        'lstm_units', [], ...
                        'num_lstm_layers', [], ...
                        'dropout_rates', [], ...
                        'fc_units', [] ...
                    );
                    
                    return;
                end
                
                % 評価結果保存用の変数 - 事前に配列を割り当て
                modelScores = zeros(numResults, 1);
                testAccuracies = zeros(numResults, 1);
                valAccuracies = zeros(numResults, 1);
                f1Scores = zeros(numResults, 1); 
                overfitPenalties = zeros(numResults, 1);
                complexityPenalties = zeros(numResults, 1);
                validIndices = zeros(numResults, 1);
                
                % サマリー情報の初期化
                summary = struct(...
                    'total_trials', length(results), ...
                    'valid_trials', numResults, ...
                    'valid_models', validModelCount, ...
                    'overfit_models', 0, ...
                    'best_accuracy', 0, ...
                    'worst_accuracy', 1, ...
                    'mean_accuracy', 0, ...
                    'learning_rates', zeros(1, numResults), ...
                    'batch_sizes', zeros(1, numResults), ...
                    'num_conv_layers', zeros(1, numResults), ...
                    'cnn_filters', zeros(1, numResults), ...
                    'filter_sizes', zeros(1, numResults), ...
                    'lstm_units', zeros(1, numResults), ...
                    'num_lstm_layers', zeros(1, numResults), ...
                    'dropout_rates', zeros(1, numResults), ...
                    'fc_units', zeros(1, numResults) ...
                );
                
                % 有効な結果数を追跡するカウンタ
                validCount = 0;
                
                obj.logMessage(2, '\n=== 各試行の詳細評価 ===\n');
                for i = 1:numResults
                    result = validResults{i};
                    try
                        % エラーチェック
                        if isfield(result, 'error') && result.error
                            obj.logMessage(2, '\n--- パラメータセット %d/%d: エラーあり (%s) ---\n', ...
                                i, numResults, result.errorMessage);
                            continue;
                        end
                        
                        % モデルと性能データの存在確認
                        if ~isfield(result, 'model') || isempty(result.model)
                            obj.logMessage(2, '\n--- パラメータセット %d/%d: モデルが空または存在しません ---\n', i, numResults);
                            continue;
                        end
                        
                        if ~isfield(result, 'performance') || isempty(result.performance)
                            obj.logMessage(2, '\n--- パラメータセット %d/%d: パフォーマンスデータが空または存在しません ---\n', i, numResults);
                            continue;
                        end
                    
                        % 基本的な精度スコア
                        testAccuracy = result.performance.accuracy;
                        
                        % 検証精度の取得
                        valAccuracy = 0;
                        if isfield(result, 'trainInfo') && isfield(result.trainInfo, 'hybridValMetrics') && ...
                           isfield(result.trainInfo.hybridValMetrics, 'accuracy')
                            valAccuracy = result.trainInfo.hybridValMetrics.accuracy;
                        elseif isfield(result, 'trainInfo') && isfield(result.trainInfo, 'cnnHistory') && ...
                               isfield(result.trainInfo.cnnHistory, 'ValidationAccuracy') && ...
                               ~isempty(result.trainInfo.cnnHistory.ValidationAccuracy)
                            cnnValAcc = result.trainInfo.cnnHistory.ValidationAccuracy;
                            lstmValAcc = result.trainInfo.lstmHistory.ValidationAccuracy;
                            cnnValAcc = cnnValAcc(~isnan(cnnValAcc));
                            lstmValAcc = lstmValAcc(~isnan(lstmValAcc));
                            
                            % 両モデルの検証精度の平均を計算（利用可能な場合のみ）
                            if ~isempty(cnnValAcc) && ~isempty(lstmValAcc)
                                meanCnnValAcc = mean(cnnValAcc(max(1, end-30):end)); % 最後の30エポックの平均
                                meanLstmValAcc = mean(lstmValAcc(max(1, end-30):end));
                                valAccuracy = (meanCnnValAcc + meanLstmValAcc) / 2;
                            elseif ~isempty(cnnValAcc)
                                valAccuracy = mean(cnnValAcc(max(1, end-30):end));
                            elseif ~isempty(lstmValAcc)
                                valAccuracy = mean(lstmValAcc(max(1, end-30):end));
                            end
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
                        if isfield(result, 'params') && length(result.params) >= 9
                            % CNN複雑性の計算
                            cnnLayers = result.params(3);
                            cnnFilters = result.params(4);
                            
                            % LSTM複雑性の計算
                            lstmLayers = result.params(7);
                            lstmUnits = result.params(6);
                            
                            % 全結合層複雑性
                            fcUnits = result.params(9);
                            
                            % 探索空間の最大値を参照して相対的な複雑さを計算
                            maxCnnLayers = obj.searchSpace.numConvLayers(2);
                            maxCnnFilters = obj.searchSpace.cnnFilters(2);
                            maxLstmLayers = obj.searchSpace.numLstmLayers(2);
                            maxLstmUnits = obj.searchSpace.lstmUnits(2);
                            maxFcUnits = obj.searchSpace.fcUnits(2);
                            
                            % CNN複雑性スコア
                            cnnComplexity = 0.5 * (cnnLayers / maxCnnLayers) + 0.5 * (cnnFilters / maxCnnFilters);
                            
                            % LSTM複雑性スコア
                            lstmComplexity = 0.5 * (lstmLayers / maxLstmLayers) + 0.5 * (lstmUnits / maxLstmUnits);
                            
                            % 全結合層複雑性スコア
                            fcComplexity = fcUnits / maxFcUnits;
                            
                            % 総合複雑性スコア
                            complexityScore = 0.4 * cnnComplexity + 0.4 * lstmComplexity + 0.2 * fcComplexity;
                            complexityPenalty = obj.evaluationWeights.complexity * complexityScore;
                        else
                            obj.logMessage(3, '  試行 %d: パラメータ情報がないため、デフォルトの複雑性ペナルティを適用します\n', i);
                        end
                        
                        % 総合スコアの計算
                        score = obj.calculateTrialScore(testAccuracy, valAccuracy, f1Score, result);
                        
                        % 配列に追加（カウンタを使用してサイズ変更を回避）
                        validCount = validCount + 1;
                        testAccuracies(validCount) = testAccuracy;
                        valAccuracies(validCount) = valAccuracy;
                        f1Scores(validCount) = f1Score;
                        overfitPenalties(validCount) = overfitPenalty;
                        complexityPenalties(validCount) = complexityPenalty;
                        modelScores(validCount) = score;
                        validIndices(validCount) = i;
                        
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
                        if isfield(result, 'params') && length(result.params) >= 9
                            summary.learning_rates(validCount) = result.params(1);
                            summary.batch_sizes(validCount) = result.params(2);
                            summary.num_conv_layers(validCount) = result.params(3);
                            summary.cnn_filters(validCount) = result.params(4);
                            summary.filter_sizes(validCount) = result.params(5);
                            summary.lstm_units(validCount) = result.params(6);
                            summary.num_lstm_layers(validCount) = result.params(7);
                            summary.dropout_rates(validCount) = result.params(8);
                            summary.fc_units(validCount) = result.params(9);
                        end
                        
                        % 結果の詳細表示（verbosity >= 2）
                        obj.logMessage(2, '\n--- パラメータセット %d/%d ---\n', i, numResults);
                        obj.logMessage(2, '性能指標:\n');
                        obj.logMessage(2, '  - テスト精度: %.4f\n', testAccuracy);
                        
                        if valAccuracy > 0
                            obj.logMessage(2, '  - 検証精度: %.4f\n', valAccuracy);
                        end
                        
                        if f1Score > 0
                            obj.logMessage(2, '  - 平均F1スコア: %.4f\n', f1Score);
                        end
                        
                        obj.logMessage(2, '  - 過学習判定: %s\n', string(isOverfit));
                        obj.logMessage(2, '  - 重大度: %s\n', severity);
                        
                        obj.logMessage(2, '複合スコア:\n');
                        obj.logMessage(2, '  - 過学習ペナルティ: %.2f\n', overfitPenalty);
                        obj.logMessage(2, '  - 複雑性ペナルティ: %.2f\n', complexityPenalty);
                        obj.logMessage(2, '  - 最終スコア: %.4f\n', score);
                    catch ME
                        obj.logMessage(1, '\n--- パラメータセット %d/%d の評価中にエラーが発生: %s ---\n', i, numResults, ME.message);
                    end
                end
                
                % 使用した有効な結果数に配列サイズを調整
                if validCount > 0
                    testAccuracies = testAccuracies(1:validCount);
                    valAccuracies = valAccuracies(1:validCount);
                    f1Scores = f1Scores(1:validCount);
                    overfitPenalties = overfitPenalties(1:validCount);
                    complexityPenalties = complexityPenalties(1:validCount);
                    modelScores = modelScores(1:validCount);
                    validIndices = validIndices(1:validCount);
                    
                    summary.learning_rates = summary.learning_rates(1:validCount);
                    summary.batch_sizes = summary.batch_sizes(1:validCount);
                    summary.num_conv_layers = summary.num_conv_layers(1:validCount);
                    summary.cnn_filters = summary.cnn_filters(1:validCount);
                    summary.filter_sizes = summary.filter_sizes(1:validCount);
                    summary.lstm_units = summary.lstm_units(1:validCount);
                    summary.num_lstm_layers = summary.num_lstm_layers(1:validCount);
                    summary.dropout_rates = summary.dropout_rates(1:validCount);
                    summary.fc_units = summary.fc_units(1:validCount);
                end
                
                % 有効な結果がない場合
                if validCount == 0
                    obj.logMessage(0, '有効な評価結果がありません。デフォルト設定を使用します。\n');
                    
                    % デフォルト結果の構築
                    bestResults = struct(...
                        'model', [], ...
                        'performance', struct('accuracy', 0), ...
                        'trainInfo', struct('cnnHistory', struct(), 'lstmHistory', struct()), ...
                        'overfitting', struct('severity', 'unknown'), ...
                        'normParams', [], ...
                        'params', [], ...
                        'isValid', false  ... % 有効な結果ではない
                    );
                    
                    return;
                end
                
                % 統計サマリーの計算
                if ~isempty(testAccuracies)
                    summary.best_accuracy = max(testAccuracies);
                    summary.worst_accuracy = min(testAccuracies);
                    summary.mean_accuracy = mean(testAccuracies);
                end
                
                % モデル選択の詳細情報
                obj.logMessage(1, '\n=== モデルスコアの分布 ===\n');
                if ~isempty(modelScores)
                    scorePercentiles = prctile(modelScores, [0, 25, 50, 75, 100]);
                    obj.logMessage(1, '  - 最小値: %.4f\n', scorePercentiles(1));
                    obj.logMessage(1, '  - 25パーセンタイル: %.4f\n', scorePercentiles(2));
                    obj.logMessage(1, '  - 中央値: %.4f\n', scorePercentiles(3));
                    obj.logMessage(1, '  - 75パーセンタイル: %.4f\n', scorePercentiles(4));
                    obj.logMessage(1, '  - 最大値: %.4f\n', scorePercentiles(5));
                end
                
                % 最良モデルの選択（最高スコア）
                [bestScore, bestLocalIdx] = max(modelScores);
                if ~isempty(bestLocalIdx) && bestLocalIdx <= length(validIndices)
                    bestIdx = validIndices(bestLocalIdx);
                    bestResults = validResults{bestIdx};
                    bestResults.isValid = true;  % これは正常な最適化結果
                    
                    obj.logMessage(1, '\n最良モデル選択 (インデックス: %d)\n', bestIdx);
                    obj.logMessage(1, '  - 最終スコア: %.4f\n', bestScore);
                    obj.logMessage(1, '  - テスト精度: %.4f\n', testAccuracies(bestLocalIdx));
                    
                    % 検証精度の表示部分を修正
                    if bestLocalIdx <= length(valAccuracies) && valAccuracies(bestLocalIdx) > 0
                        obj.logMessage(1, '  - 検証精度: %.4f\n', valAccuracies(bestLocalIdx));
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
        
                        % 上位モデルのパラメータ傾向分析
                        topN = min(5, length(validIndices));
                        if topN > 0 && obj.verbosity >= 2
                            [~, topLocalIndices] = sort(modelScores, 'descend');
                            topLocalIndices = topLocalIndices(1:topN);
                            topIndices = validIndices(topLocalIndices);
                            
                            % パラメータ情報のあるモデルを集計
                            top_params = zeros(topN, 9); % 配列を事前割り当て
                            valid_top_count = 0;
                            
                            for j = 1:length(topIndices)
                                if isfield(validResults{topIndices(j)}, 'params') && ...
                                   length(validResults{topIndices(j)}.params) >= 9
                                    valid_top_count = valid_top_count + 1;
                                    if valid_top_count <= topN  % サイズを超えないように確認
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
                                obj.logMessage(2, '  - 平均CNN層数: %.1f\n', mean(top_params(:, 3)));
                                obj.logMessage(2, '  - 平均CNNフィルタ数: %.1f\n', mean(top_params(:, 4)));
                                obj.logMessage(2, '  - 平均フィルタサイズ: %.1f\n', mean(top_params(:, 5)));
                                obj.logMessage(2, '  - 平均LSTMユニット数: %.1f\n', mean(top_params(:, 6)));
                                obj.logMessage(2, '  - 平均LSTM層数: %.1f\n', mean(top_params(:, 7)));
                                obj.logMessage(2, '  - 平均ドロップアウト率: %.2f\n', mean(top_params(:, 8)));
                                obj.logMessage(2, '  - 平均FC層ユニット数: %.1f\n', mean(top_params(:, 9)));
                            else
                                obj.logMessage(2, '\n上位モデルに有効なパラメータ情報がありません\n');
                            end
                        else
                            obj.logMessage(3, '\n上位モデル分析に十分な有効結果がありません\n');
                        end
                    else
                        obj.logMessage(1, '\n最良モデルにパラメータ情報がありません\n');
                    end
                else
                    obj.logMessage(1, '\n有効な最良モデルが見つかりませんでした。最初の有効なモデルを使用します。\n');
                    for i = 1:length(validResults)
                        if ~isfield(validResults{i}, 'error') || ~validResults{i}.error
                            if isfield(validResults{i}, 'model') && ~isempty(validResults{i}.model)
                                bestResults = validResults{i};
                                bestResults.isValid = true;
                                break;
                            end
                        end
                    end
                    
                    % それでも見つからない場合はデフォルト結果
                    if ~exist('bestResults', 'var') || isempty(bestResults)
                        obj.logMessage(0, '有効なモデルが見つかりません。デフォルト設定を使用します。\n');
                        bestResults = struct(...
                            'model', [], ...
                            'performance', struct('accuracy', 0), ...
                            'trainInfo', struct('cnnHistory', struct(), 'lstmHistory', struct()), ...
                            'overfitting', struct('severity', 'unknown'), ...
                            'normParams', [], ...
                            'params', [], ...
                            'isValid', false ...  % 有効な結果ではない
                        );
                    end
                end
        
            catch ME
                obj.logMessage(0, '結果処理中にエラーが発生: %s\n', ME.message);
                
                if obj.verbosity >= 2
                    obj.logMessage(2, 'エラー詳細:\n');
                    disp(getReport(ME, 'extended'));
                end
                
                % 最低限の結果を返す
                bestResults = struct(...
                    'model', [], ...
                    'performance', struct('accuracy', 0), ...
                    'trainInfo', struct('cnnHistory', struct(), 'lstmHistory', struct()), ...
                    'overfitting', struct('severity', 'unknown'), ...
                    'normParams', [], ...
                    'params', [], ...
                    'isValid', false ...  % 有効な結果ではない
                );
                
                summary = struct(...
                    'total_trials', length(results), ...
                    'valid_trials', length(validResults), ...
                    'valid_models', 0, ...
                    'overfit_models', 0, ...
                    'best_accuracy', 0, ...
                    'worst_accuracy', 0, ...
                    'mean_accuracy', 0, ...
                    'learning_rates', [], ...
                    'batch_sizes', [], ...
                    'num_conv_layers', [], ...
                    'cnn_filters', [], ...
                    'filter_sizes', [], ...
                    'lstm_units', [], ...
                    'num_lstm_layers', [], ...
                    'dropout_rates', [], ...
                    'fc_units', [] ...
                );
                
                % エラーを再スローせず、可能な限り処理を続行
                obj.logMessage(1, '警告: エラーが発生しましたが、可能な限り処理を続行します。\n');
            end
        end
        
        %% 最適化履歴更新メソッド
        function updateOptimizationHistory(obj, results)
            % 最適化の履歴を更新
            
            try
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
                    
                    % テスト精度と検証精度の取得
                    testAccuracy = result.performance.accuracy;
                    
                    valAccuracy = 0;
                    if isfield(result, 'trainInfo') && isfield(result.trainInfo, 'hybridValMetrics') && ...
                       isfield(result.trainInfo.hybridValMetrics, 'accuracy')
                        valAccuracy = result.trainInfo.hybridValMetrics.accuracy;
                    elseif isfield(result, 'trainInfo') && isfield(result.trainInfo, 'cnnHistory') && ...
                           isfield(result.trainInfo.cnnHistory, 'ValidationAccuracy') && ...
                           ~isempty(result.trainInfo.cnnHistory.ValidationAccuracy)
                        cnnValAcc = result.trainInfo.cnnHistory.ValidationAccuracy;
                        lstmValAcc = result.trainInfo.lstmHistory.ValidationAccuracy;
                        cnnValAcc = cnnValAcc(~isnan(cnnValAcc));
                        lstmValAcc = lstmValAcc(~isnan(lstmValAcc));
                        
                        % 両モデルの検証精度の平均を計算（利用可能な場合のみ）
                        if ~isempty(cnnValAcc) && ~isempty(lstmValAcc)
                            meanCnnValAcc = mean(cnnValAcc(max(1, end-5):end)); % 最後の5エポックの平均
                            meanLstmValAcc = mean(lstmValAcc(max(1, end-5):end));
                            valAccuracy = (meanCnnValAcc + meanLstmValAcc) / 2;
                        elseif ~isempty(cnnValAcc)
                            valAccuracy = mean(cnnValAcc(max(1, end-5):end));
                        elseif ~isempty(lstmValAcc)
                            valAccuracy = mean(lstmValAcc(max(1, end-5):end));
                        end
                    end
                    
                    f1Score = obj.calculateMeanF1Score(result.performance);
                    
                    % 総合スコアの計算
                    score = obj.calculateTrialScore(testAccuracy, valAccuracy, f1Score, result);
                    
                    % モデル構造とサイズの計算
                    modelSize = 0;
                    cnnLayerCount = 0;
                    lstmLayerCount = 0;
                    
                    if isfield(result, 'params') && length(result.params) >= 9
                        cnnLayerCount = round(result.params(3));
                        cnnFilters = round(result.params(4));
                        filterSize = round(result.params(5));
                        lstmLayerCount = round(result.params(7));
                        lstmUnits = round(result.params(6));
                        fcUnits = round(result.params(9));
                        
                        % 各コンポーネントのパラメータ数（概算）
                        cnnParams = cnnFilters * filterSize * filterSize * cnnLayerCount;
                        lstmParams = lstmUnits * lstmUnits * 4 * lstmLayerCount; % LSTM内部のゲート数を考慮
                        fcParams = fcUnits * (cnnFilters + lstmUnits);
                        
                        modelSize = (cnnParams + lstmParams + fcParams) / 1000; % KB単位
                    end
                    
                    newEntry = struct(...
                        'params', result.params, ...
                        'testAccuracy', testAccuracy, ...
                        'valAccuracy', valAccuracy, ...
                        'f1Score', f1Score, ...
                        'score', score, ...
                        'modelSize', modelSize, ...
                        'cnnLayerCount', cnnLayerCount, ...
                        'lstmLayerCount', lstmLayerCount, ...
                        'model', result.model);
                    
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
                    
                    obj.logMessage(2, '\n最適化履歴を更新しました（計 %d 個のモデル）\n', ...
                        length(obj.optimizationHistory));
                    
                    % 改善の傾向分析
                    if length(obj.optimizationHistory) >= 3
                        scores = [obj.optimizationHistory.score];
                        improvement = diff(scores(1:min(10, length(scores))));
                        avgImprovement = mean(improvement(improvement > 0));
                        
                        obj.logMessage(2, '最適化の進行状況:\n');
                        obj.logMessage(2, '  - 最良スコア: %.4f\n', scores(1));
                        obj.logMessage(2, '  - 平均改善率: %.6f\n', avgImprovement);
                        obj.logMessage(2, '  - 収束性: %s\n', string(avgImprovement < 0.001));
                    end
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
            
            obj.logMessage(1, '\n=== 最適化プロセスサマリー ===\n');
            obj.logMessage(1, '試行結果:\n');
            obj.logMessage(1, '  - 総試行回数: %d\n', summary.total_trials);
            obj.logMessage(1, '  - 有効な試行: %d\n', summary.valid_trials);
            
            if isfield(summary, 'valid_models')
                obj.logMessage(1, '  - 有効なモデル: %d\n', summary.valid_models);
            end
            
            obj.logMessage(1, '  - 過学習モデル数: %d (%.1f%%)\n', summary.overfit_models, ...
                (summary.overfit_models/max(summary.valid_trials,1))*100);
            
            obj.logMessage(1, '\n精度統計:\n');
            obj.logMessage(1, '  - 最高精度: %.4f\n', summary.best_accuracy);
            obj.logMessage(1, '  - 最低精度: %.4f\n', summary.worst_accuracy);
            obj.logMessage(1, '  - 平均精度: %.4f\n', summary.mean_accuracy);
            
            % 詳細統計情報（verbosityレベル2以上）
            if obj.verbosity >= 2 && ~isempty(summary.learning_rates)
                obj.logMessage(2, '\nパラメータ分布:\n');
                
                % 学習率の統計
                obj.logMessage(2, '\n学習率:\n');
                obj.logMessage(2, '  - 平均: %.6f\n', mean(summary.learning_rates));
                obj.logMessage(2, '  - 標準偏差: %.6f\n', std(summary.learning_rates));
                obj.logMessage(2, '  - 最小: %.6f\n', min(summary.learning_rates));
                obj.logMessage(2, '  - 最大: %.6f\n', max(summary.learning_rates));
                
                % バッチサイズの統計
                obj.logMessage(2, '\nバッチサイズ:\n');
                obj.logMessage(2, '  - 平均: %.1f\n', mean(summary.batch_sizes));
                obj.logMessage(2, '  - 標準偏差: %.1f\n', std(summary.batch_sizes));
                obj.logMessage(2, '  - 最小: %d\n', min(summary.batch_sizes));
                obj.logMessage(2, '  - 最大: %d\n', max(summary.batch_sizes));
                
                % CNN層数の統計
                obj.logMessage(2, '\nCNN層数:\n');
                obj.logMessage(2, '  - 平均: %.1f\n', mean(summary.num_conv_layers));
                obj.logMessage(2, '  - 標準偏差: %.1f\n', std(summary.num_conv_layers));
                obj.logMessage(2, '  - 最小: %d\n', min(summary.num_conv_layers));
                obj.logMessage(2, '  - 最大: %d\n', max(summary.num_conv_layers));
                
                % CNNフィルタ数の統計
                obj.logMessage(2, '\nCNNフィルタ数:\n');
                obj.logMessage(2, '  - 平均: %.1f\n', mean(summary.cnn_filters));
                obj.logMessage(2, '  - 標準偏差: %.1f\n', std(summary.cnn_filters));
                obj.logMessage(2, '  - 最小: %d\n', min(summary.cnn_filters));
                obj.logMessage(2, '  - 最大: %d\n', max(summary.cnn_filters));
                
                % フィルタサイズの統計
                obj.logMessage(2, '\nフィルタサイズ:\n');
                obj.logMessage(2, '  - 平均: %.1f\n', mean(summary.filter_sizes));
                obj.logMessage(2, '  - 標準偏差: %.1f\n', std(summary.filter_sizes));
                obj.logMessage(2, '  - 最小: %d\n', min(summary.filter_sizes));
                obj.logMessage(2, '  - 最大: %d\n', max(summary.filter_sizes));
                
                % LSTMユニット数の統計
                obj.logMessage(2, '\nLSTMユニット数:\n');
                obj.logMessage(2, '  - 平均: %.1f\n', mean(summary.lstm_units));
                obj.logMessage(2, '  - 標準偏差: %.1f\n', std(summary.lstm_units));
                obj.logMessage(2, '  - 最小: %d\n', min(summary.lstm_units));
                obj.logMessage(2, '  - 最大: %d\n', max(summary.lstm_units));
                
                % LSTM層数の統計
                obj.logMessage(2, '\nLSTM層数:\n');
                obj.logMessage(2, '  - 平均: %.1f\n', mean(summary.num_lstm_layers));
                obj.logMessage(2, '  - 標準偏差: %.1f\n', std(summary.num_lstm_layers));
                obj.logMessage(2, '  - 最小: %d\n', min(summary.num_lstm_layers));
                obj.logMessage(2, '  - 最大: %d\n', max(summary.num_lstm_layers));
                
                % ドロップアウト率の統計
                obj.logMessage(2, '\nドロップアウト率:\n');
                obj.logMessage(2, '  - 平均: %.3f\n', mean(summary.dropout_rates));
                obj.logMessage(2, '  - 標準偏差: %.3f\n', std(summary.dropout_rates));
                obj.logMessage(2, '  - 最小: %.3f\n', min(summary.dropout_rates));
                obj.logMessage(2, '  - 最大: %.3f\n', max(summary.dropout_rates));
                
                % 全結合層ユニット数の統計
                obj.logMessage(2, '\n全結合層ユニット数:\n');
                obj.logMessage(2, '  - 平均: %.1f\n', mean(summary.fc_units));
                obj.logMessage(2, '  - 標準偏差: %.1f\n', std(summary.fc_units));
                obj.logMessage(2, '  - 最小: %d\n', min(summary.fc_units));
                obj.logMessage(2, '  - 最大: %d\n', max(summary.fc_units));
                
                % パラメータ間の相関分析
                obj.logMessage(2, '\nパラメータ間の相関分析:\n');
                paramMatrix = [summary.learning_rates', summary.batch_sizes', ...
                             summary.num_conv_layers', summary.cnn_filters', ...
                             summary.filter_sizes', summary.lstm_units', ...
                             summary.num_lstm_layers', summary.dropout_rates', ...
                             summary.fc_units'];
                         
                paramNames = {'学習率', 'バッチサイズ', 'CNN層数', ...
                             'CNNフィルタ数', 'フィルタサイズ', ...
                             'LSTMユニット数', 'LSTM層数', 'ドロップアウト率', ...
                             '全結合層ユニット数'};
                         
                % 相関行列を計算
                if size(paramMatrix, 1) > 1
                    corrMatrix = corr(paramMatrix);
                    
                    % 強い相関のみを表示
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
            
            % 最適化の収束性評価（verbosityレベル2以上）
            if obj.verbosity >= 2 && length(obj.optimizationHistory) > 1
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
            
            % 最適パラメータの表示
            if ~isempty(obj.bestParams)
                obj.logMessage(1, '\n=== 最適なパラメータ ===\n');
                obj.logMessage(1, '  - 学習率: %.6f\n', obj.bestParams(1));
                obj.logMessage(1, '  - バッチサイズ: %d\n', obj.bestParams(2));
                obj.logMessage(1, '  - CNN層数: %d\n', obj.bestParams(3));
                obj.logMessage(1, '  - CNNフィルタ数: %d\n', obj.bestParams(4));
                obj.logMessage(1, '  - フィルタサイズ: %d\n', obj.bestParams(5));
                obj.logMessage(1, '  - LSTMユニット数: %d\n', obj.bestParams(6));
                obj.logMessage(1, '  - LSTM層数: %d\n', obj.bestParams(7));
                obj.logMessage(1, '  - ドロップアウト率: %.2f\n', obj.bestParams(8));
                obj.logMessage(1, '  - 全結合層ユニット数: %d\n', obj.bestParams(9));
                obj.logMessage(1, '  - 達成スコア: %.4f\n', obj.bestPerformance);
            end
        end

        %% デフォルト結果作成メソッド
        function results = createDefaultResults(obj)
            % 最適化無効時のデフォルト結果構造体を生成
            
            try
                % 基本的なハイブリッドモデルのデフォルトパラメータを使用
                obj.logMessage(1, 'デフォルトハイブリッドモデルパラメータを使用します。\n');

                % デフォルトパラメータの設定
                defaultLR = 0.001;          % 標準的な学習率
                defaultBatchSize = 32;      % 標準的なバッチサイズ
                defaultConvLayers = 2;      % 2層のCNN
                defaultCnnFilters = 32;     % 標準的なCNNフィルタ数
                defaultFilterSize = 5;      % 標準的なフィルタサイズ
                defaultLstmUnits = 64;      % 標準的なLSTMユニット数
                defaultLstmLayers = 2;      % 2層のLSTM
                defaultDropoutRate = 0.5;   % 標準的なドロップアウト率
                defaultFCUnits = 128;       % 標準的な全結合層
                
                % デフォルトパラメータの格納
                defaultParams = [
                    defaultLR; 
                    defaultBatchSize; 
                    defaultConvLayers; 
                    defaultCnnFilters;
                    defaultFilterSize;
                    defaultLstmUnits;
                    defaultLstmLayers;
                    defaultDropoutRate; 
                    defaultFCUnits
                ];
                
                obj.logMessage(1, 'デフォルトパラメータ: \n');
                obj.logMessage(1, '  - 学習率: %.6f\n', defaultLR);
                obj.logMessage(1, '  - バッチサイズ: %d\n', defaultBatchSize);
                obj.logMessage(1, '  - CNN層数: %d\n', defaultConvLayers);
                obj.logMessage(1, '  - CNNフィルタ数: %d\n', defaultCnnFilters);
                obj.logMessage(1, '  - フィルタサイズ: %d\n', defaultFilterSize);
                obj.logMessage(1, '  - LSTMユニット数: %d\n', defaultLstmUnits);
                obj.logMessage(1, '  - LSTM層数: %d\n', defaultLstmLayers);
                obj.logMessage(1, '  - ドロップアウト率: %.2f\n', defaultDropoutRate);
                obj.logMessage(1, '  - 全結合層ユニット数: %d\n', defaultFCUnits);
                
                % 結果構造体の作成
                results = struct(...
                    'model', [], ...  % モデル自体は空
                    'performance', struct('accuracy', 0), ...  % 初期パフォーマンスは0
                    'trainInfo', struct('cnnHistory', struct(), 'lstmHistory', struct()), ...
                    'overfitting', struct('severity', 'unknown'), ...
                    'normParams', [], ...
                    'defaultParams', defaultParams, ...
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
        
        %% パラメータ抽出メソッド
        function extractedParams = extractParams(obj)
            % ハイブリッドモデルのパラメータを抽出して9要素の配列として返す
            
            % デフォルト値で初期化
            extractedParams = [0, 0, 0, 0, 0, 0, 0, 0, 0];
            
            try
                % 1. 学習率
                if isfield(obj.params.classifier.hybrid.training.optimizer, 'learningRate')
                    extractedParams(1) = obj.params.classifier.hybrid.training.optimizer.learningRate;
                end
                
                % 2. バッチサイズ
                if isfield(obj.params.classifier.hybrid.training, 'miniBatchSize')
                    extractedParams(2) = obj.params.classifier.hybrid.training.miniBatchSize;
                end
                
                % 3. CNN層数
                if isfield(obj.params.classifier.hybrid.architecture.cnn, 'convLayers')
                    extractedParams(3) = length(fieldnames(obj.params.classifier.hybrid.architecture.cnn.convLayers));
                end
                
                % 4. CNNフィルタ数と5. フィルタサイズ
                if isfield(obj.params.classifier.hybrid.architecture.cnn, 'convLayers')
                    convFields = fieldnames(obj.params.classifier.hybrid.architecture.cnn.convLayers);
                    if ~isempty(convFields)
                        firstConv = obj.params.classifier.hybrid.architecture.cnn.convLayers.(convFields{1});
                        if isfield(firstConv, 'filters')
                            extractedParams(4) = firstConv.filters;
                        end
                        if isfield(firstConv, 'size')
                            if length(firstConv.size) >= 1
                                extractedParams(5) = firstConv.size(1);
                            end
                        end
                    end
                end
                
                % 6. LSTMユニット数と7. LSTM層数
                if isfield(obj.params.classifier.hybrid.architecture.lstm, 'lstmLayers')
                    lstmFields = fieldnames(obj.params.classifier.hybrid.architecture.lstm.lstmLayers);
                    extractedParams(7) = length(lstmFields);
                    if ~isempty(lstmFields)
                        firstLstm = obj.params.classifier.hybrid.architecture.lstm.lstmLayers.(lstmFields{1});
                        if isfield(firstLstm, 'numHiddenUnits')
                            extractedParams(6) = firstLstm.numHiddenUnits;
                        end
                    end
                end
                
                % 8. ドロップアウト率
                if isfield(obj.params.classifier.hybrid.architecture.cnn, 'dropoutLayers')
                    dropout = obj.params.classifier.hybrid.architecture.cnn.dropoutLayers;
                    dropoutFields = fieldnames(dropout);
                    if ~isempty(dropoutFields)
                        extractedParams(8) = dropout.(dropoutFields{1});
                    end
                end
                
                % 9. 全結合層ユニット数
                if isfield(obj.params.classifier.hybrid.architecture.cnn, 'fullyConnected')
                    fc = obj.params.classifier.hybrid.architecture.cnn.fullyConnected;
                    if ~isempty(fc)
                        extractedParams(9) = fc(1);
                    end
                end
            catch ME
                obj.logMessage(1, '警告: パラメータ抽出中にエラーが発生\n');
            end
        end

        %% ディープコピー作成メソッド
        function copy = createDeepCopy(obj, original)
            % 構造体のディープコピーを作成
            
            if ~isstruct(original)
                copy = original;
                return;
            end
            
            % 新しい構造体を作成
            copy = struct();
            
            % 各フィールドを再帰的にコピー
            fields = fieldnames(original);
            for i = 1:length(fields)
                field = fields{i};
                if isstruct(original.(field))
                    % 構造体の場合は再帰的にコピー
                    copy.(field) = obj.createDeepCopy(original.(field));
                elseif iscell(original.(field))
                    % セル配列の場合
                    cellArray = original.(field);
                    newCellArray = cell(size(cellArray));
                    for j = 1:numel(cellArray)
                        if isstruct(cellArray{j})
                            newCellArray{j} = obj.createDeepCopy(cellArray{j});
                        else
                            newCellArray{j} = cellArray{j};
                        end
                    end
                    copy.(field) = newCellArray;
                else
                    % その他のデータ型はそのままコピー
                    copy.(field) = original.(field);
                end
            end
        end

        %% 結果検証メソッド
        function validateResults(obj, results)
            % 結果構造体の妥当性を検証
            
            if ~isstruct(results)
                obj.logMessage(1, '警告: 結果が構造体ではありません\n');
                return;
            end
            
            % 必須フィールドのチェック
            requiredFields = {'model', 'performance', 'trainInfo'};
            for i = 1:length(requiredFields)
                if ~isfield(results, requiredFields{i})
                    obj.logMessage(1, '警告: 結果構造体に必須フィールド「%s」がありません\n', requiredFields{i});
                end
            end
            
            % パフォーマンスフィールドのチェック
            if isfield(results, 'performance')
                if ~isfield(results.performance, 'accuracy')
                    obj.logMessage(1, '警告: performance構造体にaccuracyフィールドがありません\n');
                end
            end
            
            % trainInfoフィールドのチェック
            if isfield(results, 'trainInfo')
                if ~isfield(results.trainInfo, 'cnnHistory') && ~isfield(results.trainInfo, 'lstmHistory')
                    obj.logMessage(1, '警告: trainInfo構造体にcnnHistoryまたはlstmHistoryフィールドがありません\n');
                end
            end
        end

        %% 結果構造体構築メソッド
        function results = buildResultsStruct(obj, hybridModel, metrics, trainInfo, normParams)
            % 結果構造体の構築
            
            % trainInfoの安全なディープコピー
            trainInfoCopy = obj.createDeepCopy(trainInfo);
            
            % パラメータ情報の抽出
            extractedParams = obj.extractParams();
            
            % 結果構造体の構築
            results = struct(...
                'model', hybridModel, ...
                'performance', metrics, ...
                'trainInfo', trainInfoCopy, ...
                'overfitting', [], ...
                'normParams', normParams, ...
                'params', extractedParams ...
            );
            
            % 出力前に結果の検証
            obj.validateResults(results);
            
            obj.logMessage(2, '結果構造体の構築完了\n');
        end
    end
end