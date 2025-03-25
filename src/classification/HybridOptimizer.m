classdef HybridOptimizer < handle
    %% HybridOptimizer - ハイブリッドモデル（CNN+LSTM）のハイパーパラメータ最適化クラス
    %
    % このクラスはハイブリッドモデルの性能を向上させるためにハイパーパラメータの
    % 最適化を行います。複数の探索アルゴリズムをサポートし、バランスの
    % 取れた評価メトリクスに基づいて最適なモデルを選択します。
    %
    % 主な機能:
    %   - Latin Hypercube Sampling, ランダム探索, グリッド探索などの実装
    %   - 複数のパラメータ組み合わせの評価
    %   - 過学習と複雑性を考慮した包括的な評価メトリクス
    %   - 早期停止による効率的な探索
    %   - 詳細な最適化結果の分析と可視化
    %
    % 使用例:
    %   params = getConfig('epocx', 'preset', 'template');
    %   optimizer = HybridOptimizer(params);
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
        earlyStopParams     % 早期停止パラメータ
        
        % 評価重み
        evaluationWeights   % 各評価指標の重みづけ
        
        % GPUメモリ監視
        gpuMemory           % GPU使用メモリ監視
    end
    
    methods (Access = public)
        %% コンストラクタ - 初期化処理
        function obj = HybridOptimizer(params)
            % HybridOptimizerのインスタンスを初期化
            %
            % 入力:
            %   params - 設定パラメータ（getConfig関数から取得）
            
            obj.params = params;
            obj.initializeSearchSpace();
            obj.bestPerformance = -inf;
            obj.optimizationHistory = struct('params', {}, 'performance', {}, 'model', {});
            obj.useGPU = params.classifier.hybrid.gpu;
            obj.maxTrials = params.classifier.hybrid.optimization.maxTrials;
            
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
            
            % GPUメモリ監視の初期化
            obj.gpuMemory = struct('total', 0, 'used', 0, 'peak', 0);
            
            % GPU利用可能性のチェック
            if obj.useGPU
                try
                    gpuInfo = gpuDevice();
                    fprintf('GPUが検出されました: %s (メモリ: %.2f GB)\n', ...
                        gpuInfo.Name, gpuInfo.TotalMemory/1e9);
                    
                    % GPU情報の初期化
                    obj.gpuMemory.total = gpuInfo.TotalMemory/1e9;
                    obj.gpuMemory.used = (gpuInfo.TotalMemory - gpuInfo.AvailableMemory)/1e9;
                    obj.gpuMemory.peak = obj.gpuMemory.used;
                catch
                    warning('GPU使用が指定されていますが、GPUが利用できません。CPUで実行します。');
                    obj.useGPU = false;
                end
            end
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
                    fprintf('ハイブリッドモデル最適化は設定で無効化されています。デフォルトパラメータを使用します。\n');
                    results = obj.createDefaultResults();
                    return;
                end
        
                fprintf('\n=== ハイブリッドモデルのハイパーパラメータ最適化を開始 ===\n');
                
                % 探索アルゴリズムの選択
                searchAlgorithm = 'lhs';  % デフォルト: Latin Hypercube Sampling
                if isfield(obj.params.classifier.hybrid.optimization, 'searchAlgorithm')
                    searchAlgorithm = obj.params.classifier.hybrid.optimization.searchAlgorithm;
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
                        
                        % GPUメモリの確認
                        if obj.useGPU
                            obj.checkGPUMemory();
                        end
                        
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
                            'crossValidation', trainResults.crossValidation ...
                        );
        
                        % モデル性能の総合スコアを計算
                        performance = trainResults.performance.accuracy;
                        
                        % 検証精度データの取得
                        valAccuracy = 0;
                        if isfield(trainResults.trainInfo, 'hybridValMetrics') && ...
                           isfield(trainResults.trainInfo.hybridValMetrics, 'accuracy')
                            valAccuracy = trainResults.trainInfo.hybridValMetrics.accuracy;
                        elseif isfield(trainResults.trainInfo.cnnHistory, 'ValidationAccuracy') && ...
                           ~isempty(trainResults.trainInfo.cnnHistory.ValidationAccuracy)
                            cnnValAcc = trainResults.trainInfo.cnnHistory.ValidationAccuracy;
                            lstmValAcc = trainResults.trainInfo.lstmHistory.ValidationAccuracy;
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
                        
                        % 早期停止のチェック
                        if obj.earlyStopParams.enable
                            obj.earlyStopParams.history = [obj.earlyStopParams.history; evaluationScore];
                            
                            if evaluationScore > obj.earlyStopParams.best_score + obj.earlyStopParams.min_delta
                                obj.earlyStopParams.best_score = evaluationScore;
                                obj.earlyStopParams.counter = 0;
                                fprintf('新しい最良スコア: %.4f\n', evaluationScore);
                            else
                                obj.earlyStopParams.counter = obj.earlyStopParams.counter + 1;
                                fprintf('スコア改善なし: %d/%d\n', ...
                                    obj.earlyStopParams.counter, obj.earlyStopParams.patience);
                                
                                if obj.earlyStopParams.counter >= obj.earlyStopParams.patience
                                    fprintf('\n=== 早期停止条件を満たしました (%d試行後) ===\n', i);
                                    fprintf('最良スコア: %.4f\n', obj.earlyStopParams.best_score);
                                    break;
                                end
                            end
                        end
                        
                        % GPUメモリの解放
                        if obj.useGPU
                            obj.resetGPUMemory();
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
        
                        % GPUメモリの解放
                        if obj.useGPU
                            obj.resetGPUMemory();
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
                
                fprintf('\n=== ハイブリッドモデル最適化が完了しました ===\n');
        
            catch ME
                fprintf('\n=== ハイブリッドモデル最適化中にエラーが発生しました ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラースタック:\n');
                disp(getReport(ME, 'extended'));
                
                % エラー回復処理
                results = obj.handleOptimizationError(ME);
            end
        end
    end
    
    methods (Access = private)
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
            
            % 探索空間の詳細表示
            fprintf('\n探索空間の範囲 (ハイブリッドモデル):\n');
            fprintf('  - 学習率: [%.6f, %.6f]\n', obj.searchSpace.learningRate(1), obj.searchSpace.learningRate(2));
            fprintf('  - バッチサイズ: [%d, %d]\n', obj.searchSpace.miniBatchSize(1), obj.searchSpace.miniBatchSize(2));
            fprintf('  - CNN層数: [%d, %d]\n', obj.searchSpace.numConvLayers(1), obj.searchSpace.numConvLayers(2));
            fprintf('  - CNNフィルタ数: [%d, %d]\n', obj.searchSpace.cnnFilters(1), obj.searchSpace.cnnFilters(2));
            fprintf('  - フィルタサイズ: [%d, %d]\n', obj.searchSpace.filterSize(1), obj.searchSpace.filterSize(2));
            fprintf('  - LSTMユニット数: [%d, %d]\n', obj.searchSpace.lstmUnits(1), obj.searchSpace.lstmUnits(2));
            fprintf('  - LSTM層数: [%d, %d]\n', obj.searchSpace.numLstmLayers(1), obj.searchSpace.numLstmLayers(2));
            fprintf('  - ドロップアウト率: [%.2f, %.2f]\n', obj.searchSpace.dropoutRate(1), obj.searchSpace.dropoutRate(2));
            fprintf('  - 全結合層ユニット数: [%d, %d]\n', obj.searchSpace.fcUnits(1), obj.searchSpace.fcUnits(2));
            
            % パラメータ空間のサイズ（9次元）
            paramSpace = 9; % 9つのパラメータを最適化
            fprintf('パラメータ空間: %d次元\n', paramSpace);
        end
        
        %% パラメータセット生成メソッド
        function paramSets = generateParameterSets(obj, numTrials, algorithm)
            % 指定されたアルゴリズムでパラメータセットを生成
            %
            % 入力:
            %   numTrials - 試行回数
            %   algorithm - 探索アルゴリズム ('lhs', 'random', 'grid')
            %
            % 出力:
            %   paramSets - パラメータセット行列 [numTrials x 9]
            
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
            
            % パラメータセット品質チェック
            obj.validateParameterSets(paramSets);
        end
        
        %% パラメータセット検証メソッド
        function validateParameterSets(obj, paramSets)
            % 生成されたパラメータセットの品質チェック
            
            % 無効な値のチェック
            if any(isnan(paramSets(:)))
                warning('生成されたパラメータセットにNaN値が含まれています');
            end
            
            if any(isinf(paramSets(:)))
                warning('生成されたパラメータセットにInf値が含まれています');
            end
            
            % 範囲外のパラメータをチェック
            if any(paramSets(:,1) < obj.searchSpace.learningRate(1)) || any(paramSets(:,1) > obj.searchSpace.learningRate(2))
                warning('学習率パラメータに範囲外の値があります');
            end
            
            % パラメータの多様性をチェック
            uniqueValues = zeros(1, size(paramSets, 2));
            for i = 1:size(paramSets, 2)
                uniqueValues(i) = length(unique(paramSets(:,i)));
            end
            
            if any(uniqueValues < 3) && size(paramSets, 1) >= 5
                warning('一部のパラメータ次元で多様性が低いです（%d次元目: %d個のユニーク値）', ...
                    find(uniqueValues < 3, 1), uniqueValues(find(uniqueValues < 3, 1)));
            end
            
            fprintf('パラメータセット検証完了\n');
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
            
            fprintf('Latin Hypercube Samplingによりパラメータセットを生成しました（%d組）\n', numTrials);
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
            
            fprintf('ランダムサンプリングによりパラメータセットを生成しました（%d組）\n', numTrials);
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
            
            % グリッド数が多すぎる場合は警告し、サブサンプリング
            if size(paramSets, 1) > 200
                warning(['グリッドサーチのパラメータ数が非常に多いです (%d組)。' ...
                    '最適化には長時間かかる可能性があります。上位200組をサンプリングします。'], size(paramSets, 1));
                rng('default'); % 再現性のため
                subsampleIdx = randperm(size(paramSets, 1), 200);
                paramSets = paramSets(subsampleIdx, :);
            end
            
            fprintf('グリッドサーチによりパラメータセットを生成しました（%d組）\n', size(paramSets, 1));
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
                
                % CNN側のドロップアウトレイヤー更新
                cnnDropoutLayers = struct();
                for j = 1:numConvLayers
                    cnnDropoutLayers.(['dropout' num2str(j)]) = dropoutRate;
                end
                params.classifier.hybrid.architecture.cnn.dropoutLayers = cnnDropoutLayers;
                
                % 5. 全結合層ユニット数の更新
                params.classifier.hybrid.architecture.cnn.fullyConnected = [round(paramSet(9))];
                params.classifier.hybrid.architecture.lstm.fullyConnected = [round(paramSet(9))];
                
                return;
                
            catch ME
                fprintf('パラメータ更新中にエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end
        
        %% 現在のパラメータ表示メソッド
        function displayCurrentParams(~, paramSet)
            % 現在評価中のパラメータセットを表示
            
            fprintf('\n評価対象のパラメータ構成:\n');
            fprintf('  - 学習率: %.6f\n', paramSet(1));
            fprintf('  - バッチサイズ: %d\n', paramSet(2));
            fprintf('  - CNN層数: %d\n', paramSet(3));
            fprintf('  - CNNフィルタ数: %d\n', paramSet(4));
            fprintf('  - フィルタサイズ: %d\n', paramSet(5));
            fprintf('  - LSTMユニット数: %d\n', paramSet(6));
            fprintf('  - LSTM層数: %d\n', paramSet(7));
            fprintf('  - ドロップアウト率: %.2f\n', paramSet(8));
            fprintf('  - 全結合層ユニット数: %d\n', paramSet(9));
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
                fprintf('F1スコア計算でエラー: %s\n', ME.message);
                f1Score = 0;
            end
        end
        
        %% モデル評価スコア計算メソッド
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
            
            try
                % 1. 基本精度スコアの計算
                testWeight = obj.evaluationWeights.test;
                valWeight = obj.evaluationWeights.validation;
                
                % 検証精度かクロスバリデーション精度のいずれかを選択
                validationScore = 0;
                if cvAccuracy > 0
                    % 交差検証があれば優先的に使用
                    validationScore = cvAccuracy;
                    valWeight = obj.evaluationWeights.crossValidation;
                    
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
                if testWeight + valWeight > 0
                    accuracyScore = (testWeight * testAccuracy + valWeight * validationScore) / (testWeight + valWeight);
                else
                    accuracyScore = testAccuracy;
                end
                
                % 2. F1スコアの統合（クラス不均衡への対応）
                combinedScore = accuracyScore;
                if f1Score > 0
                    f1Weight = obj.evaluationWeights.f1Score;
                    combinedScore = (1 - f1Weight) * accuracyScore + f1Weight * f1Score;
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
                complexityPenalty = 0;
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
                    complexityPenalty = 0.05; % デフォルトの中程度の複雑性ペナルティ
                end
                
                % 5. 最終スコアの計算 (精度 - 過学習ペナルティ - 複雑性ペナルティ)
                score = combinedScore * (1 - overfitPenalty - complexityPenalty);
                
                % 詳細なスコア計算ログ出力（デバッグ用）
                fprintf('スコア計算詳細:\n');
                fprintf('  - テスト精度: %.4f (重み: %.2f)\n', testAccuracy, testWeight);
                fprintf('  - 検証スコア: %.4f (重み: %.2f)\n', validationScore, valWeight);
                fprintf('  - F1スコア: %.4f (重み: %.2f)\n', f1Score, obj.evaluationWeights.f1Score);
                fprintf('  - 基本精度スコア: %.4f\n', accuracyScore);
                fprintf('  - 複合精度スコア: %.4f\n', combinedScore);
                fprintf('  - 過学習ペナルティ: %.4f\n', overfitPenalty);
                fprintf('  - 複雑性ペナルティ: %.4f\n', complexityPenalty);
                fprintf('  - 最終スコア: %.4f\n', score);
                
            catch ME
                fprintf('スコア計算でエラー: %s\n', ME.message);
            end
        end

        %% 最終結果処理メソッド
        function [bestResults, summary] = processFinalResults(obj, results)
            % 全ての試行結果を処理し、最良のモデルを選択
            
            try
                fprintf('\n=== パラメータ最適化の結果処理 ===\n');
                fprintf('総試行回数: %d\n', length(results));
                
                % 有効な結果のみを抽出
                validResults = results(~cellfun(@isempty, results));
                validResults = validResults(~cellfun(@(x) isfield(x, 'error') && x.error, validResults));
                numResults = length(validResults);
                
                fprintf('有効なパラメータセット数: %d\n', numResults);
                fprintf('無効な試行数: %d\n', length(results) - numResults);
                
                % 結果がない場合のエラー処理
                if numResults == 0
                    error('有効な結果がありません。全ての試行が失敗しました。');
                end
                
                % 評価結果保存用の変数
                modelScores = zeros(numResults, 1);
                testAccuracies = zeros(numResults, 1);
                valAccuracies = zeros(numResults, 1);
                cvAccuracies = zeros(numResults, 1);
                f1Scores = zeros(numResults, 1);
                overfitPenalties = zeros(numResults, 1);
                complexityPenalties = zeros(numResults, 1);
                isOverfitFlags = false(numResults, 1);
                
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
                    'num_conv_layers', [], ...
                    'cnn_filters', [], ...
                    'filter_sizes', [], ...
                    'lstm_units', [], ...
                    'num_lstm_layers', [], ...
                    'dropout_rates', [], ...
                    'fc_units', [] ...
                );
                
                fprintf('\n=== 各試行の詳細評価 ===\n');
                for i = 1:numResults
                    result = validResults{i};
                    
                    try
                        if ~isempty(result) && isfield(result, 'model') && ~isempty(result.model) && ...
                           isfield(result, 'performance') && ~isempty(result.performance)
                            
                            % 基本的な精度スコア
                            testAccuracy = result.performance.accuracy;
                            testAccuracies(i) = testAccuracy;
                            
                            % 検証精度の取得
                            valAccuracy = 0;
                            if isfield(result.trainInfo, 'hybridValMetrics') && ...
                               isfield(result.trainInfo.hybridValMetrics, 'accuracy')
                                valAccuracy = result.trainInfo.hybridValMetrics.accuracy;
                            elseif isfield(result.trainInfo, 'cnnHistory') && ...
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
                            
                            valAccuracies(i) = valAccuracy;
                            
                            % 交差検証精度の取得
                            cvAccuracy = 0;
                            if isfield(result, 'crossValidation') && ...
                               isfield(result.crossValidation, 'meanAccuracy')
                                cvAccuracy = result.crossValidation.meanAccuracy;
                                cvAccuracies(i) = cvAccuracy;
                            end
                            
                            % F1スコアの計算
                            f1Score = obj.calculateMeanF1Score(result.performance);
                            f1Scores(i) = f1Score;
                            
                            % 過学習ペナルティ計算
                            overfitPenalty = 0;
                            severity = 'none';
                            
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
                                                isOverfitFlags(i) = true;
                                            case 'severe'
                                                overfitPenalty = 0.3;  % 30%ペナルティ
                                                isOverfitFlags(i) = true;
                                            case 'moderate'
                                                overfitPenalty = 0.2;  % 20%ペナルティ
                                                isOverfitFlags(i) = true;
                                            case 'mild'
                                                overfitPenalty = 0.1;  % 10%ペナルティ
                                                isOverfitFlags(i) = true;
                                            otherwise
                                                overfitPenalty = 0;    % ペナルティなし
                                        end
                                    end
                                end
                            end
                            
                            overfitPenalties(i) = overfitPenalty;
                            
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
                            end
                            
                            complexityPenalties(i) = complexityPenalty;
                            
                            % 総合スコアの計算
                            score = obj.calculateTrialScore(testAccuracy, valAccuracy, cvAccuracy, f1Score, result);
                            modelScores(i) = score;
                            
                            % 過学習フラグの設定
                            if isOverfitFlags(i)
                                summary.overfit_models = summary.overfit_models + 1;
                            end
                            
                            % サマリー情報の更新
                            if isfield(result, 'params') && length(result.params) >= 9
                                summary.learning_rates(end+1) = result.params(1);
                                summary.batch_sizes(end+1) = result.params(2);
                                summary.num_conv_layers(end+1) = result.params(3);
                                summary.cnn_filters(end+1) = result.params(4);
                                summary.filter_sizes(end+1) = result.params(5);
                                summary.lstm_units(end+1) = result.params(6);
                                summary.num_lstm_layers(end+1) = result.params(7);
                                summary.dropout_rates(end+1) = result.params(8);
                                summary.fc_units(end+1) = result.params(9);
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
                            
                            fprintf('  - 過学習判定: %s\n', string(isOverfitFlags(i)));
                            fprintf('  - 重大度: %s\n', severity);
                            
                            fprintf('複合スコア:\n');
                            fprintf('  - 過学習ペナルティ: %.2f\n', overfitPenalty);
                            fprintf('  - 複雑性ペナルティ: %.2f\n', complexityPenalty);
                            fprintf('  - 最終スコア: %.4f\n', score);
                            
                            % 構造情報
                            fprintf('モデル構造:\n');
                            fprintf('  - CNN層数: %d\n', round(result.params(3)));
                            fprintf('  - LSTM層数: %d\n', round(result.params(7)));
                            fprintf('  - 総パラメータ数: 約 %dk\n', round((cnnFilters * filterSize * filterSize * cnnLayers + lstmUnits * lstmUnits * 4 * lstmLayers + fcUnits * (cnnFilters + lstmUnits))/1000));
                        else
                            fprintf('\n--- パラメータセット %d/%d: 有効なモデルがありません ---\n', i, numResults);
                        end
                    catch ME
                        fprintf('\n--- パラメータセット %d/%d の評価中にエラーが発生: %s ---\n', i, numResults, ME.message);
                    end
                end
                
                % 有効な結果がない場合はエラーを返す
                if all(modelScores == 0)
                    error('有効な評価結果がありません。詳細な解析ができません。');
                end
                
                % 統計サマリーの計算
                if ~isempty(testAccuracies)
                    validTestAcc = testAccuracies(testAccuracies > 0);
                    if ~isempty(validTestAcc)
                        summary.best_accuracy = max(validTestAcc);
                        summary.worst_accuracy = min(validTestAcc);
                        summary.mean_accuracy = mean(validTestAcc);
                    end
                end
                
                % モデル選択の詳細情報
                fprintf('\n=== モデルスコアの分布 ===\n');
                if ~isempty(modelScores) && any(modelScores > 0)
                    validScores = modelScores(modelScores > 0);
                    scorePercentiles = prctile(validScores, [0, 25, 50, 75, 100]);
                    fprintf('  - 最小値: %.4f\n', scorePercentiles(1));
                    fprintf('  - 25パーセンタイル: %.4f\n', scorePercentiles(2));
                    fprintf('  - 中央値: %.4f\n', scorePercentiles(3));
                    fprintf('  - 75パーセンタイル: %.4f\n', scorePercentiles(4));
                    fprintf('  - 最大値: %.4f\n', scorePercentiles(5));
                end
                
                % 最良モデルの選択戦略
                fprintf('\n=== 最良モデル選択戦略 ===\n');
                fprintf('過学習モデル数: %d/%d (%.1f%%)\n', ...
                    summary.overfit_models, numResults, (summary.overfit_models/numResults)*100);
                
                % 過学習していないモデルがあるかをチェック
                nonOverfitIndices = find(~isOverfitFlags);
                
                % 選択方法の決定
                if ~isempty(nonOverfitIndices) && length(nonOverfitIndices) >= 3
                    fprintf('過学習していないモデルが十分にあります。これらから最良モデルを選択します。\n');
                    [bestNonOverfitScore, bestLocalIdx] = max(modelScores(nonOverfitIndices));
                    bestIdx = nonOverfitIndices(bestLocalIdx);
                    fprintf('非過学習モデルから選択 - 最良スコア: %.4f\n', bestNonOverfitScore);
                    
                    % 全体最良スコアとの比較
                    [globalBestScore, globalBestIdx] = max(modelScores);
                    if globalBestScore > bestNonOverfitScore * 1.05 % 5%以上の差がある場合
                        fprintf('警告: 過学習モデルにより高いスコア (%.4f, +%.1f%%) があります\n', ...
                            globalBestScore, (globalBestScore/bestNonOverfitScore-1)*100);
                    end
                else
                    fprintf('過学習していないモデルが不足しています。全モデルから最良を選択します。\n');
                    [~, bestIdx] = max(modelScores);
                    
                    if isOverfitFlags(bestIdx)
                        fprintf('警告: 選択された最良モデルは過学習の兆候があります\n');
                    end
                end
                
                % 最良モデルの選択
                bestResults = validResults{bestIdx};
                
                fprintf('\n最良モデル選択 (インデックス: %d)\n', bestIdx);
                fprintf('  - 最終スコア: %.4f\n', modelScores(bestIdx));
                fprintf('  - テスト精度: %.4f\n', testAccuracies(bestIdx));
                
                if bestIdx <= length(valAccuracies) && valAccuracies(bestIdx) > 0
                    fprintf('  - 検証精度: %.4f\n', valAccuracies(bestIdx));
                end
                
                if bestIdx <= length(cvAccuracies) && cvAccuracies(bestIdx) > 0
                    fprintf('  - 交差検証精度: %.4f\n', cvAccuracies(bestIdx));
                end
                
                if bestIdx <= length(f1Scores) && f1Scores(bestIdx) > 0
                    fprintf('  - 平均F1スコア: %.4f\n', f1Scores(bestIdx));
                end
                
                if bestIdx <= length(overfitPenalties)
                    fprintf('  - 過学習ペナルティ: %.2f\n', overfitPenalties(bestIdx));
                end
                
                if bestIdx <= length(complexityPenalties)
                    fprintf('  - 複雑性ペナルティ: %.2f\n', complexityPenalties(bestIdx));
                end
                
                % 最良パラメータの保存
                if isfield(bestResults, 'params')
                    obj.bestParams = bestResults.params;
                    obj.bestPerformance = modelScores(bestIdx);
                    obj.optimizedModel = bestResults.model;

                    % 上位モデルのパラメータ傾向分析
                    topN = min(5, numResults);
                    if topN > 0
                        [~, topIndices] = sort(modelScores, 'descend');
                        topIndices = topIndices(1:topN);
                        
                        % パラメータ情報のあるモデルを集計
                        top_params = [];
                        valid_top_count = 0;
                        
                        for j = 1:length(topIndices)
                            if isfield(validResults{topIndices(j)}, 'params') && ...
                               length(validResults{topIndices(j)}.params) >= 9
                                valid_top_count = valid_top_count + 1;
                                top_params(valid_top_count, :) = validResults{topIndices(j)}.params;
                            end
                        end
                        
                        if valid_top_count > 0
                            fprintf('\n上位%dモデルのパラメータ傾向:\n', valid_top_count);
                            fprintf('  - 平均学習率: %.6f\n', mean(top_params(:, 1)));
                            fprintf('  - 平均バッチサイズ: %.1f\n', mean(top_params(:, 2)));
                            fprintf('  - 平均CNN層数: %.1f\n', mean(top_params(:, 3)));
                            fprintf('  - 平均CNNフィルタ数: %.1f\n', mean(top_params(:, 4)));
                            fprintf('  - 平均フィルタサイズ: %.1f\n', mean(top_params(:, 5)));
                            fprintf('  - 平均LSTMユニット数: %.1f\n', mean(top_params(:, 6)));
                            fprintf('  - 平均LSTM層数: %.1f\n', mean(top_params(:, 7)));
                            fprintf('  - 平均ドロップアウト率: %.2f\n', mean(top_params(:, 8)));
                            fprintf('  - 平均FC層ユニット数: %.1f\n', mean(top_params(:, 9)));
                        else
                            fprintf('\n上位モデルに有効なパラメータ情報がありません\n');
                        end
                    else
                        fprintf('\n上位モデル分析に十分な有効結果がありません\n');
                    end
                else
                    fprintf('\n最良モデルにパラメータ情報がありません\n');
                end
                
                return;
        
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
                        'trainInfo', struct('cnnHistory', struct(), 'lstmHistory', struct()), ...
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
                
                fprintf('警告: エラーが発生しましたが、可能な限り処理を続行します。\n');
            end
        end
        
        %% 最適化履歴更新メソッド
        function updateOptimizationHistory(obj, results)
            % 最適化の履歴を更新
            
            try
                % 有効な結果のみを抽出
                validResults = results(~cellfun(@isempty, results));
                validResults = validResults(~cellfun(@(x) isfield(x, 'error') && x.error, validResults));
                
                % 履歴に追加
                for i = 1:length(validResults)
                    result = validResults{i};
                    if isfield(result, 'model') && ~isempty(result.model) && ...
                       isfield(result, 'performance') && ~isempty(result.performance)
                        
                        % テスト精度と検証精度の取得
                        testAccuracy = result.performance.accuracy;
                        
                        valAccuracy = 0;
                        if isfield(result.trainInfo, 'hybridValMetrics') && ...
                           isfield(result.trainInfo.hybridValMetrics, 'accuracy')
                            valAccuracy = result.trainInfo.hybridValMetrics.accuracy;
                        elseif isfield(result.trainInfo, 'cnnHistory') && ...
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
                        
                        cvAccuracy = 0;
                        if isfield(result, 'crossValidation') && ...
                           isfield(result.crossValidation, 'meanAccuracy')
                            cvAccuracy = result.crossValidation.meanAccuracy;
                        end
                        
                        f1Score = obj.calculateMeanF1Score(result.performance);
                        
                        % 総合スコアの計算
                        score = obj.calculateTrialScore(testAccuracy, valAccuracy, cvAccuracy, f1Score, result);
                        
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
                        
                        % 履歴に追加
                        newEntry = struct(...
                            'params', result.params, ...
                            'testAccuracy', testAccuracy, ...
                            'valAccuracy', valAccuracy, ...
                            'cvAccuracy', cvAccuracy, ...
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
                    end
                end
                
                % スコアで降順にソート
                if ~isempty(obj.optimizationHistory)
                    [~, sortIdx] = sort([obj.optimizationHistory.score], 'descend');
                    obj.optimizationHistory = obj.optimizationHistory(sortIdx);
                    
                    fprintf('\n最適化履歴を更新しました（計 %d 個のモデル）\n', ...
                        length(obj.optimizationHistory));
                    
                    % 改善の傾向分析
                    if length(obj.optimizationHistory) >= 3
                        scores = [obj.optimizationHistory.score];
                        improvement = diff(scores(1:min(10, length(scores))));
                        avgImprovement = mean(improvement(improvement > 0));
                        
                        fprintf('最適化の進行状況:\n');
                        fprintf('  - 最良スコア: %.4f\n', scores(1));
                        fprintf('  - 平均改善率: %.6f\n', avgImprovement);
                        fprintf('  - 収束性: %s\n', string(avgImprovement < 0.001));
                    end
                end
                
            catch ME
                fprintf('最適化履歴の更新に失敗: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
            end
        end
        
        %% 最適化サマリー表示メソッド
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
            if ~isempty(summary.learning_rates)
                fprintf('\n学習率:\n');
                fprintf('  - 平均: %.6f\n', mean(summary.learning_rates));
                fprintf('  - 標準偏差: %.6f\n', std(summary.learning_rates));
                fprintf('  - 最小: %.6f\n', min(summary.learning_rates));
                fprintf('  - 最大: %.6f\n', max(summary.learning_rates));
            end
            
            % バッチサイズの統計
            if ~isempty(summary.batch_sizes)
                fprintf('\nバッチサイズ:\n');
                fprintf('  - 平均: %.1f\n', mean(summary.batch_sizes));
                fprintf('  - 標準偏差: %.1f\n', std(summary.batch_sizes));
                fprintf('  - 最小: %d\n', min(summary.batch_sizes));
                fprintf('  - 最大: %d\n', max(summary.batch_sizes));
            end
            
            % CNN層数の統計
            if ~isempty(summary.num_conv_layers)
                fprintf('\nCNN層数:\n');
                fprintf('  - 平均: %.1f\n', mean(summary.num_conv_layers));
                fprintf('  - 標準偏差: %.1f\n', std(summary.num_conv_layers));
                fprintf('  - 最小: %d\n', min(summary.num_conv_layers));
                fprintf('  - 最大: %d\n', max(summary.num_conv_layers));
            end
            
            % CNNフィルタ数の統計
            if ~isempty(summary.cnn_filters)
                fprintf('\nCNNフィルタ数:\n');
                fprintf('  - 平均: %.1f\n', mean(summary.cnn_filters));
                fprintf('  - 標準偏差: %.1f\n', std(summary.cnn_filters));
                fprintf('  - 最小: %d\n', min(summary.cnn_filters));
                fprintf('  - 最大: %d\n', max(summary.cnn_filters));
            end
            
            % フィルタサイズの統計
            if ~isempty(summary.filter_sizes)
                fprintf('\nフィルタサイズ:\n');
                fprintf('  - 平均: %.1f\n', mean(summary.filter_sizes));
                fprintf('  - 標準偏差: %.1f\n', std(summary.filter_sizes));
                fprintf('  - 最小: %d\n', min(summary.filter_sizes));
                fprintf('  - 最大: %d\n', max(summary.filter_sizes));
            end
            
            % LSTMユニット数の統計
            if ~isempty(summary.lstm_units)
                fprintf('\nLSTMユニット数:\n');
                fprintf('  - 平均: %.1f\n', mean(summary.lstm_units));
                fprintf('  - 標準偏差: %.1f\n', std(summary.lstm_units));
                fprintf('  - 最小: %d\n', min(summary.lstm_units));
                fprintf('  - 最大: %d\n', max(summary.lstm_units));
            end
            
            % LSTM層数の統計
            if ~isempty(summary.num_lstm_layers)
                fprintf('\nLSTM層数:\n');
                fprintf('  - 平均: %.1f\n', mean(summary.num_lstm_layers));
                fprintf('  - 標準偏差: %.1f\n', std(summary.num_lstm_layers));
                fprintf('  - 最小: %d\n', min(summary.num_lstm_layers));
                fprintf('  - 最大: %d\n', max(summary.num_lstm_layers));
            end
            
            % ドロップアウト率の統計
            if ~isempty(summary.dropout_rates)
                fprintf('\nドロップアウト率:\n');
                fprintf('  - 平均: %.3f\n', mean(summary.dropout_rates));
                fprintf('  - 標準偏差: %.3f\n', std(summary.dropout_rates));
                fprintf('  - 最小: %.3f\n', min(summary.dropout_rates));
                fprintf('  - 最大: %.3f\n', max(summary.dropout_rates));
            end
            
            % 全結合層ユニット数の統計
            if ~isempty(summary.fc_units)
                fprintf('\n全結合層ユニット数:\n');
                fprintf('  - 平均: %.1f\n', mean(summary.fc_units));
                fprintf('  - 標準偏差: %.1f\n', std(summary.fc_units));
                fprintf('  - 最小: %d\n', min(summary.fc_units));
                fprintf('  - 最大: %d\n', max(summary.fc_units));
            end
            
            % パラメータ間の相関分析
            if all([~isempty(summary.learning_rates), ~isempty(summary.batch_sizes), ...
                    ~isempty(summary.num_conv_layers), ~isempty(summary.cnn_filters), ...
                    ~isempty(summary.lstm_units), ~isempty(summary.num_lstm_layers), ...
                    ~isempty(summary.dropout_rates), ~isempty(summary.fc_units)])
                
                fprintf('\nパラメータ間の相関分析:\n');
                paramMatrix = [summary.learning_rates', summary.batch_sizes', ...
                             summary.num_conv_layers', summary.cnn_filters', ...
                             summary.filter_sizes', summary.lstm_units', ...
                             summary.num_lstm_layers', summary.dropout_rates', ...
                             summary.fc_units'];
                
                paramNames = {'学習率', 'バッチサイズ', 'CNN層数', ...
                             'CNNフィルタ数', 'フィルタサイズ', ...
                             'LSTMユニット数', 'LSTM層数', 'ドロップアウト率', ...
                             '全結合層ユニット数'};
                
                % 相関行列の計算
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
                
                % 最適化履歴からパラメータと性能の相関分析
                if ~isempty(obj.optimizationHistory) && length(obj.optimizationHistory) > 2
                    fprintf('\n各パラメータと性能の相関:\n');
                    scores = [obj.optimizationHistory.score]';
                    
                    for i = 1:length(paramNames)
                        if length(paramMatrix) >= i
                            correlation = corr(paramMatrix(:,i), scores);
                            if abs(correlation) > 0.2
                                direction = '';
                                if correlation > 0
                                    direction = '正の相関 (↑)';
                                else
                                    direction = '負の相関 (↓)';
                                end
                                fprintf('  - %s: %.3f (%s)\n', paramNames{i}, correlation, direction);
                            end
                        end
                    end
                end
            end
            
            % 最適化の収束性評価
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
            
            % GPU使用状況
            if obj.useGPU && obj.gpuMemory.peak > 0
                fprintf('\nGPU使用状況:\n');
                fprintf('  - 最大使用メモリ: %.2f GB\n', obj.gpuMemory.peak);
                fprintf('  - 総メモリ: %.2f GB\n', obj.gpuMemory.total);
                fprintf('  - 使用率: %.1f%%\n', (obj.gpuMemory.peak / max(obj.gpuMemory.total, 1)) * 100);
            end
            
            % 最適パラメータの表示
            if ~isempty(obj.bestParams)
                fprintf('\n=== 最適なパラメータ ===\n');
                fprintf('  - 学習率: %.6f\n', obj.bestParams(1));
                fprintf('  - バッチサイズ: %d\n', obj.bestParams(2));
                fprintf('  - CNN層数: %d\n', obj.bestParams(3));
                fprintf('  - CNNフィルタ数: %d\n', obj.bestParams(4));
                fprintf('  - フィルタサイズ: %d\n', obj.bestParams(5));
                fprintf('  - LSTMユニット数: %d\n', obj.bestParams(6));
                fprintf('  - LSTM層数: %d\n', obj.bestParams(7));
                fprintf('  - ドロップアウト率: %.2f\n', obj.bestParams(8));
                fprintf('  - 全結合層ユニット数: %d\n', obj.bestParams(9));
                fprintf('  - 達成スコア: %.4f\n', obj.bestPerformance);
            end
        end
        
        %% GPUメモリ確認メソッド
        function checkGPUMemory(obj)
            % GPU使用状況の確認とメモリ使用率の監視
            
            if obj.useGPU
                try
                    % GPUデバイス情報の取得
                    gpuInfo = gpuDevice();
                    totalMem = gpuInfo.TotalMemory / 1e9;  % GB
                    availMem = gpuInfo.AvailableMemory / 1e9;  % GB
                    usedMem = totalMem - availMem;
                    
                    % メモリ使用情報の更新
                    obj.gpuMemory.total = totalMem;
                    obj.gpuMemory.used = usedMem;
                    obj.gpuMemory.peak = max(obj.gpuMemory.peak, usedMem);
                    
                    fprintf('GPU使用状況: %.2f/%.2f GB (%.1f%%)\n', ...
                        usedMem, totalMem, (usedMem/totalMem)*100);
                    
                    % メモリ使用率が高い場合は警告と対応
                    if usedMem/totalMem > 0.8
                        warning('GPU使用率が80%%を超えています。パフォーマンスに影響する可能性があります。');
                        
                        % 適応的なバッチサイズ調整
                        if obj.params.classifier.hybrid.training.miniBatchSize > 32
                            newBatchSize = max(16, floor(obj.params.classifier.hybrid.training.miniBatchSize / 2));
                            fprintf('高メモリ使用率のため、バッチサイズを%dから%dに削減します\n', ...
                                obj.params.classifier.hybrid.training.miniBatchSize, newBatchSize);
                            obj.params.classifier.hybrid.training.miniBatchSize = newBatchSize;
                        end
                        
                        % GPUメモリの解放を試みる
                        obj.resetGPUMemory();
                    end
                catch ME
                    warning('GPUメモリチェックでエラー');
                end
            end
        end
        
        %% GPUメモリ解放メソッド
        function resetGPUMemory(obj)
            % GPUメモリのリセットと解放
            
            if obj.useGPU
                try
                    % GPUデバイスのリセット
                    currentDevice = gpuDevice();
                    reset(currentDevice);
                    
                    % リセット後のメモリ状況
                    availMem = currentDevice.AvailableMemory / 1e9;  % GB
                    totalMem = currentDevice.TotalMemory / 1e9;  % GB
                    fprintf('GPUメモリをリセットしました (利用可能: %.2f/%.2f GB)\n', ...
                        availMem, totalMem);
                        
                    % 使用状況の更新
                    obj.gpuMemory.used = totalMem - availMem;
                    
                catch ME
                    fprintf('GPUメモリのリセットに失敗: %s\n', ME.message);
                    fprintf('エラー詳細:\n');
                    disp(getReport(ME, 'extended'));
                end
            end
        end
        
        %% デフォルト結果作成メソッド
        function results = createDefaultResults(obj)
            % 最適化無効時のデフォルト結果構造体を生成
            
            fprintf('ハイブリッドモデル最適化はスキップされました\n');
            
            % デフォルトパラメータで結果構造体を作成
            results = struct(...
                'model', [], ...
                'performance', [], ...
                'trainInfo', struct('cnnHistory', struct(), 'lstmHistory', struct()), ...
                'crossValidation', [], ...
                'overfitting', [], ...
                'normParams', [] ...
            );
            
            return;
        end
        
        %% 最適化エラーハンドリングメソッド
        function results = handleOptimizationError(obj, errorObj)
            % 最適化中のエラーを処理し、回復可能な結果を返す
            %
            % 入力:
            %   errorObj - エラーオブジェクト
            %
            % 出力:
            %   results - 回復された結果構造体
            
            fprintf('\n=== 最適化エラーからの回復を試行 ===\n');
            
            % 最適化履歴に有効な結果があるか確認
            if ~isempty(obj.optimizationHistory)
                fprintf('最適化履歴から最良の結果を回復します（%d件中）\n', length(obj.optimizationHistory));
                
                % 最良スコアの結果を使用
                [~, bestIdx] = max([obj.optimizationHistory.score]);
                bestEntry = obj.optimizationHistory(bestIdx);
                
                if isfield(bestEntry, 'model') && ~isempty(bestEntry.model)
                    fprintf('履歴から最良モデルを回復しました（スコア: %.4f）\n', bestEntry.score);
                    
                    % 最低限の結果構造体を構築
                    results = struct(...
                        'model', bestEntry.model, ...
                        'performance', struct('accuracy', bestEntry.testAccuracy), ...
                        'trainInfo', struct(...
                            'cnnHistory', struct('TrainingAccuracy', [], 'ValidationAccuracy', []), ...
                            'lstmHistory', struct('TrainingAccuracy', [], 'ValidationAccuracy', []) ...
                        ), ...
                        'overfitting', struct('severity', 'unknown'), ...
                        'normParams', [], ...
                        'crossValidation', [] ...
                    );
                    
                    return;
                end
            end
            
            % 回復失敗時は空の結果を返す
            fprintf('有効な結果を回復できませんでした。最小限の結果を返します。\n');
            results = obj.createDefaultResults();
            
            return;
        end
    end
end

