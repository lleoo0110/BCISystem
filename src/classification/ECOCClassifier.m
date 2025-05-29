classdef ECOCClassifier < handle
    % ECOCClassifier - 誤り訂正出力符号を用いた多クラス分類器クラス
    %
    % このクラスはEEGデータの分類のためのECOCモデルを学習・評価・予測するための
    % 機能を提供します。ハイパーパラメータ最適化、交差検証、性能評価などの機能を含みます。
    %
    % 主な機能:
    %   - EEGデータの前処理と正規化
    %   - CSP特徴抽出
    %   - データ拡張（オプション）
    %   - ECOCモデルの学習と最適化
    %   - 詳細な過学習分析
    %   - 包括的な性能評価
    %   - マルチクラス分類対応
    %   - オンライン予測
    
    properties (Access = private)
        params              % パラメータ設定
        isOptimized         % 最適化の有無
        isEnabled           % ECOCの有効/無効
        trainResults        % 学習結果の保存
        crossValModel       % 交差検証モデル
        verbosity           % 出力詳細度 (0:最小限, 1:通常, 2:詳細, 3:デバッグ)
        
        % 学習進捗と過学習監視用
        trainingAccuracy    % 学習精度
        validationAccuracy  % 検証精度
        overfitMetrics      % 過学習メトリクス

        normalizer          % 正規化コンポーネント
    end
    
    properties (Access = public)
        ecocModel           % 学習済みECOCモデル
        performance         % 性能評価指標
        crossValidation     % 交差検証結果（専用プロパティ）
    end
    
    methods (Access = public)
        function obj = ECOCClassifier(params, verbosity)
            % コンストラクタ - ECOCClassifierのインスタンスを初期化
            % 
            % 入力:
            %   params - 設定パラメータを含む構造体
            %   verbosity - 出力詳細度 (1:通常, 2:詳細, 3:デバッグ)
            %             省略時はデフォルト値1が使用される
            
            obj.params = params;
            obj.isOptimized = params.classifier.ecoc.optimize;
            obj.isEnabled = params.classifier.ecoc.enable;
            obj.performance = struct();
            obj.crossValidation = struct(); % 交差検証結果用の専用プロパティ
            
            % verbosityレベルの設定（デフォルトは1）
            if nargin < 2
                obj.verbosity = 1;
            else
                obj.verbosity = verbosity;
            end
            
            % プロパティの初期化
            obj.initializeProperties();
            obj.normalizer = EEGNormalizer(params);
            
            obj.logMessage(2, 'ECOCClassifier初期化完了 (verbosity: %d)\n', obj.verbosity);
        end
        
        function results = trainECOC(obj, processedData, processedLabel)
            % ECOCモデルの学習と評価を行う
            %
            % 入力:
            %   processedData - 前処理済みEEGデータ [チャンネル x サンプル x エポック]
            %   processedLabel - データのクラスラベル [エポック x 1]
            %
            % 出力:
            %   results - 学習結果と性能評価指標を含む構造体
            
            try
                obj.logMessage(1, '\n=== ECOC学習処理を開始 ===\n');
                
                % 機能が無効の場合のチェック
                if ~obj.isEnabled
                    error('ECOC is disabled in configuration');
                end
                
                % データの検証
                [processedData, ~] = obj.validateAndPrepareData(processedData, processedLabel);
                
                % EEGデータの正規化
                [normalizedEEG, normParams] = obj.normalizeData(processedData);
                
                % CSP特徴抽出処理
                obj.logMessage(1, 'CSP特徴抽出を実行...\n');
                [features, filters, cspParameters] = obj.extractCSPFeatures(normalizedEEG, processedLabel);
                obj.logMessage(2, '抽出された特徴量: %d次元\n', size(features, 2));

                % データの分割（学習用と検証用）
                [trainFeatures, trainLabels, testFeatures, testLabels] = obj.splitDataset(features, processedLabel);
                
                % クラス分布の確認
                obj.checkClassDistribution('学習', trainLabels);
                obj.checkClassDistribution('検証', testLabels);

                % 交差検証の実行（学習データのみを使用 - モデル学習前に実行）
                if obj.params.classifier.ecoc.validation.enable
                    obj.performCrossValidation(trainFeatures, trainLabels);
                else
                    obj.logMessage(1, '交差検証はスキップされました（設定で無効化されています）\n');
                    obj.initializeEmptyCrossValidation();
                end

                % モデルの学習（学習データのみを使用）
                obj.logMessage(1, '\nECOCモデルの学習を開始...\n');
                obj.trainModel(trainFeatures, trainLabels);

                % 学習データでの性能評価（過学習検出用）
                obj.evaluateTrainingPerformance(trainFeatures, trainLabels);

                % テストデータでの評価
                testMetrics = obj.evaluateModel(testFeatures, testLabels);
                
                % 詳細な過学習分析（変数名を変更してプロパティ名との衝突を回避）
                [isOverfit, overfitResults] = obj.validateOverfitting(testMetrics.accuracy);
                
                % 結果の構築（過学習メトリクスを使用）
                results = obj.buildResultsStruct(testMetrics, overfitResults, filters, cspParameters, normParams);
                
                % 過学習が検出された場合の警告表示
                if isOverfit
                    obj.logMessage(1, '\n過学習が検出されました: %s\n', overfitResults.severity);
                end
                
                obj.logMessage(1, '\n=== ECOC学習処理が完了しました ===\n');

            catch ME
                obj.logMessage(0, '\n=== ECOC学習中にエラーが発生しました ===\n');
                obj.logMessage(0, 'エラーメッセージ: %s\n', ME.message);
                obj.logMessage(0, 'エラー発生場所:\n');
                for i = 1:length(ME.stack)
                    obj.logMessage(0, '  ファイル: %s\n  行: %d\n  関数: %s\n\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end
                rethrow(ME);
            end
        end

        function [label, score] = predictOnline(obj, data, ecoc)
            % オンラインでECOC予測を実行する
            %
            % 入力:
            %   data - 分類するEEGデータ [チャンネル x サンプル]
            %   ecoc - 学習済みECOCモデルと関連パラメータを含む構造体
            %
            % 出力:
            %   label - 予測されたクラスラベル
            %   score - 各クラスへの所属確率
            
            try
                % 機能が無効の場合のチェック
                if ~obj.isEnabled
                    error('ECOC is disabled');
                end
                
                % 入力データの検証
                if isempty(data)
                    error('Data cannot be empty');
                end
        
                % モデルの検証
                if isempty(ecoc) || ~isfield(ecoc, 'model') || isempty(ecoc.model)
                    error('Valid ECOC model not found');
                end

                % EEGデータの正規化
                normalizedEEG = data;
                if obj.params.classifier.normalize.enable && isfield(ecoc, 'normParams')
                    normalizedEEG = obj.normalizer.normalizeOnline(data, ecoc.normParams);
                    obj.logMessage(3, 'データ正規化完了\n');
                end

                % CSP特徴量の抽出
                if isfield(ecoc, 'cspFilters') && ~isempty(ecoc.cspFilters)
                    cspExtractor = CSPExtractor(obj.params);
                    features = cspExtractor.extractFeatures(normalizedEEG, ecoc.cspFilters);
                    obj.logMessage(3, '特徴量抽出完了: %d次元\n', size(features, 2));
                else
                    error('CSP filters not found in ECOC model');
                end
        
                % 予測の実行
                [label, score] = predict(ecoc.model, features);
                
                % スコアを確率に変換
                score = obj.convertToProb(score);
                obj.logMessage(3, '予測完了: クラス %d (確率: %.3f)\n', double(label), max(score));
        
            catch ME
                obj.logMessage(0, 'ECOC予測でエラーが発生: %s\n', ME.message);
                obj.logMessage(2, 'エラー詳細:\n');
                if obj.verbosity >= 2
                    disp(getReport(ME, 'extended'));
                end
                rethrow(ME);
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
            obj.trainingAccuracy = 0;
            obj.validationAccuracy = 0;
            obj.overfitMetrics = struct();
            
            % 交差検証結果の初期化
            obj.crossValidation = struct(...
                'enabled', false, ...
                'kfold', 0, ...
                'meanAccuracy', NaN, ...
                'stdAccuracy', NaN, ...
                'foldAccuracies', [], ...
                'meanLoss', NaN, ...
                'stdLoss', NaN, ...
                'foldLosses', [], ...
                'model', [], ...
                'completed', false ...
            );
        end
        
        function initializeEmptyCrossValidation(obj)
            % 交差検証が実行されない場合の初期化
            obj.crossValidation = struct(...
                'enabled', false, ...
                'kfold', 0, ...
                'meanAccuracy', NaN, ...
                'stdAccuracy', NaN, ...
                'foldAccuracies', [], ...
                'meanLoss', NaN, ...
                'stdLoss', NaN, ...
                'foldLosses', [], ...
                'model', [], ...
                'completed', false, ...
                'reason', 'Disabled in configuration' ...
            );
            obj.logMessage(2, '交差検証結果を空で初期化しました\n');
        end
        
        %% データ検証と準備
        function [validatedData, infoMsg] = validateAndPrepareData(obj, data, labels)
            % 入力データの検証と適切な形式への変換
            %
            % 入力:
            %   data - 検証するEEGデータ [チャンネル x サンプル x エポック]
            %   labels - 対応するクラスラベル [エポック x 1]
            %
            % 出力:
            %   validatedData - 検証済みデータ
            %   infoMsg - 検証結果メッセージ
            
            % データとラベルのサイズ確認
            if size(data, 3) ~= length(labels)
                error('データのエポック数(%d)とラベル数(%d)が一致しません', size(data, 3), length(labels));
            end
            
            % データの妥当性検証
            validateattributes(data, {'numeric'}, {'finite', 'nonnan'}, 'validateAndPrepareData', 'data');
            
            validatedData = data;
            [channels, samples, epochs] = size(data);
            infoMsg = sprintf('データ検証完了 [%d×%d×%d]', channels, samples, epochs);
            obj.logMessage(2, '%s\n', infoMsg);
        end
        
        function [normalizedEEG, normParams] = normalizeData(obj, data)
            % EEGデータの正規化を行う
            %
            % 入力:
            %   data - 正規化するEEGデータ [チャンネル x サンプル x エポック]
            %
            % 出力:
            %   normalizedEEG - 正規化されたデータ
            %   normParams - 正規化パラメータ（オンライン予測で使用）
            
            normalizedEEG = data;
            normParams = [];
            
            if obj.params.classifier.normalize.enable
                obj.logMessage(2, '正規化を実行...\n');
                [normalizedEEG, normParams] = obj.normalizer.normalize(data);
                obj.logMessage(2, '正規化完了\n');
            else
                obj.logMessage(2, '正規化はスキップされました\n');
            end
        end
        
        function [features, filters, cspParameters] = extractCSPFeatures(obj, data, labels)
            % 共通空間パターン(CSP)特徴量を抽出する
            %
            % 入力:
            %   data - EEGデータ [チャンネル x サンプル x エポック]
            %   labels - クラスラベル [エポック x 1]
            %
            % 出力:
            %   features - 抽出された特徴量 [エポック x 特徴量数]
            %   filters - CSPフィルタ [フィルタ数 x チャンネル数]
            %   cspParameters - CSP計算パラメータ
            
            cspExtractor = CSPExtractor(obj.params);
            [filters, cspParameters] = cspExtractor.trainCSP(data, labels);
            features = cspExtractor.extractFeatures(data, filters);
            
            obj.logMessage(2, 'CSP特徴抽出完了:\n');
            obj.logMessage(2, '  - フィルタ数: %d\n', size(filters, 1));
            obj.logMessage(2, '  - 特徴量次元: %d\n', size(features, 2));
        end
        
        function [trainFeatures, trainLabels, testFeatures, testLabels] = splitDataset(obj, features, labels)
            % データセットを学習用と検証用に分割する
            %
            % 入力:
            %   features - 分割する特徴量 [サンプル数 x 特徴量数]
            %   labels - 対応するラベル [サンプル数 x 1]
            %
            % 出力:
            %   trainFeatures - 学習用特徴量
            %   trainLabels - 学習用ラベル
            %   testFeatures - 検証用特徴量
            %   testLabels - 検証用ラベル
            
            obj.logMessage(1, '\nデータを学習用と検証用に分割します...\n');
            
            % 層別化サンプリングを使用してクラス分布を維持
            cv = cvpartition(labels, 'Holdout', 0.2);  % 80%学習, 20%検証
            trainIdx = cv.training;
            testIdx = cv.test;
            
            trainFeatures = features(trainIdx, :);
            trainLabels = labels(trainIdx);
            testFeatures = features(testIdx, :);
            testLabels = labels(testIdx);
            
            obj.logMessage(1, '  - 学習データ: %d サンプル (%.1f%%)\n', sum(trainIdx), (sum(trainIdx)/length(labels))*100);
            obj.logMessage(1, '  - 検証データ: %d サンプル (%.1f%%)\n', sum(testIdx), (sum(testIdx)/length(labels))*100);
        end
        
        function trainModel(obj, features, labels)
            % ECOCモデルを学習する
            %
            % 入力:
            %   features - 学習用特徴量 [サンプル数 x 特徴量数]
            %   labels - 学習用ラベル [サンプル数 x 1]
            
            obj.logMessage(1, 'ECOCモデル設定:\n');
            obj.logMessage(1, '  - コーディング方式: %s\n', obj.params.classifier.ecoc.coding);
            obj.logMessage(1, '  - 基本学習器: %s\n', obj.params.classifier.ecoc.learners);
            obj.logMessage(1, '  - カーネル関数: %s\n', obj.params.classifier.ecoc.kernel);
            
            if obj.isOptimized
                obj.trainOptimizedModel(features, labels);
            else
                obj.trainDefaultModel(features, labels);
            end

            % 確率推定の設定
            if obj.params.classifier.ecoc.probability
                obj.logMessage(2, '確率推定は学習時に有効化されています\n');
            end
            
            obj.logMessage(1, 'ECOCモデルの学習完了\n');
        end
        
        function trainOptimizedModel(obj, features, labels)
            % ハイパーパラメータ最適化を用いてECOCモデルを学習する
            %
            % 入力:
            %   features - 学習用特徴量 [サンプル数 x 特徴量数]
            %   labels - 学習用ラベル [サンプル数 x 1]
            
            obj.logMessage(1, '\nECOCハイパーパラメータ最適化を実行...\n');
            
            % 最適化方法の設定
            if isfield(obj.params.classifier.ecoc, 'hyperparameters') && ...
               isfield(obj.params.classifier.ecoc.hyperparameters, 'optimizer')
                optMethod = obj.params.classifier.ecoc.hyperparameters.optimizer;
                obj.logMessage(1, '  - 最適化手法: %s\n', optMethod);
                
                % グリッドサーチ最適化
                obj.trainWithGridSearch(features, labels);
            else
                % ハイパーパラメータフィールドがない場合はデフォルト最適化
                obj.trainWithGridSearch(features, labels);
                obj.logMessage(1, '  - デフォルトのグリッドサーチ最適化を使用\n');
            end
        end
        
        function trainWithGridSearch(obj, features, labels)
            % グリッドサーチを用いたハイパーパラメータ最適化
            %
            % 入力:
            %   features - 学習用特徴量 [サンプル数 x 特徴量数]
            %   labels - 学習用ラベル [サンプル数 x 1]
            
            obj.logMessage(1, '  - グリッドサーチ最適化を使用\n');
            
            % パラメータの設定
            boxConstraints = obj.getParameterValues('boxConstraint', [0.1, 1, 10]);
            kernelScales = obj.getParameterValues('kernelScale', [0.1, 1, 10]);
            
            obj.logMessage(2, '  - BoxConstraint候補: [%s]\n', num2str(boxConstraints));
            obj.logMessage(2, '  - KernelScale候補: [%s]\n', num2str(kernelScales));
            
            % 確率推定オプションの設定
            fitPosterior = obj.params.classifier.ecoc.probability;
            
            % グリッドサーチの実行
            bestScore = -inf;
            bestParams = struct('BoxConstraint', 1, 'KernelScale', 1);
            
            for c = boxConstraints
                for k = kernelScales
                    try
                        % 基本学習器の選択
                        if strcmpi(obj.params.classifier.ecoc.learners, 'svm')
                            template = templateSVM(...
                                'KernelFunction', obj.params.classifier.ecoc.kernel, ...
                                'BoxConstraint', c, ...
                                'KernelScale', k);
                        else
                            % 決定木 (tree) の場合
                            template = templateTree();
                        end
                        
                        mdl = fitcecoc(features, labels, ...
                            'Learners', template, ...
                            'Coding', obj.params.classifier.ecoc.coding, ...
                            'FitPosterior', fitPosterior);
                        
                        cv = crossval(mdl, 'KFold', obj.params.classifier.ecoc.validation.kfold);
                        score = 1 - kfoldLoss(cv);
                        
                        obj.logMessage(2, '  - C=%.3f, KScale=%.3f: スコア=%.4f\n', c, k, score);
                        
                        if score > bestScore
                            bestScore = score;
                            bestParams.BoxConstraint = c;
                            bestParams.KernelScale = k;
                            obj.logMessage(2, '    -> 新しい最良パラメータ\n');
                        end
                    catch ME
                        obj.logMessage(1, '    パラメータ組み合わせでエラー: %s\n', ME.message);
                    end
                end
            end
            
            obj.logMessage(1, '\n最適パラメータ: C=%.3f, KScale=%.3f, スコア=%.4f\n', ...
                bestParams.BoxConstraint, bestParams.KernelScale, bestScore);
            
            % 最適パラメータでモデルを作成
            if strcmpi(obj.params.classifier.ecoc.learners, 'svm')
                template = templateSVM(...
                    'KernelFunction', obj.params.classifier.ecoc.kernel, ...
                    'BoxConstraint', bestParams.BoxConstraint, ...
                    'KernelScale', bestParams.KernelScale);
            else
                % 決定木 (tree) の場合
                template = templateTree();
            end
            
            obj.ecocModel = fitcecoc(features, labels, ...
                'Learners', template, ...
                'Coding', obj.params.classifier.ecoc.coding, ...
                'FitPosterior', fitPosterior);
        end
        
        function values = getParameterValues(obj, paramName, defaultValues)
            % パラメータ値を設定から取得（なければデフォルト値を使用）
            %
            % 入力:
            %   paramName - パラメータ名
            %   defaultValues - デフォルト値配列
            %
            % 出力:
            %   values - パラメータ値配列
            
            if isfield(obj.params.classifier.ecoc, 'hyperparameters') && ...
               isfield(obj.params.classifier.ecoc.hyperparameters, paramName)
                values = obj.params.classifier.ecoc.hyperparameters.(paramName);
            else
                values = defaultValues;
                obj.logMessage(2, '  - %s: デフォルト値を使用\n', paramName);
            end
        end
        
        function trainDefaultModel(obj, features, labels)
            % デフォルトパラメータでECOCモデルを学習する
            %
            % 入力:
            %   features - 学習用特徴量 [サンプル数 x 特徴量数]
            %   labels - 学習用ラベル [サンプル数 x 1]
            
            obj.logMessage(1, '\nデフォルトパラメータでECOCモデルを作成...\n');
            
            % 確率推定オプションの設定
            fitPosterior = obj.params.classifier.ecoc.probability;
            
            % 基本学習器の選択
            if strcmpi(obj.params.classifier.ecoc.learners, 'svm')
                template = templateSVM(...
                    'KernelFunction', obj.params.classifier.ecoc.kernel);
            else
                % 決定木 (tree) の場合
                template = templateTree();
            end
            
            obj.ecocModel = fitcecoc(features, labels, ...
                'Learners', template, ...
                'Coding', obj.params.classifier.ecoc.coding, ...
                'FitPosterior', fitPosterior);
        end
        
        function evaluateTrainingPerformance(obj, trainFeatures, trainLabels)
            % 学習データでの性能を評価（過学習検出用）
            %
            % 入力:
            %   trainFeatures - 学習用特徴量 [サンプル数 x 特徴量数]
            %   trainLabels - 学習用ラベル [サンプル数 x 1]
            
            obj.logMessage(2, '学習データでの性能評価...\n');
            
            try
                [pred, ~] = predict(obj.ecocModel, trainFeatures);
                obj.trainingAccuracy = mean(pred == trainLabels);
                obj.logMessage(2, '学習精度: %.2f%%\n', obj.trainingAccuracy * 100);
            catch ME
                obj.logMessage(1, '学習精度の計算でエラー: %s\n', ME.message);
                obj.trainingAccuracy = NaN;
            end
        end
        
        function performCrossValidation(obj, trainFeatures, trainLabels)
            % 交差検証を実行する（パラメータ最適化を含む）
            %
            % 入力:
            %   trainFeatures - 学習用特徴量 [サンプル数 x 特徴量数]
            %   trainLabels - 学習用ラベル [サンプル数 x 1]
            
            if ~obj.params.classifier.ecoc.validation.enable
                obj.initializeEmptyCrossValidation();
                return;
            end
            
            kfold = obj.params.classifier.ecoc.validation.kfold;
            obj.logMessage(1, '\n交差検証の実行（K=%d）パラメータ最適化付き...\n', kfold);
            
            try
                % 交差検証の開始
                obj.crossValidation.enabled = true;
                obj.crossValidation.kfold = kfold;
                obj.crossValidation.completed = false;
                
                % 交差検証用のパーティション作成
                cv = cvpartition(trainLabels, 'KFold', kfold);
                
                % 各フォールドの精度と損失を計算
                foldAccuracies = zeros(kfold, 1);
                foldLosses = zeros(kfold, 1);
                optimalParams = cell(kfold, 1);  % 各フォールドの最適パラメータ
                
                obj.logMessage(2, '各フォールドの詳細結果（最適化付き）:\n');
                
                for i = 1:kfold
                    obj.logMessage(2, '\n--- フォールド %d/%d ---\n', i, kfold);
                    
                    % 現在のフォールドのデータ分割
                    foldTrainIdx = cv.training(i);
                    foldValIdx = cv.test(i);
                    
                    foldTrainFeatures = trainFeatures(foldTrainIdx, :);
                    foldTrainLabels = trainLabels(foldTrainIdx);
                    foldValFeatures = trainFeatures(foldValIdx, :);
                    foldValLabels = trainLabels(foldValIdx);
                    
                    if obj.isOptimized
                        % フォールド内でパラメータ最適化を実行
                        [bestModel, bestParams] = obj.optimizeParametersForFold(...
                            foldTrainFeatures, foldTrainLabels);
                        optimalParams{i} = bestParams;
                    else
                        % デフォルトパラメータでモデル作成
                        bestModel = obj.createDefaultModelForFold(foldTrainFeatures, foldTrainLabels);
                        optimalParams{i} = struct('type', 'default');
                    end
                    
                    % フォールドのバリデーションデータで評価
                    [pred, ~] = predict(bestModel, foldValFeatures);
                    foldAccuracies(i) = mean(pred == foldValLabels);
                    foldLosses(i) = 1 - foldAccuracies(i);  % 簡単な損失計算
                    
                    obj.logMessage(2, '  - フォールド %d: 精度 = %.4f, 損失 = %.4f\n', ...
                        i, foldAccuracies(i), foldLosses(i));
                    
                    if obj.isOptimized && obj.verbosity >= 3
                        obj.logMessage(3, '  - 最適パラメータ: C=%.3f, KScale=%.3f\n', ...
                            bestParams.BoxConstraint, bestParams.KernelScale);
                    end
                end
                
                % 統計的指標の計算
                obj.crossValidation.foldAccuracies = foldAccuracies;
                obj.crossValidation.foldLosses = foldLosses;
                obj.crossValidation.meanAccuracy = mean(foldAccuracies);
                obj.crossValidation.stdAccuracy = std(foldAccuracies);
                obj.crossValidation.meanLoss = mean(foldLosses);
                obj.crossValidation.stdLoss = std(foldLosses);
                obj.crossValidation.optimalParams = optimalParams;  % 各フォールドの最適パラメータ
                obj.crossValidation.completed = true;
                
                % validationAccuracyプロパティの更新
                obj.validationAccuracy = obj.crossValidation.meanAccuracy;
                
                % 最適パラメータの統計
                if obj.isOptimized
                    obj.analyzeOptimalParameters(optimalParams);
                end
                
                % パフォーマンス構造体への保存（後方互換性のため）
                obj.performance.cvAccuracies = foldAccuracies;
                obj.performance.cvMeanAccuracy = obj.crossValidation.meanAccuracy;
                obj.performance.cvStdAccuracy = obj.crossValidation.stdAccuracy;
                obj.performance.cvMeanLoss = obj.crossValidation.meanLoss;
                obj.performance.cvStdLoss = obj.crossValidation.stdLoss;
                
                % 結果の表示
                obj.logMessage(1, '\n交差検証結果のサマリー（最適化付き）:\n');
                obj.logMessage(1, '  - 平均精度: %.4f (±%.4f)\n', ...
                    obj.crossValidation.meanAccuracy, obj.crossValidation.stdAccuracy);
                obj.logMessage(1, '  - 平均損失: %.4f (±%.4f)\n', ...
                    obj.crossValidation.meanLoss, obj.crossValidation.stdLoss);
                obj.logMessage(1, '  - 最高精度: %.4f (フォールド %d)\n', ...
                    max(foldAccuracies), find(foldAccuracies == max(foldAccuracies), 1));
                obj.logMessage(1, '  - 最低精度: %.4f (フォールド %d)\n', ...
                    min(foldAccuracies), find(foldAccuracies == min(foldAccuracies), 1));
                
                % 信頼区間の計算（95%信頼区間）
                if kfold > 1
                    sem = obj.crossValidation.stdAccuracy / sqrt(kfold);
                    tValue = 2.262;  % t分布の95%信頼区間（自由度4、近似値）
                    if kfold > 5
                        tValue = 1.96;  % 正規分布の近似（サンプル数が多い場合）
                    end
                    ci_lower = obj.crossValidation.meanAccuracy - tValue * sem;
                    ci_upper = obj.crossValidation.meanAccuracy + tValue * sem;
                    
                    obj.crossValidation.confidenceInterval = [ci_lower, ci_upper];
                    obj.logMessage(1, '  - 95%%信頼区間: [%.4f, %.4f]\n', ci_lower, ci_upper);
                end
                
                obj.logMessage(1, '交差検証（最適化付き）が正常に完了しました\n');
                    
            catch ME
                obj.logMessage(0, '交差検証でエラーが発生: %s\n', ME.message);
                
                % エラー時の情報保存
                obj.crossValidation.completed = false;
                obj.crossValidation.error = true;
                obj.crossValidation.errorMessage = ME.message;
                obj.crossValidation.meanAccuracy = NaN;
                obj.crossValidation.stdAccuracy = NaN;
                obj.crossValidation.meanLoss = NaN;
                obj.crossValidation.stdLoss = NaN;
                
                % パフォーマンス構造体への保存（後方互換性のため）
                obj.performance.cvMeanAccuracy = NaN;
                obj.performance.cvStdAccuracy = NaN;
                
                obj.logMessage(2, 'エラー詳細:\n');
                if obj.verbosity >= 2
                    disp(getReport(ME, 'extended'));
                end
            end
        end
        
        function [bestModel, bestParams] = optimizeParametersForFold(obj, foldTrainFeatures, foldTrainLabels)
            % 単一フォールド内でのパラメータ最適化
            %
            % 入力:
            %   foldTrainFeatures - フォールド学習データ
            %   foldTrainLabels - フォールド学習ラベル
            %
            % 出力:
            %   bestModel - 最適化されたモデル
            %   bestParams - 最適パラメータ
            
            % パラメータの設定
            boxConstraints = obj.getParameterValues('boxConstraint', [0.1, 1, 10]);
            kernelScales = obj.getParameterValues('kernelScale', [0.1, 1, 10]);
            
            % 確率推定オプションの設定
            fitPosterior = obj.params.classifier.ecoc.probability;
            
            % 内側交差検証用のフォールド数（外側より少なく）
            innerKfold = max(3, obj.params.classifier.ecoc.validation.kfold - 1);
            
            % グリッドサーチの実行
            bestScore = -inf;
            bestParams = struct('BoxConstraint', 1, 'KernelScale', 1);
            bestModel = [];
            
            obj.logMessage(3, '    フォールド内最適化: %d × %d パラメータ組み合わせ\n', ...
                length(boxConstraints), length(kernelScales));
            
            for c = boxConstraints
                for k = kernelScales
                    try
                        % 基本学習器の選択
                        if strcmpi(obj.params.classifier.ecoc.learners, 'svm')
                            template = templateSVM(...
                                'KernelFunction', obj.params.classifier.ecoc.kernel, ...
                                'BoxConstraint', c, ...
                                'KernelScale', k);
                        else
                            template = templateTree();
                        end
                        
                        % モデル作成
                        mdl = fitcecoc(foldTrainFeatures, foldTrainLabels, ...
                            'Learners', template, ...
                            'Coding', obj.params.classifier.ecoc.coding, ...
                            'FitPosterior', fitPosterior);
                        
                        % 内側交差検証で評価
                        cv = crossval(mdl, 'KFold', innerKfold);
                        score = 1 - kfoldLoss(cv);
                        
                        obj.logMessage(3, '      C=%.3f, KScale=%.3f: スコア=%.4f\n', c, k, score);
                        
                        if score > bestScore
                            bestScore = score;
                            bestParams.BoxConstraint = c;
                            bestParams.KernelScale = k;
                            bestModel = mdl;  % 最適モデルを保存
                        end
                    catch ME
                        obj.logMessage(3, '      パラメータ組み合わせでエラー: %s\n', ME.message);
                    end
                end
            end
            
            obj.logMessage(3, '    最適パラメータ: C=%.3f, KScale=%.3f, スコア=%.4f\n', ...
                bestParams.BoxConstraint, bestParams.KernelScale, bestScore);
        end
        
        function model = createDefaultModelForFold(obj, foldTrainFeatures, foldTrainLabels)
            % デフォルトパラメータでのモデル作成（フォールド用）
            %
            % 入力:
            %   foldTrainFeatures - フォールド学習データ
            %   foldTrainLabels - フォールド学習ラベル
            %
            % 出力:
            %   model - 作成されたモデル
            
            % 確率推定オプションの設定
            fitPosterior = obj.params.classifier.ecoc.probability;
            
            % 基本学習器の選択
            if strcmpi(obj.params.classifier.ecoc.learners, 'svm')
                template = templateSVM(...
                    'KernelFunction', obj.params.classifier.ecoc.kernel);
            else
                template = templateTree();
            end
            
            model = fitcecoc(foldTrainFeatures, foldTrainLabels, ...
                'Learners', template, ...
                'Coding', obj.params.classifier.ecoc.coding, ...
                'FitPosterior', fitPosterior);
        end
        
        function analyzeOptimalParameters(obj, optimalParams)
            % 各フォールドの最適パラメータを分析
            %
            % 入力:
            %   optimalParams - 各フォールドの最適パラメータ
            
            if ~obj.isOptimized || isempty(optimalParams)
                return;
            end
            
            % パラメータの統計を計算
            boxConstraints = zeros(length(optimalParams), 1);
            kernelScales = zeros(length(optimalParams), 1);
            
            for i = 1:length(optimalParams)
                if isstruct(optimalParams{i}) && isfield(optimalParams{i}, 'BoxConstraint')
                    boxConstraints(i) = optimalParams{i}.BoxConstraint;
                    kernelScales(i) = optimalParams{i}.KernelScale;
                end
            end
            
            % 統計情報を交差検証結果に保存
            obj.crossValidation.parameterStats = struct(...
                'boxConstraint', struct(...
                    'mean', mean(boxConstraints), ...
                    'std', std(boxConstraints), ...
                    'min', min(boxConstraints), ...
                    'max', max(boxConstraints) ...
                ), ...
                'kernelScale', struct(...
                    'mean', mean(kernelScales), ...
                    'std', std(kernelScales), ...
                    'min', min(kernelScales), ...
                    'max', max(kernelScales) ...
                ) ...
            );
            
            obj.logMessage(2, '\n最適パラメータの統計:\n');
            obj.logMessage(2, '  BoxConstraint: 平均=%.3f, 標準偏差=%.3f, 範囲=[%.3f, %.3f]\n', ...
                mean(boxConstraints), std(boxConstraints), min(boxConstraints), max(boxConstraints));
            obj.logMessage(2, '  KernelScale: 平均=%.3f, 標準偏差=%.3f, 範囲=[%.3f, %.3f]\n', ...
                mean(kernelScales), std(kernelScales), min(kernelScales), max(kernelScales));
        end

        function metrics = evaluateModel(obj, testFeatures, testLabels)
            % テストデータでモデルの性能を評価する
            %
            % 入力:
            %   testFeatures - テスト用特徴量 [サンプル数 x 特徴量数]
            %   testLabels - テスト用ラベル [サンプル数 x 1]
            %
            % 出力:
            %   metrics - テストデータでの性能評価結果
            
            obj.logMessage(1, '\nテストデータでモデルを評価中...\n');
            
            % 初期化
            metrics = struct(...
                'accuracy', 0, ...
                'score', [], ...
                'confusionMat', [], ...
                'roc', [], ...
                'auc', [], ...
                'classwise', [], ...
                'classLabels', [] ...  % 追加
            );
        
            try
                [pred, score] = predict(obj.ecocModel, testFeatures);
                
                % スコアを確率に変換
                score = obj.convertToProb(score);
                metrics.score = score;
                
                % テストデータでの精度と混同行列
                metrics.accuracy = mean(pred == testLabels);
                metrics.confusionMat = confusionmat(testLabels, pred);
                
                % クラスラベル
                classLabels = unique(testLabels);
                metrics.classLabels = classLabels;  % metricsに追加
                
                % パフォーマンス情報の保存
                obj.performance.testAccuracy = metrics.accuracy;
                obj.performance.testConfusionMat = metrics.confusionMat;
                obj.performance.testScore = score;
                obj.performance.testPredictions = pred;
                obj.performance.classLabels = classLabels;
                
                obj.logMessage(1, 'テスト精度: %.2f%%\n', metrics.accuracy * 100);
                
                % ROC曲線とAUCの計算
                if length(classLabels) == 2
                    % 二値分類の場合
                    [X, Y, T, AUC] = perfcurve(testLabels, score(:,2), classLabels(2));
                    metrics.roc = struct('X', X, 'Y', Y, 'T', T);
                    metrics.auc = AUC;
                    obj.performance.testRoc = metrics.roc;
                    obj.performance.testAuc = AUC;
                    obj.logMessage(1, 'AUC: %.3f\n', AUC);
                elseif length(classLabels) > 2
                    % マルチクラス分類の場合: One-vs-Rest AUC
                    aucValues = zeros(1, length(classLabels));
                    for i = 1:length(classLabels)
                        binaryLabels = (testLabels == classLabels(i));
                        [~, ~, ~, aucValues(i)] = perfcurve(binaryLabels, score(:,i), true);
                    end
                    metrics.auc = mean(aucValues);
                    obj.performance.testAuc = metrics.auc;
                    obj.logMessage(1, '平均AUC (One-vs-Rest): %.3f\n', metrics.auc);
                end
                
                % クラスごとの性能評価
                metrics.classwise = obj.calculateClassMetrics(testLabels, pred, classLabels);
                obj.performance.testClasswise = metrics.classwise;
                
                % 混同行列の表示
                obj.logMessage(1, '\n混同行列:\n');
                if obj.verbosity >= 1
                    disp(metrics.confusionMat);
                end
                
            catch ME
                obj.logMessage(0, 'モデル評価でエラーが発生: %s\n', ME.message);
                obj.performance.testAccuracy = metrics.accuracy;
                rethrow(ME);
            end
        end

        function metrics = calculateClassMetrics(obj, trueLabels, predLabels, classLabels)
            % クラスごとの性能評価指標を計算する
            %
            % 入力:
            %   trueLabels - 真のラベル
            %   predLabels - 予測されたラベル
            %   classLabels - クラスのユニークラベル
            %
            % 出力:
            %   metrics - クラスごとの性能評価指標
            
            obj.logMessage(2, '\nクラスごとの性能評価:\n');
            metrics = struct();
            
            for i = 1:length(classLabels)
                className = classLabels(i);
                classIdx = (trueLabels == className);
                
                % 精度指標の計算
                TP = sum(predLabels(classIdx) == className);
                FP = sum(predLabels == className) - TP;
                FN = sum(classIdx) - TP;
                
                % ゼロ除算の防止
                if (TP + FP) > 0
                    precision = TP / (TP + FP);
                else
                    precision = 0;
                end
                
                if (TP + FN) > 0
                    recall = TP / (TP + FN);
                else
                    recall = 0;
                end
                
                if (precision + recall) > 0
                    f1score = 2 * (precision * recall) / (precision + recall);
                else
                    f1score = 0;
                end
                
                % 結果の格納
                metrics(i).precision = precision;
                metrics(i).recall = recall;
                metrics(i).f1score = f1score;
                
                obj.logMessage(2, 'クラス %d:\n', className);
                obj.logMessage(2, '  - 精度 (Precision): %.2f%%\n', precision * 100);
                obj.logMessage(2, '  - 再現率 (Recall): %.2f%%\n', recall * 100);
                obj.logMessage(2, '  - F1スコア: %.3f\n', f1score);
            end
        end
        
        function checkClassDistribution(obj, setName, labels)
            % データセット内のクラス分布を解析して表示する
            %
            % 入力:
            %   setName - データセット名（表示用）
            %   labels - クラスラベル
            
            uniqueLabels = unique(labels);
            obj.logMessage(1, '\n%sデータのクラス分布:\n', setName);
            
            % 修正: histcountsの問題を解決
            counts = zeros(1, length(uniqueLabels));
            for i = 1:length(uniqueLabels)
                counts(i) = sum(labels == uniqueLabels(i));
                obj.logMessage(1, '  - クラス %d: %d サンプル (%.1f%%)\n', ...
                    uniqueLabels(i), counts(i), (counts(i)/length(labels))*100);
            end
            
            % クラス不均衡の評価
            maxCount = max(counts);
            minCount = min(counts);
            imbalanceRatio = maxCount / max(minCount, 1);
            
            if imbalanceRatio > 3
                obj.logMessage(1, '警告: %sデータセットのクラス不均衡が大きいです (比率: %.1f:1)\n', ...
                    setName, imbalanceRatio);
            end
        end
        
        function [isOverfit, analysisResults] = validateOverfitting(obj, testAccuracy)
            % 詳細な過学習検証（SVMClassifierと同等レベル）
            %
            % 入力:
            %   testAccuracy - 検証データでの精度
            %
            % 出力:
            %   isOverfit - 過学習検出フラグ
            %   analysisResults - 詳細な過学習メトリクス（プロパティ名との衝突を回避）

            obj.logMessage(1, '\n=== 詳細な過学習検証の実行 ===\n');
            
            % 初期化
            isOverfit = false;
            analysisResults = struct(...
                'trainAccuracy', obj.trainingAccuracy, ...
                'valAccuracy', obj.validationAccuracy, ...
                'testAccuracy', testAccuracy, ...
                'trainTestGap', NaN, ...
                'valTestGap', NaN, ...
                'severity', 'none', ...
                'isCompletelyBiased', false, ...
                'gapSeverity', 'none' ...
            );
            
            try
                % 学習精度との比較
                trainTestGap = NaN;
                if ~isnan(obj.trainingAccuracy)
                    trainTestGap = abs(obj.trainingAccuracy - testAccuracy);
                    obj.logMessage(1, '学習精度: %.2f%%\n', obj.trainingAccuracy * 100);
                    analysisResults.trainTestGap = trainTestGap;
                end
                
                % 交差検証精度との比較
                valTestGap = NaN;
                if obj.crossValidation.completed && ~isnan(obj.crossValidation.meanAccuracy)
                    valTestGap = abs(obj.crossValidation.meanAccuracy - testAccuracy);
                    obj.logMessage(1, '交差検証平均精度: %.2f%%\n', obj.crossValidation.meanAccuracy * 100);
                    analysisResults.valAccuracy = obj.crossValidation.meanAccuracy;
                    analysisResults.valTestGap = valTestGap;
                elseif isfield(obj.performance, 'cvMeanAccuracy') && ~isnan(obj.performance.cvMeanAccuracy)
                    % フォールバック: パフォーマンス構造体から取得
                    valTestGap = abs(obj.performance.cvMeanAccuracy - testAccuracy);
                    obj.logMessage(1, '交差検証平均精度: %.2f%%\n', obj.performance.cvMeanAccuracy * 100);
                    analysisResults.valAccuracy = obj.performance.cvMeanAccuracy;
                    analysisResults.valTestGap = valTestGap;
                end
                
                obj.logMessage(1, 'テスト精度: %.2f%%\n', testAccuracy * 100);
                
                % ギャップの表示
                if ~isnan(trainTestGap)
                    obj.logMessage(1, '学習-テスト精度差: %.2f%%\n', trainTestGap * 100);
                end
                if ~isnan(valTestGap)
                    obj.logMessage(1, '検証-テスト精度差: %.2f%%\n', valTestGap * 100);
                end
                
                % 分類バイアスの検出
                isCompletelyBiased = obj.detectClassificationBias();
                analysisResults.isCompletelyBiased = isCompletelyBiased;
                
                % 過学習の程度を段階的に評価
                [gapOverfit, gapSeverity] = obj.evaluatePerformanceGap(trainTestGap, valTestGap);
                analysisResults.gapSeverity = gapSeverity;
                
                % 最終的な重大度判定
                if isCompletelyBiased
                    severity = 'critical';
                    isOverfit = true;
                    obj.logMessage(1, '完全な分類バイアスが検出されました\n');
                elseif gapOverfit
                    severity = gapSeverity;
                    isOverfit = true;
                else
                    severity = 'none';
                    isOverfit = false;
                end
                
                % メトリクスの更新
                analysisResults.severity = severity;
                
                % プロパティに保存
                obj.overfitMetrics = analysisResults;
                
                % 結果の表示と警告
                obj.logMessage(1, '過学習判定: %s (重大度: %s)\n', mat2str(isOverfit), severity);
                obj.displayOverfitWarning(isOverfit, severity);
                
            catch ME
                obj.logMessage(1, '過学習検証でエラーが発生: %s\n', ME.message);
                % エラー時のフォールバック - メトリクスは既に初期化済み
                analysisResults.severity = 'unknown';
            end
        end
        
        function [gapOverfit, severity] = evaluatePerformanceGap(obj, trainTestGap, valTestGap)
            % 性能ギャップに基づく過学習評価
            %
            % 入力:
            %   trainTestGap - 学習-テスト精度差
            %   valTestGap - 検証-テスト精度差
            %
            % 出力:
            %   gapOverfit - ギャップベースの過学習フラグ
            %   severity - 重大度
            
            gapOverfit = false;
            severity = 'none';
            
            % 主要なギャップを選択（利用可能な方を優先）
            primaryGap = NaN;
            if ~isnan(valTestGap)
                primaryGap = valTestGap;
                gapType = '検証-テスト';
            elseif ~isnan(trainTestGap)
                primaryGap = trainTestGap;
                gapType = '学習-テスト';
            end
            
            if ~isnan(primaryGap)
                obj.logMessage(2, '主要精度差 (%s): %.2f%%\n', gapType, primaryGap * 100);
                
                % 段階的な重大度判定
                if primaryGap > 0.20      % 20%以上の差
                    severity = 'critical';
                    gapOverfit = true;
                elseif primaryGap > 0.15  % 15%以上の差
                    severity = 'severe';
                    gapOverfit = true;
                elseif primaryGap > 0.10  % 10%以上の差
                    severity = 'moderate';
                    gapOverfit = true;
                elseif primaryGap > 0.05  % 5%以上の差
                    severity = 'mild';
                    gapOverfit = true;
                else
                    severity = 'none';
                    gapOverfit = false;
                end
            else
                obj.logMessage(2, '性能ギャップの計算に十分なデータがありません\n');
            end
        end
        
        function isCompletelyBiased = detectClassificationBias(obj)
            % 混同行列から分類バイアスを検出
            %
            % 出力:
            %   isCompletelyBiased - 完全なバイアスの有無
            
            isCompletelyBiased = false;
            
            if isfield(obj.performance, 'testConfusionMat') && ~isempty(obj.performance.testConfusionMat)
                cm = obj.performance.testConfusionMat;
                
                % 各実際のクラス（行）のサンプル数を確認
                rowSums = sum(cm, 2);
                missingActual = any(rowSums == 0);
                
                % 各予測クラス（列）の予測件数を確認
                colSums = sum(cm, 1);
                missingPredicted = any(colSums == 0);
                
                % すべての予測が1クラスに集中しているかを検出
                predictedClassCount = sum(colSums > 0);
                
                isCompletelyBiased = missingActual || missingPredicted || predictedClassCount <= 1;
                
                if isCompletelyBiased
                    obj.logMessage(1, '\n警告: 分類に完全な偏りが検出されました\n');
                    obj.logMessage(1, '  - 分類された実際のクラス数: %d / %d\n', sum(rowSums > 0), size(cm, 1));
                    obj.logMessage(1, '  - 予測されたクラス数: %d / %d\n', predictedClassCount, size(cm, 2));
                end
            end
        end
        
        function displayOverfitWarning(obj, isOverfit, severity)
            % 詳細な過学習警告とアドバイスの表示
            %
            % 入力:
            %   isOverfit - 過学習フラグ
            %   severity - 過学習の重症度
            
            if isOverfit
                obj.logMessage(1, '\n警告: モデルに過学習の兆候が検出されました (%s)\n', severity);
                
                switch severity
                    case 'critical'
                        obj.logMessage(1, '  *** 重大な過学習 ***\n');
                        obj.logMessage(1, '  緊急対策が必要です:\n');
                        obj.logMessage(1, '  - 特徴量の大幅削減\n');
                        obj.logMessage(1, '  - より強い正則化\n');
                        obj.logMessage(1, '  - 交差検証による厳密な評価\n');
                        obj.logMessage(1, '  - 別のアルゴリズムの検討\n');
                        
                    case 'severe'
                        obj.logMessage(1, '  ** 深刻な過学習 **\n');
                        obj.logMessage(1, '  対策として以下を検討してください:\n');
                        obj.logMessage(1, '  - 特徴量選択の見直し\n');
                        obj.logMessage(1, '  - 正則化パラメータの調整\n');
                        obj.logMessage(1, '  - より多くの学習データの収集\n');
                        
                    case 'moderate'
                        obj.logMessage(1, '  * 中程度の過学習 *\n');
                        obj.logMessage(1, '  推奨対策:\n');
                        obj.logMessage(1, '  - ハイパーパラメータの調整\n');
                        obj.logMessage(1, '  - データ拡張の適用\n');
                        obj.logMessage(1, '  - 交差検証の実施\n');
                        
                    case 'mild'
                        obj.logMessage(1, '  軽度の過学習\n');
                        obj.logMessage(1, '  軽微な調整で改善可能:\n');
                        obj.logMessage(1, '  - 正則化の微調整\n');
                        obj.logMessage(1, '  - 追加検証の実施\n');
                end
            else
                obj.logMessage(1, '\nモデルは良好に一般化されています。\n');
            end
        end
        
        function results = buildResultsStruct(obj, testMetrics, overfitAnalysis, filters, cspParameters, normParams)
            % 最終結果構造体を構築する
            %
            % 入力:
            %   testMetrics - テストデータでの評価結果
            %   overfitAnalysis - 過学習分析結果
            %   filters - CSPフィルタ
            %   cspParameters - CSPパラメータ
            %   normParams - 正規化パラメータ
            %
            % 出力:
            %   results - 統合された結果構造体
            
            % 交差検証結果を構築（専用プロパティから取得）
            crossValResults = struct();
            if obj.crossValidation.completed
                crossValResults = obj.crossValidation;  % 完全な交差検証結果をコピー
                
                % 後方互換性のための追加フィールド
                crossValResults.meanAccuracy = obj.crossValidation.meanAccuracy;
                crossValResults.stdAccuracy = obj.crossValidation.stdAccuracy;
                crossValResults.accuracies = obj.crossValidation.foldAccuracies;
            else
                % 交差検証が完了していない場合
                crossValResults.enabled = obj.crossValidation.enabled;
                crossValResults.completed = false;
                crossValResults.meanAccuracy = NaN;
                crossValResults.stdAccuracy = NaN;
                crossValResults.accuracies = [];
                if isfield(obj.crossValidation, 'reason')
                    crossValResults.reason = obj.crossValidation.reason;
                end
                if isfield(obj.crossValidation, 'error')
                    crossValResults.error = obj.crossValidation.error;
                    crossValResults.errorMessage = obj.crossValidation.errorMessage;
                end
            end
            
            % 結果構造体の構築
            results = struct(...
                'model', obj.ecocModel, ...
                'performance', testMetrics, ...
                'overfitting', overfitAnalysis, ...
                'cspFilters', filters, ...
                'cspParameters', cspParameters, ...
                'normParams', normParams, ...
                'trainingAccuracy', obj.trainingAccuracy, ...
                'crossValidation', crossValResults ...  % 専用の交差検証結果
            );
            
            % パフォーマンスプロパティの更新
            obj.performance = struct();  % 一旦クリア
            
            % テスト結果を正しいフィールド名で設定
            obj.performance.testAccuracy = testMetrics.accuracy;
            obj.performance.testAuc = testMetrics.auc;
            obj.performance.testConfusionMat = testMetrics.confusionMat;
            obj.performance.testScore = testMetrics.score;
            obj.performance.testClasswise = testMetrics.classwise;
            
            % classLabelsの復元（修正部分）
            if isfield(testMetrics, 'classLabels')
                obj.performance.classLabels = testMetrics.classLabels;
            else
                % フォールバック: 混同行列のサイズからクラス数を推定
                if ~isempty(testMetrics.confusionMat)
                    numClasses = size(testMetrics.confusionMat, 1);
                    obj.performance.classLabels = 1:numClasses;
                else
                    obj.performance.classLabels = [];
                end
            end
            
            % 交差検証結果を復元（後方互換性のため）
            if obj.crossValidation.completed
                obj.performance.cvMeanAccuracy = obj.crossValidation.meanAccuracy;
                obj.performance.cvStdAccuracy = obj.crossValidation.stdAccuracy;
                obj.performance.cvAccuracies = obj.crossValidation.foldAccuracies;
                obj.performance.cvMeanLoss = obj.crossValidation.meanLoss;
                obj.performance.cvStdLoss = obj.crossValidation.stdLoss;
            else
                obj.performance.cvMeanAccuracy = NaN;
                obj.performance.cvStdAccuracy = NaN;
                obj.performance.cvAccuracies = [];
                obj.performance.cvMeanLoss = NaN;
                obj.performance.cvStdLoss = NaN;
            end
            
            % 過学習情報を追加
            obj.performance.isOverfit = ~strcmp(overfitAnalysis.severity, 'none');
            obj.performance.overfitMetrics = overfitAnalysis;
            
            % 結果表示
            obj.displayResults();
            
            obj.logMessage(2, '結果構造体の構築完了\n');
            obj.logMessage(2, '交差検証結果の保存状態: completed=%s, enabled=%s\n', ...
                mat2str(obj.crossValidation.completed), mat2str(obj.crossValidation.enabled));
        end

        function displayResults(obj)
            % 分類結果の詳細概要を表示する
            
            obj.logMessage(1, '\n=== ECOC Classification Results ===\n');
            
            % 基本性能指標
            if isfield(obj.performance, 'testAccuracy')
                obj.logMessage(1, 'Overall Test Accuracy: %.2f%%\n', obj.performance.testAccuracy * 100);
            end
            
            % 学習精度
            if ~isnan(obj.trainingAccuracy)
                obj.logMessage(1, 'Training Accuracy: %.2f%%\n', obj.trainingAccuracy * 100);
            end
            
            % 交差検証結果の詳細表示
            if obj.crossValidation.enabled
                if obj.crossValidation.completed
                    obj.logMessage(1, 'Cross-validation Results:\n');
                    obj.logMessage(1, '  - Mean Accuracy: %.2f%% (±%.2f%%)\n', ...
                        obj.crossValidation.meanAccuracy * 100, ...
                        obj.crossValidation.stdAccuracy * 100);
                    obj.logMessage(1, '  - Mean Loss: %.4f (±%.4f)\n', ...
                        obj.crossValidation.meanLoss, obj.crossValidation.stdLoss);
                    obj.logMessage(1, '  - K-Fold: %d\n', obj.crossValidation.kfold);
                    
                    if isfield(obj.crossValidation, 'confidenceInterval')
                        obj.logMessage(1, '  - 95%% Confidence Interval: [%.2f%%, %.2f%%]\n', ...
                            obj.crossValidation.confidenceInterval(1) * 100, ...
                            obj.crossValidation.confidenceInterval(2) * 100);
                    end
                    
                    % 各フォールドの詳細（verbosity >= 2）
                    if obj.verbosity >= 2 && ~isempty(obj.crossValidation.foldAccuracies)
                        obj.logMessage(2, '  - Individual Fold Accuracies:\n');
                        for i = 1:length(obj.crossValidation.foldAccuracies)
                            obj.logMessage(2, '    Fold %d: %.4f\n', i, obj.crossValidation.foldAccuracies(i));
                        end
                    end
                else
                    obj.logMessage(1, 'Cross-validation: Failed to complete\n');
                    if isfield(obj.crossValidation, 'errorMessage')
                        obj.logMessage(1, '  - Error: %s\n', obj.crossValidation.errorMessage);
                    end
                end
            else
                obj.logMessage(1, 'Cross-validation: Disabled\n');
                if isfield(obj.crossValidation, 'reason')
                    obj.logMessage(1, '  - Reason: %s\n', obj.crossValidation.reason);
                end
            end
            
            % AUC表示
            if isfield(obj.performance, 'testAuc')
                obj.logMessage(1, 'AUC: %.3f\n', obj.performance.testAuc);
            end
            
            % 過学習評価結果
            if isfield(obj.performance, 'isOverfit') && obj.performance.isOverfit
                obj.logMessage(1, '\n*** Overfitting Detected: %s ***\n', obj.performance.overfitMetrics.severity);
                if isfield(obj.performance.overfitMetrics, 'trainTestGap') && ~isnan(obj.performance.overfitMetrics.trainTestGap)
                    obj.logMessage(1, 'Training-Test gap: %.2f%%\n', obj.performance.overfitMetrics.trainTestGap * 100);
                end
                if isfield(obj.performance.overfitMetrics, 'valTestGap') && ~isnan(obj.performance.overfitMetrics.valTestGap)
                    obj.logMessage(1, 'Validation-Test gap: %.2f%%\n', obj.performance.overfitMetrics.valTestGap * 100);
                end
            end
            
            % クラスごとの性能（詳細表示はverbosity >= 2）
            if isfield(obj.performance, 'testClasswise') && obj.verbosity >= 2
                obj.logMessage(2, '\nClass-wise Performance (Test data):\n');
                for i = 1:length(obj.performance.classLabels)
                    obj.logMessage(2, 'Class %d:\n', obj.performance.classLabels(i));
                    obj.logMessage(2, '  - Precision: %.2f%%\n', obj.performance.testClasswise(i).precision * 100);
                    obj.logMessage(2, '  - Recall: %.2f%%\n', obj.performance.testClasswise(i).recall * 100);
                    obj.logMessage(2, '  - F1-Score: %.3f\n', obj.performance.testClasswise(i).f1score);
                end
            end
            
            % システム情報（verbosity >= 3）
            if obj.verbosity >= 3
                obj.logMessage(3, '\nSystem Information:\n');
                obj.logMessage(3, '  - ECOC Coding: %s\n', obj.params.classifier.ecoc.coding);
                obj.logMessage(3, '  - Base Learners: %s\n', obj.params.classifier.ecoc.learners);
                obj.logMessage(3, '  - Kernel Function: %s\n', obj.params.classifier.ecoc.kernel);
                obj.logMessage(3, '  - Optimization: %s\n', mat2str(obj.isOptimized));
                obj.logMessage(3, '  - Probability Estimation: %s\n', mat2str(obj.params.classifier.ecoc.probability));
                if isfield(obj.performance, 'classLabels')
                    obj.logMessage(3, '  - Number of Classes: %d\n', length(obj.performance.classLabels));
                end
                obj.logMessage(3, '  - Cross-validation Status: %s\n', mat2str(obj.crossValidation.completed));
            end
        end
        
        function probScores = convertToProb(~, scores)
            % 符号反転をしない版でテスト
            % ソフトマックス関数で確率に変換
            expScores = exp(scores - max(scores, [], 2));
            probScores = expScores ./ sum(expScores, 2);
        end
    end
end