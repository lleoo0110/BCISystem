classdef SVMClassifier < handle
    % SVMClassifier - サポートベクターマシンを用いた分類器クラス
    %
    % このクラスはEEGデータの分類のためのSVMモデルを学習・評価・予測するための
    % 機能を提供します。ハイパーパラメータ最適化、交差検証、性能評価などの機能を含みます。
    
    properties (Access = private)
        params          % パラメータ設定
        isOptimized     % 最適化の有無
        isEnabled       % SVMの有効/無効
        trainResults    % 学習結果の保存
        crossValModel   % 交差検証モデル
    end
    
    properties (Access = public)
        svmModel        % 学習済みSVMモデル
        performance     % 性能評価指標
    end
    
    methods (Access = public)
        function obj = SVMClassifier(params)
            % コンストラクタ
            % 
            % 入力:
            %   params - 設定パラメータを含む構造体
            
            obj.params = params;
            obj.isOptimized = params.classifier.svm.optimize;
            obj.isEnabled = params.classifier.svm.enable;
            obj.performance = struct();
        end
        
        function results = trainSVM(obj, processedData, processedLabel)
            % SVMモデルの学習と評価を行う
            %
            % 入力:
            %   processedData - 前処理済みEEGデータ
            %   processedLabel - データのクラスラベル
            %
            % 出力:
            %   results - 学習結果と性能評価指標を含む構造体
            
            try
                % EEGデータの正規化
                [normalizedEEG, normParams] = obj.normalizeData(processedData);
                
                % CSP特徴抽出処理
                [features, filters, cspParameters] = obj.extractCSPFeatures(normalizedEEG, processedLabel);
      
                % データの分割（学習用と検証用）
                [trainFeatures, trainLabels, testFeatures, testLabels] = obj.splitDataset(features, processedLabel);
                
                % クラス分布の確認
                obj.checkClassDistribution('学習', trainLabels);
                obj.checkClassDistribution('検証', testLabels);

                % モデルの学習（学習データのみを使用）
                fprintf('\nSVMモデルの学習を開始...\n');
                obj.trainModel(trainFeatures, trainLabels);

                % 交差検証の実行（学習データのみを使用）
                if obj.params.classifier.evaluation.enable
                    obj.performCrossValidation();
                end

                % テストデータでの評価
                testMetrics = obj.evaluateModel(testFeatures, testLabels);
                
                % 過学習の検出
                [isOverfit, overfitMetrics] = obj.validateOverfitting(testMetrics.accuracy);
                
                % 結果の構築
                results = struct(...
                    'model', obj.svmModel, ...
                    'performance', testMetrics, ...
                    'overfitMetrics', overfitMetrics, ...
                    'cspFilters', filters, ...
                    'cspParameters', cspParameters, ...
                    'normParams', normParams ...
                );
                
                % パフォーマンス情報を保存
                obj.performance = results.performance;
                
                % 結果表示
                obj.displayResults();

            catch ME
                error('SVM training failed: %s', ME.message);
            end
        end
        
        function [label, score] = predictOnline(obj, data, svm)
            % オンラインでSVM予測を実行する
            %
            % 入力:
            %   data - 分類するEEGデータ
            %   svm - 学習済みSVMモデルと関連パラメータを含む構造体
            %
            % 出力:
            %   label - 予測されたクラスラベル
            %   score - 各クラスへの所属確率
            
            try
                % 入力データの検証
                if isempty(data)
                    error('Data cannot be empty');
                end
        
                % モデルの検証
                if isempty(svm)
                    error('Model cannot be empty');
                end

                % モデルのクラスラベル数の検証
                classLabels = svm.model.ClassNames;
                if length(classLabels) ~= 2
                    error('SVMClassifier only supports binary classification');
                end

                % EEGデータの正規化
                normalizedEEG = data;
                if obj.params.classifier.normalize.enable && isfield(svm, 'normParams')
                    eegNormalizer = EEGNormalizer(obj.params);
                    normalizedEEG = eegNormalizer.normalizeOnline(data, svm.normParams);
                end

                % CSP特徴量の抽出
                if isfield(svm, 'cspFilters') && ~isempty(svm.cspFilters)
                    cspExtractor = CSPExtractor(obj.params);
                    features = cspExtractor.extractFeatures(normalizedEEG, svm.cspFilters);
                else
                    error('CSP filters not found in SVM model');
                end
        
                % 予測の実行
                [label, score] = predict(svm.model, features);
        
            catch ME
                error('Error in SVM online prediction: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function [normalizedEEG, normParams] = normalizeData(obj, data)
            % EEGデータの正規化を行う
            %
            % 入力:
            %   data - 正規化するEEGデータ
            %
            % 出力:
            %   normalizedEEG - 正規化されたデータ
            %   normParams - 正規化パラメータ（オンライン予測で使用）
            
            normalizedEEG = data;
            normParams = [];
            
            if obj.params.classifier.normalize.enable
                eegNormalizer = EEGNormalizer(obj.params);
                [normalizedEEG, normParams] = eegNormalizer.normalize(data);
            end
        end
        
        function [features, filters, cspParameters] = extractCSPFeatures(obj, data, labels)
            % 共通空間パターン(CSP)特徴量を抽出する
            %
            % 入力:
            %   data - EEGデータ
            %   labels - クラスラベル
            %
            % 出力:
            %   features - 抽出された特徴量
            %   filters - CSPフィルタ
            %   cspParameters - CSP計算パラメータ
            
            cspExtractor = CSPExtractor(obj.params);
            [filters, cspParameters] = cspExtractor.trainCSP(data, labels);
            features = cspExtractor.extractFeatures(data, filters);
        end
        
        function [trainFeatures, trainLabels, testFeatures, testLabels] = splitDataset(~, features, labels)
            % データセットを学習用と検証用に分割する
            %
            % 入力:
            %   features - 分割する特徴量
            %   labels - 対応するラベル
            %
            % 出力:
            %   trainFeatures - 学習用特徴量
            %   trainLabels - 学習用ラベル
            %   testFeatures - 検証用特徴量
            %   testLabels - 検証用ラベル
            
            fprintf('\nデータを学習用と検証用に分割します...\n');
            cv = cvpartition(labels, 'Holdout', 0.2);  % 80%学習, 20%検証
            trainIdx = cv.training;
            testIdx = cv.test;
            
            trainFeatures = features(trainIdx, :);
            trainLabels = labels(trainIdx);
            testFeatures = features(testIdx, :);
            testLabels = labels(testIdx);
            
            fprintf('  - 学習データ: %d サンプル\n', sum(trainIdx));
            fprintf('  - 検証データ: %d サンプル\n', sum(testIdx));
        end
        
        function trainModel(obj, features, labels)
            % SVMモデルを学習する
            %
            % 入力:
            %   features - 学習用特徴量
            %   labels - 学習用ラベル
            
            if obj.isOptimized
                obj.trainOptimizedModel(features, labels);
            else
                obj.trainDefaultModel(features, labels);
            end

            % 確率推定を有効にする
            if obj.params.classifier.svm.probability
                fprintf('確率推定を有効化...\n');
                obj.svmModel = fitPosterior(obj.svmModel);
            end
            
            fprintf('SVMモデルの学習完了\n');
        end
        
        function trainOptimizedModel(obj, features, labels)
            % ハイパーパラメータ最適化を用いてSVMモデルを学習する
            %
            % 入力:
            %   features - 学習用特徴量
            %   labels - 学習用ラベル
            
            fprintf('\nSVMハイパーパラメータ最適化を実行...\n');
            
            % 最適化方法の設定
            if isfield(obj.params.classifier.svm, 'hyperparameters') && ...
               isfield(obj.params.classifier.svm.hyperparameters, 'optimizer')
                optMethod = obj.params.classifier.svm.hyperparameters.optimizer;
                fprintf('  - 最適化手法: %s\n', optMethod);
                
                % 自動最適化を使用
                if strcmpi(optMethod, 'auto')
                    obj.trainWithAutoOptimization(features, labels);
                else
                    % グリッドサーチ最適化
                    obj.trainWithGridSearch(features, labels);
                end
            else
                % ハイパーパラメータフィールドがない場合はデフォルト最適化
                obj.trainWithAutoOptimization(features, labels);
                fprintf('  - デフォルトの自動最適化を使用\n');
            end
        end
        
        function trainWithAutoOptimization(obj, features, labels)
            % MATLABの自動最適化を使用してSVMを学習する
            %
            % 入力:
            %   features - 学習用特徴量
            %   labels - 学習用ラベル
            
            fprintf('  - MATLABの自動最適化を使用\n');
            obj.svmModel = fitcsvm(features, labels, ...
                'KernelFunction', obj.params.classifier.svm.kernel, ...
                'OptimizeHyperparameters', 'auto', ...
                'HyperparameterOptimizationOptions', struct(...
                    'ShowPlots', false, ...
                    'Verbose', 1, ...
                    'MaxObjectiveEvaluations', 30 ...
                ) ...
            );
        end
        
        function trainWithGridSearch(obj, features, labels)
            % グリッドサーチを用いたハイパーパラメータ最適化
            %
            % 入力:
            %   features - 学習用特徴量
            %   labels - 学習用ラベル
            
            fprintf('  - グリッドサーチ最適化を使用\n');
            
            % BoxConstraintパラメータの設定
            if ~isfield(obj.params.classifier.svm.hyperparameters, 'boxConstraint')
                boxConstraints = [0.1, 1, 10];
                fprintf('  - BoxConstraint候補: デフォルト値を使用 [%s]\n', num2str(boxConstraints));
            else
                boxConstraints = obj.params.classifier.svm.hyperparameters.boxConstraint;
                fprintf('  - BoxConstraint候補: [%s]\n', num2str(boxConstraints));
            end
            
            % KernelScaleパラメータの設定
            if ~isfield(obj.params.classifier.svm.hyperparameters, 'kernelScale')
                kernelScales = [0.1, 1, 10];
                fprintf('  - KernelScale候補: デフォルト値を使用 [%s]\n', num2str(kernelScales));
            else
                kernelScales = obj.params.classifier.svm.hyperparameters.kernelScale;
                fprintf('  - KernelScale候補: [%s]\n', num2str(kernelScales));
            end
            
            % グリッドサーチの実行
            bestScore = -inf;
            bestParams = struct('BoxConstraint', 1, 'KernelScale', 1);
            
            for c = boxConstraints
                for k = kernelScales
                    mdl = fitcsvm(features, labels, ...
                        'KernelFunction', obj.params.classifier.svm.kernel, ...
                        'BoxConstraint', c, ...
                        'KernelScale', k);
                    
                    cv = crossval(mdl, 'KFold', obj.params.classifier.svm.validation.kfold);
                    score = 1 - kfoldLoss(cv);
                    
                    fprintf('  - C=%.3f, KScale=%.3f: スコア=%.4f\n', c, k, score);
                    
                    if score > bestScore
                        bestScore = score;
                        bestParams.BoxConstraint = c;
                        bestParams.KernelScale = k;
                        fprintf('    -> 新しい最良パラメータ\n');
                    end
                end
            end
            
            fprintf('\n最適パラメータ: C=%.3f, KScale=%.3f, スコア=%.4f\n', ...
                bestParams.BoxConstraint, bestParams.KernelScale, bestScore);
            
            % 最適パラメータでモデルを作成
            obj.svmModel = fitcsvm(features, labels, ...
                'KernelFunction', obj.params.classifier.svm.kernel, ...
                'BoxConstraint', bestParams.BoxConstraint, ...
                'KernelScale', bestParams.KernelScale);
        end
        
        function trainDefaultModel(obj, features, labels)
            % デフォルトパラメータでSVMモデルを学習する
            %
            % 入力:
            %   features - 学習用特徴量
            %   labels - 学習用ラベル
            
            fprintf('\nデフォルトパラメータでSVMモデルを作成...\n');
            obj.svmModel = fitcsvm(features, labels, ...
                'KernelFunction', obj.params.classifier.svm.kernel);
        end
        
        function performCrossValidation(obj)
            % 交差検証を実行する
            
            if ~obj.params.classifier.evaluation.enable
                return;
            end
            
            kfold = obj.params.classifier.svm.validation.kfold;
            fprintf('\n交差検証の実行（K=%d）...\n', kfold);
            obj.crossValModel = crossval(obj.svmModel, 'KFold', kfold);
            
            obj.performance.cvAccuracies = zeros(kfold, 1);
            
            % 各フォールドの精度を計算
            for i = 1:kfold
                obj.performance.cvAccuracies(i) = 1 - kfoldLoss(obj.crossValModel, 'Folds', i);
                fprintf('  - フォールド %d: 精度 = %.4f\n', i, obj.performance.cvAccuracies(i));
            end
            
            obj.performance.cvMeanAccuracy = mean(obj.performance.cvAccuracies);
            obj.performance.cvStdAccuracy = std(obj.performance.cvAccuracies);
            
            fprintf('平均交差検証精度: %.4f (±%.4f)\n', ...
                obj.performance.cvMeanAccuracy, obj.performance.cvStdAccuracy);
        end

        function metrics = evaluateModel(obj, testFeatures, testLabels)
            % テストデータでモデルの性能を評価する
            %
            % 入力:
            %   testFeatures - 検証用特徴量
            %   testLabels - 検証用ラベル
            %
            % 出力:
            %   metrics - テストデータでの性能評価結果
            
            fprintf('\n検証データでモデルを評価中...\n');
            metrics = struct(...
                'accuracy', [], ...
                'score', [], ...
                'confusionMat', [], ....
                'roc', [], ...
                'auc', [], ...
                'classwise', [] ...
            );

            [pred, score] = predict(obj.svmModel, testFeatures);
            metrics.score = score;
            
            % 検証データでの精度と混同行列
            metrics.accuracy = mean(pred == testLabels);
            metrics.confusionMat = confusionmat(testLabels, pred);
            
            % クラスラベル
            classLabels = unique(testLabels);
            
            % ROC曲線とAUCの計算（2クラス分類の場合）
            if length(classLabels) == 2
                [X, Y, T, AUC] = perfcurve(testLabels, score(:,2), classLabels(2));
                metrics.roc = struct('X', X, 'Y', Y, 'T', T);
                metrics.auc = AUC;
                obj.performance.testRoc = metrics.roc;
                obj.performance.testAuc = AUC;
            end
            
            % クラスごとの性能評価
            metrics.classwise = obj.calculateClassMetrics(testLabels, pred, classLabels);
            obj.performance.testClasswise = metrics.classwise;
        end

        function metrics = calculateClassMetrics(~, trueLabels, predLabels, classLabels)
            % クラスごとの性能評価指標を計算する
            %
            % 入力:
            %   trueLabels - 真のラベル
            %   predLabels - 予測されたラベル
            %   classLabels - クラスのユニークラベル
            %
            % 出力:
            %   metrics - クラスごとの性能評価指標
            
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
            end
        end
        
        function checkClassDistribution(~, setName, labels)
            % データセット内のクラス分布を解析して表示する
            %
            % 入力:
            %   setName - データセット名（表示用）
            %   labels - クラスラベル
            
            uniqueLabels = unique(labels);
            fprintf('\n%sデータのクラス分布:\n', setName);
            
            for i = 1:length(uniqueLabels)
                count = sum(labels == uniqueLabels(i));
                fprintf('  - クラス %d: %d サンプル (%.1f%%)\n', ...
                    uniqueLabels(i), count, (count/length(labels))*100);
            end
            
            % クラス不均衡の評価
            counts = histcounts(labels, 'BinMethod', 'integers');
            maxCount = max(counts);
            minCount = min(counts);
            imbalanceRatio = maxCount / max(minCount, 1);
            
            if imbalanceRatio > 3
                warning('%sデータセットのクラス不均衡が大きいです (比率: %.1f:1)', ...
                    setName, imbalanceRatio);
            end
        end
        
        function [isOverfit, metrics] = validateOverfitting(obj, testAccuracy)
            % 学習データと検証データの性能比較に基づく過学習検出
            %
            % 入力:
            %   testAccuracy - 検証データでの精度
            %
            % 出力:
            %   isOverfit - 過学習検出フラグ
            %   metrics - 過学習メトリクス

            fprintf('\n=== 過学習検証の実行 ===\n');
            
            % 交差検証精度との比較
            if isfield(obj.performance, 'cvMeanAccuracy')
                valAccuracy = obj.performance.cvMeanAccuracy;
                perfGap = abs(valAccuracy - testAccuracy);
                
                fprintf('交差検証平均精度: %.2f%%\n', valAccuracy * 100);
                fprintf('検証データ精度: %.2f%%\n', testAccuracy * 100);
                fprintf('精度差: %.2f%%\n', perfGap * 100);
                
                % 過学習の程度を評価
                if perfGap > 0.15  % 15%以上の差は重度の過学習と判定
                    severity = 'severe';
                elseif perfGap > 0.10  % 10%以上の差は中程度の過学習
                    severity = 'moderate';
                elseif perfGap > 0.05  % 5%以上の差は軽度の過学習
                    severity = 'mild';
                else
                    severity = 'none';
                end
            else
                % 交差検証データがない場合はテスト精度のみで判断
                valAccuracy = NaN;
                perfGap = NaN;
                
                if testAccuracy < 0.6
                    severity = 'poor_performance';
                elseif testAccuracy > 0.95
                    severity = 'potential_overfit';
                else
                    severity = 'unknown';
                end
            end
            
            % 過学習フラグの設定
            isOverfit = ~strcmp(severity, 'none') && ~strcmp(severity, 'unknown');
            
            % メトリクスの構築
            metrics = struct(...
                'valAccuracy', valAccuracy, ...
                'testAccuracy', testAccuracy, ...
                'accuracyGap', perfGap, ...
                'severity', severity);
            
            % 結果の表示
            fprintf('過学習評価結果: %s\n', severity);
            
            % 過学習や性能問題の警告とアドバイス
            obj.displayOverfitWarning(isOverfit, severity);
        end
        
        function displayOverfitWarning(~, isOverfit, severity)
            % 過学習や性能問題の警告を表示する
            %
            % 入力:
            %   isOverfit - 過学習フラグ
            %   severity - 過学習または性能問題の重症度
            
            if isOverfit
                fprintf('\n警告: モデルに過学習または性能問題の兆候が検出されました (%s)\n', severity);
                fprintf('  対策として以下を検討してください:\n');
                fprintf('  - 特徴量の削減\n');
                fprintf('  - 正則化パラメータの調整\n');
                fprintf('  - データ拡張\n');
                fprintf('  - より多くのトレーニングデータの使用\n');
            elseif strcmp(severity, 'poor_performance')
                fprintf('\n警告: モデルの性能が十分ではありません。\n');
                fprintf('  対策として以下を検討してください:\n');
                fprintf('  - より適切な特徴量の選択\n');
                fprintf('  - ハイパーパラメータの最適化\n');
                fprintf('  - 別のカーネル関数の試行\n');
            else
                fprintf('\nモデルは良好に一般化されています。\n');
            end
        end

        function displayResults(obj)
            % 分類結果の概要を表示する
            
            fprintf('\n=== SVM Classification Results ===\n');
            
            % 検証データでの結果
            if isfield(obj.performance, 'testAccuracy')
                fprintf('Overall Accuracy: %.2f%%\n', obj.performance.testAccuracy * 100);
            end
            
            % 交差検証結果
            if obj.params.classifier.evaluation.enable && isfield(obj.performance, 'cvMeanAccuracy')
                fprintf('Cross-validation Accuracy: %.2f%% (±%.2f%%)\n', ...
                    obj.performance.cvMeanAccuracy * 100, ...
                    obj.performance.cvStdAccuracy * 100);
            end
            
            % AUC表示
            if isfield(obj.performance, 'testAuc')
                fprintf('AUC: %.3f\n', obj.performance.testAuc);
            end
            
            % 混同行列表示
            if isfield(obj.performance, 'testConfusionMat')
                fprintf('\nConfusion Matrix:\n');
                disp(obj.performance.testConfusionMat);
            end
            
            % 過学習評価結果
            if isfield(obj.performance, 'isOverfit')
                if obj.performance.isOverfit
                    fprintf('\nOverfitting detected: %s\n', obj.performance.overfitMetrics.severity);
                    fprintf('Accuracy gap: %.2f%%\n', obj.performance.overfitMetrics.accuracyGap * 100);
                else
                    fprintf('\nNo significant overfitting detected.\n');
                end
            end
            
            % クラスごとの性能（検証データ）
            if isfield(obj.performance, 'testClasswise')
                fprintf('\nClass-wise Performance (Test data):\n');
                for i = 1:length(obj.performance.classLabels)
                    fprintf('Class %d:\n', obj.performance.classLabels(i));
                    fprintf('  - Precision: %.2f%%\n', obj.performance.testClasswise(i).precision * 100);
                    fprintf('  - Recall: %.2f%%\n', obj.performance.testClasswise(i).recall * 100);
                    fprintf('  - F1-Score: %.2f\n', obj.performance.testClasswise(i).f1score);
                end
            end
        end
    end
end