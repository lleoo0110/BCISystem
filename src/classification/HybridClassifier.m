classdef HybridClassifier < handle
    %% HybridClassifier - CNN+LSTM統合ハイブリッド分類器
    %
    % このクラスはEEGデータに対してCNNとLSTMを組み合わせた
    % ハイブリッド深層学習モデルを実装します。時空間的特徴の両方を
    % 効果的に捉え、高精度な分類を実現します。
    %
    % 主な機能:
    %   - EEGデータの前処理と変換
    %   - CNN・LSTMの並列学習とモデル統合
    %   - 学習済みモデルによるオンライン予測
    %   - 過学習検出と詳細な性能評価
    %   - ハイパーパラメータの最適化サポート
    %
    % 使用例:
    %   params = getConfig('epocx');
    %   hybrid = HybridClassifier(params);
    %   results = hybrid.trainHybrid(processedData, processedLabel);
    %   [label, score] = hybrid.predictOnline(newData, results.model);
    
    properties (Access = private)
        params              % システム設定パラメータ
        netCNN             % 学習済みCNNネットワーク
        netLSTM            % 学習済みLSTMネットワーク
        adaModel           % 学習済みAdaBoost統合分類器
        isEnabled          % 有効/無効フラグ
        isInitialized      % 初期化フラグ
        useGPU             % GPU使用の有無

        % 学習進捗の追跡用
        trainingHistory     % 学習履歴
        validationHistory   % 検証履歴
        bestValAccuracy     % 最良の検証精度
        patienceCounter     % Early Stopping用カウンター
        currentEpoch        % 現在のエポック

        % 過学習監視用
        overfitMetrics      % 過学習メトリクス

        % コンポーネント
        dataAugmenter       % データ拡張処理
        normalizer          % 正規化処理
    end
    
    properties (Access = public)
        performance         % 性能評価指標
    end
    
    methods (Access = public)
        %% コンストラクタ - 初期化処理
        function obj = HybridClassifier(params)
            % HybridClassifierのインスタンスを初期化
            %
            % 入力:
            %   params - 設定パラメータ（getConfig関数から取得）
            
            obj.params = params;
            obj.isInitialized = false;
            obj.useGPU = params.classifier.hybrid.gpu;
            
            % プロパティの初期化
            obj.initializeProperties();
            
            % コンポーネントの初期化
            obj.dataAugmenter = DataAugmenter(params);
            obj.normalizer = EEGNormalizer(params);
            
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

        %% ハイブリッドモデル学習メソッド
        function results = trainHybrid(obj, processedData, processedLabel)
            % EEGデータを使用してハイブリッドモデルを学習
            %
            % 入力:
            %   processedData - 前処理済みEEGデータ [チャンネル x サンプル x エポック]
            %   processedLabel - クラスラベル [エポック x 1]
            %
            % 出力:
            %   results - 学習結果を含む構造体（モデル、性能評価、正規化パラメータなど）
            try                
                fprintf('\n=== ハイブリッドモデル学習処理を開始 ===\n');
                
                % データの次元を確認し、必要に応じて調整
                [processedData, processInfo] = obj.validateAndPrepareData(processedData);
                fprintf('データ検証: %s\n', processInfo);
                
                % データを学習・検証・テストセットに分割
                [trainData, trainLabels, valData, valLabels, testData, testLabels] = obj.splitDataset(processedData, processedLabel);

                % 学習データの拡張と正規化処理
                [trainData, trainLabels, normParams] = obj.preprocessTrainingData(trainData, trainLabels);
                
                % 検証・テストデータも同じ正規化パラメータで処理
                valData = obj.normalizer.normalizeOnline(valData, normParams);
                testData = obj.normalizer.normalizeOnline(testData, normParams);
                
                % CNN用にデータ形式を変換
                prepCnnTrainData = obj.prepareDataForCNN(trainData);
                prepCnnValData = obj.prepareDataForCNN(valData);
                prepCnnTestData = obj.prepareDataForCNN(testData);
                
                % LSTM用にデータ形式を変換
                prepLstmTrainData = obj.prepareDataForLSTM(trainData);
                prepLstmValData = obj.prepareDataForLSTM(valData);
                prepLstmTestData = obj.prepareDataForLSTM(testData);
                
                % モデルの学習
                [hybridModel, trainInfo] = obj.trainHybridModel( ...
                    prepCnnTrainData, prepLstmTrainData, trainLabels, ...
                    prepCnnValData, prepLstmValData, valLabels);
                
                % テストデータでの最終評価
                testMetrics = obj.evaluateModel(hybridModel, prepCnnTestData, prepLstmTestData, testLabels);
                
                % 過学習の分析
                [isOverfit, obj.overfitMetrics] = obj.validateOverfitting(trainInfo, testMetrics);
                
                if isOverfit
                    fprintf('\n警告: モデルに過学習の兆候が検出されました (%s)\n', obj.overfitMetrics.severity);
                end

                % 性能指標の更新
                obj.updatePerformanceMetrics(testMetrics);
                
                % 交差検証の実行
                crossValidation = obj.performCrossValidationIfEnabled(processedData, processedLabel);
                
                % 結果構造体の構築
                results = obj.buildResultsStruct(hybridModel, testMetrics, trainInfo, ...
                    crossValidation, normParams);

                % 結果のサマリー表示
                obj.displayResults();
                
                % 使用リソースのクリーンアップ
                obj.resetGPUMemory();
                fprintf('\n=== ハイブリッドモデル学習処理が完了しました ===\n');

            catch ME
                % エラー発生時の詳細情報出力
                fprintf('\n=== ハイブリッドモデル学習中にエラーが発生しました ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラー発生場所:\n');
                for i = 1:length(ME.stack)
                    fprintf('  ファイル: %s\n  行: %d\n  関数: %s\n\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end

                % クリーンアップ処理
                obj.resetGPUMemory();
                rethrow(ME);
            end
        end

        %% オンライン予測メソッド
        function [label, score] = predictOnline(obj, data, hybrid)
            % 学習済みモデルを使用して新しいEEGデータを分類
            %
            % 入力:
            %   data - 分類するEEGデータ [チャンネル x サンプル]
            %   hybridModel - 学習済みハイブリッドモデル
            %
            % 出力:
            %   label - 予測クラスラベル
            %   score - 予測確率スコア
            try
                % 正規化パラメータを使用してデータを正規化
                if isfield(hybrid, 'normParams') && ~isempty(hybrid.normParams)
                    data = obj.normalizer.normalizeOnline(data, hybrid.normParams);
                else
                    warning('正規化パラメータが見つかりません。正規化をスキップします。');
                end
        
                % CNN用とLSTM用にデータを変換
                cnnData = obj.prepareDataForCNN(data);
                lstmData = obj.prepareDataForLSTM(data);
                
                % 各モデルから特徴抽出
                cnnFeatures = activations(hybrid.netCNN, cnnData, 'fc_cnn', 'OutputAs', 'rows');
                
                % LSTMからの特徴抽出
                if iscell(lstmData)
                    lstmFeatures = [];
                    for i = 1:length(lstmData)
                        lstmOut = predict(hybrid.netLSTM, lstmData{i}, 'MiniBatchSize', 1);
                        
                        % 最初のサンプルで配列を初期化
                        if i == 1
                            lstmFeatures = zeros(length(lstmData), size(lstmOut, 2));
                        end
                        
                        lstmFeatures(i,:) = lstmOut;
                    end
                else
                    lstmOut = predict(hybrid.netLSTM, lstmData, 'MiniBatchSize', 1);
                    lstmFeatures = lstmOut;
                end
                
                % 特徴を結合
                combinedFeatures = [cnnFeatures, lstmFeatures];
                
                % AdaBoostで最終予測
                [label, score] = predict(hybridModel.adaModel, combinedFeatures);
        
            catch ME
                fprintf('Error in hybrid online prediction: %s\n', ME.message);
                fprintf('Error details:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end
    end
    
    methods (Access = private)
        %% プロパティ初期化メソッド
        function initializeProperties(obj)
            % クラスプロパティの初期化
            obj.trainingHistory = struct('loss', [], 'accuracy', []);
            obj.validationHistory = struct('loss', [], 'accuracy', []);
            obj.bestValAccuracy = 0;
            obj.patienceCounter = 0;
            obj.currentEpoch = 0;
            obj.overfitMetrics = struct();
        end
        
        %% データ検証と準備メソッド
        function [validatedData, infoMsg] = validateAndPrepareData(~, data)
            % 入力データの検証と適切な形式への変換
            
            % データの次元と形状を確認
            dataSize = size(data);
            dimCount = ndims(data);
            
            if dimCount > 3
                error('対応していないデータ次元数: %d (最大3次元まで対応)', dimCount);
            end
            
            validatedData = data;
            
            % 必要に応じてデータ次元を調整
            if dimCount == 2
                [channels, samples] = size(data);
                validatedData = reshape(data, [channels, samples, 1]);
                infoMsg = sprintf('2次元データを3次元に変換 [%d×%d] → [%d×%d×1]', ...
                    channels, samples, channels, samples);
            else
                [channels, samples, epochs] = size(data);
                infoMsg = sprintf('3次元データを検証 [%d×%d×%d]', channels, samples, epochs);
            end
            
            % データの妥当性検証
            validateattributes(validatedData, {'numeric'}, {'finite', 'nonnan'}, ...
                'validateAndPrepareData', 'data');
            
            return;
        end
        
        %% 学習データの前処理
        function [procTrainData, procTrainLabels, normParams] = preprocessTrainingData(obj, trainData, trainLabels)
            % 学習データの拡張と正規化を実行
            
            procTrainData = trainData;
            procTrainLabels = trainLabels;
            normParams = [];
            
            % データ拡張処理
            if obj.params.classifier.augmentation.enable
                fprintf('\nデータ拡張を実行...\n');
                [procTrainData, procTrainLabels, ~] = obj.dataAugmenter.augmentData(trainData, trainLabels);
                fprintf('  - 拡張前: %d サンプル\n', length(trainLabels));
                fprintf('  - 拡張後: %d サンプル (%.1f倍)\n', length(procTrainLabels), ... 
                    length(procTrainLabels)/length(trainLabels));
            end
            
            % 正規化処理
            if obj.params.classifier.normalize.enable
                [procTrainData, normParams] = obj.normalizer.normalize(procTrainData);
            end
            
            return;
        end
        
        %% データセット分割メソッド
        function [trainData, trainLabels, valData, valLabels, testData, testLabels] = splitDataset(obj, data, labels)
            % データを学習・検証・テストセットに分割
            %
            % 入力:
            %   data - 前処理済みEEGデータ [チャンネル x サンプル x エポック]
            %   labels - クラスラベル [エポック x 1]
            %
            % 出力:
            %   trainData - 学習データ
            %   trainLabels - 学習ラベル
            %   valData - 検証データ
            %   valLabels - 検証ラベル
            %   testData - テストデータ
            %   testLabels - テストラベル
            
            try
                % 分割数の取得
                k = obj.params.classifier.hybrid.training.validation.kfold;
                
                % データサイズの取得
                [~, ~, numEpochs] = size(data);
                fprintf('\nデータセット分割 (k=%d):\n', k);
                fprintf('  - 総エポック数: %d\n', numEpochs);
        
                % インデックスのシャッフル
                rng('default'); % 再現性のため
                shuffledIdx = randperm(numEpochs);
        
                % 分割比率の計算
                trainRatio = (k-1)/k;  % (k-1)/k
                valRatio = 1/(2*k);    % 0.5/k
                testRatio = 1/(2*k);   % 0.5/k
        
                % データ数の計算
                numTrain = floor(numEpochs * trainRatio);
                numVal = floor(numEpochs * valRatio);
                
                % インデックスの分割
                trainIdx = shuffledIdx(1:numTrain);
                valIdx = shuffledIdx(numTrain+1:numTrain+numVal);
                testIdx = shuffledIdx(numTrain+numVal+1:end);
        
                % データの分割
                trainData = data(:,:,trainIdx);
                trainLabels = labels(trainIdx);
                
                valData = data(:,:,valIdx);
                valLabels = labels(valIdx);
                
                testData = data(:,:,testIdx);
                testLabels = labels(testIdx);

                % 分割結果のサマリー表示
                fprintf('  - 学習データ: %d サンプル (%.1f%%)\n', ...
                    length(trainIdx), (length(trainIdx)/numEpochs)*100);
                fprintf('  - 検証データ: %d サンプル (%.1f%%)\n', ...
                    length(valIdx), (length(valIdx)/numEpochs)*100);
                fprintf('  - テストデータ: %d サンプル (%.1f%%)\n', ...
                    length(testIdx), (length(testIdx)/numEpochs)*100);
        
                % データの検証
                if isempty(trainData) || isempty(valData) || isempty(testData)
                    error('分割後に空のデータセットが存在します');
                end
        
                % クラスの分布を確認
                obj.checkClassDistribution('学習', trainLabels);
                obj.checkClassDistribution('検証', valLabels);
                obj.checkClassDistribution('テスト', testLabels);
        
            catch ME
                error('データ分割でエラーが発生しました: %s', ME.message);
            end
        end
        
        %% クラス分布確認メソッド
        function checkClassDistribution(~, setName, labels)
            % データセット内のクラス分布を解析して表示
            
            uniqueLabels = unique(labels);
            fprintf('\n%sデータのクラス分布:\n', setName);
            
            for i = 1:length(uniqueLabels)
                count = sum(labels == uniqueLabels(i));
                fprintf('  - クラス %d: %d サンプル (%.1f%%)\n', ...
                    uniqueLabels(i), count, (count/length(labels))*100);
            end
            
            % クラス不均衡の評価
            maxCount = max(histcounts(labels));
            minCount = min(histcounts(labels));
            imbalanceRatio = maxCount / max(minCount, 1);
            
            if imbalanceRatio > 3
                warning('%sデータセットのクラス不均衡が大きいです (比率: %.1f:1)', ...
                    setName, imbalanceRatio);
            end
        end

        %% データのCNN形式への変換
        function preparedData = prepareDataForCNN(~, data)
            % データをCNNに適した形式に変換
            %
            % 入力:
            %   data - 入力データ
            %
            % 出力:
            %   preparedData - CNN用に整形されたデータ
            
            try
                % データ次元数の確認
                if ndims(data) == 3
                    % 3次元データ（チャンネル x サンプル x エポック）の場合
                    [channels, samples, epochs] = size(data);
                    preparedData = zeros(samples, channels, 1, epochs);
                    
                    for i = 1:epochs
                        preparedData(:,:,1,i) = data(:,:,i)';
                    end
                        
                elseif ismatrix(data)
                    % 2次元データ（チャンネル x サンプル）の場合
                    [channels, samples] = size(data);
                    preparedData = permute(data, [2, 1, 3]);
                    preparedData = reshape(preparedData, [samples, channels, 1, 1]);
                        
                else
                    % その他の次元数は対応外
                    error('対応していないデータ次元数: %d', ndims(data));
                end
                
                % NaN/Infチェック
                if any(isnan(preparedData(:)))
                    error('変換後のデータにNaN値が含まれています');
                end
                
                if any(isinf(preparedData(:)))
                    error('変換後のデータにInf値が含まれています');
                end
                
            catch ME
                fprintf('データ形式変換でエラーが発生: %s\n', ME.message);
                fprintf('入力データサイズ: [%s]\n', num2str(size(data)));
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% データのLSTM形式への変換
        function preparedData = prepareDataForLSTM(obj, data)
            % データをLSTMに適した形式に変換
            %
            % 入力:
            %   data - 入力データ (3次元数値配列、2次元配列、またはセル配列)
            %
            % 出力:
            %   preparedData - LSTM用に整形されたデータ (セル配列)
            
            try
                fprintf('LSTM用データ変換を開始...\n');
                
                if iscell(data)
                    % 入力が既にセル配列の場合
                    trials = numel(data);
                    preparedData = cell(trials, 1);
                    fprintf('セル配列入力: %d試行\n', trials);
                    
                    for i = 1:trials
                        currentData = data{i};
                        if ~isa(currentData, 'double')
                            currentData = double(currentData);
                        end
                        
                        % 重要: LSTM用に時間軸と特徴量軸を転置
                        % [チャンネル × 時間] → [時間 × チャンネル]
                        currentData = currentData';
                        
                        % NaN/Inf値の検出と補間処理
                        currentData = obj.interpolateInvalidValues(currentData, i);
                        
                        preparedData{i} = currentData;
                    end
                    
                elseif ndims(data) == 3
                    % 3次元数値配列の処理 [チャンネル × 時間ポイント × 試行]
                    [channels, timepoints, trials] = size(data);
                    preparedData = cell(trials, 1);
                    fprintf('3次元データ入力: [%dチャンネル × %d時間ポイント × %d試行]\n', ...
                        channels, timepoints, trials);
                    
                    for i = 1:trials
                        currentData = data(:, :, i);
                        if ~isa(currentData, 'double')
                            currentData = double(currentData);
                        end
                        
                        % 重要: LSTM用に時間軸と特徴量軸を転置
                        currentData = currentData';  % [時間 × チャンネル]
                        
                        % NaN/Inf値の検出と補間処理
                        currentData = obj.interpolateInvalidValues(currentData, i);
                        
                        preparedData{i} = currentData;
                    end
                    
                elseif ismatrix(data)
                    % 2次元データ（単一試行）の処理
                    [channels, timepoints] = size(data);
                    fprintf('2次元データ入力: [%dチャンネル × %d時間ポイント]\n', channels, timepoints);
                    
                    currentData = data;
                    if ~isa(currentData, 'double')
                        currentData = double(currentData);
                    end
                    
                    % 重要: LSTM用に時間軸と特徴量軸を転置
                    currentData = currentData';  % [時間 × チャンネル]
                    
                    % NaN/Inf値の検出と補間処理
                    currentData = obj.interpolateInvalidValues(currentData, 1);
                    
                    preparedData = {currentData};
                else
                    error('対応していないデータ次元数: %d (最大3次元まで対応)', ndims(data));
                end
                
                % 結果検証
                if isempty(preparedData)
                    error('変換後のデータが空です');
                end
                
                % サンプルとなる試行のサイズを表示（デバッグ用）
                sampleData = preparedData{1};
                [timepoints, features] = size(sampleData);
                fprintf('LSTM用データ変換完了: 各試行は [%d時間ポイント × %d特徴量] の形式\n', ...
                    timepoints, features);
                
            catch ME
                fprintf('LSTM用データ準備でエラーが発生: %s\n', ME.message);
                fprintf('入力データサイズ: [%s]\n', num2str(size(data)));
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end
        
        function processedData = interpolateInvalidValues(~, data, trialIndex)
            % NaN/Infなどの無効値を線形補間で処理
            %
            % 入力:
            %   data - 処理するデータ [時間 × チャンネル]
            %   trialIndex - 試行インデックス（デバッグ情報用）
            %
            % 出力:
            %   processedData - 補間処理済みデータ
            
            processedData = data;
            [timepoints, channels] = size(data);
            
            % 無効値の検出
            hasInvalidData = false;
            invalidCount = 0;
            
            for ch = 1:channels
                channelData = data(:, ch);
                invalidIndices = isnan(channelData) | isinf(channelData);
                invalidCount = invalidCount + sum(invalidIndices);
                
                if any(invalidIndices)
                    hasInvalidData = true;
                    validIndices = ~invalidIndices;
                    
                    % 有効なデータポイントが十分ある場合は線形補間を使用
                    if sum(validIndices) > 1
                        % 補間のための準備
                        validTimePoints = find(validIndices);
                        validValues = channelData(validIndices);
                        invalidTimePoints = find(invalidIndices);
                        
                        % 線形補間を適用
                        interpolatedValues = interp1(validTimePoints, validValues, invalidTimePoints, 'linear', 'extrap');
                        channelData(invalidIndices) = interpolatedValues;
                    else
                        % 有効データが不足している場合はチャンネルの平均値または0で置換
                        if sum(validIndices) == 1
                            % 1点のみ有効な場合はその値を使用
                            replacementValue = channelData(validIndices);
                        else
                            % 全て無効な場合は0を使用
                            replacementValue = 0;
                            fprintf('警告: 試行 %d, チャンネル %d の全データポイントが無効です。0で置換します。\n', ...
                                trialIndex, ch);
                        end
                        channelData(invalidIndices) = replacementValue;
                    end
                    
                    processedData(:, ch) = channelData;
                end
            end
            
            % 無効値があった場合に情報を表示
            if hasInvalidData
                fprintf('試行 %d: %d個の無効値を検出し補間処理しました (%.1f%%)\n', ...
                    trialIndex, invalidCount, (invalidCount/(timepoints*channels))*100);
            end
        end

        %% ハイブリッドモデル学習メソッド
        function [hybridModel, trainInfo] = trainHybridModel(obj, cnnTrainData, lstmTrainData, trainLabels, cnnValData, lstmValData, valLabels)
            try        
                fprintf('\n=== ハイブリッドモデル学習開始 ===\n');
                
                % --- CNNモデルの学習 ---
                fprintf('\n--- CNNモデル学習開始 ---\n');
                
                % CNNアーキテクチャの構築
                cnnArchitecture = obj.params.classifier.hybrid.architecture.cnn;
                cnnLayers = obj.buildCNNLayers(cnnTrainData, cnnArchitecture);
                
                % CNN学習オプションの設定
                cnnTrainOptions = obj.getCNNTrainingOptions(cnnValData, valLabels);
                
                % ラベルのカテゴリカル変換
                uniqueLabels = unique(trainLabels);
                trainLabels_cat = categorical(trainLabels, uniqueLabels);
                valLabels_cat = categorical(valLabels, uniqueLabels);
                
                % CNNモデルの学習
                [cnnModel, cnnTrainInfo] = trainNetwork(cnnTrainData, trainLabels_cat, cnnLayers, cnnTrainOptions);
                fprintf('CNNモデル学習完了\n');
                
                % --- LSTMモデルの学習 ---
                fprintf('\n--- LSTMモデル学習開始 ---\n');
                
                % LSTMアーキテクチャの構築
                lstmArchitecture = obj.params.classifier.hybrid.architecture.lstm;
                lstmLayers = obj.buildLSTMLayers(lstmTrainData, lstmArchitecture);
                
                % LSTM学習オプションの設定
                lstmTrainOptions = obj.getLSTMTrainingOptions(lstmValData, valLabels);
                
                % LSTMモデルの学習
                [lstmModel, lstmTrainInfo] = trainNetwork(lstmTrainData, trainLabels_cat, lstmLayers, lstmTrainOptions);
                fprintf('LSTMモデル学習完了\n');
                
                % --- 特徴抽出と統合モデルの学習 ---
                fprintf('\n--- 特徴統合と最終分類器の学習 ---\n');
                
                % CNNからの特徴抽出
                cnnFeatures = activations(cnnModel, cnnTrainData, 'fc_cnn', 'OutputAs', 'rows');
                
                % LSTMからの特徴抽出 - 動的にサイズを決定
                lstmFeatures = [];
                for i = 1:length(lstmTrainData)
                    % LSTM予測実行
                    lstmOut = predict(lstmModel, lstmTrainData{i}, 'MiniBatchSize', 1);
                    
                    % 最初のサンプルで配列を初期化
                    if i == 1
                        lstmFeatureSize = size(lstmOut, 2);
                        lstmFeatures = zeros(length(lstmTrainData), lstmFeatureSize);
                    end
                    
                    % 特徴の格納
                    lstmFeatures(i,:) = lstmOut;
                end
                
                % 特徴の統合
                combinedFeatures = [cnnFeatures, lstmFeatures];
                
                % AdaBoost分類器のパラメータ設定
                adaParams = obj.params.classifier.hybrid.adaBoost;
                
                % AdaBoost分類器の学習
                adaBoostModel = fitcensemble(combinedFeatures, trainLabels, ...
                    'Method', 'AdaBoostM1', ...
                    'NumLearningCycles', adaParams.numLearners, ...
                    'Learners', 'tree', ...
                    'LearnRate', adaParams.learnRate);
                
                fprintf('AdaBoost統合分類器の学習完了\n');
                
                % --- ここから新規追加部分: 検証データでのハイブリッドモデル全体の評価 ---
                fprintf('\n--- 検証データでのハイブリッドモデル評価 ---\n');
                
                % 検証データからの特徴抽出 (CNN)
                cnnValFeatures = activations(cnnModel, cnnValData, 'fc_cnn', 'OutputAs', 'rows');
                
                % 検証データからの特徴抽出 (LSTM)
                lstmValFeatures = [];
                for i = 1:length(lstmValData)
                    lstmOut = predict(lstmModel, lstmValData{i}, 'MiniBatchSize', 1);
                    
                    % 最初のサンプルで配列を初期化
                    if i == 1
                        lstmValFeatures = zeros(length(lstmValData), size(lstmOut, 2));
                    end
                    
                    % 特徴の格納
                    lstmValFeatures(i,:) = lstmOut;
                end
                
                % 特徴の統合 (検証データ)
                combinedValFeatures = [cnnValFeatures, lstmValFeatures];
                
                % 検証データでの予測
                [valPred, valScores] = predict(adaBoostModel, combinedValFeatures);
                
                % 検証精度の計算
                hybridValAccuracy = mean(valPred == valLabels) * 100;  % パーセントに変換
                fprintf('検証精度: %.2f%%\n', hybridValAccuracy);
                
                % クラスごとの評価指標 (検証データ)
                classes = unique(valLabels);
                classwise_val = struct();
                
                for i = 1:length(classes)
                    className = classes(i);
                    classIdx = (valLabels == className);
                    
                    % クラスごとの指標計算
                    TP = sum(valPred(classIdx) == className);
                    FP = sum(valPred == className) - TP;
                    FN = sum(classIdx) - TP;
                    
                    % ゼロ除算防止
                    precision = 0;
                    recall = 0;
                    f1score = 0;
                    
                    if (TP + FP) > 0
                        precision = TP / (TP + FP);
                    end
                    
                    if (TP + FN) > 0
                        recall = TP / (TP + FN);
                    end
                    
                    if (precision + recall) > 0
                        f1score = 2 * (precision * recall) / (precision + recall);
                    end
                    
                    % クラス指標の保存
                    classwise_val(i).precision = precision;
                    classwise_val(i).recall = recall;
                    classwise_val(i).f1score = f1score;
                end
                
                % 検証結果の詳細構造体
                hybridValMetrics = struct(...
                    'accuracy', hybridValAccuracy / 100, ...  % 比率に戻す
                    'prediction', valPred, ...
                    'score', valScores, ...
                    'classwise', classwise_val);
                
                % ハイブリッドモデル情報を構築
                hybridModel = struct(...
                    'netCNN', cnnModel, ...
                    'netLSTM', lstmModel, ...
                    'adaModel', adaBoostModel, ...
                    'lstmFeatureSize', lstmFeatureSize);
                
                % 学習履歴情報の構築
                trainInfo = struct(...
                    'cnnHistory', cnnTrainInfo, ...
                    'lstmHistory', lstmTrainInfo, ...
                    'hybridValAccuracy', hybridValAccuracy, ...
                    'hybridValMetrics', hybridValMetrics, ...
                    'FinalEpoch', length(cnnTrainInfo.TrainingLoss));
                
                fprintf('\n=== ハイブリッドモデル学習が完了しました ===\n');
                
            catch ME
                fprintf('\n=== ハイブリッドモデル学習中にエラーが発生 ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% CNNレイヤー構築メソッド
        function layers = buildCNNLayers(obj, data, architecture)
            % ハイブリッドモデルのCNN部分のアーキテクチャを構築
            
            fprintf('CNNアーキテクチャを構築中...\n');
            
            % 入力サイズの決定
            inputSize = size(data);
            if length(inputSize) >= 4
                layerInputSize = inputSize(1:3);  % [サンプル数, チャンネル数, 1]
            else
                error('CNNデータの形式が不正です');
            end
            
            % 入力層
            layers = [
                imageInputLayer(layerInputSize, 'Name', 'input', 'Normalization', 'none')
            ];
            
            % 畳み込み層の追加
            convFields = fieldnames(architecture.convLayers);
            for i = 1:length(convFields)
                convName = convFields{i};
                convParams = architecture.convLayers.(convName);
                
                % 畳み込み層
                layers = [layers
                    convolution2dLayer(...
                        convParams.size, convParams.filters, ...
                        'Stride', convParams.stride, ...
                        'Padding', convParams.padding, ...
                        'Name', convName)
                ];
                
                % バッチ正規化層（設定に応じて）
                if architecture.batchNorm
                    layers = [layers
                        batchNormalizationLayer('Name', ['bn_' convName])
                    ];
                end
                
                % 活性化関数
                layers = [layers
                    reluLayer('Name', ['relu_' convName])
                ];
                
                % プーリング層
                poolName = ['pool' num2str(i)];
                if isfield(architecture.poolLayers, poolName)
                    poolParams = architecture.poolLayers.(poolName);
                    layers = [layers
                        maxPooling2dLayer(...
                            poolParams.size, 'Stride', poolParams.stride, ...
                            'Name', poolName)
                    ];
                end
                
                % ドロップアウト層
                dropoutName = ['dropout' num2str(i)];
                if isfield(architecture.dropoutLayers, dropoutName)
                    dropoutRate = architecture.dropoutLayers.(dropoutName);
                    layers = [layers
                        dropoutLayer(dropoutRate, 'Name', dropoutName)
                    ];
                end
            end
            
            % 全結合層（特徴出力用）
            layers = [layers
                globalAveragePooling2dLayer('Name', 'gap')
                fullyConnectedLayer(architecture.fullyConnected, 'Name', 'fc_cnn')
                reluLayer('Name', 'relu_fc')
            ];
            
            % 出力層
            layers = [layers
                fullyConnectedLayer(obj.params.classifier.hybrid.architecture.numClasses, 'Name', 'fc_output')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];
            
            fprintf('CNNアーキテクチャ構築完了: %dレイヤー\n', length(layers));
        end

        %% LSTMレイヤー構築メソッド
        function layers = buildLSTMLayers(obj, data, architecture)
            % ハイブリッドモデルのLSTM部分のアーキテクチャを構築
            
            fprintf('LSTMアーキテクチャを構築中...\n');
            
            % 入力特徴量サイズの決定
            if iscell(data)
                % 最初のセルを使用
                sampleData = data{1};
                inputSize = size(sampleData, 1);  % チャンネル数
            else
                error('LSTMデータの形式が不正です');
            end
            
            % 入力層
            layers = [
                sequenceInputLayer(inputSize, ...
                    'Name', 'sequence_input', ...
                    'Normalization', architecture.sequenceInputLayer.normalization)
            ];
            
            % LSTM層の追加
            lstmFields = fieldnames(architecture.lstmLayers);
            for i = 1:length(lstmFields)
                lstmName = lstmFields{i};
                lstmParams = architecture.lstmLayers.(lstmName);
                
                % LSTM層
                layers = [layers
                    lstmLayer(lstmParams.numHiddenUnits, ...
                        'OutputMode', lstmParams.OutputMode, ...
                        'Name', lstmName)
                ];
                
                % バッチ正規化層（設定に応じて）
                if architecture.batchNorm
                    layers = [layers
                        batchNormalizationLayer('Name', ['bn_' lstmName])
                    ];
                end
                
                % ドロップアウト層
                dropoutName = ['dropout' num2str(i)];
                if isfield(architecture.dropoutLayers, dropoutName)
                    dropoutRate = architecture.dropoutLayers.(dropoutName);
                    layers = [layers
                        dropoutLayer(dropoutRate, 'Name', dropoutName)
                    ];
                end
            end
            
            % 全結合層（特徴出力用）
            if isfield(architecture, 'fullyConnected')
                layers = [layers
                    fullyConnectedLayer(architecture.fullyConnected, 'Name', 'fc_lstm')
                    reluLayer('Name', 'relu_fc')
                ];
            end
            
            % 出力層
            layers = [layers
                fullyConnectedLayer(obj.params.classifier.hybrid.architecture.numClasses, 'Name', 'fc_output')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'output')
            ];
            
            fprintf('LSTMアーキテクチャ構築完了: %dレイヤー\n', length(layers));
        end

        %% CNNトレーニングオプション設定メソッド
        function options = getCNNTrainingOptions(obj, valData, valLabels)
            % CNNのトレーニングオプションを設定
            
            % 実行環境の選択
            executionEnvironment = 'cpu';
            if obj.useGPU
                executionEnvironment = 'gpu';
            end
            
            % 検証データの準備
            valLabels_cat = categorical(valLabels, unique(valLabels));
            valDS = {valData, valLabels_cat};
            
            % トレーニングオプションの設定
            options = trainingOptions(obj.params.classifier.hybrid.training.optimizer.type, ...
                'InitialLearnRate', obj.params.classifier.hybrid.training.optimizer.learningRate, ...
                'MaxEpochs', obj.params.classifier.hybrid.training.maxEpochs, ...
                'MiniBatchSize', obj.params.classifier.hybrid.training.miniBatchSize, ...
                'Plots', 'none', ...
                'Shuffle', obj.params.classifier.hybrid.training.shuffle, ...
                'ExecutionEnvironment', executionEnvironment, ...
                'OutputNetwork', 'best-validation', ...
                'Verbose', true, ...
                'ValidationData', valDS, ...
                'ValidationFrequency', obj.params.classifier.hybrid.training.frequency, ...
                'ValidationPatience', obj.params.classifier.hybrid.training.patience, ...
                'GradientThreshold', 1);
        end

        %% LSTMトレーニングオプション設定メソッド
        function options = getLSTMTrainingOptions(obj, valData, valLabels)
            % LSTMのトレーニングオプションを設定
            
            % 実行環境の選択
            executionEnvironment = 'cpu';
            if obj.useGPU
                executionEnvironment = 'gpu';
            end
            
            % 検証データの準備
            valLabels_cat = categorical(valLabels, unique(valLabels));
            valDS = {valData, valLabels_cat};
            
            % トレーニングオプションの設定
            options = trainingOptions(obj.params.classifier.hybrid.training.optimizer.type, ...
                'InitialLearnRate', obj.params.classifier.hybrid.training.optimizer.learningRate, ...
                'MaxEpochs', obj.params.classifier.hybrid.training.maxEpochs, ...
                'MiniBatchSize', obj.params.classifier.hybrid.training.miniBatchSize, ...
                'Plots', 'none', ...
                'Shuffle', obj.params.classifier.hybrid.training.shuffle, ...
                'ExecutionEnvironment', executionEnvironment, ...
                'OutputNetwork', 'best-validation', ...
                'Verbose', true, ...
                'ValidationData', valDS, ...
                'ValidationFrequency', obj.params.classifier.hybrid.training.frequency, ...
                'ValidationPatience', obj.params.classifier.hybrid.training.patience, ...
                'GradientThreshold', obj.params.classifier.hybrid.training.optimizer.gradientThreshold);
        end

        %% モデル評価メソッド
        function metrics = evaluateModel(~, model, cnnTestData, lstmTestData, testLabels)
            % 学習済みハイブリッドモデルの性能を評価
            %
            % 入力:
            %   model - 学習済みハイブリッドモデル
            %   cnnTestData - CNN用テストデータ
            %   lstmTestData - LSTM用テストデータ
            %   testLabels - テストラベル
            %
            % 出力:
            %   metrics - 詳細な評価メトリクス
            
            fprintf('\n=== モデル評価を実行 ===\n');
            metrics = struct(...
                'accuracy', [], ...
                'score', [], ...
                'confusionMat', [], ...
                'classwise', [], ...
                'roc', [], ...
                'auc', [] ...
            );
        
            try                
                % CNNからの特徴抽出
                cnnFeatures = activations(model.netCNN, cnnTestData, 'fc_cnn', 'OutputAs', 'rows');
                
                % LSTMからの特徴抽出
                lstmFeatures = [];
                for i = 1:length(lstmTestData)
                    lstmOut = predict(model.netLSTM, lstmTestData{i}, 'MiniBatchSize', 1);
                    
                    % 最初のサンプルで配列を初期化
                    if i == 1
                        lstmFeatures = zeros(length(lstmTestData), size(lstmOut, 2));
                    end
                    
                    lstmFeatures(i,:) = lstmOut;
                end
                
                % 特徴の統合
                combinedFeatures = [cnnFeatures, lstmFeatures];
                
                % AdaBoostによる予測
                [pred, score] = predict(model.adaModel, combinedFeatures);
                metrics.score =score;
                
                % 基本的な指標の計算
                metrics.accuracy = mean(pred == testLabels);
                metrics.confusionMat = confusionmat(testLabels, pred);
                
                fprintf('テスト精度: %.2f%%\n', metrics.accuracy * 100);
                
                % クラスごとの性能評価
                classes = unique(testLabels);
                metrics.classwise = struct('precision', zeros(1,length(classes)), ...
                                        'recall', zeros(1,length(classes)), ...
                                        'f1score', zeros(1,length(classes)));
                
                fprintf('\nクラスごとの評価:\n');
                for i = 1:length(classes)
                    className = classes(i);
                    classIdx = (testLabels == className);
                    
                    % 各クラスの指標計算
                    TP = sum(pred(classIdx) == className);
                    FP = sum(pred == className) - TP;
                    FN = sum(classIdx) - TP;
                    
                    % 0による除算を回避
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
                        f1 = 2 * (precision * recall) / (precision + recall);
                    else
                        f1 = 0;
                    end
                    
                    metrics.classwise(i).precision = precision;
                    metrics.classwise(i).recall = recall;
                    metrics.classwise(i).f1score = f1;
                    
                    fprintf('  - クラス %d:\n', i);
                    fprintf('    - 精度 (Precision): %.2f%%\n', precision * 100);
                    fprintf('    - 再現率 (Recall): %.2f%%\n', recall * 100);
                    fprintf('    - F1スコア: %.2f\n', f1);
                end
                
                % 2クラス分類の場合のROC曲線とAUC
                if length(classes) == 2
                    [X, Y, T, AUC] = perfcurve(testLabels, score(:,2), classes(2));
                    metrics.roc = struct('X', X, 'Y', Y, 'T', T);
                    metrics.auc = AUC;
                    fprintf('\nAUC: %.3f\n', AUC);
                end
                
                % 混同行列の表示
                fprintf('\n混同行列:\n');
                disp(metrics.confusionMat);
                
            catch ME
                fprintf('モデル評価でエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                rethrow(ME);
            end
        end

        %% 交差検証メソッド（有効時のみ実行）
        function results = performCrossValidationIfEnabled(obj, data, labels)
            % 交差検証が有効な場合に実行
            
            results = struct('meanAccuracy', [], 'stdAccuracy', []);
            
            % 交差検証の実行     
            if obj.params.classifier.hybrid.training.validation.enable
                results = obj.performCrossValidation(data, labels);
                fprintf('交差検証平均精度: %.2f%% (±%.2f%%)\n', ...
                    results.meanAccuracy * 100, ...
                    results.stdAccuracy * 100);
            else
                fprintf('交差検証はスキップされました（設定で無効）\n');
            end
            
            return;
        end
        
        %% 交差検証メソッド
        function results = performCrossValidation(obj, data, labels)
            % k分割交差検証の実行
            
            try
                % k-fold cross validationのパラメータ取得
                k = obj.params.classifier.hybrid.training.validation.kfold;
                fprintf('\n=== %d分割交差検証開始 ===\n', k);
                
                % 結果初期化
                successCount = 0;
                allAccuracies = [];
                
                % 交差検証分割の作成
                cvp = cvpartition(length(labels), 'KFold', k);
                
                % 各フォールドの処理
                for i = 1:k
                    fprintf('\nフォールド %d/%d の処理を開始\n', i, k);
                    
                    try
                        % データの分割
                        trainIdx = cvp.training(i);
                        testIdx = cvp.test(i);
                        
                        % 学習・検証データの準備
                        foldTrainData = data(:,:,trainIdx);
                        foldTrainLabels = labels(trainIdx);
                        foldTestData = data(:,:,testIdx);
                        foldTestLabels = labels(testIdx);
                        
                        % 内部検証分割の作成
                        innerCVP = cvpartition(length(foldTrainLabels), 'Holdout', 0.2);
                        valIdx = innerCVP.test;
                        trainIdxInner = innerCVP.training;
                        
                        foldInnerTrainData = foldTrainData(:,:,trainIdxInner);
                        foldInnerTrainLabels = foldTrainLabels(trainIdxInner);
                        foldValData = foldTrainData(:,:,valIdx);
                        foldValLabels = foldTrainLabels(valIdx);
                        
                        % データ準備（CNN用）
                        cnnTrainData = obj.prepareDataForCNN(foldInnerTrainData);
                        cnnValData = obj.prepareDataForCNN(foldValData);
                        cnnTestData = obj.prepareDataForCNN(foldTestData);
                        
                        % データ準備（LSTM用）
                        lstmTrainData = obj.prepareDataForLSTM(foldInnerTrainData);
                        lstmValData = obj.prepareDataForLSTM(foldValData);
                        lstmTestData = obj.prepareDataForLSTM(foldTestData);
                        
                        % ハイブリッドモデルの学習
                        [foldModel, ~] = obj.trainHybridModel( ...
                            cnnTrainData, lstmTrainData, foldInnerTrainLabels,cnnValData, lstmValData, foldValLabels);
                        
                        % テストデータでの評価
                        metrics = obj.evaluateModel(foldModel, cnnTestData, lstmTestData, foldTestLabels);
                        
                        % 結果の保存
                        successCount = successCount + 1;
                        allAccuracies(successCount) = metrics.accuracy;
                        
                        fprintf('フォールド %d の精度: %.2f%%\n', i, metrics.accuracy * 100);
                        
                    catch ME
                        warning('フォールド %d でエラーが発生: %s', i, ME.message);
                        fprintf('エラー詳細:\n');
                        disp(getReport(ME, 'extended'));
                    end
                    
                    % GPUメモリの解放
                    if obj.useGPU
                        reset(gpuDevice);
                    end
                end
                
                % 結果の集計
                if successCount > 0
                    meanAcc = mean(allAccuracies);
                    stdAcc = std(allAccuracies);
                else
                    meanAcc = 0;
                    stdAcc = 0;
                end
                
                % 結果構造体の構築
                results = struct(...
                    'meanAccuracy', meanAcc, ...
                    'stdAccuracy', stdAcc, ...
                    'successCount', successCount, ...
                    'totalFolds', k);
                
                fprintf('\n交差検証結果:\n');
                fprintf('  - 平均精度: %.2f%% (±%.2f%%)\n', meanAcc * 100, stdAcc * 100);
                fprintf('  - 有効フォールド数: %d/%d\n', successCount, k);
                
            catch ME
                fprintf('\n=== 交差検証中にエラーが発生 ===\n');
                fprintf('エラーメッセージ: %s\n', ME.message);
                fprintf('エラースタック:\n');
                disp(getReport(ME, 'extended'));
                
                % 最小限の結果構造体を返す
                results = struct('meanAccuracy', 0, 'stdAccuracy', 0);
            end
        end

        %% 過学習検証メソッド
        function [isOverfit, metrics] = validateOverfitting(obj, trainInfo, testMetrics)
            % トレーニング結果とテスト結果から過学習を分析
            %
            % 入力:
            %   trainInfo - トレーニング情報（学習曲線を含む）
            %   testMetrics - テストデータでの評価結果
            %
            % 出力:
            %   isOverfit - 過学習の有無（論理値）
            %   metrics - 詳細な過学習メトリクス
            
            fprintf('\n=== 過学習検証の実行 ===\n');
            
            try
                % 初期化
                isOverfit = false;
                metrics = struct();
                
                % --- ハイブリッドモデル全体の検証精度を使用 ---
                if isfield(trainInfo, 'hybridValAccuracy')
                    % 統合ハイブリッドモデルの検証精度を使用
                    meanValAcc = trainInfo.hybridValAccuracy;
                    fprintf('検証精度: %.2f%%\n', meanValAcc);
                    
                    % テスト精度との比較
                    testAcc = testMetrics.accuracy * 100;  % パーセントに変換
                    fprintf('テスト精度: %.2f%%\n', testAcc);
                    
                    % 精度ギャップの計算
                    perfGap = abs(meanValAcc - testAcc);
                    fprintf('精度ギャップ: %.2f%%\n', perfGap);
                    
                    % 過学習の重大度判定
                    if perfGap > 15
                        severity = 'severe';      % 15%以上の差は重度の過学習
                    elseif perfGap > 10
                        severity = 'moderate';    % 10%以上の差は中程度の過学習
                    elseif perfGap > 5
                        severity = 'mild';        % 5%以上の差は軽度の過学習
                    else
                        severity = 'none';        % 5%未満の差は許容範囲
                    end
                else
                    % 統合モデルの検証精度が利用できない場合はCNNとLSTMのデータを試用
                    warning('検証精度がありません。CNNとLSTMのデータを使用します。');
                    
                    % CNN検証精度の取得
                    cnnValAcc = [];
                    if isfield(trainInfo, 'cnnHistory') && isfield(trainInfo.cnnHistory, 'ValidationAccuracy')
                        cnnValAcc = trainInfo.cnnHistory.ValidationAccuracy;
                        % NaN値をフィルタリング
                        cnnValAcc = cnnValAcc(~isnan(cnnValAcc));
                    end
                    
                    % LSTM検証精度の取得
                    lstmValAcc = [];
                    if isfield(trainInfo, 'lstmHistory') && isfield(trainInfo.lstmHistory, 'ValidationAccuracy')
                        lstmValAcc = trainInfo.lstmHistory.ValidationAccuracy;
                        % NaN値をフィルタリング
                        lstmValAcc = lstmValAcc(~isnan(lstmValAcc));
                    end
                    
                    % 利用可能なデータを使用
                    if ~isempty(cnnValAcc) && ~isempty(lstmValAcc)
                        % 両方あれば平均を使用
                        meanCnnValAcc = mean(cnnValAcc(max(1, end-5):end));
                        meanLstmValAcc = mean(lstmValAcc(max(1, end-5):end));
                        meanValAcc = (meanCnnValAcc + meanLstmValAcc) / 2;
                        fprintf('平均検証精度 (CNN+LSTM): %.2f%%\n', meanValAcc);
                    elseif ~isempty(cnnValAcc)
                        meanValAcc = mean(cnnValAcc(max(1, end-5):end));
                        fprintf('CNN検証精度: %.2f%%\n', meanValAcc);
                    elseif ~isempty(lstmValAcc)
                        meanValAcc = mean(lstmValAcc(max(1, end-5):end));
                        fprintf('LSTM検証精度: %.2f%%\n', meanValAcc);
                    else
                        % どのデータもなければテスト精度をそのまま使用
                        warning('利用可能な検証精度データがありません。過学習判定をスキップします。');
                        meanValAcc = testMetrics.accuracy * 100;
                        fprintf('参考値として使用するテスト精度: %.2f%%\n', meanValAcc);
                        
                        % 過学習判定不可
                        severity = 'unknown';
                        perfGap = 0;
                    end
                    
                    % テスト精度との比較（上記で判定できなかった場合のみ）
                    if ~exist('severity', 'var')
                        testAcc = testMetrics.accuracy * 100;
                        fprintf('テスト精度: %.2f%%\n', testAcc);
                        
                        perfGap = abs(meanValAcc - testAcc);
                        fprintf('精度ギャップ: %.2f%%\n', perfGap);
                        
                        % 過学習の重大度判定
                        if perfGap > 15
                            severity = 'severe';
                        elseif perfGap > 10
                            severity = 'moderate';
                        elseif perfGap > 5
                            severity = 'mild';
                        else
                            severity = 'none';
                        end
                    end
                end
                
                % 分類バイアスの検出
                isCompletelyBiased = obj.detectClassificationBias(testMetrics);
                
                % 学習曲線の分析（CNN）
                trainTrend = struct('mean_change', 0, 'volatility', 0);
                valTrend = struct('mean_change', 0, 'volatility', 0);
                
                if isfield(trainInfo, 'cnnHistory') && isfield(trainInfo.cnnHistory, 'TrainingAccuracy')
                    trainAcc = trainInfo.cnnHistory.TrainingAccuracy;
                    trainAcc = trainAcc(~isnan(trainAcc)); % NaN値を除去
                    if ~isempty(trainAcc)
                        trainTrend = obj.analyzeLearningCurve(trainAcc);
                    end
                end
                
                % 検証精度トレンド分析
                if isfield(trainInfo, 'cnnHistory') && isfield(trainInfo.cnnHistory, 'ValidationAccuracy')
                    valAcc = trainInfo.cnnHistory.ValidationAccuracy;
                    valAcc = valAcc(~isnan(valAcc)); % NaN値を除去
                    if ~isempty(valAcc)
                        valTrend = obj.analyzeLearningCurve(valAcc);
                    end
                end
                
                % 学習が進行中かどうかチェック
                isLearningProgressing = (trainTrend.mean_change > 0.001) || (valTrend.mean_change > 0.001);
                
                % 最適エポックの検出
                if isfield(trainInfo, 'cnnHistory') && isfield(trainInfo.cnnHistory, 'ValidationAccuracy')
                    valAcc = trainInfo.cnnHistory.ValidationAccuracy;
                    valAcc = valAcc(~isnan(valAcc)); % NaN値を除去
                    if ~isempty(valAcc)
                        [optimalEpoch, totalEpochs] = obj.findOptimalEpoch(valAcc);
                    else
                        optimalEpoch = 0;
                        totalEpochs = 0;
                    end
                else
                    optimalEpoch = 0;
                    totalEpochs = 0;
                end
                
                % 結果の格納
                metrics = struct(...
                    'performanceGap', perfGap, ...
                    'meanValAcc', meanValAcc, ...
                    'testAcc', testMetrics.accuracy * 100, ...
                    'isCompletelyBiased', isCompletelyBiased, ...
                    'isLearningProgressing', isLearningProgressing, ...
                    'severity', severity, ...
                    'trainingTrend', trainTrend, ...
                    'validationTrend', valTrend, ...
                    'optimalEpoch', optimalEpoch, ...
                    'totalEpochs', totalEpochs);
                
                % 過学習の最終判定（バイアスも加味）
                isOverfit = strcmp(severity, 'severe') || ...
                            strcmp(severity, 'moderate') || ...
                            isCompletelyBiased;
                
                fprintf('過学習判定: %s (重大度: %s)\n', mat2str(isOverfit), severity);
                
            catch ME
                fprintf('過学習検証でエラーが発生: %s\n', ME.message);
                fprintf('エラー詳細:\n');
                disp(getReport(ME, 'extended'));
                
                % エラー時のフォールバック値
                metrics = struct(...
                    'performanceGap', 0, ...
                    'isCompletelyBiased', false, ...
                    'isLearningProgressing', false, ...
                    'severity', 'error', ...
                    'optimalEpoch', 0, ...
                    'totalEpochs', 0);
                isOverfit = false;
            end
        end

        %% 学習曲線分析メソッド
        function trend = analyzeLearningCurve(~, acc)
            % 学習曲線の変化率と変動性を分析
            
            if length(acc) < 2
                trend = struct('mean_change', NaN, 'volatility', NaN);
                return;
            end
            
            % 各エポック間の変化量を計算
            diffValues = diff(acc);
            
            % 平均変化量とボラティリティ（標準偏差）を計算
            trend = struct(...
                'mean_change', mean(diffValues), ...
                'volatility', std(diffValues), ...
                'increasing_ratio', sum(diffValues > 0) / length(diffValues));
        end

        %% 最適エポック検出メソッド
        function [optimalEpoch, totalEpochs] = findOptimalEpoch(~, valAcc)
            % 検証精度が最大となるエポックを検出
            
            totalEpochs = length(valAcc);
            [~, optimalEpoch] = max(valAcc);
            
            % 最適エポックが最後のエポックの場合、学習が不十分の可能性
            if optimalEpoch == totalEpochs
                fprintf('警告: 最適エポックが最後のエポックです。より長い学習が必要かもしれません。\n');
            end
        end

        %% 分類バイアス検出メソッド
        function isCompletelyBiased = detectClassificationBias(~, testMetrics)
            % 混同行列から分類バイアスを検出
            
            isCompletelyBiased = false;
    
            if isfield(testMetrics, 'confusionMat')
                cm = testMetrics.confusionMat;
                
                % 各実際のクラス（行）のサンプル数を確認
                rowSums = sum(cm, 2);
                missingActual = any(rowSums == 0);
                
                % 各予測クラス（列）の予測件数を確認
                colSums = sum(cm, 1);
                missingPredicted = any(colSums == 0);
                
                % すべての予測が1クラスに集中しているかを検出
                predictedClassCount = sum(colSums > 0);
                
                % いずれかが true ならば、全く現れないクラスがあると判断
                isCompletelyBiased = missingActual || missingPredicted || predictedClassCount <= 1;
                
                if isCompletelyBiased
                    fprintf('\n警告: 分類に完全な偏りが検出されました\n');
                    fprintf('  - 分類された実際のクラス数: %d / %d\n', sum(rowSums > 0), size(cm, 1));
                    fprintf('  - 予測されたクラス数: %d / %d\n', predictedClassCount, size(cm, 2));
                end
            end
        end

        %% 性能メトリクス更新メソッド
        function updatePerformanceMetrics(obj, testMetrics)
            % 評価結果から性能メトリクスを更新
            
            obj.performance = testMetrics;
        end

        %% GPUメモリ解放メソッド
        function resetGPUMemory(obj)
            % GPUメモリのリセットと解放
            
            if obj.useGPU
                try
                    % GPUデバイスのリセット
                    reset(gpuDevice);
                    fprintf('GPUメモリをリセットしました\n');
                catch ME
                    fprintf('GPUメモリのリセットに失敗: %s', ME.message);
                end
            end
        end

        %% 結果表示メソッド
        function displayResults(obj)
            % 総合的な結果サマリーの表示
            
            try
                fprintf('\n=== ハイブリッド分類結果サマリー ===\n');
                
                if ~isempty(obj.performance)
                    % 精度情報
                    fprintf('全体精度: %.2f%%\n', obj.performance.accuracy * 100);

                    if isfield(obj.performance, 'auc')
                        fprintf('AUC: %.3f\n', obj.performance.auc);
                    end

                    % 混同行列
                    if ~isempty(obj.performance.confusionMat)
                        fprintf('\n混同行列:\n');
                        disp(obj.performance.confusionMat);
                    end

                    % クラスごとの性能
                    if isfield(obj.performance, 'classwise') && ~isempty(obj.performance.classwise)
                        fprintf('\nクラスごとの性能:\n');
                        for i = 1:length(obj.performance.classwise)
                            fprintf('クラス %d:\n', i);
                            fprintf('  - 精度: %.2f%%\n', ...
                                obj.performance.classwise(i).precision * 100);
                            fprintf('  - 再現率: %.2f%%\n', ...
                                obj.performance.classwise(i).recall * 100);
                            fprintf('  - F1スコア: %.2f\n', ...
                                obj.performance.classwise(i).f1score);
                        end
                    end
                end

                % 過学習分析結果の表示
                if ~isempty(obj.overfitMetrics) && isstruct(obj.overfitMetrics)
                    fprintf('\n過学習分析:\n');
                    fprintf('  - 性能ギャップ: %.2f%%\n', obj.overfitMetrics.performanceGap);
                    fprintf('  - 重大度: %s\n', obj.overfitMetrics.severity);
                    
                    if isfield(obj.overfitMetrics, 'trainingTrend')
                        fprintf('\n学習カーブ分析:\n');
                        fprintf('  - 学習の進行: %s\n', ...
                            mat2str(obj.overfitMetrics.isLearningProgressing));
                        
                        if isfield(obj.overfitMetrics, 'validationTrend')
                            trend = obj.overfitMetrics.validationTrend;
                            fprintf('  - 検証平均変化率: %.4f\n', trend.mean_change);
                            fprintf('  - 検証変動性: %.4f\n', trend.volatility);
                        end
                    end
                    
                    if isfield(obj.overfitMetrics, 'optimalEpoch')
                        fprintf('  - 最適エポック: %d/%d (%.2f%%)\n', ...
                            obj.overfitMetrics.optimalEpoch, ...
                            obj.overfitMetrics.totalEpochs, ...
                            (obj.overfitMetrics.optimalEpoch / ...
                             max(obj.overfitMetrics.totalEpochs, 1)) * 100);
                    end
                end

            catch ME
                fprintf('結果表示でエラーが発生: %s', ME.message);
            end
        end

        %% 結果構造体構築メソッド
        function results = buildResultsStruct(obj, hybridModel, metrics, trainInfo, ...
            crossValidation, normParams)
            % 結果構造体の構築
            results = struct(...
                'model', hybridModel, ...
                'performance', metrics, ...
                'crossValidation', crossValidation, ...
                'trainInfo', trainInfo, ...
                'overfitting', obj.overfitMetrics, ...
                'normParams', normParams ...
            );
        end
    end
end