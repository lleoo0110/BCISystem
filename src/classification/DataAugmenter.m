classdef DataAugmenter < handle
    % DataAugmenter - EEGデータ拡張のためのクラス
    %
    % このクラスはEEGデータの拡張（データオーグメンテーション）を行い、
    % 訓練データセットを増強するための様々な手法を提供します。
    %
    % 実装されている拡張手法:
    %   - ノイズ付加 (ガウシアン, ピンクノイズ等)
    %   - スケーリング（振幅調整）
    %   - 時間シフト
    %   - 振幅反転（ミラーリング）
    %   - チャンネル交換
    %   - 周波数選択的雑音 (周波数領域での増強)
    %   - 時間反転
    %   - バンドパワー調整
    %   - フィルタリング
    
    properties (Access = private)
        params              % システム設定パラメータ
        channelMap         % チャンネル名とインデックスのマッピング
        rng                % 乱数生成器
        augmentationMethods % 利用可能な拡張手法のマップ
    end
    
    methods (Access = public)
        function obj = DataAugmenter(params)
            % DataAugmenterのコンストラクタ
            %
            % 入力:
            %   params - 設定パラメータ構造体
            
            obj.params = params;
            % 再現性のために固定シードの乱数生成器を使用
            obj.rng = RandStream('mlfg6331_64');
            % チャンネルマップの初期化
            obj.initializeChannelMap();
            % 拡張手法のマップを初期化
            obj.initializeAugmentationMethods();
        end
        
        function [augData, augLabels, info] = augmentData(obj, data, labels)
            % EEGデータとラベルを拡張する
            %
            % 入力:
            %   data - 原データ [チャンネル x サンプル x エポック]
            %   labels - ラベルデータ [エポック x 1]
            %
            % 出力:
            %   augData - 拡張後のデータ
            %   augLabels - 拡張後のラベル
            %   info - 拡張処理に関する情報構造体
            
            try
                % データサイズの取得
                [channels, samples, epochs] = size(data);
                
                % 情報構造体の初期化
                info = struct(...
                    'originalSize', size(data), ...
                    'augmentationApplied', false);
                
                % 拡張処理が無効の場合は元データをそのまま返す
                if ~obj.params.classifier.augmentation.enable
                    augData = data;
                    augLabels = labels;
                    return;
                end
                
                % 拡張比率の取得とターゲットエポック数の計算
                augRatio = obj.params.classifier.augmentation.augmentationRatio;
                targetTotalEpochs = round(epochs * (1 + augRatio));
                
                % ユニークなラベルとクラスごとのサンプル数を取得
                uniqueLabels = unique(labels);
                labelCounts = zeros(length(uniqueLabels), 1);
                for i = 1:length(uniqueLabels)
                    labelCounts(i) = sum(labels == uniqueLabels(i));
                end
                
                % --- クラス不均衡への対応 ---
                
                % 1. クラス均衡化のための拡張数を計算
                maxCount = max(labelCounts);
                balanceAugPerClass = maxCount - labelCounts;
                
                % 2. 残りの拡張サンプル数を計算
                remainingAugs = targetTotalEpochs - (epochs + sum(balanceAugPerClass));
                
                % 3. 残りの拡張をクラスの元のサンプル数比率に基づいて分配
                if remainingAugs > 0
                    classRatios = labelCounts / sum(labelCounts);
                    additionalAugs = round(remainingAugs * classRatios);
                    augCountsPerClass = balanceAugPerClass + additionalAugs;
                else
                    augCountsPerClass = balanceAugPerClass;
                end
                
                % 総拡張サンプル数
                totalAugmentations = sum(augCountsPerClass);
                
                % --- 出力配列の初期化 ---
                
                augData = zeros(channels, samples, epochs + totalAugmentations);
                augData(:,:,1:epochs) = data;  % 元データをコピー
                
                augLabels = zeros(epochs + totalAugmentations, 1);
                augLabels(1:epochs) = labels;  % 元ラベルをコピー
                
                % 拡張情報の更新
                info.augmentationApplied = true;
                info.methodsUsed = cell(totalAugmentations, 1);
                info.sourceSamples = zeros(totalAugmentations, 1);
                
                % --- 拡張データの生成 ---
                
                currentIdx = epochs + 1;
                for i = 1:length(uniqueLabels)
                    classLabel = uniqueLabels(i);
                    classEpochs = find(labels == classLabel);
                    numAugNeeded = augCountsPerClass(i);
                    
                    % このクラスの各拡張サンプルを生成
                    for j = 1:numAugNeeded
                        % 元データからランダムに1エポックを選択
                        baseIdx = classEpochs(randi(obj.rng, length(classEpochs)));
                        info.sourceSamples(currentIdx-epochs) = baseIdx;
                        
                        % 拡張手法の選択
                        methods = obj.selectAugmentationMethods();
                        info.methodsUsed{currentIdx-epochs} = methods;
                        
                        % 選択した拡張手法を適用
                        augmentedEpoch = data(:,:,baseIdx);
                        for methodIdx = 1:length(methods)
                            % 各手法を適用した結果で更新
                            augmentedEpoch = obj.applyAugmentation(augmentedEpoch, methods{methodIdx});
                        end
                        
                        % ノイズ除去や異常値対応のための最終処理
                        augmentedEpoch = obj.postprocessAugmentation(augmentedEpoch);
                        
                        % 拡張データとラベルの保存
                        augData(:,:,currentIdx) = augmentedEpoch;
                        augLabels(currentIdx) = classLabel;
                        currentIdx = currentIdx + 1;
                    end
                end
                
                % --- 拡張結果の情報を設定 ---
                
                info.finalSize = size(augData);
                info.totalAugmentations = totalAugmentations;
                info.augmentationRatio = totalAugmentations / epochs;
                info.classBalance = struct(...
                    'original', labelCounts, ...
                    'added', augCountsPerClass, ...
                    'final', labelCounts + augCountsPerClass);
                info.augmentationDetails = obj.calculateMethodStats(info.methodsUsed);
                
                % クラス間バランス評価
                info.balanceMetrics = obj.evaluateClassBalance(augLabels);
                
            catch ME
                % エラーハンドリング
                errorMsg = sprintf('データ拡張処理でエラーが発生しました: %s', ME.message);
                fprintf('%s\n', errorMsg);
                fprintf('エラー詳細:\n%s\n', getReport(ME, 'extended'));
            end
        end
    end
    
    methods (Access = private)
        function initializeChannelMap(obj)
            % チャンネル名とインデックスのマッピングを初期化
            try
                channels = obj.params.device.channels;
                if ~iscell(channels)
                    warning('チャンネル設定が正しくありません。デフォルト値を使用します。');
                    channels = arrayfun(@(x) sprintf('Ch%d', x), 1:32, 'UniformOutput', false);
                end
                obj.channelMap = containers.Map(channels, 1:length(channels));
            catch ME
                fprintf('チャンネルマップの初期化に失敗しました: %s', ME.message);
                % デフォルトマップ作成（空のマップを避ける）
                obj.channelMap = containers.Map({'Ch1'}, {1});
            end
        end
        
        function initializeAugmentationMethods(obj)
            % 拡張手法のマップを初期化
            obj.augmentationMethods = containers.Map();
            
            % 基本拡張手法
            obj.augmentationMethods('noise') = @(data, config) obj.addNoise(data, config);
            obj.augmentationMethods('scaling') = @(data, config) obj.applyScaling(data, config);
            obj.augmentationMethods('timeshift') = @(data, config) obj.applyTimeshift(data, config);
            obj.augmentationMethods('mirror') = @(data, ~) obj.applyMirror(data);
            obj.augmentationMethods('channelSwap') = @(data, config) obj.swapChannels(data, config);
            
            % 拡張手法の追加（設定に存在する場合のみ）
            if isfield(obj.params.classifier.augmentation.methods, 'frequencyNoise')
                obj.augmentationMethods('frequencyNoise') = @(data, config) obj.addFrequencyNoise(data, config);
            end
            
            if isfield(obj.params.classifier.augmentation.methods, 'timeReverse')
                obj.augmentationMethods('timeReverse') = @(data, ~) obj.applyTimeReverse(data);
            end
            
            if isfield(obj.params.classifier.augmentation.methods, 'bandpowerAdjust')
                obj.augmentationMethods('bandpowerAdjust') = @(data, config) obj.adjustBandpower(data, config);
            end
            
            if isfield(obj.params.classifier.augmentation.methods, 'filtering')
                obj.augmentationMethods('filtering') = @(data, config) obj.applyFiltering(data, config);
            end
        end
        
        function methods = selectAugmentationMethods(obj)
            % 適用する拡張手法の組み合わせを選択
            %
            % 出力:
            %   methods - 選択された拡張手法のセル配列
            
            augParams = obj.params.classifier.augmentation;
            availableMethods = fieldnames(augParams.methods);
            methods = {};
            
            % 必ず1つ以上の拡張手法を適用するために、少なくとも1回はループを回す
            while isempty(methods)
                % 設定で有効化されている手法のみを考慮
                enabledMethods = availableMethods(cellfun(@(m) ...
                    isfield(augParams.methods.(m), 'enable') && ...
                    augParams.methods.(m).enable, availableMethods));
                
                if ~isempty(enabledMethods)
                    % 適用する手法数をランダム選択（上限あり）
                    maxMethods = min(augParams.combinationLimit, length(enabledMethods));
                    numMethods = randi(obj.rng, [1, maxMethods]);
                    
                    % 手法をランダムに並べ替え
                    shuffledMethods = enabledMethods(randperm(obj.rng, length(enabledMethods)));
                    
                    % 確率に基づいて各手法を適用するか決定
                    for i = 1:min(numMethods, length(shuffledMethods))
                        method = shuffledMethods{i};
                        if obj.shouldApplyMethod(method)
                            methods{end+1} = method;
                        end
                    end
                end
            end
        end
        
        function shouldApply = shouldApplyMethod(obj, method)
            % 特定の拡張手法を適用するかを確率に基づいて決定
            %
            % 入力:
            %   method - 拡張手法名
            %
            % 出力:
            %   shouldApply - 適用するかどうかの論理値
            
            try
                methodParams = obj.params.classifier.augmentation.methods.(method);
                if isfield(methodParams, 'probability')
                    shouldApply = methodParams.enable && rand(obj.rng) < methodParams.probability;
                else
                    shouldApply = methodParams.enable;
                end
            catch
                % エラー発生時はデフォルトで適用しない
                shouldApply = false;
            end
        end
        
        function augmented = applyAugmentation(obj, data, method)
            % 指定された拡張手法を適用
            %
            % 入力:
            %   data - 入力データ [チャンネル x サンプル]
            %   method - 適用する拡張手法名
            %
            % 出力:
            %   augmented - 拡張されたデータ
            
            try
                if obj.augmentationMethods.isKey(method)
                    % 該当する拡張手法が登録されている場合
                    methodFunc = obj.augmentationMethods(method);
                    if isfield(obj.params.classifier.augmentation.methods, method)
                        methodParams = obj.params.classifier.augmentation.methods.(method);
                        augmented = methodFunc(data, methodParams);
                    else
                        % パラメータがない場合は空のstructを渡す
                        augmented = methodFunc(data, struct());
                    end
                else
                    % 該当する拡張手法が登録されていない場合
                    warning('未登録の拡張手法: %s', method);
                    augmented = data;
                end
            catch ME
                warning('拡張手法の適用に失敗しました (%s): %s', method, ME.message);
                augmented = data;
            end
        end
        
        function augmented = postprocessAugmentation(obj, data)
            % 拡張後のデータに対する後処理
            %
            % 入力:
            %   data - 拡張されたデータ
            %
            % 出力:
            %   augmented - 後処理されたデータ
            
            augmented = data;
            
            % 異常値（NaN, Inf）の除去
            if any(isnan(augmented(:))) || any(isinf(augmented(:)))
                warning('拡張データに異常値が含まれています。置換します。');
                augmented(isnan(augmented)) = 0;
                augmented(isinf(augmented)) = 0;
            end
            
            % 振幅が極端に大きい場合はクリッピング
            maxAllowedAmplitude = 1000;  % 設定に応じて調整可能
            if max(abs(augmented(:))) > maxAllowedAmplitude
                warning('極端に大きい振幅値を検出しました。クリッピングを適用します。');
                augmented(augmented > maxAllowedAmplitude) = maxAllowedAmplitude;
                augmented(augmented < -maxAllowedAmplitude) = -maxAllowedAmplitude;
            end
            
            return;
        end
        
        %% --- 基本拡張手法 ---
        function augmented = addNoise(obj, data, config)
            % データにノイズを追加する拡張手法
            %
            % 入力:
            %   data - 入力データ [チャンネル x サンプル]
            %   config - ノイズ設定パラメータ
            %
            % 出力:
            %   augmented - ノイズが追加されたデータ [チャンネル x サンプル]
            
            % 設定が無効の場合は元データを返す
            if ~config.enable
                augmented = data;
                return;
            end
            
            try
                % データサイズの取得
                [channels, samples] = size(data);
                
                % ノイズタイプの選択
                if isfield(config, 'types') && ~isempty(config.types)
                    noiseTypes = config.types;
                    if ~iscell(noiseTypes)
                        noiseTypes = {noiseTypes}; % 単一値の場合はセル配列に変換
                    end
                    selectedType = noiseTypes{randi(obj.rng, length(noiseTypes))};
                else
                    selectedType = 'gaussian'; % デフォルトタイプ
                end
                
                % ノイズ強度の設定
                if isfield(config, 'variance') && ~isempty(config.variance)
                    noiseVariance = config.variance;
                else
                    noiseVariance = 0.05; % デフォルト値
                end
                
                % 元信号パワーに対するノイズ強度の調整
                signalPower = mean(var(data, 0, 2));
                scaledVariance = noiseVariance * signalPower;
                
                % ノイズ生成
                switch selectedType
                    case 'gaussian'
                        % ガウシアンノイズ（正規分布）
                        noise = obj.rng.randn(channels, samples) * sqrt(scaledVariance);
                        
                    case 'pink'
                        % ピンクノイズ（1/f特性）
                        noise = zeros(channels, samples);
                        for ch = 1:channels
                            % 時間領域で単一チャンネルのノイズを生成
                            noise(ch, :) = obj.generatePinkNoise(samples);
                        end
                        % ノイズの強度を調整
                        noise = noise .* sqrt(scaledVariance) / std(noise(:));
                        
                    case 'brown'
                        % ブラウンノイズ（1/f^2特性）- 累積和で生成
                        noise = zeros(channels, samples);
                        for ch = 1:channels
                            % ガウシアンノイズを生成
                            temp = obj.rng.randn(1, samples);
                            % 累積和を取る
                            noise(ch, :) = cumsum(temp);
                            % スケーリング
                            noise(ch, :) = noise(ch, :) - mean(noise(ch, :));
                            noise(ch, :) = noise(ch, :) / std(noise(ch, :));
                        end
                        % ノイズの強度を調整
                        noise = noise .* sqrt(scaledVariance);
                        
                    case 'white'
                        % ホワイトノイズ（フラットスペクトル）
                        noise = obj.rng.randn(channels, samples) * sqrt(scaledVariance);
                        
                    otherwise
                        % 不明なタイプの場合はガウシアンノイズを使用
                        warning('不明なノイズタイプ: %s - ガウシアンノイズを使用します', selectedType);
                        noise = obj.rng.randn(channels, samples) * sqrt(scaledVariance);
                end
                
                % ノイズと元データの形状確認（デバッグ用）
                if size(noise, 1) ~= channels || size(noise, 2) ~= samples
                    error('生成されたノイズ(%dx%d)と元データ(%dx%d)のサイズが一致しません', ...
                        size(noise, 1), size(noise, 2), channels, samples);
                end
                
                % ノイズ適用（加算）
                augmented = data + noise;
                
            catch ME
                % エラー処理
                fprintf('ノイズ追加処理でエラーが発生しました: %s', ME.message);
                % 元データをそのまま返す
                augmented = data;
            end
        end
        
        function pinkNoise = generatePinkNoise(obj, length)
            % ピンクノイズ（1/f特性）の生成 - 修正版
            %
            % 入力:
            %   length - 生成するノイズの長さ
            %
            % 出力:
            %   pinkNoise - 生成されたピンクノイズ [1 x length]
            
            try
                % FFTサイズを2のべき乗に設定（効率化）
                fftSize = 2^nextpow2(length);
                
                % ホワイトノイズの生成
                whiteNoise = obj.rng.randn(1, fftSize);
                
                % FFT変換
                fftNoise = fft(whiteNoise);
                
                % 周波数インデックス（直流成分を除く）
                halfN = fftSize / 2;
                freqIndices = 2:halfN+1;  % インデックス2からN/2+1まで
                
                % 1/f^0.5 スペクトルの作成
                freqScale = (1:halfN) / halfN;
                filter = 1 ./ sqrt(freqScale + eps);  % 0除算回避のためeps追加
                
                % 周波数領域でフィルタ適用（直流成分は変更せず）
                fftNoise(freqIndices) = fftNoise(freqIndices) .* filter;
                
                % 共役対称性の維持（実数信号のため必要）
                symIndices = fftSize:-1:halfN+2;  % 対称インデックス
                fftNoise(symIndices) = conj(fftNoise(2:halfN));
                
                % 逆FFT変換で時間領域に戻す
                noise = real(ifft(fftNoise));
                
                % 正規化
                noise = noise - mean(noise);
                stdNoise = std(noise);
                if stdNoise > 0
                    noise = noise / stdNoise;
                else
                    % 標準偏差が0の場合はホワイトノイズに戻す
                    noise = obj.rng.randn(1, fftSize);
                    noise = noise - mean(noise);
                    noise = noise / std(noise);
                end
                
                % 必要な長さを抽出
                pinkNoise = noise(1:length);
                
            catch ME
                % エラー発生時はランダムノイズを代わりに生成
                error('ピンクノイズ生成でエラー発生: %s - ガウシアンノイズを代替使用', ME.message);
            end
        end
        
        function augmented = applyScaling(obj, data, config)
            % データのスケーリング（振幅調整）
            %
            % 入力:
            %   data - 入力データ
            %   config - スケーリング設定
            %
            % 出力:
            %   augmented - スケーリングされたデータ
            
            if ~config.enable
                augmented = data;
                return;
            end
            
            % 指定された範囲からランダムにスケール係数を選択
            scale = config.range(1) + (config.range(2) - config.range(1)) * rand(obj.rng);
            
            % チャンネルごとに異なるスケールを適用するオプション
            if isfield(config, 'perChannel') && config.perChannel
                % チャンネルごとに異なるスケールを適用
                numChannels = size(data, 1);
                channelScales = scale + (rand(obj.rng, numChannels, 1) - 0.5) * 0.2;
                augmented = data .* channelScales;
            else
                % 全チャンネル同じスケールを適用
                augmented = data * scale;
            end
        end
        
        function augmented = applyTimeshift(obj, data, config)
            % 時間軸方向のシフト処理
            %
            % 入力:
            %   data - 入力データ
            %   config - タイムシフト設定
            %
            % 出力:
            %   augmented - 時間シフトされたデータ
            
            if ~config.enable
                augmented = data;
                return;
            end
            
            % 最大シフト量をサンプル数に変換してランダム選択
            maxShiftSamples = round(config.maxShift * size(data, 2));
            shiftAmount = randi(obj.rng, [-maxShiftSamples, maxShiftSamples]);
            
            % チャンネルごとに異なるシフト量を適用するオプション
            if isfield(config, 'perChannel') && config.perChannel
                augmented = zeros(size(data));
                for ch = 1:size(data, 1)
                    channelShift = shiftAmount + randi(obj.rng, [-5, 5]);
                    augmented(ch, :) = circshift(data(ch, :), channelShift);
                end
            else
                % 全チャンネル同じシフト量を適用
                augmented = circshift(data, [0, shiftAmount]);
            end
        end
        
        function augmented = applyMirror(~, data)
            % 振幅の反転（ミラーリング）
            %
            % 入力:
            %   data - 入力データ
            %
            % 出力:
            %   augmented - 振幅反転されたデータ
            
            augmented = -data;
        end
        
        function augmented = swapChannels(obj, data, config)
            % 指定されたチャンネルペアの交換
            %
            % 入力:
            %   data - 入力データ
            %   config - チャンネル交換設定
            %
            % 出力:
            %   augmented - チャンネル交換されたデータ
            
            if ~config.enable || isempty(config.pairs)
                augmented = data;
                return;
            end
            
            augmented = data;
            
            % 交換ペアをランダム選択
            pairs = config.pairs;
            selectedPair = pairs{randi(obj.rng, length(pairs))};
            
            % チャンネルのインデックスを取得
            ch1Idx = obj.getChannelIndex(selectedPair{1});
            ch2Idx = obj.getChannelIndex(selectedPair{2});
            
            if ch1Idx > 0 && ch2Idx > 0 && ch1Idx <= size(data, 1) && ch2Idx <= size(data, 1)
                % チャンネルデータの交換
                temp = augmented(ch1Idx, :);
                augmented(ch1Idx, :) = augmented(ch2Idx, :);
                augmented(ch2Idx, :) = temp;
            end
        end
        
        %% --- 追加拡張手法 ---
        
        function augmented = addFrequencyNoise(obj, data, config)
            % 周波数領域でノイズを追加
            %
            % 入力:
            %   data - 入力データ
            %   config - 周波数ノイズ設定
            %
            % 出力:
            %   augmented - 周波数ノイズが追加されたデータ
            
            if ~config.enable
                augmented = data;
                return;
            end
            
            [numChannels, numSamples] = size(data);
            augmented = zeros(size(data));
            
            % 各チャンネルに対して処理
            for ch = 1:numChannels
                % データをFFT変換
                fftData = fft(data(ch, :));
                
                % 周波数帯域をランダムに選択
                if isfield(config, 'freqBands') && ~isempty(config.freqBands)
                    bands = config.freqBands;
                    bandIdx = randi(obj.rng, length(bands));
                    freqRange = bands{bandIdx};
                else
                    % デフォルト周波数範囲
                    freqRange = [8, 13];  % アルファ帯域
                end
                
                % サンプリングレートから周波数インデックスに変換
                fs = obj.params.device.sampleRate;
                freqIdxRange = round(freqRange / fs * numSamples) + 1;
                freqIdxRange = min(max(freqIdxRange, 1), floor(numSamples/2));
                
                % 選択した周波数帯域にノイズを追加
                noiseLevel = config.level * mean(abs(fftData(freqIdxRange(1):freqIdxRange(2))));
                noise = (rand(obj.rng, 1, freqIdxRange(2)-freqIdxRange(1)+1) - 0.5) * noiseLevel;
                
                % ノイズを適用（対称性を保持）
                fftData(freqIdxRange(1):freqIdxRange(2)) = fftData(freqIdxRange(1):freqIdxRange(2)) + noise;
                fftData(numSamples-freqIdxRange(2)+2:numSamples-freqIdxRange(1)+2) = ...
                    conj(flipud(fftData(freqIdxRange(1):freqIdxRange(2))));
                
                % 逆変換
                augmented(ch, :) = real(ifft(fftData));
            end
        end
        
        function augmented = applyTimeReverse(~, data)
            % 時間軸の反転
            %
            % 入力:
            %   data - 入力データ
            %
            % 出力:
            %   augmented - 時間反転されたデータ
            
            % 各チャンネルのデータを時間軸で反転
            augmented = fliplr(data);
        end
        
        function augmented = adjustBandpower(obj, data, config)
            % 特定の周波数帯域のパワーを調整する拡張手法
            %
            % 入力:
            %   data - 入力データ [チャンネル x サンプル]
            %   config - バンドパワー調整設定
            %
            % 出力:
            %   augmented - バンドパワーが調整されたデータ
            
            % 設定が無効の場合は元データを返す
            if ~config.enable
                augmented = data;
                return;
            end
            
            try
                % サイズ情報の取得
                [numChannels, numSamples] = size(data);
                
                % サンプリングレートの取得
                if isfield(obj.params, 'device') && isfield(obj.params.device, 'sampleRate')
                    fs = obj.params.device.sampleRate;
                else
                    fs = 256; % デフォルト値
                    warning('サンプリングレートが見つかりません。デフォルト値(256Hz)を使用します。');
                end
                
                % 出力配列の初期化
                augmented = zeros(size(data));
                
                % 各チャンネルに対してバンドパワー調整を実行
                for ch = 1:numChannels
                    % FFT変換のサイズを決定（2のべき乗）
                    fftSize = 2^nextpow2(numSamples);
                    
                    % FFT変換
                    fftData = fft(data(ch, :), fftSize);
                    
                    % 調整する周波数帯域を選択
                    if isfield(config, 'bands') && ~isempty(config.bands)
                        bandNames = fieldnames(config.bands);
                        if ~isempty(bandNames)
                            % ランダムに帯域を選択
                            selectedBand = bandNames{randi(obj.rng, length(bandNames))};
                            freqRange = config.bands.(selectedBand);
                        else
                            % デフォルト帯域（アルファ帯域）
                            freqRange = [8, 13];
                        end
                    else
                        % デフォルト帯域（アルファ帯域）
                        freqRange = [8, 13];
                    end
                    
                    % 周波数インデックスの計算
                    freqIdx = round(freqRange / fs * fftSize) + 1;
                    % 境界チェック
                    freqIdx = min(max(freqIdx, 1), floor(fftSize/2) + 1);
                    
                    % スケール係数の決定（増幅または減衰）
                    if isfield(config, 'scaleRange') && length(config.scaleRange) == 2
                        scaleRange = config.scaleRange;
                    else
                        scaleRange = [0.5, 2.0]; % デフォルト: 0.5倍～2倍
                    end
                    
                    % ランダムなスケール係数を生成
                    scale = scaleRange(1) + (scaleRange(2) - scaleRange(1)) * rand(obj.rng);
                    
                    % 対象周波数帯域のインデックス範囲
                    idxRange = freqIdx(1):freqIdx(2);
                    
                    % 選択した周波数帯域のパワーを調整
                    if ~isempty(idxRange) && idxRange(end) <= length(fftData)
                        fftData(idxRange) = fftData(idxRange) * scale;
                        
                        % 共役対称性を保持（実信号の要件）
                        if fftSize - freqIdx(2) + 2 <= fftSize && freqIdx(1) <= freqIdx(2)
                            symIdx = fftSize - idxRange + 2;
                            symIdx = symIdx(symIdx <= fftSize & symIdx >= 1);
                            if ~isempty(symIdx) && max(symIdx) <= length(fftData)
                                fftData(symIdx) = conj(fftData(idxRange(end:-1:1)));
                            end
                        end
                    end
                    
                    % 逆FFT変換して実部を取得
                    timeData = real(ifft(fftData, fftSize));
                    
                    % 元のデータ長に合わせて切り出し
                    augmented(ch, :) = timeData(1:numSamples);
                end
                
            catch ME
                % エラー処理
                fprintf('バンドパワー調整処理でエラーが発生しました: %s', ME.message);
                % 元データをそのまま返す
                augmented = data;
            end
        end
        
        function augmented = applyFiltering(obj, data, config)
            % フィルタ処理による拡張
            %
            % 入力:
            %   data - 入力データ
            %   config - フィルタリング設定
            %
            % 出力:
            %   augmented - フィルタリングされたデータ
            
            if ~config.enable
                augmented = data;
                return;
            end
            
            % サンプリングレートの取得
            fs = obj.params.device.sampleRate;
            
            % フィルタのタイプをランダムに選択
            if isfield(config, 'types')
                filterTypes = config.types;
                selectedType = filterTypes{randi(obj.rng, length(filterTypes))};
            else
                % デフォルトはバンドパスフィルタ
                selectedType = 'bandpass';
            end
            
            % フィルタ次数の設定
            if isfield(config, 'order')
                order = config.order;
            else
                order = 4; % デフォルト次数
            end
            
            % フィルタ周波数の設定
            if isfield(config, 'freqRange')
                freqRange = config.freqRange;
            else
                % デフォルト周波数範囲はランダム選択
                allRanges = {[0.5, 4], [4, 8], [8, 13], [13, 30], [30, 50]};
                freqRange = allRanges{randi(obj.rng, length(allRanges))};
            end
            
            % 正規化周波数に変換
            Wn = freqRange / (fs/2);
            
            % フィルタ設計
            switch selectedType
                case 'bandpass'
                    [b, a] = butter(order, Wn, 'bandpass');
                case 'lowpass'
                    [b, a] = butter(order, Wn(2), 'low');
                case 'highpass'
                    [b, a] = butter(order, Wn(1), 'high');
                case 'bandstop'
                    [b, a] = butter(order, Wn, 'stop');
                otherwise
                    [b, a] = butter(order, Wn, 'bandpass');
            end
            
            % フィルタ適用
            augmented = zeros(size(data));
            for ch = 1:size(data, 1)
                augmented(ch, :) = filtfilt(b, a, data(ch, :));
            end
        end
        
        %% --- ヘルパーメソッド ---
        function idx = getChannelIndex(obj, channelName)
            % チャンネル名からインデックスを取得
            %
            % 入力:
            %   channelName - チャンネル名
            %
            % 出力:
            %   idx - チャンネルインデックス
            
            if obj.channelMap.isKey(channelName)
                idx = obj.channelMap(channelName);
            else
                % チャンネル名が見つからない場合は警告とデフォルト値
                warning('チャンネル名 "%s" が見つかりません', channelName);
                idx = 0;
            end
        end
        
        function stats = calculateMethodStats(~, methodsUsed)
            % 使用された拡張手法の統計情報を計算
            %
            % 入力:
            %   methodsUsed - 使用された拡張手法のリスト
            %
            % 出力:
            %   stats - 拡張手法の使用統計
            
            stats = struct();
            
            % 基本拡張手法リスト
            allMethods = {'noise', 'scaling', 'timeshift', 'mirror', 'channelSwap', ...
                'frequencyNoise', 'timeReverse', 'bandpowerAdjust', 'filtering'};
            
            % 各手法の使用回数と割合を計算
            for i = 1:length(allMethods)
                method = allMethods{i};
                
                % 方法をカウント
                count = 0;
                for j = 1:length(methodsUsed)
                    if any(strcmp(methodsUsed{j}, method))
                        count = count + 1;
                    end
                end
                
                % 統計情報を保存
                stats.(method) = struct(...
                    'count', count, ...
                    'percentage', count / length(methodsUsed) * 100);
            end
            
            % 組み合わせの情報も追加
            combinationCounts = zeros(1, 5);
            for i = 1:length(methodsUsed)
                numMethods = length(methodsUsed{i});
                if numMethods <= 5
                    combinationCounts(numMethods) = combinationCounts(numMethods) + 1;
                end
            end
            
            stats.combinations = struct(...
                'counts', combinationCounts, ...
                'average', mean(cellfun(@length, methodsUsed)));
        end
        
        function metrics = evaluateClassBalance(~, labels)
            % クラスバランスの評価指標を計算
            %
            % 入力:
            %   labels - クラスラベル
            %
            % 出力:
            %   metrics - クラスバランス評価指標
            
            uniqueLabels = unique(labels);
            counts = histcounts(labels, [uniqueLabels; max(uniqueLabels)+1] - 0.5);
            
            % クラス分布のエントロピー（均一性の指標）
            probs = counts / sum(counts);
            entropy = -sum(probs .* log2(probs + eps));
            maxEntropy = log2(length(uniqueLabels));
            
            % インバランス比（最大/最小）
            imbalanceRatio = max(counts) / min(counts);
            
            % バランス性指標（1に近いほどバランスが良い）
            balanceIndex = entropy / maxEntropy;
            
            % 結果をまとめる
            metrics = struct(...
                'counts', counts, ...
                'entropy', entropy, ...
                'maxPossibleEntropy', maxEntropy, ...
                'balanceIndex', balanceIndex, ...
                'imbalanceRatio', imbalanceRatio);
        end
    end
end