classdef PowerFeatureExtractor < handle
    properties (Access = private)
        params          % パラメータ設定
        powerFunction   % パワー値計算関数
        freqBands      % 周波数帯域情報
        usedParams     % 使用したパラメータ
        leftChannels   % 左前頭葉のチャンネル番号
        rightChannels  % 右前頭葉のチャンネル番号
        faaThreshold   % FAA判定の閾値
        baselinePower  % 各周波数帯のベースラインパワー
        isBaselineSet  % ベースライン設定フラグ
        abRatioThreshold
        emotionThreshold
        
        % 感情分類用プロパティ
        emotionLabels     % 感情状態ラベル
        neutralLabel      % 中性状態ラベル
        normalizeMethod   % 正規化方法
        scalingFactor     % スケーリング係数
    end
    
    methods (Access = public)
        function obj = PowerFeatureExtractor(params)
            % コンストラクタ
            obj.params = params;
            obj.initializeFreqBands();
            
            % 感情分類用パラメータの初期化
            if isfield(params.feature, 'emotion') && params.feature.emotion.enable
                % チャンネル設定
                obj.leftChannels = params.feature.emotion.channels.left;
                obj.rightChannels = params.feature.emotion.channels.right;

                % 閾値設定
                obj.abRatioThreshold = params.feature.emotion.thresholds.abRatio;
                obj.emotionThreshold = params.feature.emotion.thresholds.centerRegion;
                obj.faaThreshold = params.feature.emotion.thresholds.faa;

                % ラベル設定 - セル配列の入れ子構造を解消
                if iscell(params.feature.emotion.labels.states) && ...
                   iscell(params.feature.emotion.labels.states{1})
                    obj.emotionLabels = params.feature.emotion.labels.states{1};
                else
                    obj.emotionLabels = params.feature.emotion.labels.states;
                end
                obj.neutralLabel = params.feature.emotion.labels.neutral;

                % 座標変換設定
                obj.normalizeMethod = params.feature.emotion.coordinates.normalizeMethod;
                obj.scalingFactor = params.feature.emotion.coordinates.scaling;
            else
                % デフォルト値の設定
                obj.leftChannels = [1, 3];
                obj.rightChannels = [14, 12];
                obj.abRatioThreshold = 1.0;
                obj.emotionThreshold = 0.3;
                obj.faaThreshold = 0.5;
                obj.emotionLabels = {'興奮', '喜び', '快適', 'リラックス', ...
                                    '眠気', '憂鬱', '不快', '緊張', '安静'};
                obj.neutralLabel = '安静';
                obj.normalizeMethod = 'tanh';
                obj.scalingFactor = 1.0;
            end
            
            % ERD関連の初期化
            obj.baselinePower = struct();
            obj.isBaselineSet = false;
        end
        
        function features = extractFeatures(obj, data)
            try
                % データサイズの取得
                [nChannels, ~, nTrials] = size(data);
                nBands = length(obj.freqBands);
                
                % 特徴量格納用配列の初期化
                features = zeros(nTrials, nChannels * nBands);
                
                % 各試行に対してパワー値を計算
                for trial = 1:nTrials
                    % bandPowersの事前割り当て
                    bandPowers = zeros(nChannels, nBands);
                    
                    for band = 1:nBands
                        % 周波数帯域の取得
                        bandName = obj.freqBands{band};
                        freqRange = obj.params.feature.power.bands.(bandName);
                        
                        % パワー値の計算
                        bandPowers(:, band) = obj.calculatePower(data(:,:,trial), freqRange);
                    end
                    
                    % 特徴量の格納
                    features(trial, :) = bandPowers(:)';
                end
                
                % 標準化処理
                if obj.params.feature.power.normalize
                    features = obj.normalizeFeatures(features);
                end
                
            catch ME
                error('Feature extraction failed: %s', ME.message);
            end
        end
        
        function [faaResults] = calculateFAA(obj, data)
            try
                % データ形式に応じて処理を分岐
                if iscell(data)
                    numEpochs = length(data);
                    faaResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        % 左右それぞれのパワー値計算
                        leftPower = obj.calculatePower(data{epoch}(obj.leftChannels,:), [8 13]);
                        rightPower = obj.calculatePower(data{epoch}(obj.rightChannels,:), [8 13]);
                        
                        % FAA値の計算
                        [faaValue, arousalState] = obj.computeFAAValue(leftPower, rightPower);
                        faaResults{epoch} = struct(...
                            'faa', faaValue, ...
                            'arousal', arousalState, ...
                            'leftPower', mean(leftPower), ...
                            'rightPower', mean(rightPower));
                    end
                    
                else  % 3次元配列の場合
                    numEpochs = size(data, 3);
                    faaResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        leftPower = obj.calculatePower(data(obj.leftChannels,:,epoch), [8 13]);
                        rightPower = obj.calculatePower(data(obj.rightChannels,:,epoch), [8 13]);
                        
                        [faaValue, arousalState] = obj.computeFAAValue(leftPower, rightPower);
                        faaResults{epoch} = struct(...
                            'faa', faaValue, ...
                            'arousal', arousalState, ...
                            'leftPower', mean(leftPower), ...
                            'rightPower', mean(rightPower));
                    end
                end
                
            catch ME
                fprintf('Error in calculateFAA: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        function [emotionResults] = classifyEmotion(obj, data)
            try
                if iscell(data)
                    numEpochs = length(data);
                    emotionResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        % 各エポックの感情状態を分類
                        [emotionState, coordinates, emotionCoords] = obj.computeEmotionState(data{epoch});
                        
                        emotionResults{epoch} = struct(...
                            'state', emotionState, ...
                            'coordinates', coordinates, ...
                            'emotionCoords', emotionCoords);
                    end
                    
                else  % 3次元配列の場合
                    numEpochs = size(data, 3);
                    emotionResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        % 各エポックの感情状態を分類
                        [emotionState, coordinates, emotionCoords] = obj.computeEmotionState(data(:,:,epoch));
                        
                        emotionResults{epoch} = struct(...
                            'state', emotionState, ...
                            'coordinates', coordinates, ...
                            'emotionCoords', emotionCoords);
                    end
                end
                
            catch ME
                fprintf('Error in classifyEmotion: %s\n', ME.message);
                fprintf('Stack trace:\n');
                disp(ME.stack);
                rethrow(ME);
            end
        end
        
        function [abRatio, arousalState] = calculateABRatio(obj, data)
            try
                % 前頭葉のチャンネルのみを使用（FAA計算と同じチャンネル）
                frontalData = data([obj.leftChannels, obj.rightChannels],:);

                % アルファ波(8-13Hz)とベータ波(13-30Hz)のパワーを計算
                alphaPower = obj.calculatePower(frontalData, [8 13]);
                betaPower = obj.calculatePower(frontalData, [13 30]);

                % チャンネルごとのα/β比を計算
                abRatio = mean(alphaPower) / mean(betaPower);

                % 覚醒状態の判定
                if abRatio < obj.abRatioThreshold
                    arousalState = 'high';
                else
                    arousalState = 'low';
                end

            catch ME
                fprintf('Error in calculateABRatio: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        function labels = getEmotionLabels(obj)
            labels = obj.emotionLabels;
        end
        
        function power = calculatePower(obj, signal, freqBand)
            fs = obj.params.device.sampleRate;
            method = obj.params.feature.power.method;
            
            switch lower(method)
                case 'fft'
                    % FFT解析のパラメータ設定
                    windowType = obj.params.feature.power.fft.windowType;
                    nfft = obj.params.feature.power.fft.nfft;
                    
                    [nChannels, nSamples] = size(signal);
                    power = zeros(nChannels, 1);
                    window = obj.getWindow(windowType, nSamples);
                    
                    % 周波数軸の作成
                    freq = (0:nfft-1)*fs/nfft;
                    idx_nyq = ceil(nfft/2);
                    freq = freq(1:idx_nyq);
                    idx = freq >= freqBand(1) & freq <= freqBand(2);
                    
                    for ch = 1:nChannels
                        windowed_signal = signal(ch,:) .* window';
                        fftdata = fft(windowed_signal, nfft);
                        pow_fftdata = abs(fftdata).^2/nfft;
                        singlePow = [pow_fftdata(1), 2*pow_fftdata(2:idx_nyq)];
                        power(ch) = sum(singlePow(idx));
                    end
                    
                case 'welch'
                    % Welch法のパラメータ設定
                    windowType = obj.params.feature.power.welch.windowType;
                    windowLength = obj.params.feature.power.welch.windowLength;
                    overlap = obj.params.feature.power.welch.overlap;
                    nfft = obj.params.feature.power.welch.nfft;
                    
                    window = obj.getWindow(windowType, windowLength);
                    noverlap = round(windowLength * overlap);
                    
                    [nChannels, ~] = size(signal);
                    power = zeros(nChannels, 1);
                    
                    for ch = 1:nChannels
                        [pxx, f] = pwelch(signal(ch,:), window, noverlap, nfft, fs);
                        idx = f >= freqBand(1) & f <= freqBand(2);
                        df = f(2) - f(1);
                        power(ch) = sum(pxx(idx)) * df;
                    end
                    
                case 'filter'
                    % フィルタリング法のパラメータ設定
                    filterType = obj.params.feature.power.filter.type;
                    filterOrder = obj.params.feature.power.filter.order;
                    
                    Wn = [freqBand(1)/(fs/2), freqBand(2)/(fs/2)];
                    if any(Wn <= 0) || any(Wn >= 1)
                        error('Frequency band must be within (0, fs/2) Hz');
                    end
                    
                    % フィルタの設計
                    switch lower(filterType)
                        case 'butter'
                            [b, a] = butter(filterOrder, Wn, 'bandpass');
                        case 'cheby1'
                            [b, a] = cheby1(filterOrder, 0.5, Wn, 'bandpass');
                        case 'ellip'
                            [b, a] = ellip(filterOrder, 0.5, 40, Wn, 'bandpass');
                        otherwise
                            error('Unknown filter type');
                    end
                    
                    % フィルタリングとパワー計算
                    [nChannels, ~] = size(signal);
                    power = zeros(nChannels, 1);
                    
                    for ch = 1:nChannels
                        filtered = filtfilt(b, a, signal(ch,:));
                        power(ch) = mean(filtered.^2);
                    end
                    
                case 'wavelet'
                    % Wavelet解析のパラメータ設定
                    waveletType = obj.params.feature.power.wavelet.type;
                    scaleNum = obj.params.feature.power.wavelet.scaleNum;
                    
                    [nChannels, ~] = size(signal);
                    power = zeros(nChannels, 1);
                    
                    for ch = 1:nChannels
                        % 中心周波数の取得
                        fc = centfrq(waveletType);
                        
                        % スケールの計算
                        scales = fs * fc ./ linspace(freqBand(1), freqBand(2), scaleNum);
                        
                        % 連続ウェーブレット変換の計算
                        coefs = cwt(signal(ch,:), scales, waveletType);
                        
                        % 周波数の計算
                        freq = scal2frq(scales, waveletType, 1/fs);
                        
                        % パワー計算
                        idx = freq >= freqBand(1) & freq <= freqBand(2);
                        df = mean(diff(freq(idx)));
                        power(ch) = sum(mean(abs(coefs(idx,:)).^2)) * df;
                    end
                    
                case 'hilbert'
                    % Hilbert変換のパラメータ設定
                    filterOrder = obj.params.feature.power.hilbert.filterOrder;
                    
                    [nChannels, ~] = size(signal);
                    power = zeros(nChannels, 1);
                    
                    % バンドパスフィルタの設計
                    Wn = freqBand / (fs/2);
                    [b, a] = butter(filterOrder, Wn, 'bandpass');
                    
                    for ch = 1:nChannels
                        % フィルタリング
                        filtered = filtfilt(b, a, signal(ch,:));
                        
                        % Hilbert変換とパワー計算
                        analytic = hilbert(filtered);
                        envelope = abs(analytic);
                        power(ch) = mean(envelope.^2);
                    end
                    
                otherwise
                    error('Unknown method: %s', method);
            end
        end
        
        function [powerResults] = calculatePowerForAllBands(obj, data)
            try
                bandNames = obj.params.feature.power.bands.names;
                if iscell(bandNames{1})
                    bandNames = bandNames{1};
                end
                
                if iscell(data)
                    numEpochs = length(data);
                    powerResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        bandPowers = struct();
                        for i = 1:length(bandNames)
                            bandName = bandNames{i};
                            freqRange = obj.params.feature.power.bands.(bandName);
                            bandPower = obj.calculatePower(data{epoch}, freqRange);
                            bandPowers.(bandName) = bandPower;
                        end
                        powerResults{epoch} = bandPowers;
                    end
                    
                else
                    numEpochs = size(data, 3);
                    powerResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        bandPowers = struct();
                        for i = 1:length(bandNames)
                            bandName = bandNames{i};
                            freqRange = obj.params.feature.power.bands.(bandName);
                            bandPower = obj.calculatePower(data(:,:,epoch), freqRange);
                            bandPowers.(bandName) = bandPower;
                        end
                        powerResults{epoch} = bandPowers;
                    end
                end
                
            catch ME
                fprintf('Error in calculatePowerForAllBands: %s\n', ME.message);
                rethrow(ME);
            end
        end
        
        function calculateBaseline(obj, baselineData)
            % 安静時データからベースラインパワーを計算
            try
                obj.baselinePower = struct();
                
                % 各周波数帯のベースラインパワーを計算
                for i = 1:length(obj.freqBands)
                    bandName = obj.freqBands{i};
                    freqRange = obj.params.feature.power.bands.(bandName);
                    obj.baselinePower.(bandName) = obj.calculatePower(baselineData, freqRange);
                end
                
                % ベースライン設定フラグを更新
                obj.isBaselineSet = true;
                
                % デバッグ情報の出力
                fprintf('Baseline power calculated for bands: ');
                fprintf('%s ', obj.freqBands{:});
                fprintf('\n');
                
                % 各帯域のベースラインパワー値を表示
                for i = 1:length(obj.freqBands)
                    bandName = obj.freqBands{i};
                    fprintf('%s band baseline power: %.4f\n', ...
                        bandName, mean(obj.baselinePower.(bandName)));
                end
                
            catch ME
                obj.isBaselineSet = false;
                error('Failed to calculate baseline power: %s', ME.message);
            end
        end
        
        function [erdResults] = calculateERD(obj, data)
            try
                % ベースラインが未設定の場合はエラー
                if ~obj.isBaselineSet
                    error('Baseline power not set. Call updateBaseline first.');
                end
                
                if iscell(data)
                    numEpochs = length(data);
                    erdResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        % 各周波数帯でのERD計算
                        [erdValues, erdPercent] = obj.computeERDValues(data{epoch});
                        erdResults{epoch} = struct(...
                            'values', erdValues, ...
                            'percent', erdPercent);
                    end
                    
                else
                    numEpochs = size(data, 3);
                    erdResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        [erdValues, erdPercent] = obj.computeERDValues(data(:,:,epoch));
                        erdResults{epoch} = struct(...
                            'values', erdValues, ...
                            'percent', erdPercent);
                    end
                end
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
                rethrow(ME);
            end
        end
        
        function [pxx, f] = calculateSpectrum(obj, data)
            try
                % Welch法によるスペクトル計算
                fs = obj.params.device.sampleRate;
                windowLength = obj.params.feature.power.welch.windowLength;
                overlap = obj.params.feature.power.welch.overlap;
                nfft = obj.params.feature.power.welch.nfft;
                windowType = obj.params.feature.power.welch.windowType;

                % 窓関数の取得
                window = obj.getWindow(windowType, windowLength);
                noverlap = round(windowLength * overlap);

                % チャンネル1のデータでスペクトル計算
                [pxx, f] = pwelch(data(1,:), window, noverlap, nfft, fs);

            catch ME
                warning(ME.identifier, '%s', ME.message);
                pxx = [];
                f = [];
            end
        end
        
        function [ersp, times, freqs] = calculateERSP(obj, data)
            try
                % ERSP計算用パラメータ
                fs = obj.params.device.sampleRate;
                windowSize = obj.params.gui.display.visualization.ersp.numFreqs;
                freqRange = obj.params.gui.display.visualization.ersp.freqRange;

                % 周波数軸の作成
                freqs = linspace(freqRange(1), freqRange(2), windowSize);

                % 短時間フーリエ変換の実行（timesを出力として使用）
                [S, freqs, times] = spectrogram(data(1,:), ...
                    hamming(windowSize), ...
                    round(windowSize*0.75), ...
                    freqs, ...
                    fs, ...
                    'yaxis');

                % パワーの計算とdB変換
                ersp = 10*log10(abs(S).^2);

            catch ME
                warning(ME.identifier, '%s', ME.message);
                ersp = [];
                times = [];
                freqs = [];
            end
        end
        
        function window = getWindow(~, windowType, windowLength)
            switch lower(windowType)
                case 'rectangular'
                    window = ones(windowLength, 1);
                case 'hamming'
                    window = hamming(windowLength);
                case 'hann'
                    window = hann(windowLength);
                case 'blackman'
                    window = blackman(windowLength);
                case 'kaiser'
                    window = kaiser(windowLength, 5);
                case 'gaussian'
                    window = gausswin(windowLength);
                otherwise
                    error('Unknown window type: %s', windowType);
            end
        end
        
        function params = getUsedParams(obj)
            % 使用したパラメータの取得
            params = obj.usedParams;
        end
        
        function setFAAThreshold(obj, threshold)
            % FAA閾値の設定
            obj.faaThreshold = threshold;
        end
    end
    
    methods (Access = private)
        function initializeFreqBands(obj)
            % 周波数帯域の初期化
            if isfield(obj.params.feature.power.bands, 'names')
                % パラメータで指定された周波数帯域を使用
                bandNames = obj.params.feature.power.bands.names;
                if ischar(bandNames)
                    obj.freqBands = {bandNames};
                elseif iscell(bandNames)
                    if iscell(bandNames{1})  % 入れ子のセル配列の場合
                        obj.freqBands = bandNames{1};
                    else
                        obj.freqBands = bandNames;
                    end
                else
                    error('Invalid frequency band specification');
                end
                
                % 指定された帯域が定義されているか確認
                for i = 1:length(obj.freqBands)
                    bandName = obj.freqBands{i};
                    if ~isfield(obj.params.feature.power.bands, bandName)
                        error('Frequency band %s is not defined', bandName);
                    end
                    % 周波数範囲の妥当性確認
                    freqRange = obj.params.feature.power.bands.(bandName);
                    if ~isnumeric(freqRange) || length(freqRange) ~= 2
                        error('Invalid frequency range for band %s', bandName);
                    end
                end
            else
                % デフォルトの周波数帯域を使用
                obj.freqBands = {'alpha', 'beta'};
            end
            
            % デバッグ情報の出力
            fprintf('Initialized frequency bands: ');
            fprintf('%s ', obj.freqBands{:});
            fprintf('\n');
        end
        
        function features = normalizeFeatures(~, features)
            % 特徴量の標準化
            features = (features - mean(features)) ./ std(features);
        end
        
        function [faaValue, arousalState] = computeFAAValue(obj, leftPower, rightPower)
            % FAA値計算のヘルパーメソッド
            leftMean = mean(leftPower);
            rightMean = mean(rightPower);
            
            faaValue = log(rightMean) - log(leftMean);
            
            if faaValue > obj.faaThreshold
                arousalState = 'aroused';
            else
                arousalState = 'non-aroused';
            end
        end
        
        function [erdValues, erdPercent] = computeERDValues(obj, epochData)
            % ERD計算のヘルパーメソッド
            erdValues = struct();
            erdPercent = struct();
            
            for i = 1:length(obj.freqBands)
                bandName = obj.freqBands{i};
                freqRange = obj.params.feature.power.bands.(bandName);
                
                % 現在のパワー値を計算
                currentPower = obj.calculatePower(epochData, freqRange);
                basePower = obj.baselinePower.(bandName);
                
                % ERDを計算
                erdValue = basePower - currentPower;
                erdPct = (erdValue ./ basePower) * 100;
                
                erdValues.(bandName) = erdValue;
                erdPercent.(bandName) = erdPct;
            end
        end
        
        function [emotionState, coordinates, emotionCoords] = computeEmotionState(obj, epochData)
            % FAA値とα/β比の計算
            [faaValue, ~] = obj.computeFAAValue(...
                obj.calculatePower(epochData(obj.leftChannels,:), [8 13]), ...
                obj.calculatePower(epochData(obj.rightChannels,:), [8 13]));
            
            % 前頭葉のチャンネルのデータを取得
            frontalData = epochData([obj.leftChannels, obj.rightChannels],:);
            
            % アルファ波とベータ波のパワーを計算
            alphaPower = obj.calculatePower(frontalData, [8 13]);
            betaPower = obj.calculatePower(frontalData, [13 30]);
            
            % α/β比の計算
            abRatio = mean(alphaPower) / mean(betaPower);
            
            % 感情座標の計算
            valence = tanh(faaValue * obj.scalingFactor);
            arousal = -tanh((abRatio - 1) * obj.scalingFactor);
            
            % 極座標への変換
            radius = sqrt(valence^2 + arousal^2);
            angle = atan2(arousal, valence);
            
            % 感情状態の分類
            if radius < obj.emotionThreshold
                emotionState = obj.neutralLabel;
            else
                angles = linspace(-pi, pi, length(obj.emotionLabels));
                angleIdx = find(angle >= angles, 1, 'last');
                if isempty(angleIdx) || angleIdx > length(obj.emotionLabels)
                    angleIdx = 1;
                end
                emotionState = obj.emotionLabels{angleIdx};
            end
            
            % 返却値の設定
            coordinates = struct('valence', valence, ...
                'arousal', arousal, ...
                'radius', radius, ...
                'angle', angle);
            
            % 4次元感情座標の取得
            emotionCoords = obj.getEmotionCoordinates(emotionState);
        end
        
        function coords = getEmotionCoordinates(obj, emotionState)
            % 感情状態を4次元座標に変換するメソッド
            % 座標は [快活性, 快不活性, 不快不活性, 不快活性] を表す
            emotionMap = containers.Map(...
                obj.emotionLabels, ...
                {[0 0 0 0], [100 0 0 100], [100 0 0 0], [100 100 0 0], ...
                [0 100 0 0], [0 100 100 0], [0 0 100 0], [0 0 100 100], [0 0 0 100]});
            
            if emotionMap.isKey(emotionState)
                coords = emotionMap(emotionState);
            else
                coords = [0 0 0 0];  % デフォルト値（安静状態）
                warning('Emotion state "%s" not found. Using default coordinates.', emotionState);
            end
        end
        
    end
    
    methods (Static)
        function featureNames = getFeatureNames(channels, freqBands)
            % 特徴量の名前を生成（事前割り当て）
            totalFeatures = length(channels) * length(freqBands);
            featureNames = cell(1, totalFeatures);

            featureIdx = 1;
            for ch = 1:length(channels)
                for band = 1:length(freqBands)
                    featureNames{featureIdx} = sprintf('Ch%d_%s', ch, freqBands{band});
                    featureIdx = featureIdx + 1;
                end
            end
        end

        function [meanPower, stdPower] = calculateChannelStats(powers)
            % チャンネルごとのパワー値統計量を計算
            meanPower = mean(powers, 1);
            stdPower = std(powers, 0, 1);
        end
    end
end