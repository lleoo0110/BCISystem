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
   end
   
   methods (Access = public)
       function obj = PowerFeatureExtractor(params)
           % コンストラクタ
           obj.params = params;
           obj.initializeFreqBands();
           
           % EPOCXの場合のチャンネル番号設定
           % AF3=1, F3=3 が左前頭葉
           % AF4=14, F4=12 が右前頭葉
           obj.leftChannels = [1, 3];   % AF3, F3
           obj.rightChannels = [14, 12]; % AF4, F4
           obj.faaThreshold = 0.5;
           
           % ERD関連の初期化
           obj.baselinePower = struct();
           obj.isBaselineSet = false;
       end
       
       function features = extractFeatures(obj, data)
           % 特徴量抽出メイン関数
           try
               % データサイズの取得
               [nChannels, nSamples, nTrials] = size(data);
               nBands = length(obj.freqBands);
               
               % 特徴量格納用配列の初期化
               features = zeros(nTrials, nChannels * nBands);
               
               % 各試行に対してパワー値を計算
               for trial = 1:nTrials
                   bandPowers = [];
                   for band = 1:nBands
                       % 周波数帯域の取得
                       bandName = obj.freqBands{band};
                       freqRange = obj.params.feature.power.bands.(bandName);
                       
                       % パワー値の計算
                       power = obj.calculatePower(data(:,:,trial), freqRange);
                       
                       % 結果の保存
                       bandPowers = [bandPowers, power];
                   end
                   
                   % 特徴量の格納
                   features(trial, :) = bandPowers(:)';
               end
               
               % 標準化処理
               if obj.params.feature.standardize
                   features = obj.standardizeFeatures(features);
               end
               
           catch ME
               error('Feature extraction failed: %s', ME.message);
           end
       end
       
       function [faaValue, arousalState] = calculateFAA(obj, data)
           try
               % データサイズの確認と出力
               [nChannels, nSamples] = size(data);
               fprintf('Data size: %d channels x %d samples\n', nChannels, nSamples);
               
               % 左右それぞれのパワー値計算
               leftPower = obj.calculatePower(data(obj.leftChannels,:), [8 13]);
               rightPower = obj.calculatePower(data(obj.rightChannels,:), [8 13]);
               
               % デバッグ情報の出力
               fprintf('Left channels power: %f\n', mean(leftPower));
               fprintf('Right channels power: %f\n', mean(rightPower));
               
               % 左右のパワー値の平均を計算
               leftMean = mean(leftPower);
               rightMean = mean(rightPower);
               
               % FAA値の計算 (log(右) - log(左))
               faaValue = log(rightMean) - log(leftMean);
               
               % 覚醒状態の判定
               if faaValue > obj.faaThreshold
                   arousalState = 'aroused';
               else
                   arousalState = 'non-aroused';
               end
              
               
           catch ME
               fprintf('Error in calculateFAA: %s\n', ME.message);
               rethrow(ME);
           end
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
        
       function [erdValues, erdPercent] = calculateERD(obj, data)
            try
                % ベースラインが未設定の場合はエラー
                if ~obj.isBaselineSet
                    error('Baseline power not set. Call updateBaseline first.');
                end
                
                % 結果格納用の初期化
                erdValues = struct();
                erdPercent = struct();
                
                % 各周波数帯でのERD計算
                for i = 1:length(obj.freqBands)
                    bandName = obj.freqBands{i};
                    freqRange = obj.params.feature.power.bands.(bandName);
                    
                    % 現在のパワー値を計算
                    currentPower = obj.calculatePower(data, freqRange);
                    baselinePower = obj.baselinePower.(bandName);
                    
                    % ERDを計算 (baselinePower - currentPower) / baselinePower * 100
                    erdValue = baselinePower - currentPower;
                    erdPct = (erdValue ./ baselinePower) * 100;
                    
                    % 結果を保存
                    erdValues.(bandName) = erdValue;
                    erdPercent.(bandName) = erdPct;
                end
                
            catch ME
                fprintf('Error calculating ERD: %s\n', ME.message);
                rethrow(ME);
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
       
       function features = standardizeFeatures(~, features)
           % 特徴量の標準化
           features = (features - mean(features)) ./ std(features);
       end
   end
   
   methods (Static)
       function featureNames = getFeatureNames(channels, freqBands)
           % 特徴量の名前を生成
           featureNames = {};
           for ch = 1:length(channels)
               for band = 1:length(freqBands)
                   featureName = sprintf('Ch%d_%s', ch, freqBands{band});
                   featureNames{end+1} = featureName;
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