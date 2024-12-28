classdef PowerExtractor < handle
    properties (Access = private)
        params           % パラメータ設定
        powerFunction    % パワー値計算関数
        freqBands       % 周波数帯域情報
        usedParams      % 使用したパラメータ
    end
    
    methods (Access = public)
        function obj = PowerExtractor(params)
            obj.params = params;
            obj.initializeFreqBands();
        end
        
        function [powers] = calculatePower(obj, signal, freqBand)
            % パワー値の計算
            fs = obj.params.device.sampleRate;
            method = obj.params.feature.power.method;
            
            switch lower(method)
                case 'welch'
                    powers = obj.calculateWelchPower(signal, freqBand, fs);
                case 'filter'
                    powers = obj.calculateFilterPower(signal, freqBand, fs);
                otherwise
                    error('Unknown power calculation method');
            end
            
            % 正規化の適用
            if obj.params.feature.power.normalize.enable
                powers = obj.applyNormalizations(powers, signal);
            end
        end
        
        function [pxx, f] = calculateSpectrum(obj, data)
            % Welch法によるスペクトル計算
            fs = obj.params.device.sampleRate;
            windowLength = obj.params.feature.power.welch.windowLength;
            overlap = obj.params.feature.power.welch.overlap;
            nfft = obj.params.feature.power.welch.nfft;
            windowType = obj.params.feature.power.welch.windowType;
            
            window = obj.getWindow(windowType, windowLength);
            noverlap = round(windowLength * overlap);
            
            [pxx, f] = pwelch(data(1,:), window, noverlap, nfft, fs);
        end
        
        function [ersp, times, freqs] = calculateERSP(obj, data)
            % パラメータ設定
            fs = obj.params.device.sampleRate;
            freqRange = obj.params.gui.display.visualization.ersp.freqRange;
            numFreqs = obj.params.gui.display.visualization.ersp.numFreqs;

            % 周波数軸の設定（表示用に線形スケールを使用）
            freqs = linspace(freqRange(1), freqRange(2), numFreqs);

            % spectrogram計算
            [S, freqs, times] = spectrogram(data(1,:), ...
                hamming(numFreqs), ...
                round(numFreqs*0.75), ...
                freqs, ...
                fs, ...
                'yaxis');

            % パワー計算とdB変換
            ersp = 10*log10(abs(S).^2);

            % サイズの確認と必要な場合の調整
            if size(ersp, 2) ~= length(times)
                ersp = ersp(:, 1:length(times));
            end
        end
    end
    
    methods (Access = private)
        % パワー計算関連メソッド
        function powers = calculateWelchPower(obj, signal, freqBand, fs)
            welchParams = obj.params.feature.power.welch;
            window = obj.getWindow(welchParams.windowType, welchParams.windowLength);
            noverlap = round(welchParams.windowLength * welchParams.overlap);
            nfft = welchParams.nfft;
            
            [nChannels, ~] = size(signal);
            powers = zeros(nChannels, 1);
            
            for ch = 1:nChannels
                [pxx, f] = pwelch(signal(ch,:), window, noverlap, nfft, fs);
                idx = f >= freqBand(1) & f <= freqBand(2);
                df = f(2) - f(1);
                powers(ch) = sum(pxx(idx)) * df;
            end
        end
        
        function powers = calculateFilterPower(obj, signal, freqBand, fs)
            filterParams = obj.params.feature.power.filter;
            Wn = [freqBand(1)/(fs/2), freqBand(2)/(fs/2)];
            [b, a] = butter(filterParams.order, Wn, 'bandpass');
            
            [nChannels, ~] = size(signal);
            powers = zeros(nChannels, 1);
            
            for ch = 1:nChannels
                filtered = filtfilt(b, a, signal(ch,:));
                powers(ch) = mean(filtered.^2);
            end
        end

        function powers = applyNormalizations(obj, powers, signal)
            normMethods = obj.params.feature.power.normalize.methods;
            for i = 1:length(normMethods)
                switch normMethods{i}
                    case 'relative'
                        powers = obj.normalizeRelative(powers, signal);
                    case 'log'
                        powers = obj.normalizeLog(powers);
                    case 'zscore'
                        powers = obj.normalizeZscore(powers);
                    case 'robust'
                        powers = obj.normalizeRobust(powers);
                end
            end
        end

        function powers = normalizeRelative(obj, powers, signal)
            % 全周波数帯域のパワー総和による正規化
            totalPower = obj.computeTotalPower(signal);
            powers = powers ./ (totalPower + eps);
        end

        function powers = normalizeLog(~, powers)
            % ログ変換による正規化
            powers = log10(powers + eps);
        end

        function powers = normalizeZscore(~, powers)
            % Z-score正規化（平均0、標準偏差1）
            mu = mean(powers);
            sigma = std(powers);
            powers = (powers - mu) ./ (sigma + eps);
        end

        function powers = normalizeRobust(~, powers)
            % ロバスト正規化（中央値とMAD）
            med = median(powers);
            mad_val = 1.4826 * mad(powers, 1);
            powers = (powers - med) ./ (mad_val + eps);
        end
        
        function totalPower = computeTotalPower(obj, signal)
            totalPower = 0;
            for i = 1:length(obj.freqBands)
                bandName = obj.freqBands{i};
                freqRange = obj.params.feature.power.bands.(bandName);
                
                if strcmpi(obj.params.feature.power.method, 'welch')
                    powers = obj.calculateWelchPower(signal, freqRange, obj.params.device.sampleRate);
                else
                    powers = obj.calculateFilterPower(signal, freqRange, obj.params.device.sampleRate);
                end
                totalPower = totalPower + sum(powers);
            end
        end

        function initializeFreqBands(obj)
            if isfield(obj.params.feature.power.bands, 'names')
                bandNames = obj.params.feature.power.bands.names;
                if iscell(bandNames{1})
                    obj.freqBands = bandNames{1};
                else
                    obj.freqBands = bandNames;
                end
            else
                obj.freqBands = {'delta', 'theta', 'alpha', 'beta', 'gamma'};
            end
        end
        
        function window = getWindow(~, windowType, windowLength)
            switch lower(windowType)
                case 'hamming'
                    window = hamming(windowLength);
                case 'hann'
                    window = hann(windowLength);
                case 'blackman'
                    window = blackman(windowLength);
            end
        end
    end
end