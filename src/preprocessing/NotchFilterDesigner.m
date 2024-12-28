classdef NotchFilterDesigner < handle
    properties (Access = private)
        params
        filterInfo
        
        % フィルタ設計パラメータ
        notchFrequencies
        bandwidth
    end
    
    methods (Access = public)
        function obj = NotchFilterDesigner(params)
            obj.params = params;
            obj.initializeParameters();
        end
        
        function [filteredData, filterInfo] = designAndApplyFilter(obj, data)
            try
                % データの検証
                obj.validateData(data);
                
                % 出力データの初期化
                filteredData = data;
                filterInfo = struct('notchFilters', cell(1, length(obj.notchFrequencies)));
                
                % 各ノッチ周波数に対してフィルタを適用
                for i = 1:length(obj.notchFrequencies)
                    [b, a, info] = obj.designNotchFilter(obj.notchFrequencies(i));
                    filterInfo.notchFilters{i} = info;
                    
                    % フィルタの適用
                    for ch = 1:size(data, 1)
                        filteredData(ch, :) = filtfilt(b, a, filteredData(ch, :));
                    end
                end
                
                obj.filterInfo = filterInfo;
                
            catch ME
                error('NotchFilterDesigner:FilterError', 'Filtering failed: %s', ME.message);
            end
        end
        
        function info = getFilterInfo(obj)
            info = obj.filterInfo;
        end
    end
    
    methods (Access = private)
        function initializeParameters(obj)
            notch_params = obj.params.signal.preprocessing.filter.notch;
            obj.notchFrequencies = notch_params.frequency;
            obj.bandwidth = notch_params.bandwidth;
        end
        
        function validateData(obj, data)
            validateattributes(data, {'numeric'}, ...
                {'2d', 'nrows', obj.params.device.channelCount}, ...
                'NotchFilterDesigner', 'data');
        end
        
        function [b, a, filterInfo] = designNotchFilter(obj, freq)
            fs = obj.params.device.sampleRate;
            
            % 正規化周波数の計算
            w0 = freq/(fs/2);  % 正規化中心周波数
            bw = obj.bandwidth/(fs/2);  % 正規化帯域幅
            
            % IIRノッチフィルタの設計
            [b, a] = iirnotch(w0, bw);
            
            % フィルタ情報の記録
            filterInfo = struct();
            filterInfo.frequency = freq;
            filterInfo.bandwidth = obj.bandwidth;
            filterInfo.sampleRate = fs;
            filterInfo.coefficients = struct('b', b, 'a', a);
            
            % 周波数応答の計算
            [h, w] = freqz(b, a, 1024, fs);
            filterInfo.response = struct(...
                'frequency', w, ...
                'magnitude', 20*log10(abs(h)), ...
                'phase', angle(h));
            
            % -3dBポイントの検出
            mag_db = filterInfo.response.magnitude;
            notch_idx = find(w >= freq, 1);
            bw_low = find(mag_db(1:notch_idx) <= -3, 1, 'last');
            bw_high = find(mag_db(notch_idx:end) <= -3, 1) + notch_idx - 1;
            
            filterInfo.actualBandwidth = w(bw_high) - w(bw_low);
            filterInfo.attenuation = min(mag_db(notch_idx-5:notch_idx+5));
        end
        
        function analyzeFilterPerformance(obj, data, filteredData)
            % フィルタ性能の解析
            for i = 1:length(obj.notchFrequencies)
                freq = obj.notchFrequencies(i);
                
                % スペクトル解析
                [pxx_orig, f] = pwelch(data', [], [], [], obj.params.device.sampleRate);
                [pxx_filt, ~] = pwelch(filteredData', [], [], [], obj.params.device.sampleRate);
                
                % ノッチ周波数付近のパワー減衰を計算
                freq_idx = find(f >= freq-2 & f <= freq+2);
                attenuation = mean(10*log10(pxx_orig(freq_idx,:)) - 10*log10(pxx_filt(freq_idx,:)));
                
                % 結果の記録
                obj.filterInfo.notchFilters{i}.performance = struct(...
                    'powerAttenuation', attenuation, ...
                    'spectralLeakage', max(pxx_filt(freq_idx,:)) / max(pxx_filt));
            end
        end
        
        function validateFilterStability(~, ~, a)
            % フィルタの安定性チェック
            poles = roots(a);
            if any(abs(poles) >= 1)
                warning('NotchFilterDesigner:UnstableFilter', ...
                    'Designed filter may be unstable. Consider adjusting parameters.');
            end
        end
    end
end