classdef NotchFilterDesigner < handle
    properties (Access = private)
        params
    end
    
    methods (Access = public)
        function obj = NotchFilterDesigner(params)
            obj.params = params;
        end
        
        function [filteredData, filterInfo] = designAndApplyFilter(obj, data)
            filteredData = data;
            
            % filterInfoの初期化を構造体として行う
            filterInfo = struct();
            filterInfo.notchFilters = cell(1, length(obj.params.signal.filter.notch.frequency));
            
            for i = 1:length(obj.params.signal.filter.notch.frequency)
                freq = obj.params.signal.filter.notch.frequency(i);
                [b, a, info] = obj.designNotchFilter(freq);
                filterInfo.notchFilters{i} = info;
                
                % フィルタの適用
                for ch = 1:size(data, 1)
                    filteredData(ch, :) = filtfilt(b, a, filteredData(ch, :));
                end
            end
        end
    end
    
    methods (Access = private)
        function [b, a, filterInfo] = designNotchFilter(obj, freq)
            fs = obj.params.device.sampleRate;
            wo = freq/(fs/2);  % 正規化周波数
            bw = obj.params.signal.filter.notch.bandwidth/(fs/2);  % 正規化帯域幅
            
            % IIRノッチフィルタの設計
            [b, a] = iirnotch(wo, bw);
            
            % フィルタ情報の記録
            filterInfo = struct(...
                'frequency', freq, ...
                'bandwidth', obj.params.signal.filter.notch.bandwidth, ...
                'sampleRate', fs, ...
                'coefficients', struct('b', b, 'a', a) ...
            );
            
            % 周波数応答の計算
            [h, w] = freqz(b, a, 1024, fs);
            filterInfo.response = struct(...
                'frequency', w, ...
                'magnitude', 20*log10(abs(h)), ...
                'phase', angle(h) ...
            );
        end
    end
end