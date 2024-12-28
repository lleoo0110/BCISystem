classdef FIRFilterDesigner < handle
    properties (Access = private)
        params
        filterInfo
        
        % フィルタ設計パラメータ
        fmin
        fmax
        filterOrder
        designMethod
        windowType
        passbandRipple
        stopbandAttenuation
    end
    
    methods (Access = public)
        function obj = FIRFilterDesigner(params)
            obj.params = params;
            obj.initializeParameters();
        end
        
        function [filteredData, filterInfo] = designAndApplyFilter(obj, data)
            try
                % データの検証
                obj.validateData(data);
                
                % フィルタの設計
                bpFilt = obj.designFilter();
                
                % フィルタ特性の解析
                filterInfo = obj.analyzeFilter(bpFilt);
                
                % フィルタの適用
                filteredData = obj.applyFilter(data, bpFilt);
                
                % フィルタ情報の保存
                filterInfo.filter = bpFilt;
                obj.filterInfo = filterInfo;
                
            catch ME
                error('FIRFilterDesigner:FilterError', 'Filtering failed: %s', ME.message);
            end
        end
        
        function info = getFilterInfo(obj)
            info = obj.filterInfo;
        end
    end
    
    methods (Access = private)
        function initializeParameters(obj)
            % フィルタパラメータの初期化
            filter_params = obj.params.signal.preprocessing.filter.fir;
            freq_params = obj.params.signal.frequency;

            obj.fmin = freq_params.min;
            obj.fmax = freq_params.max;
            obj.filterOrder = filter_params.filterOrder;
            obj.designMethod = filter_params.designMethod;
            obj.windowType = filter_params.windowType;
            obj.passbandRipple = filter_params.passbandRipple;
            obj.stopbandAttenuation = filter_params.stopbandAttenuation;
        end
        
        function validateData(obj, data)
            validateattributes(data, {'numeric'}, ...
                {'2d', 'nrows', obj.params.device.channelCount}, ...
                'FIRFilterDesigner', 'data');
        end
        
        function bpFilt = designFilter(obj)
            fs = obj.params.device.sampleRate;
            
            % 正規化周波数の計算
            wn = [obj.fmin, obj.fmax] / (fs/2);
            
            if strcmpi(obj.designMethod, 'window')
                % ウィンドウ法によるフィルタ設計
                win = obj.getWindowFunction(obj.windowType, obj.filterOrder+1);
                bpFilt = fir1(obj.filterOrder, wn, 'bandpass', win);
                
            else  % equiripple method
                % Parks-McClellan法によるフィルタ設計
                f = [0, wn(1)-0.1, wn(1), wn(2), wn(2)+0.1, 1];
                a = [0, 0, 1, 1, 0, 0];
                dev = [(10^(obj.stopbandAttenuation/20)), (10^(-obj.passbandRipple/20)), ...
                      (10^(-obj.passbandRipple/20)), (10^(obj.stopbandAttenuation/20))];
                [n,fo,ao,w] = firpmord(f,a,dev);
                n = min(n, obj.filterOrder);  % フィルタ次数の制限
                bpFilt = firpm(n,fo,ao,w);
            end
        end
        
        function filterInfo = analyzeFilter(obj, bpFilt)
            fs = obj.params.device.sampleRate;
            
            % 周波数応答の計算
            [h, f] = freqz(bpFilt, 1, 1024, fs);
            
            filterInfo = struct();
            filterInfo.frequency = f;
            filterInfo.magnitude = 20 * log10(abs(h));
            filterInfo.phase = angle(h);
            filterInfo.groupDelay = grpdelay(bpFilt, 1, 1024);
            
            % -3dBポイントの検出
            mag_db = filterInfo.magnitude;
            passband_idx = find(f >= obj.fmin & f <= obj.fmax);
            cutoff_low = find(mag_db(1:passband_idx(1)) >= -3, 1, 'last');
            cutoff_high = find(mag_db(passband_idx(end):end) >= -3, 1) + passband_idx(end) - 1;
            
            filterInfo.cutoffFrequencies = [f(cutoff_low), f(cutoff_high)];
            filterInfo.params = struct(...
                'order', obj.filterOrder, ...
                'method', obj.designMethod, ...
                'window', obj.windowType, ...
                'fmin', obj.fmin, ...
                'fmax', obj.fmax);
        end
        
        function win = getWindowFunction(~, windowType, N)
            switch lower(windowType)
                case 'hamming'
                    win = hamming(N);
                case 'hanning'
                    win = hanning(N);
                case 'blackman'
                    win = blackman(N);
                case 'kaiser'
                    win = kaiser(N, 4);
                otherwise
                    win = hamming(N);
            end
        end
        
        function filteredData = applyFilter(~, data, bpFilt)
            filteredData = zeros(size(data));
            
            % 各チャンネルにフィルタを適用
            for ch = 1:size(data, 1)
                filteredData(ch, :) = filtfilt(bpFilt, 1, data(ch, :));
            end
        end
    end
end