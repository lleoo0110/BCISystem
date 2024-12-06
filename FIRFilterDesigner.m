classdef FIRFilterDesigner < handle
    properties (Access = private)
        params
    end
    
    methods (Access = public)
        function obj = FIRFilterDesigner(params)
            obj.params = params;
        end
                
        function [filteredData, filterInfo] = designAndApplyFilter(obj, data)
            % フィルタ設計
            bpFilt = obj.designFilter();

            % フィルタ特性の取得
            filterInfo = obj.analyzeFilter(bpFilt);
            filterInfo.filter = bpFilt;  % フィルタオブジェクトを保存

            % フィルタリングの適用
            filteredData = obj.applyFilter(data, bpFilt);
        end
        
        function visualizeFilter(obj, bpFilt)
            % フィルタ特性の可視化
            figure('Name', 'FIRバンドパスフィルタの特性');

            % 周波数応答の計算
            fs = obj.params.device.sampleRate;
            [h, f] = freqz(bpFilt, fs);

            % 振幅応答の表示
            subplot(2,1,1);
            plot(f, 20*log10(abs(h)));
            title('振幅応答');
            xlabel('周波数 (Hz)');
            ylabel('振幅 (dB)');
            grid on;

            % 位相応答の表示
            subplot(2,1,2);
            plot(f, unwrap(angle(h))*180/pi);
            title('位相応答');
            xlabel('周波数 (Hz)');
            ylabel('位相 (度)');
            grid on;
        end
    end
    
    methods (Access = private)
        function bpFilt = designFilter(obj)
            fs = obj.params.device.sampleRate;
            fmin = obj.params.signal.frequency.min;
            fmax = obj.params.signal.frequency.max;
            designMethod = obj.params.signal.filter.fir.designMethod;
            filterOrder = obj.params.signal.filter.fir.filterOrder;
            
            if strcmpi(obj.params.signal.filter.fir.designMethod, 'cls')
                % CLSメソッドの場合
                bpFilt = designfilt('bandpassfir', ...
                    'FilterOrder', filterOrder, ...
                    'CutoffFrequency1', fmin, ...
                    'CutoffFrequency2', fmax, ...
                    'SampleRate', fs, ...
                    'StopbandAttenuation1', obj.params.signal.filter.fir.stopbandAttenuation, ...
                    'PassbandRipple', obj.params.signal.filter.fir.passbandRipple, ...
                    'StopbandAttenuation2', obj.params.signal.filter.fir.stopbandAttenuation, ...
                    'DesignMethod', 'equiripple');
            else
                % Windowメソッドの場合
                win = obj.getWindowFunction(obj.params.signal.filter.fir.windowType, filterOrder+1);
                bpFilt = designfilt('bandpassfir', ...
                    'FilterOrder', filterOrder, ...
                    'CutoffFrequency1', fmin, ...
                    'CutoffFrequency2', fmax, ...
                    'SampleRate', fs, ...
                    'DesignMethod', designMethod, ...
                    'Window', win);
            end
        end
        
        function win = getWindowFunction(obj, windowType, N)
            % ウィンドウ関数の生成
            switch lower(windowType)
                case 'hamming'
                    win = hamming(N);
                case 'hanning'
                    win = hanning(N);
                case 'blackman'
                    win = blackman(N);
                case 'kaiser'
                    win = kaiser(N, 4);  % ベータパラメータは4を使用
                otherwise
                    win = hamming(N);  % デフォルトはハミング窓
            end
        end
        
        function filterInfo = analyzeFilter(obj, bpFilt)
            % フィルタの特性解析
            filterInfo = struct();
            
            % 周波数応答の計算
            [h, w] = freqz(bpFilt);
            
            % 振幅応答（dB）
            filterInfo.magnitude = 20 * log10(abs(h));
            
            % 位相応答
            filterInfo.phase = angle(h);
            
            % 群遅延の計算
            gd = grpdelay(bpFilt);
            filterInfo.groupDelay = mean(gd);
            
            % -3dBポイントの検出
            [~, cutoffLow] = min(abs(filterInfo.magnitude + 3));
            [~, cutoffHigh] = min(abs(filterInfo.magnitude(cutoffLow:end) + 3));
            filterInfo.cutoffFrequencies = [w(cutoffLow), w(cutoffLow + cutoffHigh)] * ...
                (obj.params.device.sampleRate/(2*pi));
        end
        
        function filteredData = applyFilter(obj, data, bpFilt)
            % フィルタの適用
            filteredData = zeros(size(data));
            for ch = 1:size(data, 1)
                filteredData(ch, :) = filtfilt(bpFilt, data(ch, :));
            end
        end
    end
end