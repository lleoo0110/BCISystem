classdef DownSampler < handle
    properties (Access = private)
        params              % 設定パラメータ
        
        % パラメータ
        originalRate      % 元のサンプリングレート
        targetRate       % 目標サンプリングレート
        decimationFactor  % デシメーション係数
        filterOrder      % フィルタ次数
        
        % 結果保存用
        samplingInfo     % サンプリング情報
    end
    
    methods (Access = public)
        function obj = DownSampler(params)
            obj.params = params;
            obj.initializeParameters();
        end
        
        function [downsampledData, samplingInfo] = downsample(obj, data, targetRate)
            try
                % データとパラメータの検証
                obj.validateData(data);
                obj.validateRate(targetRate);
                
                % デシメーション係数の計算
                obj.calculateDecimationFactor(targetRate);
                
                % アンチエイリアシングフィルタの適用
                filteredData = obj.applyAntiAliasingFilter(data);
                
                % ダウンサンプリングの実行
                downsampledData = obj.executeDownsampling(filteredData);
                
                % サンプリング情報の更新
                samplingInfo = obj.updateSamplingInfo();
                obj.samplingInfo = samplingInfo;
                
            catch ME
                error('DownSampler:ProcessError', 'Downsampling failed: %s', ME.message);
            end
        end
        
        function info = getSamplingInfo(obj)
            info = obj.samplingInfo;
        end
    end
    
    methods (Access = private)
        function initializeParameters(obj)
            downsample_params = obj.params.signal.preprocessing.downsample;
            obj.originalRate = obj.params.device.sampleRate;
            obj.filterOrder = downsample_params.filterOrder;
        end
        
        function validateData(obj, data)
            validateattributes(data, {'numeric'}, ...
                {'2d', 'nrows', obj.params.device.channelCount}, ...
                'DownSampler', 'data');
        end
        
        function validateRate(obj, targetRate)
            validateattributes(targetRate, {'numeric'}, ...
                {'scalar', 'positive', '<', obj.originalRate}, ...
                'DownSampler', 'targetRate');
            obj.targetRate = targetRate;
        end
        
        function calculateDecimationFactor(obj, targetRate)
            factor = floor(obj.originalRate / targetRate);
            
            if factor < 2
                error('Target rate too high for effective downsampling');
            end
            
            obj.decimationFactor = factor;
        end
        
        function filteredData = applyAntiAliasingFilter(obj, data)
            % カットオフ周波数の計算（ナイキスト周波数の0.8倍）
            cutoffFreq = 0.8 * (obj.targetRate / 2);
            normalizedCutoff = cutoffFreq / (obj.originalRate / 2);
            
            % FIRフィルタの設計
            b = fir1(obj.filterOrder, normalizedCutoff, 'low');
            
            % フィルタの適用
            filteredData = zeros(size(data));
            for ch = 1:size(data, 1)
                filteredData(ch,:) = filtfilt(b, 1, data(ch,:));
            end
        end
        
        function downsampledData = executeDownsampling(obj, data)
            % 出力サイズの計算
            newLength = floor(size(data,2) / obj.decimationFactor);
            downsampledData = zeros(size(data,1), newLength);
            
            % データの間引き
            for ch = 1:size(data, 1)
                downsampledData(ch,:) = data(ch, 1:obj.decimationFactor:obj.decimationFactor*newLength);
            end
        end
        
        function samplingInfo = updateSamplingInfo(obj)
            samplingInfo = struct(...
                'originalRate', obj.originalRate, ...
                'targetRate', obj.targetRate, ...
                'decimationFactor', obj.decimationFactor, ...
                'filterOrder', obj.filterOrder);
        end
    end
end