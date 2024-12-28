classdef BaselineCorrector < handle
    properties (Access = private)
        params              % 設定パラメータ
        
        % 処理パラメータ
        windowSize        % 補正窓サイズ
        overlap          % オーバーラップ率
        
        % 結果保存用
        correctionInfo    % 補正情報
    end
    
    methods (Access = public)
        function obj = BaselineCorrector(params)
            obj.params = params;
            obj.initializeParameters();
        end
        
        function [correctedData, correctionInfo] = correctBaseline(obj, data, method)
            try
                % データの検証
                obj.validateData(data);
                
                % 補正方法に応じた処理
                switch lower(method)
                    case 'interval'
                        [correctedData, correctionInfo] = obj.intervalCorrection(data);
                    case 'trend'
                        [correctedData, correctionInfo] = obj.trendRemoval(data);
                    case 'dc'
                        [correctedData, correctionInfo] = obj.dcRemoval(data);
                    case 'moving'
                        [correctedData, correctionInfo] = obj.movingAverageCorrection(data);
                    otherwise
                        error('Unknown correction method');
                end
                
                obj.correctionInfo = correctionInfo;
                
            catch ME
                error('BaselineCorrector:CorrectionError', 'Correction failed: %s', ME.message);
            end
        end
        
        function info = getCorrectionInfo(obj)
            info = obj.correctionInfo;
        end
    end
    
    methods (Access = private)
        function initializeParameters(obj)
            baseline_params = obj.params.signal.preprocessing.baseline;
            obj.windowSize = round(baseline_params.windowSize * obj.params.device.sampleRate);
            obj.overlap = baseline_params.overlap;
        end
        
        function validateData(obj, data)
            validateattributes(data, {'numeric'}, ...
                {'2d', 'nrows', obj.params.device.channelCount}, ...
                'BaselineCorrector', 'data');
        end
        
        function [correctedData, correctionInfo] = intervalCorrection(obj, data)
            correctedData = data;
            correctionInfo = struct('method', 'interval', 'corrections', []);
            corrections = zeros(obj.params.device.channelCount, 1);
            
            for ch = 1:size(data, 1)
                % 区間ごとの平均値を計算
                intervalStart = 1:round((1-obj.overlap)*obj.windowSize):size(data,2)-obj.windowSize+1;
                baselineValues = zeros(length(intervalStart), 1);
                
                for i = 1:length(intervalStart)
                    idx = intervalStart(i):min(intervalStart(i)+obj.windowSize-1, size(data,2));
                    baselineValues(i) = mean(data(ch, idx));
                end
                
                % 平均ベースライン値を使用して補正
                baselineValue = mean(baselineValues);
                correctedData(ch,:) = data(ch,:) - baselineValue;
                corrections(ch) = baselineValue;
            end
            
            correctionInfo.corrections = corrections;
        end
        
        function [correctedData, correctionInfo] = trendRemoval(~, data)
            correctedData = data;
            correctionInfo = struct('method', 'trend', 'polynomials', cell(1, size(data,1)));
            
            for ch = 1:size(data, 1)
                % 3次多項式フィッティング
                x = 1:size(data,2);
                [p, ~] = polyfit(x, data(ch,:), 3);
                trend = polyval(p, x);
                
                % トレンド除去
                correctedData(ch,:) = data(ch,:) - trend;
                correctionInfo.polynomials{ch} = p;
            end
        end
        
        function [correctedData, correctionInfo] = dcRemoval(~, data)
            correctedData = data;
            correctionInfo = struct('method', 'dc', 'means', zeros(size(data,1), 1));
            
            for ch = 1:size(data, 1)
                dcValue = mean(data(ch,:));
                correctedData(ch,:) = data(ch,:) - dcValue;
                correctionInfo.means(ch) = dcValue;
            end
        end
        
        function [correctedData, correctionInfo] = movingAverageCorrection(obj, data)
            correctedData = data;
            correctionInfo = struct('method', 'moving', 'windowSize', obj.windowSize);
            
            % 移動平均フィルタの設計
            b = ones(1, obj.windowSize) / obj.windowSize;
            a = 1;
            
            for ch = 1:size(data, 1)
                % 移動平均の計算と補正
                baseline = filtfilt(b, a, data(ch,:));
                correctedData(ch,:) = data(ch,:) - baseline;
            end
        end
    end
end