classdef ArtifactRemover < handle
    properties (Access = private)
        % 設定パラメータ
        params
        
        % 処理パラメータ
        thresholds
        windowSize     % 解析窓サイズ
        
        % 状態管理
        isInitialized
        removeInfo
    end
    
    methods (Access = public)
        function obj = ArtifactRemover(params)
            obj.params = params;
            obj.isInitialized = false;
            obj.initializeParameters();
        end
        
        function [cleanedData, artifactInfo] = removeArtifacts(obj, data, method)
            try
                % データの検証
                obj.validateData(data);
                
                switch lower(method)
                    case 'all'
                        [cleanedData, artifactInfo] = obj.removeAllArtifacts(data);
                    case 'eog'
                        [cleanedData, artifactInfo] = obj.removeEOG(data);
                    case 'emg'
                        [cleanedData, artifactInfo] = obj.removeEMG(data);
                    case 'baseline'
                        [cleanedData, artifactInfo] = obj.removeBaseline(data);
                    case 'threshold'
                        [cleanedData, artifactInfo] = obj.removeByThreshold(data);
                    otherwise
                        error('Unknown artifact removal method');
                end
                
                obj.removeInfo = artifactInfo;
                
            catch ME
                error('ArtifactRemover:RemovalError', 'Artifact removal failed: %s', ME.message);
            end
        end
        
        function info = getRemovalInfo(obj)
            info = obj.removeInfo;
        end
    end
    
    methods (Access = private)
        function initializeParameters(obj)
            % パラメータの初期化
            artifact_params = obj.params.signal.preprocessing.artifact;
            obj.thresholds = artifact_params.thresholds;
            obj.windowSize = round(artifact_params.windowSize * obj.params.device.sampleRate);
            obj.isInitialized = true;
        end
        
        function validateData(obj, data)
            validateattributes(data, {'numeric'}, ...
                {'2d', 'nrows', obj.params.device.channelCount}, ...
                'ArtifactRemover', 'data');
        end
        
        function [cleanedData, artifactInfo] = removeAllArtifacts(obj, data)
            artifactInfo = struct();
            
            % EOGの除去
            [cleanedData, artifactInfo.eog] = obj.removeEOG(data);
            
            % EMGの除去
            [cleanedData, artifactInfo.emg] = obj.removeEMG(cleanedData);
            
            % 基線変動の除去
            [cleanedData, artifactInfo.baseline] = obj.removeBaseline(cleanedData);
            
            % 閾値による異常値の除去
            [cleanedData, artifactInfo.threshold] = obj.removeByThreshold(cleanedData);
        end
        
        function [cleanedData, artifactInfo] = removeEOG(obj, data)
            cleanedData = data;
            artifactInfo = struct('method', 'EOG', 'removedSegments', []);
            
            for ch = 1:size(data, 1)
                if obj.isFrontalChannel(ch)
                    % 移動窓による処理
                    for i = 1:obj.windowSize:size(data,2)-obj.windowSize+1
                        segment = data(ch, i:i+obj.windowSize-1);
                        if max(abs(diff(segment))) > obj.thresholds.eog
                            cleanedData(ch, i:i+obj.windowSize-1) = ...
                                obj.interpolateSegment(segment);
                            artifactInfo.removedSegments = [artifactInfo.removedSegments; i, i+obj.windowSize-1];
                        end
                    end
                end
            end
        end
        
        function [cleanedData, artifactInfo] = removeEMG(obj, data)
            cleanedData = data;
            artifactInfo = struct('method', 'EMG', 'removedSegments', []);
            segments = [];
            
            for ch = 1:size(data, 1)
                % 移動窓による処理
                for i = 1:obj.windowSize:size(data,2)-obj.windowSize+1
                    segment = data(ch, i:i+obj.windowSize-1);
                    
                    % 高周波パワーの計算
                    [pxx, f] = pwelch(segment, [], [], [], obj.params.device.sampleRate);
                    highFreqPower = mean(pxx(f > 30));  % 30Hz以上を高周波とする
                    
                    if highFreqPower > obj.thresholds.emg
                        segments = [segments; i, i+obj.windowSize-1];
                        
                        % ローパスフィルタの適用
                        cleanedData(ch, i:i+obj.windowSize-1) = ...
                            obj.applyLowPassFilter(segment);
                    end
                end
            end
            
            artifactInfo.removedSegments = segments;
        end
        
        function [cleanedData, artifactInfo] = removeBaseline(~, data)
            cleanedData = data;
            artifactInfo = struct('method', 'Baseline', 'correction', []);
            corrections = zeros(size(data, 1), 1);

            for ch = 1:size(data, 1)
                try
                    % データの正規化とスケーリング
                    x = (1:size(data,2))';
                    x = (x - mean(x)) / std(x);  % データのセンタリングとスケーリング
                    y = data(ch,:)';
                    y = (y - mean(y)) / std(y);

                    % より低次の多項式フィットを使用
                    [p, ~] = polyfit(x, y, 2);  % 3次から2次に変更
                    trend = polyval(p, x) * std(data(ch,:)) + mean(data(ch,:));

                    cleanedData(ch,:) = data(ch,:) - trend';
                    corrections(ch) = mean(abs(trend));
                catch ME
                    warning('Baseline removal failed for channel %d: %s', ch, ME.message);
                    % エラーが発生した場合は元のデータを保持
                    cleanedData(ch,:) = data(ch,:);
                    corrections(ch) = 0;
                end
            end

            artifactInfo.correction = corrections;
        end
        
        function [cleanedData, artifactInfo] = removeByThreshold(obj, data)
            cleanedData = data;
            artifactInfo = struct('method', 'Threshold', 'removedPoints', []);
            removedPoints = [];
            
            for ch = 1:size(data, 1)
                % 異常値の検出
                outliers = abs(data(ch,:)) > obj.thresholds.amplitude;
                
                if any(outliers)
                    % 異常値の位置を記録
                    removedPoints = [removedPoints; ch * ones(sum(outliers),1), find(outliers)];
                    
                    % 異常値の補間
                    cleanedData(ch, outliers) = ...
                        obj.interpolateOutliers(data(ch,:), outliers);
                end
            end
            
            artifactInfo.removedPoints = removedPoints;
        end
        
        % ヘルパーメソッド
        function interpolated = interpolateSegment(~, segment)
            x = 1:length(segment);
            interpolated = interp1(x([1,end]), segment([1,end]), x, 'pchip');
        end
        
        function filtered = applyLowPassFilter(obj, segment)
            [b, a] = butter(4, 30/(obj.params.device.sampleRate/2), 'low');
            filtered = filtfilt(b, a, segment);
        end
        
        function interpolated = interpolateOutliers(~, data, outliers)
            x = 1:length(data);
            validPoints = ~outliers;
            interpolated = interp1(x(validPoints), data(validPoints), ...
                x(outliers), 'pchip');
        end
        
        function isFrontal = isFrontalChannel(obj, channelIndex)
            channels = obj.params.device.channels;
            channelName = channels{channelIndex};
            isFrontal = any(strncmp(channelName, {'Fp', 'F'}, 1));
        end
    end
end