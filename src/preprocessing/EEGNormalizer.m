classdef EEGNormalizer < handle
    properties (Access = private)
        params
        method
        normalizationInfo
    end
    
    methods (Access = public)
        function obj = EEGNormalizer(params)
            obj.params = params;
            obj.method = params.signal.preprocessing.normalize.method;
        end
        
        function [normalizedData, normParams] = normalize(obj, data)
            try
                % データ形式を判別
                dataType = obj.getDataType(data);
                
                % データ形式に応じた処理
                switch dataType
                    case 'cell'
                        [normalizedData, normParams] = obj.normalizeCellData(data);
                    case '3D array'
                        [normalizedData, normParams] = obj.normalizeArrayData(data);
                    case '2D array'
                        [normalizedData, normParams] = obj.normalize2DArrayData(data);
                    otherwise
                        error('EEGNORMALIZER:INVALIDINPUT', 'Invalid input data format');
                end
                
                % メタ情報の追加
                normParams.dataType = dataType;
                normParams.method = obj.method;
                obj.normalizationInfo = normParams;
                
            catch ME
                error('EEGNORMALIZER:NORMALIZATION', 'Normalization failed: %s', ME.message);
            end
        end
        
        function normalizedData = normalizeOnline(obj, data, normParams)
            try
                % 2D配列として正規化
                switch normParams.method
                    case 'zscore'
                        % Z-score正規化
                        normalizedData = (data - normParams.mean) ./ normParams.std;

                    case 'minmax'
                        % Min-max正規化
                        normalizedData = (data - normParams.min) ./ (normParams.max - normParams.min);

                    case 'robust'
                        % Robust正規化（MADを使用）
                        normalizedData = (data - normParams.median) ./ normParams.mad;

                    otherwise
                        error('EEGNORMALIZER:INVALIDMETHOD', 'Invalid normalization method: %s', obj.method);
                end

            catch ME
                error('EEGNORMALIZER:NORMALIZATIONFAILED', 'Online normalization failed: %s', ME.message);
            end
        end
        
        function info = getNormalizationInfo(obj)
            info = obj.normalizationInfo;
        end
    end
    
    methods (Access = private)
        function dataType = getDataType(~, data)
            if iscell(data)
                dataType = 'cell';
            elseif isnumeric(data) && ndims(data) == 3
                dataType = '3D array';
            elseif isnumeric(data) && ismatrix(data)
                dataType = '2D array';
            else
                error('EEGNORMALIZER:INVALIDINPUT', 'Invalid input data format');
            end
        end
        
        function [normalizedData, normParams] = normalize2DArrayData(obj, data)
            [numChannels, ~] = size(data);
            
            switch obj.method
                case 'zscore'
                    normParams.mean = mean(data, 2);
                    normParams.std = std(data, 0, 2);
                    % 標準偏差が0の場合の処理
                    normParams.std(normParams.std == 0) = 1;
                    normalizedData = (data - normParams.mean) ./ normParams.std;
                    
                case 'minmax'
                    normParams.min = min(data, [], 2);
                    normParams.max = max(data, [], 2);
                    % 最大値と最小値が同じ場合の処理
                    sameValues = normParams.max == normParams.min;
                    normParams.max(sameValues) = normParams.min(sameValues) + 1;
                    normalizedData = (data - normParams.min) ./ (normParams.max - normParams.min);
                    
                case 'robust'
                    normParams.median = median(data, 2);
                    normParams.mad = mad(data, 1, 2);
                    % MADが0の場合の処理
                    normParams.mad(normParams.mad == 0) = 1;
                    normalizedData = (data - normParams.median) ./ normParams.mad;
                    
                otherwise
                    error('EEGNORMALIZER:INVALIDMETHOD', 'Invalid normalization method');
            end
            
            % チャンネル情報の追加
            normParams.channelCount = numChannels;
        end
        
        function [normalizedData, normParams] = normalizeArrayData(obj, data)
            [numChannels, numSamples, numEpochs] = size(data);
            reshapedData = reshape(data, numChannels, []);
            
            % 2D配列として正規化
            [normalizedReshapedData, normParams] = obj.normalize2DArrayData(reshapedData);
            
            % 3D形状に戻す
            normalizedData = reshape(normalizedReshapedData, numChannels, numSamples, numEpochs);
            
            % エポック情報の追加
            normParams.epochCount = numEpochs;
            normParams.samplesPerEpoch = numSamples;
        end
        
        function [normalizedData, normParams] = normalizeCellData(obj, cellData)
            numEpochs = length(cellData);
            [numChannels, numSamplesPerEpoch] = size(cellData{1});
            
            % 全データを2D配列に変換
            reshapedData = zeros(numChannels, numSamplesPerEpoch * numEpochs);
            for ep = 1:numEpochs
                startIdx = (ep-1) * numSamplesPerEpoch + 1;
                endIdx = ep * numSamplesPerEpoch;
                reshapedData(:, startIdx:endIdx) = cellData{ep};
            end
            
            % 2D配列として正規化
            [normalizedReshapedData, normParams] = obj.normalize2DArrayData(reshapedData);
            
            % セル配列に戻す
            normalizedData = cell(size(cellData));
            for ep = 1:numEpochs
                startIdx = (ep-1) * numSamplesPerEpoch + 1;
                endIdx = ep * numSamplesPerEpoch;
                normalizedData{ep} = normalizedReshapedData(:, startIdx:endIdx);
            end
            
            % セル配列情報の追加
            normParams.epochCount = numEpochs;
            normParams.samplesPerEpoch = numSamplesPerEpoch;
        end
    end
end