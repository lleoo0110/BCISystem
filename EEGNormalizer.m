classdef EEGNormalizer < handle
    properties (Access = private)
        params
    end
    
    methods (Access = public)
        function obj = EEGNormalizer(params)
            obj.params = params;
        end
        
        function [normalizedData, normParams] = normalize(obj, data)
            try
                % データ形式を判別
                dataType = obj.getDataType(data);
                
                % 正規化手法の取得
                method = obj.params.signal.normalize.method;
                
                % データ形式に応じた処理
                switch dataType
                    case 'cell'
                        [normalizedData, normParams] = obj.normalizeCellData(data, method);
                    case '3D array'
                        [normalizedData, normParams] = obj.normalizeArrayData(data, method);
                    case '2D array'
                        [normalizedData, normParams] = obj.normalize2DArrayData(data, method);
                    otherwise
                        error('EEGNORMALIZER:INVALIDINPUT', 'Invalid input data format');
                end
                
                % メタ情報の追加
                normParams.dataType = dataType;
                normParams.method = method;
                
            catch ME
                error('EEGNORMALIZER:NORMALIZATION', 'Normalization failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        % データタイプの判別（既存のまま）
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
        
        % 2D配列データの正規化
        function [normalizedData, normParams] = normalize2DArrayData(~, data, method)
            try
                [numChannels, numSamples] = size(data);
                if any([numChannels, numSamples] == 0)
                    error('Invalid array dimensions');
                end
                
                % 正規化パラメータの計算
                switch method
                    case 'zscore'
                        normParams.mean = mean(data, 2);
                        normParams.std = std(data, 0, 2);
                        
                        % 標準偏差が0のチャンネルの処理
                        zeroStdChannels = normParams.std == 0;
                        if any(zeroStdChannels)
                            warning('EEGNORMALIZER:ZEROSTD', ...
                                'Channels with zero standard deviation detected. Using std = 1.');
                            normParams.std(zeroStdChannels) = 1;
                        end
                        
                        % 正規化の適用
                        normalizedData = (data - normParams.mean) ./ normParams.std;
                        
                    case 'minmax'
                        normParams.min = min(data, [], 2);
                        normParams.max = max(data, [], 2);
                        
                        % 最大値と最小値が同じチャンネルの処理
                        zeroRangeChannels = normParams.max == normParams.min;
                        if any(zeroRangeChannels)
                            warning('EEGNORMALIZER:ZERORANGE', ...
                                'Channels with zero range detected. Using range = 1.');
                            normParams.max(zeroRangeChannels) = normParams.min(zeroRangeChannels) + 1;
                        end
                        
                        % 正規化の適用
                        normalizedData = (data - normParams.min) ./ (normParams.max - normParams.min);
                        
                    case 'robust'
                        normParams.median = median(data, 2);
                        normParams.mad = mad(data, 1, 2);
                        
                        % MADが0のチャンネルの処理
                        zeroMadChannels = normParams.mad == 0;
                        if any(zeroMadChannels)
                            warning('EEGNORMALIZER:ZEROMAD', ...
                                'Channels with zero MAD detected. Using MAD = 1.');
                            normParams.mad(zeroMadChannels) = 1;
                        end
                        
                        % 正規化の適用
                        normalizedData = (data - normParams.median) ./ normParams.mad;
                        
                    otherwise
                        error('EEGNORMALIZER:INVALIDMETHOD', ...
                            'Invalid normalization method: %s', method);
                end
                
            catch ME
                error('EEGNORMALIZER:2DARRAYNORM', ...
                    'Error in 2D array data normalization: %s', ME.message);
            end
        end
        
        % 3D配列データの正規化
        function [normalizedData, normParams] = normalizeArrayData(obj, data, method)
            try
                [numChannels, numSamples, numEpochs] = size(data);
                if any([numChannels, numSamples, numEpochs] == 0)
                    error('Invalid array dimensions');
                end
                
                % チャンネルデータを2D配列に変換
                reshapedData = reshape(data, numChannels, []);
                
                % 2D配列として正規化
                [normalizedReshapedData, normParams] = obj.normalize2DArrayData(reshapedData, method);
                
                % 3D形状に戻す
                normalizedData = reshape(normalizedReshapedData, numChannels, numSamples, numEpochs);
                
            catch ME
                error('EEGNORMALIZER:ARRAYNORM', ...
                    'Error in array data normalization: %s', ME.message);
            end
        end
        
        % セルデータの正規化
        function [normalizedData, normParams] = normalizeCellData(obj, cellData, method)
            try
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
                [normalizedReshapedData, normParams] = obj.normalize2DArrayData(reshapedData, method);
                
                % セル配列に戻す
                normalizedData = cell(size(cellData));
                for ep = 1:numEpochs
                    startIdx = (ep-1) * numSamplesPerEpoch + 1;
                    endIdx = ep * numSamplesPerEpoch;
                    normalizedData{ep} = normalizedReshapedData(:, startIdx:endIdx);
                end
                
            catch ME
                error('EEGNORMALIZER:CELLNORM', ...
                    'Error in cell data normalization: %s', ME.message);
            end
        end
    end
end