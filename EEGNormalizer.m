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
                % データ形式を最初に判別
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
                        error('EEGNORMALIZER:INVALIDINPUT', ...
                            'Invalid input data format');
                end

                % データ形式の情報を追加
                normParams.dataType = dataType;

            catch ME
                error('EEGNORMALIZER:NORMALIZATION', ...
                    'Normalization failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function dataType = getDataType(obj, data)
            if iscell(data)
                dataType = 'cell';
            elseif isnumeric(data) && ndims(data) == 3
                dataType = '3D array';
            elseif isnumeric(data) && ismatrix(data)
                dataType = '2D array';
            else
                error('EEGNORMALIZER:INVALIDINPUT', ...
                    'Invalid input data format');
            end
        end

        function [normalizedData, normParams] = normalizeCellData(obj, cellData)
            try
                numEpochs = length(cellData);
                [numChannels, numSamplesPerEpoch] = size(cellData{1});
                
                normParams.mean = zeros(numChannels, 1);
                normParams.std = zeros(numChannels, 1);
                normalizedData = cell(size(cellData));
                
                for ep = 1:numEpochs
                    normalizedData{ep} = zeros(size(cellData{ep}));
                end
                
                totalSamples = numSamplesPerEpoch * numEpochs;
                
                for ch = 1:numChannels
                    channelData = zeros(1, totalSamples);
                    startIdx = 1;
                    
                    for ep = 1:numEpochs
                        endIdx = startIdx + numSamplesPerEpoch - 1;
                        channelData(startIdx:endIdx) = cellData{ep}(ch, :);
                        startIdx = endIdx + 1;
                    end
                    
                    normParams.mean(ch) = mean(channelData);
                    normParams.std(ch) = std(channelData);
                    
                    if normParams.std(ch) == 0
                        warning('EEGNORMALIZER:ZEROSTD', ...
                            'Channel %d has zero standard deviation. Using std = 1.', ch);
                        normParams.std(ch) = 1;
                    end
                    
                    for ep = 1:numEpochs
                        normalizedData{ep}(ch, :) = ...
                            (cellData{ep}(ch, :) - normParams.mean(ch)) ./ normParams.std(ch);
                    end
                end
                
            catch ME
                error('EEGNORMALIZER:CELLNORM', ...
                    'Error in cell data normalization: %s', ME.message);
            end
        end

        function [normalizedData, normParams] = normalizeArrayData(obj, arrayData)
            try
                [numChannels, numSamples, numEpochs] = size(arrayData);
                if any([numChannels, numSamples, numEpochs] == 0)
                    error('Invalid array dimensions');
                end
                
                normParams.mean = zeros(numChannels, 1);
                normParams.std = zeros(numChannels, 1);
                normalizedData = zeros(size(arrayData));
                
                for ch = 1:numChannels
                    channelData = reshape(arrayData(ch, :, :), 1, []);
                    normParams.mean(ch) = mean(channelData);
                    normParams.std(ch) = std(channelData);
                    
                    if normParams.std(ch) == 0
                        warning('EEGNORMALIZER:ZEROSTD', ...
                            'Channel %d has zero standard deviation. Using std = 1.', ch);
                        normParams.std(ch) = 1;
                    end
                    
                    normalizedData(ch, :, :) = ...
                        (arrayData(ch, :, :) - normParams.mean(ch)) ./ normParams.std(ch);
                end
                
            catch ME
                error('EEGNORMALIZER:ARRAYNORM', ...
                    'Error in array data normalization: %s', ME.message);
            end
        end
        
        function [normalizedData, normParams] = normalize2DArrayData(obj, arrayData)
            try
                [numChannels, numSamples] = size(arrayData);
                if any([numChannels, numSamples] == 0)
                    error('Invalid array dimensions');
                end
                
                normParams.mean = mean(arrayData, 2);
                normParams.std = std(arrayData, 0, 2);
                
                zeroStdChannels = normParams.std == 0;
                if any(zeroStdChannels)
                    warning('EEGNORMALIZER:ZEROSTD', ...
                        'Channels with zero standard deviation detected. Using std = 1.');
                    normParams.std(zeroStdChannels) = 1;
                end
                
                normalizedData = zeros(size(arrayData));
                for ch = 1:numChannels
                    normalizedData(ch, :) = ...
                        (arrayData(ch, :) - normParams.mean(ch)) ./ normParams.std(ch);
                end
                
            catch ME
                error('EEGNORMALIZER:2DARRAYNORM', ...
                    'Error in 2D array data normalization: %s', ME.message);
            end
        end
    end
end