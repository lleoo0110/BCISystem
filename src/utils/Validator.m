classdef Validator < handle
    methods (Static)
        function validateEEGData(data, channelCount, sampleRate)
            % EEGデータの検証
            validateattributes(data, {'numeric'}, ...
                {'2d', 'nrows', channelCount}, ...
                'validateEEGData', 'data');
            
            if size(data, 2) < sampleRate
                error('Validator:InvalidData', ...
                    'Data length must be at least 1 second (%d samples)', sampleRate);
            end
        end
        
        function validateLabels(labels, numSamples)
            % ラベルの検証
            if isempty(labels)
                return;
            end
            
            % 構造体の場合
            if isstruct(labels)
                required = {'value', 'time', 'sample'};
                for i = 1:length(required)
                    if ~isfield(labels, required{i})
                        error('Validator:InvalidLabels', ...
                            'Labels must contain field: %s', required{i});
                    end
                end
                
                % サンプル番号の検証
                samples = [labels.sample];
                if any(samples > numSamples)
                    error('Validator:InvalidLabels', ...
                        'Label sample indices exceed data length');
                end
            end
        end
        
        function validateFeatures(features)
            % 特徴量の検証
            validateattributes(features, {'numeric'}, ...
                {'2d', 'finite', 'nonnan'}, ...
                'validateFeatures', 'features');
        end
        
        function validateConfig(config)
            % 設定の検証
            required = {'acquisition', 'signal', 'feature', 'classifier'};
            for i = 1:length(required)
                if ~isfield(config, required{i})
                    error('Validator:InvalidConfig', ...
                        'Configuration must contain field: %s', required{i});
                end
            end
        end
    end
end