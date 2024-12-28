classdef EmotionExtractor < handle
    properties (Access = private)
        params          % パラメータ設定
        faaExtractor    % FAA特徴抽出器
        abExtractor     % α/β比特徴抽出器
        emotionLabels   % 感情状態ラベル
        neutralLabel    % 中性状態ラベル
        emotionThreshold % 感情判定閾値
        normalizeMethod % 正規化方法
        scalingFactor   % スケーリング係数
    end
    
    methods (Access = public)
        function obj = EmotionExtractor(params)
            obj.params = params;
            obj.faaExtractor = FAAExtractor(params);
            obj.abExtractor = ABRatioExtractor(params);
            
            % 感情分類パラメータの設定
            obj.emotionLabels = params.feature.emotion.labels.states;
            if iscell(obj.emotionLabels{1})
                obj.emotionLabels = obj.emotionLabels{1};
            end
            obj.neutralLabel = params.feature.emotion.labels.neutral;
            obj.emotionThreshold = params.feature.emotion.threshold;
            obj.normalizeMethod = params.feature.emotion.coordinates.normalizeMethod;
            obj.scalingFactor = params.feature.emotion.coordinates.scaling;
        end
        
        function [emotionResults] = classifyEmotion(obj, data)
            try
                if iscell(data)
                    numEpochs = length(data);
                else
                    numEpochs = size(data, 3);
                end
                
                emotionResults = cell(numEpochs, 1);
                
                for epoch = 1:numEpochs
                    if iscell(data)
                        epochData = data{epoch};
                    else
                        epochData = data(:,:,epoch);
                    end
                    
                    % FAA値とα/β比の取得
                    faaResult = obj.faaExtractor.calculateFAA(epochData);
                    [abRatio, ~] = obj.abExtractor.calculateABRatio(epochData);
                    
                    % 感情座標の計算
                    valence = tanh(faaResult{1}.faa * obj.scalingFactor);
                    arousal = -tanh((abRatio - 1) * obj.scalingFactor);
                    
                    % 極座標への変換
                    radius = sqrt(valence^2 + arousal^2);
                    angle = atan2(arousal, valence);
                    
                    % 感情状態の分類
                    emotionState = obj.classifyEmotionState(radius, angle);
                    
                    % 結果の格納
                    emotionResults{epoch} = struct(...
                        'state', emotionState, ...
                        'coordinates', struct(...
                            'valence', valence, ...
                            'arousal', arousal, ...
                            'radius', radius, ...
                            'angle', angle), ...
                        'emotionCoords', obj.getEmotionCoordinates(emotionState));
                end
                
            catch ME
                error('Emotion classification failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function emotionState = classifyEmotionState(obj, radius, angle)
            if radius < obj.emotionThreshold
                emotionState = obj.neutralLabel;
            else
                angles = linspace(-pi, pi, length(obj.emotionLabels));
                angleIdx = find(angle >= angles, 1, 'last');
                if isempty(angleIdx) || angleIdx > length(obj.emotionLabels)
                    angleIdx = 1;
                end
                emotionState = obj.emotionLabels{angleIdx};
            end
        end
        
        function coords = getEmotionCoordinates(obj, emotionState)
            emotionMap = containers.Map(...
                obj.emotionLabels, ...
                {[0 0 0 0], [100 0 0 100], [100 0 0 0], [100 100 0 0], ...
                [0 100 0 0], [0 100 100 0], [0 0 100 0], [0 0 100 100], [0 0 0 100]});
            
            if emotionMap.isKey(emotionState)
                coords = emotionMap(emotionState);
            else
                coords = [0 0 0 0];
            end
        end
    end
end