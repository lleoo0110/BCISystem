classdef FAAExtractor < handle
    properties (Access = private)
        params          % パラメータ設定
        powerExtractor  % パワー計算用オブジェクト
        leftChannels   % 左前頭葉のチャンネル番号
        rightChannels  % 右前頭葉のチャンネル番号
        faaThreshold   % FAA判定の閾値
    end
    
    methods (Access = public)
        function obj = FAAExtractor(params)
            obj.params = params;
            obj.powerExtractor = PowerExtractor(params);
            
            % チャンネル設定
            obj.leftChannels = params.feature.faa.channels.left;
            obj.rightChannels = params.feature.faa.channels.right;
            
            % 閾値設定
            obj.faaThreshold = params.feature.faa.threshold;
        end
        
        function [faaResults] = calculateFAA(obj, data)
            try
                if iscell(data)
                    numEpochs = length(data);
                    faaResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        [faaValue, arousalState] = obj.computeFAAForEpoch(data{epoch});
                        faaResults{epoch} = struct(...
                            'faa', faaValue, ...
                            'arousal', arousalState);
                    end
                else
                    numEpochs = size(data, 3);
                    faaResults = cell(numEpochs, 1);
                    
                    for epoch = 1:numEpochs
                        [faaValue, arousalState] = obj.computeFAAForEpoch(data(:,:,epoch));
                        faaResults{epoch} = struct(...
                            'faa', faaValue, ...
                            'arousal', arousalState);
                    end
                end
            catch ME
                error('FAA calculation failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function [faaValue, arousalState] = computeFAAForEpoch(obj, epochData)
            % アルファ帯域のパワー計算
            leftPower = obj.powerExtractor.calculatePower(...
                epochData(obj.leftChannels,:), [8 13]);
            rightPower = obj.powerExtractor.calculatePower(...
                epochData(obj.rightChannels,:), [8 13]);
            
            % FAA値の計算
            leftMean = mean(leftPower);
            rightMean = mean(rightPower);
            faaValue = log(rightMean) - log(leftMean);
            
            % 状態判定
            if faaValue > obj.faaThreshold
                arousalState = 'aroused';
            else
                arousalState = 'non-aroused';
            end
        end
    end
end