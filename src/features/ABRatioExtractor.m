classdef ABRatioExtractor < handle
    properties (Access = private)
        params          % パラメータ設定
        powerExtractor  % パワー計算用オブジェクト
        abRatioThreshold % α/β比の閾値
        channels        % 使用するチャンネル
    end
    
    methods (Access = public)
        function obj = ABRatioExtractor(params)
            obj.params = params;
            obj.powerExtractor = PowerExtractor(params);
            obj.abRatioThreshold = params.feature.abRatio.threshold;
            obj.channels = [params.feature.abRatio.channels.left, ...
                          params.feature.abRatio.channels.right];
        end
        
        function [abRatio, arousalState] = calculateABRatio(obj, data)
            try
                % アルファ波とベータ波のパワーを計算
                alphaPower = obj.powerExtractor.calculatePower(data(obj.channels,:), [8 13]);
                betaPower = obj.powerExtractor.calculatePower(data(obj.channels,:), [13 30]);
                
                % α/β比の計算
                abRatio = mean(alphaPower) / mean(betaPower);
                
                % 覚醒状態の判定
                if abRatio < obj.abRatioThreshold
                    arousalState = 'arousal';
                else
                    arousalState = 'no-arousal';
                end
                
            catch ME
                error('AB ratio calculation failed: %s', ME.message);
            end
        end
    end
end