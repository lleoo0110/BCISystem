classdef EOGExtractor < handle
    properties (Access = private)
        params              % システムパラメータ
        filterCoeffs       % フィルタ係数
        baselineBuffer     % ベースライン計算用バッファ
        lastDirection      % 前回の検出方向
        directionHistory   % 方向履歴
        channelIndices     % チャンネルインデックス
    end
    
    methods (Access = public)
        function obj = EOGExtractor(params)
            obj.params = params;
            
            % チャンネルインデックスの設定
            obj.channelIndices = struct(...
                'primary', struct(...
                    'left', params.device.eog.pairs.primary.left, ...
                    'right', params.device.eog.pairs.primary.right ...
                ), ...
                'secondary', struct(...
                    'left', params.device.eog.pairs.secondary.left, ...
                    'right', params.device.eog.pairs.secondary.right ...
                ) ...
            );
            
            % フィルタと各種バッファの初期化
            obj.initializeFilters();
            obj.resetBuffers();
            
            % 設定情報の表示
            obj.displayConfiguration();
        end
        
        function direction = detectGazeDirection(obj, data)
            try
                % プライマリペアからのEOG信号抽出
                eogSignal = obj.extractEOGSignal(data, 'primary');
                
                % 信号が不明確な場合はセカンダリペアを使用
                if ~obj.isValidSignal(eogSignal)
                    eogSignal = obj.extractEOGSignal(data, 'secondary');
                end
                
                % フィルタリングとベースライン補正
                eogSignal = obj.filterEOGSignal(eogSignal);
                eogSignal = obj.correctBaseline(eogSignal);
                
                % 瞬きの検出と除去
                if obj.detectBlink(eogSignal)
                    direction = obj.lastDirection;  % 瞬き中は前回の方向を維持
                    return;
                end
                
                % 方向検出
                direction = obj.determineDirection(eogSignal);
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
                direction = 'error';
                eogSignal = [];
            end
        end
    end
    
    methods (Access = private)
        function initializeFilters(obj)
            % バンドパスフィルタの設計
            fs = obj.params.device.sampleRate;
            nyquist = fs/2;
            
            % フィルタパラメータの取得
            filterParams = obj.params.acquisition.eog.filter;
            [b, a] = butter(filterParams.order, ...
                [filterParams.bandpass.low filterParams.bandpass.high]/nyquist, ...
                'bandpass');
            
            obj.filterCoeffs = struct('b', b, 'a', a);
        end
        
        function eogSignal = extractEOGSignal(obj, data, pairType)
            % 指定されたペアのインデックスを使用して信号を抽出
            leftIdx = obj.channelIndices.(pairType).left;
            rightIdx = obj.channelIndices.(pairType).right;
            
            % インデックスの範囲チェック
            if leftIdx > size(data,1) || rightIdx > size(data,1)
                error('チャンネルインデックスが範囲外です: left=%d, right=%d', leftIdx, rightIdx);
            end
            
            % 左右の電極間の電位差を計算
            eogSignal = data(leftIdx, :) - data(rightIdx, :);
        end
        
        function resetBuffers(obj)
            % バッファの初期化
            bufferSize = round(obj.params.device.sampleRate * 1); % 1秒分
            obj.baselineBuffer = zeros(1, bufferSize);
            obj.directionHistory = struct('time', [], 'direction', {}, 'confidence', []);
            obj.lastDirection = 'center';
        end
        
        function valid = isValidSignal(~, signal)
            % 信号の妥当性チェック
            valid = all(abs(signal) < 200) && std(signal) > 0.1;
        end
        
        function filteredSignal = filterEOGSignal(obj, signal)
            % EOG信号のフィルタリング
            filteredSignal = filtfilt(obj.filterCoeffs.b, obj.filterCoeffs.a, signal);
        end
        
        function correctedSignal = correctBaseline(obj, signal)
            % ベースライン補正
            windowSize = round(obj.params.device.sampleRate * ...
                obj.params.acquisition.eog.baseline);
            baseline = movmean(signal, windowSize);
            correctedSignal = signal - baseline;
        end
        
        function [direction, confidence] = determineDirection(obj, signal)
            % 閾値の取得
            thresholds = obj.params.acquisition.eog.thresholds;
            
            % 方向判定のロジック
            meanAmplitude = mean(signal);
            peakAmplitude = max(abs(signal));
            
            % 信頼度の計算
            confidence = min(1, abs(meanAmplitude) / ...
                max(abs([thresholds.left, thresholds.right])));
            
            % 方向の判定
            if meanAmplitude < thresholds.left
                direction = 'left';
            elseif meanAmplitude > thresholds.right
                direction = 'right';
            else
                direction = 'center';
                confidence = 1 - confidence;
            end
            
            % 時間的な安定性の考慮
            if strcmp(direction, obj.lastDirection)
                confidence = min(1, confidence * 1.2);
            else
                confidence = confidence * 0.8;
            end
            
            obj.lastDirection = direction;
        end
        
        function isBlink = detectBlink(obj, signal)
            % 瞬きの検出
            derivSignal = diff(signal);
            threshold = obj.params.acquisition.eog.thresholds.blink;
            
            isBlink = any(abs(derivSignal) > threshold) && ...
                     max(signal) - min(signal) > threshold;
        end
        
        function displayConfiguration(obj)
            fprintf('\n=== EOG Configuration ===\n');
            fprintf('Primary pair - Left: %d, Right: %d\n', ...
                obj.channelIndices.primary.left, ...
                obj.channelIndices.primary.right);
            fprintf('Secondary pair - Left: %d, Right: %d\n', ...
                obj.channelIndices.secondary.left, ...
                obj.channelIndices.secondary.right);
            fprintf('\nThresholds:\n');
            fprintf('  Left: %d μV\n', obj.params.acquisition.eog.thresholds.left);
            fprintf('  Right: %d μV\n', obj.params.acquisition.eog.thresholds.right);
            fprintf('  Blink: %d μV\n', obj.params.acquisition.eog.thresholds.blink);
            fprintf('===========================\n\n');
        end
    end
end