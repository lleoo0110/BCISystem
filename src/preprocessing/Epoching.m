classdef Epoching < handle
    properties (Access = private)
        params          % 設定パラメータ
        epochs          % エポック分割されたデータ
        epochLabels     % エポック化後のラベル情報
        processingInfo  % 処理情報
        baselineWindow  % ベースライン区間
    end
    
    methods (Access = public)
        function obj = Epoching(params)
            % コンストラクタ
            obj.params = params;
            
            % エポックパラメータの初期化
            if isfield(params.signal.epoch, 'baseline') && ~isempty(params.signal.epoch.baseline)
                obj.baselineWindow = params.signal.epoch.baseline;
            else
                obj.baselineWindow = [-0.5 0];
            end
            
            % 処理情報の初期化
            obj.initializeProcessingInfo();
        end
        
        function [epochs, epochLabels, epochTimes] = epoching(obj, data, labels)
            try
                if isempty(labels)
                    epochs = {};
                    epochLabels = [];
                    epochTimes = [];
                    return;
                end
                
                % エポック化方法の取得
                epochMethod = obj.params.signal.epoch.method;
                
                switch lower(epochMethod)
                    case 'time'
                        % 時間窓ベースのエポック化
                        obj.epochingByTime(data, labels);
                    case 'odd-even'
                        % 奇数-偶数ペアによるエポック化
                        obj.epochingByOddEven(data, labels);
                    otherwise
                        error('Unknown epoching method: %s', epochMethod);
                end
                
                % 結果の返却
                epochs = obj.epochs;
                epochLabels = obj.epochLabels;
                epochTimes = obj.processingInfo.epoching.epochTimes;
                
            catch ME
                error('Epoch creation failed: %s', ME.message);
            end
        end

        function info = getProcessingInfo(obj)
            % 処理情報を返すメソッド
            info = obj.processingInfo;
        end
    end
    
    methods (Access = private)
        function initializeProcessingInfo(obj)
            % エポック化処理情報の初期化
            epochInfo = struct(...
                'method', '', ...
                'analysisWindow', 0, ...
                'stimulusWindow', 0, ...
                'overlapRatio', 0, ...
                'samplesPerEpoch', 0, ...
                'stepSize', 0, ...
                'nSteps', 0, ...
                'totalEpochs', 0, ...
                'storageType', '', ...
                'epochTimes', []);
            
            % processingInfo構造体の作成
            obj.processingInfo.epoching = epochInfo;
        end

        function epochingByTime(obj, data, labels)
            try
                % パラメータの取得
                analysisWindow = obj.params.signal.window.analysis;
                stimulusWindow = obj.params.signal.window.stimulus;
                overlapRatio = obj.params.signal.epoch.overlap;
                fs = obj.params.device.sampleRate;

                % サンプル数の計算
                samplesPerEpoch = round(fs * analysisWindow);
                stepSize = round(analysisWindow * (1 - overlapRatio) * fs);

                % 1トリガーあたりのステップ数を計算
                nSteps = floor((stimulusWindow - analysisWindow) / (analysisWindow * (1 - overlapRatio))) + 1;

                % 総エポック数の計算
                nTrials = length(labels);

                % 保存形式に応じた処理
                if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                    obj.epochingCell(nTrials, nSteps, fs, samplesPerEpoch, stepSize, labels, data);
                else
                    obj.epochingArray(nTrials, nSteps, size(data,1), fs, samplesPerEpoch, stepSize, labels, data);
                end

                % 処理情報の更新
                obj.updateEpochingInfo(analysisWindow, stimulusWindow, overlapRatio, ...
                    samplesPerEpoch, stepSize, nSteps);

            catch ME
                error('Time-based epoching failed: %s', ME.message);
            end
        end
        
        function epochingByOddEven(obj, data, labels)
            try
                % ラベルの数を確認
                nLabels = length(labels);
                if mod(nLabels, 2) ~= 0
                    warning('Odd number of labels. Last label will be ignored.');
                    nLabels = nLabels - 1;
                end

                % ペア数の計算
                nPairs = floor(nLabels / 2);
                fs = obj.params.device.sampleRate;

                if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                    % セル配列形式での初期化
                    obj.epochs = cell(nPairs, 1);
                    obj.epochLabels = zeros(nPairs, 1);
                    epochTimes = zeros(nPairs, 2);

                    for pair = 1:nPairs
                        oddIdx = 2*pair - 1;
                        evenIdx = 2*pair;

                        startSample = labels(oddIdx).sample;
                        endSample = labels(evenIdx).sample;

                        obj.epochs{pair} = data(:, startSample:endSample);
                        obj.epochLabels(pair) = labels(oddIdx).value;
                        epochTimes(pair,:) = [startSample/fs, endSample/fs];
                    end
                else
                    % 配列形式での初期化
                    firstStart = labels(1).sample;
                    lastEnd = labels(2).sample;
                    samplesPerEpoch = lastEnd - firstStart + 1;

                    obj.epochs = zeros(size(data, 1), samplesPerEpoch, nPairs);
                    obj.epochLabels = zeros(nPairs, 1);
                    epochTimes = zeros(nPairs, 2);

                    for pair = 1:nPairs
                        oddIdx = 2*pair - 1;
                        evenIdx = 2*pair;

                        startSample = labels(oddIdx).sample;
                        endSample = labels(evenIdx).sample;

                        obj.epochs(:, :, pair) = data(:, startSample:endSample);
                        obj.epochLabels(pair) = labels(oddIdx).value;
                        epochTimes(pair,:) = [startSample/fs, endSample/fs];
                    end
                end

                % 処理情報の更新
                obj.processingInfo.epoching.epochTimes = epochTimes;
                obj.processingInfo.epoching.method = 'odd-even';
                obj.processingInfo.epoching.totalEpochs = nPairs;

            catch ME
                error('Odd-even epoching failed: %s', ME.message);
            end
        end
        
        function epochingCell(obj, nTrials, nSteps, fs, samplesPerEpoch, stepSize, labels, data)
            obj.epochs = cell(nTrials * nSteps, 1);
            obj.epochLabels = zeros(nTrials * nSteps, 1);
            epochTimes = zeros(nTrials * nSteps, 2);
            
            epochIndex = 1;
            for trial = 1:nTrials
                startSample = labels(trial).sample;
                
                for step = 1:nSteps
                    epochStartSample = startSample + (step-1) * stepSize;
                    epochEndSample = epochStartSample + samplesPerEpoch - 1;
                    
                    if epochEndSample <= size(data, 2)
                        obj.epochs{epochIndex} = data(:, epochStartSample:epochEndSample);
                        obj.epochLabels(epochIndex) = labels(trial).value;
                        epochTimes(epochIndex,:) = [epochStartSample/fs, epochEndSample/fs];
                        epochIndex = epochIndex + 1;
                    end
                end
            end
            
            % 未使用部分の削除
            obj.epochs = obj.epochs(1:epochIndex-1);
            obj.epochLabels = obj.epochLabels(1:epochIndex-1);
            obj.processingInfo.epoching.epochTimes = epochTimes(1:epochIndex-1,:);
        end
        
        function epochingArray(obj, nTrials, nSteps, nChannels, fs, samplesPerEpoch, stepSize, labels, data)
            maxEpochs = nTrials * nSteps;
            obj.epochs = zeros(nChannels, samplesPerEpoch, maxEpochs);
            obj.epochLabels = zeros(maxEpochs, 1);
            epochTimes = zeros(maxEpochs, 2);
            
            epochIndex = 1;
            for trial = 1:nTrials
                startSample = labels(trial).sample;
                
                for step = 1:nSteps
                    epochStartSample = startSample + (step-1) * stepSize;
                    epochEndSample = epochStartSample + samplesPerEpoch - 1;
                    
                    if epochEndSample <= size(data, 2)
                        obj.epochs(:, :, epochIndex) = data(:, epochStartSample:epochEndSample);
                        obj.epochLabels(epochIndex) = labels(trial).value;
                        epochTimes(epochIndex,:) = [epochStartSample/fs, epochEndSample/fs];
                        epochIndex = epochIndex + 1;
                    end
                end
            end
            
            % 未使用部分の削除
            obj.epochs = obj.epochs(:, :, 1:epochIndex-1);
            obj.epochLabels = obj.epochLabels(1:epochIndex-1);
            obj.processingInfo.epoching.epochTimes = epochTimes(1:epochIndex-1,:);
        end

        function updateEpochingInfo(obj, analysisWindow, stimulusWindow, overlapRatio, ...
                samplesPerEpoch, stepSize, nSteps)
            obj.processingInfo.epoching.method = 'time';
            obj.processingInfo.epoching.analysisWindow = analysisWindow;
            obj.processingInfo.epoching.stimulusWindow = stimulusWindow;
            obj.processingInfo.epoching.overlapRatio = overlapRatio;
            obj.processingInfo.epoching.samplesPerEpoch = samplesPerEpoch;
            obj.processingInfo.epoching.stepSize = stepSize;
            obj.processingInfo.epoching.nSteps = nSteps;
            obj.processingInfo.epoching.storageType = obj.params.signal.epoch.storageType;
            obj.processingInfo.epoching.totalEpochs = length(obj.epochLabels);
        end
    end
end