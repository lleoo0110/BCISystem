classdef Epoching < handle
    properties (Access = private)
        params          % 設定パラメータ
        epochs          % エポック分割されたデータ
        epochLabels     % エポック化後のラベル情報（valueのみ）
        processingInfo  % 処理情報
    end
    
    methods (Access = public)
        function obj = Epoching(params)
            % コンストラクタ
            obj.params = params;
            obj.initializeProcessingInfo();
        end
        
        function [epochs, epochLabels, epochInfo] = epoching(obj, data, labels)
            try
                if isempty(labels)
                    epochs = {};
                    epochLabels = [];
                    return;
                end
                
                % エポック化方法の取得
                epochMethod = obj.params.signal.epoch.method;
                
                switch lower(epochMethod)
                    case 'time'
                        % 時間窓ベースのエポック化（視覚タスクの場合も内部で分岐）
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
                epochInfo = obj.processingInfo.epoching;
                
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
                'observationDuration', 0, ...
                'signalDuration', 0, ...
                'imageryDuration', 0, ...
                'overlapRatio', 0, ...
                'samplesPerEpoch', 0, ...
                'stepSize', 0, ...
                'nSteps', 0, ...           % 通常はスカラー（非視覚の場合）
                'totalEpochs', 0, ...
                'storageType', '', ...
                'epochTimes', []);
            
            % processingInfo構造体の作成
            obj.processingInfo.epoching = epochInfo;
        end

        function epochingByTime(obj, data, labels)
            try
                fs = obj.params.device.sampleRate;
                overlapRatio = obj.params.signal.epoch.overlap;
                analysisWindow = obj.params.signal.window.analysis; % 例: 2秒
                samplesPerEpoch = round(fs * analysisWindow);
                
                % 視覚タスクが有効なら視覚用エポック抽出を実行
                if isfield(obj.params.signal.epoch, 'visual') && obj.params.signal.epoch.visual.enable
                    % 視覚用パラメータの取得
                    observationDuration = obj.params.signal.epoch.visual.observationDuration;
                    signalDuration = obj.params.signal.epoch.visual.signalDuration;
                    imageryDuration = obj.params.signal.epoch.visual.imageryDuration;
                    
                    % 各期間における分析窓のシフト（ステップ）サイズ
                    stepSize = round(analysisWindow * (1 - overlapRatio) * fs);
                    % 各期間で抽出可能なエポック数（オーバーラップ考慮）
                    nStepsObs = floor((observationDuration - analysisWindow) / (analysisWindow*(1-overlapRatio))) + 1;
                    nStepsImg = floor((imageryDuration - analysisWindow) / (analysisWindow*(1-overlapRatio))) + 1;
                    
                    % taskTypes により抽出対象を決定（例: {'observation','imagery'} or 片方のみ）
                    selectedTaskTypes = obj.params.signal.epoch.visual.taskTypes;
                    if ~any(strcmpi(selectedTaskTypes, 'observation'))
                        nStepsObs = 0;
                    end
                    if ~any(strcmpi(selectedTaskTypes, 'imagery'))
                        nStepsImg = 0;
                    end
                    nTrials = length(labels);
                    
                    % 保存形式に応じた処理
                    if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                        obj.epochingCellVisual(nTrials, fs, samplesPerEpoch, stepSize, nStepsObs, nStepsImg, labels, data);
                    else
                        obj.epochingArrayVisual(nTrials, fs, samplesPerEpoch, stepSize, nStepsObs, nStepsImg, size(data,1), labels, data);
                    end
                    
                    % 処理情報の更新（視覚タスク用）
                    totalEpochs = nTrials * (nStepsObs + nStepsImg);
                    obj.processingInfo.epoching.method = 'time-visual';
                    obj.processingInfo.epoching.analysisWindow = analysisWindow;
                    obj.processingInfo.epoching.observationDuration = observationDuration;
                    obj.processingInfo.epoching.signalDuration = signalDuration;
                    obj.processingInfo.epoching.imageryDuration = imageryDuration;
                    obj.processingInfo.epoching.overlapRatio = overlapRatio;
                    obj.processingInfo.epoching.samplesPerEpoch = samplesPerEpoch;
                    obj.processingInfo.epoching.stepSize = stepSize;
                    obj.processingInfo.epoching.nSteps = [nStepsObs, nStepsImg];
                    obj.processingInfo.epoching.totalEpochs = totalEpochs;
                    
                else
                    % 通常の時間窓エポック抽出（非視覚タスク）
                    stimulusWindow = obj.params.signal.window.stimulus;
                    nSteps = floor((stimulusWindow - analysisWindow) / (analysisWindow * (1 - overlapRatio))) + 1;
                    nTrials = length(labels);
                    stepSize = round(analysisWindow * (1 - overlapRatio) * fs);
                    
                    if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                        obj.epochingCell(nTrials, nSteps, fs, samplesPerEpoch, stepSize, labels, data);
                    else
                        obj.epochingArray(nTrials, nSteps, size(data,1), fs, samplesPerEpoch, stepSize, labels, data);
                    end
                    
                    obj.updateEpochingInfo(analysisWindow, stimulusWindow, overlapRatio, ...
                        samplesPerEpoch, stepSize, nSteps);
                end

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
                    % processedLabelは値のみ
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
        
        %% --- 視覚タスク用のエポック抽出（Cell形式） ---
        function epochingCellVisual(obj, nTrials, fs, samplesPerEpoch, stepSize, nStepsObs, nStepsImg, labels, data)
            % taskTypes の設定に基づき抽出するか判定
            selectedTaskTypes = obj.params.signal.epoch.visual.taskTypes;
            extractObservation = any(strcmpi(selectedTaskTypes, 'observation'));
            extractImagery     = any(strcmpi(selectedTaskTypes, 'imagery'));
            
            totalEpochs = nTrials * (nStepsObs + nStepsImg);
            obj.epochs = cell(totalEpochs, 1);
            % processedLabelは value のみ（数値や文字列）
            obj.epochLabels = zeros(totalEpochs, 1);
            epochTimes = zeros(totalEpochs, 2);
            
            observationDuration = obj.params.signal.epoch.visual.observationDuration;
            signalDuration = obj.params.signal.epoch.visual.signalDuration;
            imageryDuration = obj.params.signal.epoch.visual.imageryDuration;
            
            epochIndex = 1;
            for trial = 1:nTrials
                trialStart = labels(trial).sample;
                trialValue = labels(trial).value;
                
                % 【観察期間】(抽出対象の場合のみ)
                if extractObservation
                    obsStart = trialStart;
                    obsEnd = trialStart + round(observationDuration * fs) - 1;
                    for step = 1:nStepsObs
                        currentStart = obsStart + (step-1)*stepSize;
                        currentEnd = currentStart + samplesPerEpoch - 1;
                        if currentEnd <= obsEnd
                            obj.epochs{epochIndex} = data(:, currentStart:currentEnd);
                            % processedLabel は値のみ
                            obj.epochLabels(epochIndex) = trialValue;
                            epochTimes(epochIndex,:) = [currentStart/fs, currentEnd/fs];
                            epochIndex = epochIndex + 1;
                        end
                    end
                end
                
                % 【イメージ期間】(抽出対象の場合のみ、合図期間はスキップ)
                if extractImagery
                    imgStart = trialStart + round((observationDuration + signalDuration) * fs);
                    imgEnd = imgStart + round(imageryDuration * fs) - 1;
                    for step = 1:nStepsImg
                        currentStart = imgStart + (step-1)*stepSize;
                        currentEnd = currentStart + samplesPerEpoch - 1;
                        if currentEnd <= imgEnd
                            obj.epochs{epochIndex} = data(:, currentStart:currentEnd);
                            obj.epochLabels(epochIndex) = trialValue;
                            epochTimes(epochIndex,:) = [currentStart/fs, currentEnd/fs];
                            epochIndex = epochIndex + 1;
                        end
                    end
                end
            end
            
            % 未使用部分の削除
            obj.epochs = obj.epochs(1:epochIndex-1);
            obj.epochLabels = obj.epochLabels(1:epochIndex-1);
            obj.processingInfo.epoching.epochTimes = epochTimes(1:epochIndex-1,:);
        end
        
        %% --- 視覚タスク用のエポック抽出（Array形式） ---
        function epochingArrayVisual(obj, nTrials, fs, samplesPerEpoch, stepSize, nStepsObs, nStepsImg, nChannels, labels, data)
            % taskTypes の設定に基づき抽出するか判定
            selectedTaskTypes = obj.params.signal.epoch.visual.taskTypes;
            extractObservation = any(strcmpi(selectedTaskTypes, 'observation'));
            extractImagery     = any(strcmpi(selectedTaskTypes, 'imagery'));
            
            maxEpochs = nTrials * (nStepsObs + nStepsImg);
            obj.epochs = zeros(nChannels, samplesPerEpoch, maxEpochs);
            % processedLabelは value のみ
            epochLabelsTmp = zeros(maxEpochs, 1);
            epochTimes = zeros(maxEpochs, 2);
            
            observationDuration = obj.params.signal.epoch.visual.observationDuration;
            signalDuration = obj.params.signal.epoch.visual.signalDuration;
            imageryDuration = obj.params.signal.epoch.visual.imageryDuration;
            
            epochIndex = 1;
            for trial = 1:nTrials
                trialStart = labels(trial).sample;
                trialValue = labels(trial).value;
                
                % 【観察期間】(抽出対象の場合のみ)
                if extractObservation
                    obsStart = trialStart;
                    obsEnd = trialStart + round(observationDuration * fs) - 1;
                    for step = 1:nStepsObs
                        currentStart = obsStart + (step-1)*stepSize;
                        currentEnd = currentStart + samplesPerEpoch - 1;
                        if currentEnd <= obsEnd
                            obj.epochs(:, :, epochIndex) = data(:, currentStart:currentEnd);
                            epochLabelsTmp(epochIndex) = trialValue;
                            epochTimes(epochIndex,:) = [currentStart/fs, currentEnd/fs];
                            epochIndex = epochIndex + 1;
                        end
                    end
                end
                
                % 【イメージ期間】(抽出対象の場合のみ)
                if extractImagery
                    imgStart = trialStart + round((observationDuration + signalDuration) * fs);
                    imgEnd = imgStart + round(imageryDuration * fs) - 1;
                    for step = 1:nStepsImg
                        currentStart = imgStart + (step-1)*stepSize;
                        currentEnd = currentStart + samplesPerEpoch - 1;
                        if currentEnd <= imgEnd
                            obj.epochs(:, :, epochIndex) = data(:, currentStart:currentEnd);
                            epochLabelsTmp(epochIndex) = trialValue;
                            epochTimes(epochIndex,:) = [currentStart/fs, currentEnd/fs];
                            epochIndex = epochIndex + 1;
                        end
                    end
                end
            end
            
            % 未使用部分の削除
            obj.epochs = obj.epochs(:, :, 1:epochIndex-1);
            epochLabelsTmp = epochLabelsTmp(1:epochIndex-1);
            obj.epochLabels = epochLabelsTmp;
            obj.processingInfo.epoching.epochTimes = epochTimes(1:epochIndex-1,:);
        end
        
    end
end