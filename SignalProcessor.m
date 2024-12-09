classdef SignalProcessor < handle
    properties (Access = private)
        params              % 設定パラメータ
        rawData            % 生データ
        preprocessedData   % 前処理済みデータ
        epochs             % エポック分割されたデータ
        labels             % トリガー情報 (value, time, sampleを含むstruct配列)
        epochLabels        % エポック化後のラベル情報
        filterDesigner     % フィルタ設計器
        notchFilterDesigner % ノッチフィルタ設計器
        normalizer         % 正規化器
        processingInfo     % 処理情報
        processedData      % 処理済みデータ
        processedLabel     % 処理済みラベル
    end
    
    methods (Access = public)
        function obj = SignalProcessor(params)
            obj.params = params;
            obj.filterDesigner = FIRFilterDesigner(params);
            obj.notchFilterDesigner = NotchFilterDesigner(params);
            obj.normalizer = EEGNormalizer(params);
            obj.initializeProcessingInfo();
        end
        
        function [processedData, processedLabel, processingInfo] = process(obj, eegData, labels)
            try
                % 処理情報の初期化
                obj.initializeProcessingInfo();
                
                % データの検証
                if isempty(eegData)
                    error('SIGNALPROCESSOR:EMPTYDATA', 'Input EEG data is empty');
                end
                
                % 入力データ情報の記録
                obj.updateInputDataInfo(eegData, labels);
                
                % データの設定と前処理
                obj.rawData = eegData;
                obj.preprocess(eegData);
                    
                % 正規化処理
                if obj.params.signal.normalize.enabled
                    switch obj.params.signal.normalize.type
                        case 'epoch'
                            obj.normalizePerEpoch();
                        case 'all'
                            [obj.preprocessedData, normParams] = obj.normalizer.normalize(obj.preprocessedData);
                            obj.processingInfo.normalize = struct(...
                                'type', 'all', ...
                                'method', obj.params.signal.normalize.method, ...
                                'normParams', normParams);
                    end
                end
                
                if ~isempty(labels)
                    % トリガー情報が指定されている場合のエポック処理
                    obj.labels = labels;
                    obj.epoching(labels);
                    
                    % データ拡張処理
                    if obj.params.signal.augmentation.enabled
                        obj.augmentData(obj.epochs);
                    end
                    
                    obj.organizeData();
                else
                    % オンライン処理またはトリガーなしの場合
                    obj.processedData = obj.preprocessedData;
                    obj.processedLabel = [];  % 明示的に空を設定
                end

                % 処理情報の最終更新
                obj.updateFinalProcessingInfo();
                
                % 結果の設定
                processedData = obj.processedData;
                processedLabel = obj.processedLabel;
                processingInfo = obj.processingInfo;
                
            catch ME
                error('Signal processing failed: %s', ME.message);
            end
        end
        
        function preprocessedData = preprocess(obj, data)
            try
                tmpData = data;
                
                % ノッチフィルタ処理
                if obj.params.signal.filter.notch.enabled
                    [tmpData, notchInfo] = obj.notchFilterDesigner.designAndApplyFilter(tmpData);
                    obj.processingInfo.filter.notch = notchInfo;
                    obj.processingInfo.processingOrder{end+1} = 'notch_filter';
                end
                
                % FIRフィルタ処理
                if obj.params.signal.filter.fir.enabled
                    [tmpData, firInfo] = obj.filterDesigner.designAndApplyFilter(tmpData);
                    obj.processingInfo.filter.fir = firInfo;
                    obj.processingInfo.processingOrder{end+1} = 'fir_filter';
                end
                
                obj.preprocessedData = tmpData;
                preprocessedData = obj.preprocessedData;
                
                % フィルタ処理情報の更新
                obj.updateFilteringInfo();
                
            catch ME
                error('SIGNALPROCESSOR:PREPROCESS', 'Preprocessing failed: %s', ME.message);
            end
        end
        
        function normalizedData = normalizeOnline(obj, data, normParams)
            % オンライン処理用の正規化関数
            try
                if obj.params.signal.normalize.enabled && ~isempty(data) && ~isempty(normParams)
                    % 正規化パラメータを使用してデータを正規化
                    switch obj.params.signal.normalize.method
                        case 'zscore'
                            normalizedData = (data - normParams.mean) ./ normParams.std;
                        case 'minmax'
                            normalizedData = (data - normParams.min) ./ (normParams.max - normParams.min);
                        case 'robust'
                            normalizedData = (data - normParams.median) ./ normParams.mad;
                        otherwise
                            error('SIGNALPROCESSOR:NORMALIZE', '未知の正規化方法: %s', obj.params.signal.normalize.method);
                    end
                    
                    % 処理情報の更新
                    obj.processingInfo.normalize.appliedParams = normParams;
                    obj.processingInfo.normalize.method = obj.params.signal.normalize.method;
                    obj.processingInfo.processingOrder{end+1} = 'normalize_online';
                else
                    normalizedData = data;
                    if obj.params.signal.normalize.enabled
                        warning('正規化パラメータが指定されていません．正規化をスキップします．');
                    end
                end
            catch ME
                error('SIGNALPROCESSOR:NORMALIZE', '正規化処理に失敗しました: %s', ME.message);
            end
        end
        
        function visualizeProcessing(obj)
            if isempty(obj.processedData)
                error('No processed data available for visualization');
            end
            
            % 生データと処理済みデータの比較プロット
            figure('Name', 'Signal Processing Results');
            
            subplot(2,1,1);
            plot(obj.rawData(1,:));
            title('Raw EEG Data (Channel 1)');
            xlabel('Samples');
            ylabel('Amplitude');
            grid on;
            
            subplot(2,1,2);
            plot(obj.preprocessedData(1,:));
            title('Preprocessed EEG Data (Channel 1)');
            xlabel('Samples');
            ylabel('Amplitude');
            grid on;
            
            % フィルタ特性の可視化
            if isfield(obj.processingInfo.filter, 'fir')
                obj.filterDesigner.visualizeFilter(obj.processingInfo.filter.fir.filter);
            end
            
            % ノッチフィルタの特性表示
            if isfield(obj.processingInfo.filter, 'notch')
                figure('Name', 'Notch Filter Response');
                notchInfo = obj.processingInfo.filter.notch;
                if isfield(notchInfo, 'response')
                    plot(notchInfo.response.frequency, notchInfo.response.magnitude);
                    title('Notch Filter Response');
                    xlabel('Frequency (Hz)');
                    ylabel('Magnitude (dB)');
                    grid on;
                end
            end
        end
        
        function dataInfo = getProcessingInfo(obj)
            dataInfo = obj.processingInfo;
        end
    end
    
    methods (Access = private)
        function initializeProcessingInfo(obj)
            % 処理情報の初期化
            obj.processingInfo = struct(...
                'startTime', datetime('now'), ...
                'deviceInfo', struct(...
                    'name', obj.params.device.name, ...
                    'sampleRate', obj.params.device.sampleRate, ...
                    'channelCount', obj.params.device.channelCount, ...
                    'channels', {obj.params.device.channels}), ...
                'filter', struct(), ...
                'normalize', struct(), ...
                'epoching', struct(), ...
                'augmentation', struct(), ...
                'processingOrder', {{}}, ...
                'version', '1.0');
        end
        
        function updateInputDataInfo(obj, eegData, labels)
            % 入力データ情報の記録
            obj.processingInfo.inputData = struct(...
                'size', size(eegData), ...
                'duration', size(eegData, 2) / obj.params.device.sampleRate, ...
                'numChannels', size(eegData, 1));
            
            if ~isempty(labels)
                obj.processingInfo.inputData.numLabels = length(labels);
                obj.processingInfo.inputData.uniqueLabels = unique([labels.value]);
            end
        end
        
        function updateFilteringInfo(obj)
            % フィルタ処理情報の更新
            obj.processingInfo.filter.parameters = struct(...
                'notch', obj.params.signal.filter.notch, ...
                'fir', obj.params.signal.filter.fir);
        end
        
        function epoching(obj, labels)
            if isempty(labels)
                obj.epochs = {};
                obj.epochLabels = [];
                return;
            end
            
            nTrials = size(labels, 2);
            analysisWindow = obj.params.signal.window.analysis;
            stimulusWindow = obj.params.signal.window.stimulus;
            overlapRatio = obj.params.signal.epoch.overlap;
            
            fs = obj.params.device.sampleRate;
            nChannels = size(obj.preprocessedData, 1);
            samplesPerEpoch = round(fs * analysisWindow);
            
            stepSize = analysisWindow * (1 - overlapRatio);
            nSteps = floor((stimulusWindow - analysisWindow) / stepSize) + 1;
            
            % エポック化の方法を選択
            if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                obj.epochingCell(nTrials, nSteps, fs, samplesPerEpoch, stepSize, labels);
            else
                obj.epochingArray(nTrials, nSteps, nChannels, fs, samplesPerEpoch, stepSize, labels);
            end
            
            % エポック化情報の記録
            obj.processingInfo.epoching = struct(...
                'numTrials', nTrials, ...
                'analysisWindow', analysisWindow, ...
                'stimulusWindow', stimulusWindow, ...
                'overlapRatio', overlapRatio, ...
                'samplesPerEpoch', samplesPerEpoch, ...
                'stepSize', stepSize, ...
                'nSteps', nSteps);
            
            obj.processingInfo.processingOrder{end+1} = 'epoching';
        end
        
        function epochingCell(obj, nTrials, nSteps, fs, samplesPerEpoch, stepSize, labels)
            obj.epochs = cell(nTrials * nSteps, 1);
            obj.epochLabels = zeros(nTrials * nSteps, 1);
            
            epochIndex = 1;
            for trial = 1:nTrials
                startIdx = labels(trial).sample;
                
                for step = 1:nSteps
                    shiftAmount = round((step - 1) * fs * stepSize);
                    epochStartIdx = startIdx + shiftAmount;
                    epochEndIdx = epochStartIdx + samplesPerEpoch - 1;
                    
                    if epochEndIdx <= size(obj.preprocessedData, 2)
                        obj.epochs{epochIndex} = obj.preprocessedData(:, epochStartIdx:epochEndIdx);
                        obj.epochLabels(epochIndex) = labels(trial).value;
                        epochIndex = epochIndex + 1;
                    end
                end
            end
            
            obj.epochs = obj.epochs(1:epochIndex-1);
            obj.epochLabels = obj.epochLabels(1:epochIndex-1);
        end
        
        function epochingArray(obj, nTrials, nSteps, nChannels, fs, samplesPerEpoch, stepSize, labels)
            maxEpochs = nTrials * nSteps;
            obj.epochs = zeros(nChannels, samplesPerEpoch, maxEpochs);
            obj.epochLabels = zeros(maxEpochs, 1);
            
            epochIndex = 1;
            for trial = 1:nTrials
                startIdx = labels(trial).sample;
                
                for step = 1:nSteps
                    shiftAmount = round((step - 1) * fs * stepSize);
                    epochStartIdx = startIdx + shiftAmount;
                    epochEndIdx = epochStartIdx + samplesPerEpoch - 1;
                    
                    if epochEndIdx <= size(obj.preprocessedData, 2)
                        obj.epochs(:, :, epochIndex) = obj.preprocessedData(:, epochStartIdx:epochEndIdx);
                        obj.epochLabels(epochIndex) = labels(trial).value;
                        epochIndex = epochIndex + 1;
                    end
                end
            end
            
            obj.epochs = obj.epochs(:, :, 1:epochIndex-1);
            obj.epochLabels = obj.epochLabels(1:epochIndex-1);
        end
        
        function augmentData(obj, data)
            if iscell(data)
                [augData, augLabels] = obj.applyAugmentationCell(data, obj.epochLabels);
            else
                [augData, augLabels] = obj.applyAugmentationArray(data, obj.epochLabels);
            end
            
            obj.epochs = augData;
            obj.epochLabels = augLabels;
            
            % データ拡張情報の記録
            obj.processingInfo.augmentation = struct(...
                'enabled', true, ...
                'numAugmentations', obj.params.signal.augmentation.numAugmentations, ...
                'maxShiftRatio', obj.params.signal.augmentation.maxShiftRatio, ...
                'noiseLevel', obj.params.signal.augmentation.noiseLevel, ...
                'resultSize', size(augData));
            
            obj.processingInfo.processingOrder{end+1} = 'augmentation';
        end
        
        function [augmented_data, augmented_labels] = applyAugmentationCell(obj, data, labels)
            nTrials = length(data);
            nAug = obj.params.signal.augmentation.numAugmentations;
            maxShiftRatio = obj.params.signal.augmentation.maxShiftRatio;
            noiseLevel = obj.params.signal.augmentation.noiseLevel;
            
            totalTrials = nTrials * (nAug + 1);
            augmented_data = cell(totalTrials, 1);
            augmented_labels = zeros(totalTrials, 1);
            
            % オリジナルデータのコピー
            augmented_data(1:nTrials) = data;
            augmented_labels(1:nTrials) = labels;
            
            idx = nTrials + 1;
            for i = 1:nTrials
                orig_data = data{i};
                [nChannels, nSamples] = size(orig_data);
                max_shift = round(nSamples * maxShiftRatio);
                label = labels(i);
                
                for j = 1:nAug
                    aug_data = zeros(nChannels, nSamples);
                    
                    for ch = 1:nChannels
                        % シフトとノイズの適用
                        shift = randi([-max_shift, max_shift]);
                        shifted_data = circshift(orig_data(ch,:), shift);
                        noise = noiseLevel * std(orig_data(ch,:)) * randn(1, nSamples);
                        aug_data(ch,:) = shifted_data + noise;
                    end
                    
                    augmented_data{idx} = aug_data;
                    augmented_labels(idx) = label;
                    idx = idx + 1;
                end
            end
        end
        
        function [augmented_data, augmented_labels] = applyAugmentationArray(obj, data, labels)
            [nChannels, nSamples, nTrials] = size(data);
            nAug = obj.params.signal.augmentation.numAugmentations;
            maxShiftRatio = obj.params.signal.augmentation.maxShiftRatio;
            noiseLevel = obj.params.signal.augmentation.noiseLevel;
            
            totalTrials = nTrials * (nAug + 1);
            augmented_data = zeros(nChannels, nSamples, totalTrials);
            augmented_labels = zeros(totalTrials, 1);
            
            % オリジナルデータのコピー
            augmented_data(:, :, 1:nTrials) = data;
            augmented_labels(1:nTrials) = labels;
            
            max_shift = round(nSamples * maxShiftRatio);
            idx = nTrials + 1;
            
            for i = 1:nTrials
                orig_data = data(:, :, i);
                label = labels(i);
                
                for j = 1:nAug
                    aug_data = zeros(nChannels, nSamples);
                    
                    for ch = 1:nChannels
                        % シフトとノイズの適用
                        shift = randi([-max_shift, max_shift]);
                        shifted_data = circshift(orig_data(ch,:), shift);
                        noise = noiseLevel * std(orig_data(ch,:)) * randn(1, nSamples);
                        aug_data(ch,:) = shifted_data + noise;
                    end
                    
                    augmented_data(:, :, idx) = aug_data;
                    augmented_labels(idx) = label;
                    idx = idx + 1;
                end
            end
        end
        
        function normalizePerEpoch(obj)
            try
                if iscell(obj.epochs)
                    for i = 1:length(obj.epochs)
                        [obj.epochs{i}, normParams] = obj.normalizer.normalize(obj.epochs{i});
                        obj.processingInfo.normalize.perEpoch{i} = normParams;
                    end
                else
                    for i = 1:size(obj.epochs, 3)
                        [normalizedEpoch, normParams] = obj.normalizer.normalize(obj.epochs(:,:,i));
                        obj.epochs(:,:,i) = normalizedEpoch;
                        obj.processingInfo.normalize.perEpoch{i} = normParams;
                    end
                end
                
                obj.processingInfo.normalize.type = 'epoch';
                obj.processingInfo.normalize.method = obj.params.signal.normalize.method;
                obj.processingInfo.processingOrder{end+1} = 'normalize_per_epoch';
                
            catch ME
                warning('SIGNALPROCESSOR:NORMALIZATION', 'Per-epoch normalization failed: %s', ME.message);
            end
        end
        
        function organizeData(obj)
            % エポック化されたデータとラベルを処理済みデータとして設定
            obj.processedData = obj.epochs;
            obj.processedLabel = obj.epochLabels;
        end
        
        function updateFinalProcessingInfo(obj)
            % 最終的な処理情報の更新
            obj.processingInfo.endTime = datetime('now');
            obj.processingInfo.processingDuration = seconds(obj.processingInfo.endTime - obj.processingInfo.startTime);
            
            if ~isempty(obj.processedData)
                if iscell(obj.processedData)
                    obj.processingInfo.outputData.format = 'cell';
                    obj.processingInfo.outputData.numEpochs = length(obj.processedData);
                    if ~isempty(obj.processedData)
                        obj.processingInfo.outputData.epochSize = size(obj.processedData{1});
                    end
                else
                    obj.processingInfo.outputData.format = 'array';
                    obj.processingInfo.outputData.size = size(obj.processedData);
                end
            end
            
            % 信号処理パラメータの記録
            obj.processingInfo.parameters = obj.params.signal;
        end
    end
end