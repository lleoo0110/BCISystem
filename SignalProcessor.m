classdef SignalProcessor < handle
    properties (Access = private)
        params              % 設定パラメータ
        rawData            % 生データ
        preprocessedData   % 前処理済みデータ
        epochs             % エポック分割されたデータ
        labels        % トリガー情報 (value, time, sampleを含むstruct配列)
        epochLabels        % エポック化後のラベル情報
        filterDesigner     % フィルタ設計器
        notchFilterDesigner % ノッチフィルタ設計器
        normalizer         % 正規化器
        processingInfo     % 処理情報（フィルタ情報を含む）
        processedData      % 処理済みデータ
        processedLabel    % 処理済みラベル
    end
    
    methods (Access = public)
        function obj = SignalProcessor(params)
            obj.params = params;
            obj.filterDesigner = FIRFilterDesigner(params);
            obj.notchFilterDesigner = NotchFilterDesigner(params);
            obj.normalizer = EEGNormalizer(params);
            obj.processingInfo = struct();
        end
        
        function [processedData, processedLabel, processingInfo] = process(obj, eegData, labels)
            try
                % 入力データの検証
                if isempty(eegData)
                    error('SIGNALPROCESSOR:EMPTYDATA', 'Input EEG data is empty');
                end

                % データの設定と前処理
                obj.rawData = eegData;
                obj.preprocess(eegData);  % フィルタリング

                if ~isempty(labels)
                    % トリガー情報が指定されている場合のみエポック分割を実行
                    obj.labels = labels;  % トリガー情報を保存
                    obj.epoching(labels);  % エポック分割

                    % データ拡張（エポック単位）
                    if isfield(obj.params.signal, 'augmentation') && ...
                       isfield(obj.params.signal.augmentation, 'enabled') && ...
                       obj.params.signal.augmentation.enabled
                        obj.augmentData(obj.epochs);
                    end

                    % 正規化処理
                    if isfield(obj.params.signal, 'normalize') && ...
                       isfield(obj.params.signal.normalize, 'enabled') && ...
                       obj.params.signal.normalize.enabled
                        if obj.params.signal.normalize.perEpoch
                            obj.normalizePerEpoch();  % エポックごとの正規化
                        else
                            obj.normalizeAllData();   % 全データの正規化
                        end
                    end

                    obj.organizeData();  % エポック化されたデータの整理
                else
                    % オンライン処理またはトリガーなしの場合
                    obj.processedData = obj.preprocessedData;
                    obj.processedLabel = [];  % 明示的に空を設定
                end

                % 処理情報の記録
                obj.processingInfo.processedTime = datetime('now');
                obj.processingInfo.dataSize = size(obj.processedData);
                if ~isempty(labels)
                    obj.processingInfo.numTrials = size(labels, 1);
                else
                    obj.processingInfo.numTrials = 0;
                end
                
                 % 処理結果の設定
                processedData = obj.processedData;
                processedLabel = obj.processedLabel;
                processingInfo = obj.processingInfo;

            catch ME
                % エラーを再スロー
                error('Signal processing failed: %s', ME.message);
            end
        end
        
        function preprocessedData = preprocess(obj, data)
            try
                tmpData = data;
                obj.processingInfo = struct();
                obj.processingInfo.processingOrder = {};

                % ノッチフィルタの適用
                if obj.params.signal.filter.notch.enabled
                    [tmpData, notchInfo] = obj.notchFilterDesigner.designAndApplyFilter(tmpData);
                    obj.processingInfo.notch = notchInfo;
                    obj.processingInfo.processingOrder{end+1} = 'notch';
                end

                % FIRフィルタの適用
                if obj.params.signal.filter.fir.enabled
                    [tmpData, firInfo] = obj.filterDesigner.designAndApplyFilter(tmpData);
                    obj.processingInfo.fir = firInfo;
                    obj.processingInfo.processingOrder{end+1} = 'fir';
                end
                
                obj.preprocessedData = tmpData;
                preprocessedData = obj.preprocessedData;
            catch ME
                error('SIGNALPROCESSOR:PREPROCESS', ...
                    'Preprocessing failed: %s', ME.message);
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
            if isfield(obj.processingInfo, 'fir')
                obj.filterDesigner.visualizeFilter(obj.processingInfo.fir.filter);
            end

            % ノッチフィルタの特性表示
            if isfield(obj.processingInfo, 'notch')
                figure('Name', 'Notch Filter Response');
                for i = 1:length(obj.processingInfo.notch.notchFilters)
                    notchInfo = obj.processingInfo.notch.notchFilters{i};
                    subplot(length(obj.processingInfo.notch.notchFilters), 1, i);
                    plot(notchInfo.response.frequency, notchInfo.response.magnitude);
                    title(sprintf('Notch Filter at %d Hz', notchInfo.frequency));
                    xlabel('Frequency (Hz)');
                    ylabel('Magnitude (dB)');
                    grid on;
                end
            end

            % 正規化情報の表示
            if isfield(obj.processingInfo, 'normalize')
                figure('Name', 'Normalization Parameters');
                if isfield(obj.processingInfo.normalize, 'perEpoch')
                    % エポックごとの正規化の場合
                    lastEpoch = obj.processingInfo.normalize.perEpoch{end};
                    subplot(2,1,1);
                    bar(lastEpoch.mean);
                    title('Channel Means (Last Epoch)');
                    xlabel('Channel');
                    ylabel('Mean Value');
                    grid on;

                    subplot(2,1,2);
                    bar(lastEpoch.std);
                    title('Channel Standard Deviations (Last Epoch)');
                    xlabel('Channel');
                    ylabel('Std Value');
                    grid on;
                else
                    % 全データの正規化の場合
                    subplot(2,1,1);
                    bar(obj.processingInfo.normalize.normParams.mean);
                    title('Channel Means (All Data)');
                    xlabel('Channel');
                    ylabel('Mean Value');
                    grid on;

                    subplot(2,1,2);
                    bar(obj.processingInfo.normalize.normParams.std);
                    title('Channel Standard Deviations (All Data)');
                    xlabel('Channel');
                    ylabel('Std Value');
                    grid on;
                end
            end
        end
        
        function dataInfo = getProcessingInfo(obj)
            dataInfo = struct();
            if ~isempty(obj.processedData)
                if iscell(obj.processedData)
                    dataInfo.numTrials = length(obj.processedData);
                    dataInfo.epochLength = size(obj.processedData{1}, 2);
                    dataInfo.numChannels = size(obj.processedData{1}, 1);
                else
                    dataInfo.numTrials = size(obj.processedData, 3);
                    dataInfo.epochLength = size(obj.processedData, 2);
                    dataInfo.numChannels = size(obj.processedData, 1);
                end
                if ~isempty(obj.labels)
                    dataInfo.uniqueLabels = unique([obj.labels.value]);
                end
                dataInfo.samplingRate = obj.params.device.sampleRate;
                dataInfo.totalDuration = dataInfo.epochLength / dataInfo.samplingRate;
                dataInfo.processingInfo = obj.processingInfo;
            else
                error('SIGNALPROCESSOR:NODATA', 'No processed data available');
            end
        end
    end
    
    methods (Access = private)        
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
        end
        
        function epochingCell(obj, nTrials, nSteps, fs, samplesPerEpoch, stepSize, labels)
            obj.epochs = cell(nTrials * nSteps, 1);
            obj.epochLabels = zeros(nTrials * nSteps, 1);
            
            epochIndex = 1;
            for trial = 1:nTrials
                % トリガー情報から直接サンプル位置を使用
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
                % トリガー情報から直接サンプル位置を使用
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
           obj.processingInfo.processingOrder{end+1} = 'augmentation';
       end
       
       function organizeData(obj)
           % エポック化されたデータとラベルを処理済みデータとして設定
           obj.processedData = obj.epochs;
           obj.processedLabel = obj.epochLabels;
       end

       function [augmented_data, augmented_labels] = applyAugmentationCell(obj, data, labels)
           nTrials = length(data);
           nAug = obj.params.signal.augmentation.numAugmentations;
           maxShiftRatio = obj.params.signal.augmentation.maxShiftRatio;
           noiseLevel = obj.params.signal.augmentation.noiseLevel;

           totalTrials = nTrials * (nAug + 1);
           augmented_data = cell(totalTrials, 1);
           augmented_labels = zeros(totalTrials, 1);

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
       
       % エポックごとの正規化
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
               obj.processingInfo.processingOrder{end+1} = 'normalize_per_epoch';
           catch ME
               warning('SIGNALPROCESSOR:NORMALIZATION', ...
                   'Per-epoch normalization failed: %s', ME.message);
           end
       end

       % 全データの正規化
       function normalizeAllData(obj)
           try
               [normalizedData, normParams] = obj.normalizer.normalize(obj.epochs);
               obj.epochs = normalizedData;
               obj.processingInfo.normalize.normParams = normParams;  % 正規化パラメータを保存
               obj.processingInfo.processingOrder{end+1} = 'normalize_all';
           catch ME
               warning('SIGNALPROCESSOR:NORMALIZATION', ...
                   'Global normalization failed: %s', ME.message);
           end
       end
   end
end