classdef Epoching < handle
    % Epochingクラス：時系列データのエポック分割を行うクラス
    properties (Access = private)
        params          % 設定パラメータを格納する構造体
        epochs          % エポック分割されたデータを格納する変数 (cellまたはarray)
        epochLabels     % エポック化後のラベル情報を格納する変数（valueのみ、数値またはカテゴリカル）
        processingInfo  % エポック分割処理に関する情報を格納する構造体
    end

    methods (Access = public)
        function obj = Epoching(params)
            % Epochingコンストラクタ：Epochingクラスのインスタンスを作成し、パラメータを設定、処理情報を初期化する
            % Input:
            %   params: 設定パラメータを格納した構造体。以下のフィールドを含む必要がある:
            %       - device.sampleRate: サンプリングレート (Hz)
            %       - signal.epoch.method: エポック分割方法 ('time' or 'odd-even')
            %       - signal.epoch.overlap: 時間窓エポック分割時のオーバーラップ率 (0 ~ 1)
            %       - signal.epoch.storageType: エポックデータの保存形式 ('cell' or 'array')
            %       - signal.window.epochDuration: 各エポックの時間長 (秒) ※旧signal.window.analysis
            %       - signal.window.timeRange: 刺激提示からの時間範囲 
            %                                  単一範囲: [開始時間, 終了時間] (秒)
            %                                  複数範囲: {[開始時間1, 終了時間1], [開始時間2, 終了時間2], ...}

            obj.params = params; % パラメータ構造体をオブジェクトのプロパティに格納
            obj.initializeProcessingInfo(); % 処理情報構造体を初期化
        end

        function [epochs, epochLabels, epochInfo] = epoching(obj, data, labels)
            % epochingメソッド：データとラベルを基にエポック分割を実行し、結果と処理情報を返す
            % Input:
            %   data: エポック分割対象のデータ (チャンネル x サンプル)
            %   labels: ラベル情報 (構造体配列)。各要素は少なくとも .sample (サンプルインデックス) と .value (ラベル値) フィールドを持つ必要がある
            % Output:
            %   epochs: エポック分割されたデータ (cell配列またはarray)。storageTypeの設定に依存
            %   epochLabels: エポックに対応するラベル (数値またはカテゴリカル配列)
            %   epochInfo: エポック分割処理に関する情報 (構造体)

            try
                if isempty(labels)
                    % ラベルが空の場合、空のエポックとラベルを返す
                    epochs = {};
                    epochLabels = [];
                    epochInfo = obj.processingInfo.epoching;
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
                epochInfo = obj.processingInfo.epoching;

            catch ME
                error('Epoch creation failed: %s', ME.message);
            end
        end

        function info = getProcessingInfo(obj)
            % getProcessingInfoメソッド：現在の処理情報を返す
            % Output:
            %   info: 処理情報構造体 (processingInfoプロパティの内容)
            info = obj.processingInfo;
        end
    end

    methods (Access = private)
        
        function timeRanges = standardizeTimeRanges(~, inputRanges)
            % 時間範囲を標準形式に変換（セル配列形式）
            % Input:
            %   inputRanges: 入力時間範囲 (数値配列または複数配列を含むセル配列)
            % Output:
            %   timeRanges: 標準形式の時間範囲 (セル配列形式)
            
            if isempty(inputRanges)
                timeRanges = {};
                return;
            end
            
            % セル配列でない場合はセル配列に変換
            if ~iscell(inputRanges)
                if size(inputRanges, 1) == 1 && size(inputRanges, 2) == 2
                    % 単一の時間範囲 [start, end]
                    timeRanges = {inputRanges};
                elseif size(inputRanges, 1) > 1 && size(inputRanges, 2) == 2
                    % 複数の時間範囲 [[start1, end1]; [start2, end2]; ...]
                    timeRanges = cell(size(inputRanges, 1), 1);
                    for i = 1:size(inputRanges, 1)
                        timeRanges{i} = inputRanges(i, :);
                    end
                else
                    error('Invalid time range format. Expected [start, end] or multiple [start, end] rows.');
                end
            else
                % セル配列の場合は各要素をチェック
                timeRanges = inputRanges;
                for i = 1:length(timeRanges)
                    range = timeRanges{i};
                    if ~isnumeric(range) || numel(range) ~= 2
                        error('Invalid time range in cell array. Expected [start, end] for each range.');
                    end
                end
            end
        end
        
        function initializeProcessingInfo(obj)
            % initializeProcessingInfoメソッド：processingInfo構造体を初期化する
            % エポック化処理に関する情報を格納するための構造体を初期化
            epochInfo = struct(...
                'method', '', ...               % エポック分割方法 ('time', 'odd-even', 'time-multi' など)
                'epochDuration', 0, ...         % エポック時間長 (秒)
                'timeRanges', {{}}, ...         % 時間範囲 (複数の[開始時間, 終了時間]を格納)
                'overlapRatio', 0, ...          % オーバーラップ率 ('time'メソッド)
                'samplesPerEpoch', 0, ...       % 1エポックあたりのサンプル数 ('time'メソッド)
                'stepSize', 0, ...              % ステップサイズ (サンプル数, 'time'メソッド)
                'nSteps', [], ...               % 各時間範囲のステップ数 ('time'メソッド)
                'totalEpochs', 0, ...           % 総エポック数
                'storageType', '', ...          % エポックデータの保存形式 ('cell' or 'array')
                'epochTimes', [], ...           % 各エポックの開始・終了時間 (秒, [nEpochs x 2]行列)
                'rangeIndices', []);            % 各エポックが属する時間範囲のインデックス

            % processingInfo構造体の作成。epochingフィールドにepochInfo構造体を格納
            obj.processingInfo.epoching = epochInfo;
        end

        function epochingByTime(obj, data, labels)
            % epochingByTimeメソッド：時間窓ベースのエポック分割を実行する。複数の時間範囲に対応
            % Input:
            %   data: エポック分割対象のデータ (チャンネル x サンプル)
            %   labels: ラベル情報 (構造体配列)。各要素は少なくとも .sample (サンプルインデックス) と .value (ラベル値) フィールドを持つ必要がある

            try
                % 共通パラメータを取得
                fs = obj.params.device.sampleRate;
                overlapRatio = obj.params.signal.epoch.overlap;
                epochDuration = obj.params.signal.window.epochDuration;
                samplesPerEpoch = round(fs * epochDuration);
                stepSize = round(epochDuration * (1 - overlapRatio) * fs);

<<<<<<< HEAD
                % 視覚タスクが有効かどうかをparamsから確認
                if isfield(obj.params.signal.epoch, 'visual') && obj.params.signal.epoch.visual.enable
                    % 【視覚タスクの場合】
                    % 視覚タスク用パラメータの取得
                    observationDuration = obj.params.signal.epoch.visual.observationDuration; % 観察期間 (秒)
                    signalDuration = obj.params.signal.epoch.visual.signalDuration; % 合図期間 (秒)
                    imageryDuration = obj.params.signal.epoch.visual.imageryDuration; % イメージ期間 (秒)

                    % 各期間における分析窓のシフト（ステップ）サイズを計算
                    stepSize = round(analysisWindow * (1 - overlapRatio) * fs);
                    % 各期間で抽出可能なエポック数（オーバーラップを考慮）を計算
                    nStepsObs = floor((observationDuration - analysisWindow) / (analysisWindow*(1-overlapRatio))) + 1; % 観察期間のステップ数
                    nStepsImg = floor((imageryDuration - analysisWindow) / (analysisWindow*(1-overlapRatio))) + 1; % イメージ期間のステップ数

                    % taskTypes により抽出対象を決定（例: {'observation','imagery'} or 片方のみ）
                    selectedTaskTypes = obj.params.signal.epoch.visual.taskTypes;
                    if ~any(strcmpi(selectedTaskTypes, 'observation'))
                        nStepsObs = 0; % 観察期間を抽出しない場合はステップ数を0に
                    end
                    if ~any(strcmpi(selectedTaskTypes, 'imagery'))
                        nStepsImg = 0; % イメージ期間を抽出しない場合はステップ数を0に
                    end
                    nTrials = length(labels); % トライアル数を取得

                    % 保存形式 (cell or array) に応じてエポック抽出関数を呼び分け
                    if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                        obj.epochingCellVisual(nTrials, fs, samplesPerEpoch, stepSize, nStepsObs, nStepsImg, labels, data); % Cell形式で視覚タスクのエポック抽出
                    else
                        obj.epochingArrayVisual(nTrials, fs, samplesPerEpoch, stepSize, nStepsObs, nStepsImg, size(data,1), labels, data); % Array形式で視覚タスクのエポック抽出
                    end

                    % 処理情報の更新（視覚タスク用）
                    totalEpochs = nTrials * (nStepsObs + nStepsImg); % 総エポック数を計算
                    obj.processingInfo.epoching.method = 'time-visual'; % エポック分割方法を 'time-visual' に設定
                    obj.processingInfo.epoching.analysisWindow = analysisWindow; % 分析窓長を記録
                    obj.processingInfo.epoching.observationDuration = observationDuration; % 観察期間を記録
                    obj.processingInfo.epoching.signalDuration = signalDuration; % 合図期間を記録
                    obj.processingInfo.epoching.imageryDuration = imageryDuration; % イメージ期間を記録
                    obj.processingInfo.epoching.overlapRatio = overlapRatio; % オーバーラップ率を記録
                    obj.processingInfo.epoching.samplesPerEpoch = samplesPerEpoch; % 1エポックあたりサンプル数を記録
                    obj.processingInfo.epoching.stepSize = stepSize; % ステップサイズを記録
                    obj.processingInfo.epoching.nSteps = [nStepsObs, nStepsImg]; % ステップ数 (観察期間, イメージ期間) を記録
                    obj.processingInfo.epoching.totalEpochs = totalEpochs; % 総エポック数を記録

                else
                    % 【非視覚タスクの場合】（通常の時間窓エポック抽出）    
                    startTime = obj.params.signal.window.stimulus(1);
                    endTime = obj.params.signal.window.stimulus(2);
                    stimulusWindow = endTime - startTime; % 刺激提示窓長 (秒) を取得
                    % stimulusWindow = obj.params.signal.window.stimulus;
                    nSteps = floor((stimulusWindow - analysisWindow) / (analysisWindow * (1 - overlapRatio))) + 1; % ステップ数を計算
                    nTrials = length(labels); % トライアル数を取得
                    stepSize = round(analysisWindow * (1 - overlapRatio) * fs); % ステップサイズを計算

                    startOffset = round(startTime * fs);
                    if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                        obj.epochingCell(nTrials, nSteps, fs, samplesPerEpoch, stepSize, labels, data,startOffset); % Cell形式でエポック抽出
                    else
                        obj.epochingArray(nTrials, nSteps, size(data,1), fs, samplesPerEpoch, stepSize, labels, data,startOffset); % Array形式でエポック抽出
                    end

                    obj.updateEpochingInfo(analysisWindow, stimulusWindow, overlapRatio, ...
                        samplesPerEpoch, stepSize, nSteps); % 処理情報を更新 (非視覚タスク用)
=======
                % パラメータの検証
                if ~isnumeric(epochDuration) || epochDuration <= 0
                    error('Invalid epoch duration: must be a positive number');
                end
                
                if ~isnumeric(overlapRatio) || overlapRatio < 0 || overlapRatio >= 1
                    error('Invalid overlap ratio: must be between 0 and 1');
>>>>>>> 77be08a1646b3e6dca6b51b459a238fdd6ad0b8a
                end

                % 時間範囲を取得し、標準形式に変換
                timeRanges = obj.standardizeTimeRanges(obj.params.signal.window.timeRange);
                
                if isempty(timeRanges)
                    error('Time ranges are empty or invalid.');
                end
                
                % 複数時間範囲からのエポック抽出を実行
                obj.epochingMultipleTimeRanges(data, labels, fs, timeRanges, samplesPerEpoch, stepSize);

            catch ME
                error('Time-based epoching failed: %s', ME.message);
            end
        end

        function epochingMultipleTimeRanges(obj, data, labels, fs, timeRanges, samplesPerEpoch, stepSize)
            % epochingMultipleTimeRangesメソッド：複数の時間範囲からエポック分割を実行する
            % Input:
            %   data: エポック分割対象のデータ (チャンネル x サンプル)
            %   labels: ラベル情報 (構造体配列)
            %   fs: サンプリングレート
            %   timeRanges: 時間範囲のセル配列 (各要素は [開始時間, 終了時間])
            %   samplesPerEpoch: 1エポックあたりのサンプル数
            %   stepSize: ステップサイズ (サンプル数)

            % 各時間範囲のステップ数を計算
            nRanges = length(timeRanges);
            nSteps = zeros(nRanges, 1);
            totalDurations = zeros(nRanges, 1);
            overlapRatio = obj.params.signal.epoch.overlap;
            epochDuration = obj.params.signal.window.epochDuration;
            
            for i = 1:nRanges
                timeRange = timeRanges{i};
                totalDurations(i) = timeRange(2) - timeRange(1);
                nSteps(i) = floor((totalDurations(i) - epochDuration) / (epochDuration * (1 - overlapRatio))) + 1;
                
                % 有効なステップ数をチェック
                if nSteps(i) < 0
                    nSteps(i) = 0;
                    warning('Time range %d is too short for the epoch duration. No epochs will be extracted from this range.', i);
                end
            end
            
            % トライアル数
            nTrials = length(labels);
            
            % 最大エポック数を計算
            maxEpochs = nTrials * sum(nSteps);
            
            % storageTypeに応じてエポック抽出方法を選択
            if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                obj.extractMultiRangeEpochsCellFormat(data, labels, nTrials, nRanges, nSteps, timeRanges, samplesPerEpoch, stepSize, fs);
            else
                obj.extractMultiRangeEpochsArrayFormat(data, labels, nTrials, nRanges, nSteps, timeRanges, size(data,1), samplesPerEpoch, stepSize, fs);
            end
            
            % 処理情報の更新
            obj.updateMultiRangeEpochingInfo(epochDuration, timeRanges, nSteps, overlapRatio, samplesPerEpoch, stepSize);
        end

        function extractMultiRangeEpochsCellFormat(obj, data, labels, nTrials, nRanges, nSteps, timeRanges, samplesPerEpoch, stepSize, fs)
            % extractMultiRangeEpochsCellFormatメソッド：複数時間範囲からのエポックをセル配列形式で抽出する
            % Input:
            %   data: エポック分割対象のデータ (チャンネル x サンプル)
            %   labels: ラベル情報 (構造体配列)
            %   nTrials: トライアル数
            %   nRanges: 時間範囲の数
            %   nSteps: 各時間範囲のステップ数の配列
            %   timeRanges: 時間範囲のセル配列 (各要素は [開始時間, 終了時間])
            %   samplesPerEpoch: 1エポックあたりのサンプル数
            %   stepSize: ステップサイズ (サンプル数)
            %   fs: サンプリングレート

            % 最大エポック数を計算
            maxEpochs = nTrials * sum(nSteps);
            
            % セル配列の初期化
            obj.epochs = cell(maxEpochs, 1);
            obj.epochLabels = zeros(maxEpochs, 1);
            epochTimes = zeros(maxEpochs, 2);
            rangeIndices = zeros(maxEpochs, 1); % 各エポックがどの時間範囲から抽出されたかを記録
            
            % エポックのインデックスを初期化
            epochIndex = 1;
            
            % 各トライアルについて処理
            for trial = 1:nTrials
                trialSample = labels(trial).sample;
                trialValue = labels(trial).value;
                
                % 各時間範囲について処理
                for rangeIdx = 1:nRanges
                    timeRange = timeRanges{rangeIdx};
                    rangeSteps = nSteps(rangeIdx);
                    
                    % 時間範囲が有効な場合のみ処理
                    if rangeSteps > 0
                        % 開始サンプルのオフセットを計算
                        startOffsetSamples = round(timeRange(1) * fs);
                        
                        % 各ステップについて処理
                        for step = 1:rangeSteps
                            % エポックの開始・終了サンプルを計算
                            epochStartSample = trialSample + startOffsetSamples + (step-1) * stepSize;
                            epochEndSample = epochStartSample + samplesPerEpoch - 1;
                            
                            % データ範囲内かチェック
                            if epochStartSample >= 1 && epochEndSample <= size(data, 2)
                                % エポックを抽出して格納
                                obj.epochs{epochIndex} = data(:, epochStartSample:epochEndSample);
                                obj.epochLabels(epochIndex) = trialValue;
                                epochTimes(epochIndex, :) = [epochStartSample/fs, epochEndSample/fs];
                                rangeIndices(epochIndex) = rangeIdx;
                                epochIndex = epochIndex + 1;
                            end
                        end
                    end
                end
            end
            
            % 未使用部分の削除
            if epochIndex > 1
                obj.epochs = obj.epochs(1:epochIndex-1);
                obj.epochLabels = obj.epochLabels(1:epochIndex-1);
                epochTimes = epochTimes(1:epochIndex-1, :);
                rangeIndices = rangeIndices(1:epochIndex-1);
            else
                obj.epochs = {};
                obj.epochLabels = [];
                epochTimes = [];
                rangeIndices = [];
                warning('No valid epochs were extracted.');
            end
            
            % 処理情報に格納
            obj.processingInfo.epoching.epochTimes = epochTimes;
            obj.processingInfo.epoching.rangeIndices = rangeIndices;
        end

        function extractMultiRangeEpochsArrayFormat(obj, data, labels, nTrials, nRanges, nSteps, timeRanges, nChannels, samplesPerEpoch, stepSize, fs)
            % extractMultiRangeEpochsArrayFormatメソッド：複数時間範囲からのエポックをnumeric配列形式で抽出する
            % Input:
            %   data: エポック分割対象のデータ (チャンネル x サンプル)
            %   labels: ラベル情報 (構造体配列)
            %   nTrials: トライアル数
            %   nRanges: 時間範囲の数
            %   nSteps: 各時間範囲のステップ数の配列
            %   timeRanges: 時間範囲のセル配列 (各要素は [開始時間, 終了時間])
            %   nChannels: チャンネル数
            %   samplesPerEpoch: 1エポックあたりのサンプル数
            %   stepSize: ステップサイズ (サンプル数)
            %   fs: サンプリングレート

            % 最大エポック数を計算
            maxEpochs = nTrials * sum(nSteps);
            
            % numeric配列の初期化
            obj.epochs = zeros(nChannels, samplesPerEpoch, maxEpochs);
            obj.epochLabels = zeros(maxEpochs, 1);
            epochTimes = zeros(maxEpochs, 2);
            rangeIndices = zeros(maxEpochs, 1); % 各エポックがどの時間範囲から抽出されたかを記録
            
            % エポックのインデックスを初期化
            epochIndex = 1;
            
            % 各トライアルについて処理
            for trial = 1:nTrials
                trialSample = labels(trial).sample;
                trialValue = labels(trial).value;
                
                % 各時間範囲について処理
                for rangeIdx = 1:nRanges
                    timeRange = timeRanges{rangeIdx};
                    rangeSteps = nSteps(rangeIdx);
                    
                    % 時間範囲が有効な場合のみ処理
                    if rangeSteps > 0
                        % 開始サンプルのオフセットを計算
                        startOffsetSamples = round(timeRange(1) * fs);
                        
                        % 各ステップについて処理
                        for step = 1:rangeSteps
                            % エポックの開始・終了サンプルを計算
                            epochStartSample = trialSample + startOffsetSamples + (step-1) * stepSize;
                            epochEndSample = epochStartSample + samplesPerEpoch - 1;
                            
                            % データ範囲内かチェック
                            if epochStartSample >= 1 && epochEndSample <= size(data, 2)
                                % エポックを抽出して格納
                                obj.epochs(:, :, epochIndex) = data(:, epochStartSample:epochEndSample);
                                obj.epochLabels(epochIndex) = trialValue;
                                epochTimes(epochIndex, :) = [epochStartSample /fs, epochEndSample/fs];
                                rangeIndices(epochIndex) = rangeIdx;
                                epochIndex = epochIndex + 1;
                            end
                        end
                    end
                end
            end
            
            % 未使用部分の削除
            if epochIndex > 1
                obj.epochs = obj.epochs(:, :, 1:epochIndex-1);
                obj.epochLabels = obj.epochLabels(1:epochIndex-1);
                epochTimes = epochTimes(1:epochIndex-1, :);
                rangeIndices = rangeIndices(1:epochIndex-1);
            else
                obj.epochs = zeros(nChannels, samplesPerEpoch, 0);
                obj.epochLabels = [];
                epochTimes = [];
                rangeIndices = [];
                warning('No valid epochs were extracted.');
            end
            
            % 処理情報に格納
            obj.processingInfo.epoching.epochTimes = epochTimes;
            obj.processingInfo.epoching.rangeIndices = rangeIndices;
        end

        function updateMultiRangeEpochingInfo(obj, epochDuration, timeRanges, nSteps, overlapRatio, samplesPerEpoch, stepSize)
            % updateMultiRangeEpochingInfoメソッド：複数時間範囲エポック分割の処理情報を更新する
            % Input:
            %   epochDuration: エポック時間長 (秒)
            %   timeRanges: 時間範囲のセル配列
            %   nSteps: 各時間範囲のステップ数の配列
            %   overlapRatio: オーバーラップ率
            %   samplesPerEpoch: 1エポックあたりのサンプル数
            %   stepSize: ステップサイズ (サンプル数)

            obj.processingInfo.epoching.method = 'time-multi';
            obj.processingInfo.epoching.epochDuration = epochDuration;
            obj.processingInfo.epoching.timeRanges = timeRanges;
            obj.processingInfo.epoching.overlapRatio = overlapRatio;
            obj.processingInfo.epoching.samplesPerEpoch = samplesPerEpoch;
            obj.processingInfo.epoching.stepSize = stepSize;
            obj.processingInfo.epoching.nSteps = nSteps;
            obj.processingInfo.epoching.storageType = obj.params.signal.epoch.storageType;
            obj.processingInfo.epoching.totalEpochs = length(obj.epochLabels);
        end

        function epochingByOddEven(obj, data, labels)
            % epochingByOddEvenメソッド：奇数-偶数ペアによるエポック分割を実行する
            % Input:
            %   data: エポック分割対象のデータ (チャンネル x サンプル)
            %   labels: ラベル情報 (構造体配列)。奇数番目のラベルがエポック開始、偶数番目のラベルがエポック終了を示す

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
                    % セル配列形式でのエポック抽出
                    obj.extractOddEvenEpochsCellFormat(data, labels, nPairs, fs);
                else
                    % numeric配列形式でのエポック抽出
                    obj.extractOddEvenEpochsArrayFormat(data, labels, nPairs, fs);
                end

            catch ME
                error('Odd-even epoching failed: %s', ME.message);
            end
        end

        function extractOddEvenEpochsCellFormat(obj, data, labels, nPairs, fs)
            % extractOddEvenEpochsCellFormatメソッド：奇数-偶数ペアのエポックをセル配列形式で抽出する
            % Input:
            %   data: エポック分割対象のデータ (チャンネル x サンプル)
            %   labels: ラベル情報 (構造体配列)
            %   nPairs: ペア数
            %   fs: サンプリングレート

            obj.epochs = cell(nPairs, 1);
            obj.epochLabels = zeros(nPairs, 1);
            epochTimes = zeros(nPairs, 2);

            for pair = 1:nPairs
                oddIdx = 2*pair - 1;
                evenIdx = 2*pair;

                startSample = labels(oddIdx).sample;
                endSample = labels(evenIdx).sample;

                % データ範囲内かチェック
                if startSample >= 1 && endSample <= size(data, 2)
                    obj.epochs{pair} = data(:, startSample:endSample);
                    obj.epochLabels(pair) = labels(oddIdx).value;
                    epochTimes(pair,:) = [startSample/fs, endSample/fs];
                else
                    warning('Pair %d: Epoch outside data range.', pair);
                    obj.epochs{pair} = [];
                    obj.epochLabels(pair) = labels(oddIdx).value;
                    epochTimes(pair,:) = [NaN, NaN];
                end
            end

            % 空のエポックを削除
            validIndices = ~cellfun(@isempty, obj.epochs);
            obj.epochs = obj.epochs(validIndices);
            obj.epochLabels = obj.epochLabels(validIndices);
            epochTimes = epochTimes(validIndices, :);

            obj.processingInfo.epoching.epochTimes = epochTimes;
            obj.processingInfo.epoching.method = 'odd-even';
            obj.processingInfo.epoching.totalEpochs = length(obj.epochs);
            obj.processingInfo.epoching.storageType = 'cell';
        end

        function extractOddEvenEpochsArrayFormat(obj, data, labels, nPairs, fs)
            % extractOddEvenEpochsArrayFormatメソッド：奇数-偶数ペアのエポックをnumeric配列形式で抽出する
            % Input:
            %   data: エポック分割対象のデータ (チャンネル x サンプル)
            %   labels: ラベル情報 (構造体配列)
            %   nPairs: ペア数
            %   fs: サンプリングレート

            % 各ペアの長さを確認
            pairLengths = zeros(nPairs, 1);
            validPairs = false(nPairs, 1);
            
            for pair = 1:nPairs
                oddIdx = 2*pair - 1;
                evenIdx = 2*pair;
                
                startSample = labels(oddIdx).sample;
                endSample = labels(evenIdx).sample;
                
                if startSample >= 1 && endSample <= size(data, 2)
                    pairLengths(pair) = endSample - startSample + 1;
                    validPairs(pair) = true;
                else
                    warning('Pair %d: Epoch outside data range.', pair);
                    pairLengths(pair) = 0;
                    validPairs(pair) = false;
                end
            end
            
            % 有効なペアのみを抽出
            validPairIndices = find(validPairs);
            nValidPairs = length(validPairIndices);
            
            if nValidPairs == 0
                obj.epochs = zeros(size(data, 1), 0, 0);
                obj.epochLabels = [];
                obj.processingInfo.epoching.epochTimes = [];
                warning('No valid epochs were extracted.');
                return;
            end

            % 有効なペアの中で最長のエポック長を決定（または固定長を選択）
            samplesPerEpoch = max(pairLengths(validPairs));
            
            % 配列の初期化
            obj.epochs = zeros(size(data, 1), samplesPerEpoch, nValidPairs);
            obj.epochLabels = zeros(nValidPairs, 1);
            epochTimes = zeros(nValidPairs, 2);
            
            % 有効なペアのデータを抽出
            for i = 1:nValidPairs
                pair = validPairIndices(i);
                oddIdx = 2*pair - 1;
                evenIdx = 2*pair;
                
                startSample = labels(oddIdx).sample;
                endSample = labels(evenIdx).sample;
                currentLength = endSample - startSample + 1;
                
                % 長さの違いを処理（切り詰めまたはゼロパディング）
                if currentLength > samplesPerEpoch
                    obj.epochs(:, :, i) = data(:, startSample:startSample+samplesPerEpoch-1);
                    warningMessage = sprintf('Pair %d: Epoch truncated from %d to %d samples.', pair, currentLength, samplesPerEpoch);
                    warning(warningMessage);
                else
                    obj.epochs(:, 1:currentLength, i) = data(:, startSample:endSample);
                    if currentLength < samplesPerEpoch
                        obj.epochs(:, currentLength+1:end, i) = 0;
                        warningMessage = sprintf('Pair %d: Epoch padded from %d to %d samples.', pair, currentLength, samplesPerEpoch);
                        warning(warningMessage);
                    end
                end
                
                obj.epochLabels(i) = labels(oddIdx).value;
                epochTimes(i,:) = [startSample/fs, endSample/fs];
            end

            obj.processingInfo.epoching.epochTimes = epochTimes;
            obj.processingInfo.epoching.method = 'odd-even';
            obj.processingInfo.epoching.totalEpochs = nValidPairs;
            obj.processingInfo.epoching.samplesPerEpoch = samplesPerEpoch;
            obj.processingInfo.epoching.storageType = 'array';
        end
    end
end