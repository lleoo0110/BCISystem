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
            %       - signal.window.analysis: 分析窓の長さ (秒)
            %       - signal.window.stimulus: 刺激提示窓の長さ (秒, 'time'メソッド、非視覚タスク時)
            %       - signal.epoch.visual.enable: 視覚タスクモードの有効/無効フラグ (true/false, 'time'メソッド)
            %       - signal.epoch.visual.observationDuration: 観察期間 (秒, 視覚タスク時)
            %       - signal.epoch.visual.signalDuration: 合図期間 (秒, 視覚タスク時)
            %       - signal.epoch.visual.imageryDuration: イメージ期間 (秒, 視覚タスク時)
            %       - signal.epoch.visual.taskTypes: 抽出するタスクタイプ ({'observation', 'imagery'} など, 視覚タスク時)

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
                    return;
                end

                % エポック化方法の取得 (params.signal.epoch.methodを参照)
                epochMethod = obj.params.signal.epoch.method;

                switch lower(epochMethod)
                    case 'time'
                        % 時間窓ベースのエポック化
                        obj.epochingByTime(data, labels); % 時間窓ベースのエポック分割を実行 (視覚タスク対応)
                    case 'odd-even'
                        % 奇数-偶数ペアによるエポック化
                        obj.epochingByOddEven(data, labels); % 奇数-偶数ペアベースのエポック分割を実行
                    otherwise
                        error('Unknown epoching method: %s', epochMethod); % 不明なエポック化方法が指定された場合はエラー
                end

                % 結果の返却
                epochs = obj.epochs; % エポック分割されたデータをepochs変数に代入
                epochLabels = obj.epochLabels; % エポックラベルをepochLabels変数に代入
                epochInfo = obj.processingInfo.epoching; % エポック分割処理情報をepochInfo変数に代入

            catch ME
                error('Epoch creation failed: %s', ME.message); % エポック作成中にエラーが発生した場合、エラーメッセージを表示
            end
        end

        function info = getProcessingInfo(obj)
            % getProcessingInfoメソッド：現在の処理情報を返す
            % Output:
            %   info: 処理情報構造体 (processingInfoプロパティの内容)
            info = obj.processingInfo; % processingInfoプロパティの内容をinfo変数に代入して返す
        end
    end

    methods (Access = private)
        function initializeProcessingInfo(obj)
            % initializeProcessingInfoメソッド：processingInfo構造体を初期化する
            % エポック化処理に関する情報を格納するための構造体を初期化
            epochInfo = struct(...
                'method', '', ...               % エポック分割方法 ('time', 'odd-even', 'time-visual' など)
                'analysisWindow', 0, ...       % 分析窓長 (秒, 'time'メソッド)
                'stimulusWindow', 0, ...       % 刺激提示窓長 (秒, 'time'メソッド、非視覚タスク時)
                'observationDuration', 0, ...  % 観察期間 (秒, 'time-visual'メソッド)
                'signalDuration', 0, ...       % 合図期間 (秒, 'time-visual'メソッド)
                'imageryDuration', 0, ...      % イメージ期間 (秒, 'time-visual'メソッド)
                'overlapRatio', 0, ...         % オーバーラップ率 ('time'メソッド)
                'samplesPerEpoch', 0, ...      % 1エポックあたりのサンプル数 ('time'メソッド)
                'stepSize', 0, ...            % ステップサイズ (サンプル数, 'time'メソッド)
                'nSteps', 0, ...              % ステップ数 ('time'メソッド, 通常はスカラー、視覚タスク時は[nStepsObs, nStepsImg]ベクトル)
                'totalEpochs', 0, ...         % 総エポック数
                'storageType', '', ...         % エポックデータの保存形式 ('cell' or 'array')
                'epochTimes', []);            % 各エポックの開始・終了時間 (秒, [nEpochs x 2]行列)

            % processingInfo構造体の作成。epochingフィールドにepochInfo構造体を格納
            obj.processingInfo.epoching = epochInfo;
        end

        function epochingByTime(obj, data, labels)
            % epochingByTimeメソッド：時間窓ベースのエポック分割を実行する。視覚タスクと非視覚タスクで処理を分岐
            % Input:
            %   data: エポック分割対象のデータ (チャンネル x サンプル)
            %   labels: ラベル情報 (構造体配列)。各要素は少なくとも .sample (サンプルインデックス) と .value (ラベル値) フィールドを持つ必要がある

            try
                fs = obj.params.device.sampleRate; % サンプリングレートを取得
                overlapRatio = obj.params.signal.epoch.overlap; % オーバーラップ率を取得
                analysisWindow = obj.params.signal.window.analysis; % 分析窓長 (秒) を取得
                samplesPerEpoch = round(fs * analysisWindow); % 1エポックあたりのサンプル数を計算

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

                    startOffset = startTime * fs;
                    if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                        obj.epochingCell(nTrials, nSteps, fs, samplesPerEpoch, stepSize, labels, data,startOffset); % Cell形式でエポック抽出
                    else
                        obj.epochingArray(nTrials, nSteps, size(data,1), fs, samplesPerEpoch, stepSize, labels, data,startOffset); % Array形式でエポック抽出
                    end

                    obj.updateEpochingInfo(analysisWindow, stimulusWindow, overlapRatio, ...
                        samplesPerEpoch, stepSize, nSteps); % 処理情報を更新 (非視覚タスク用)
                end

            catch ME
                error('Time-based epoching failed: %s', ME.message); % 時間窓ベースのエポック分割中にエラーが発生した場合、エラーメッセージを表示
            end
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
                    warning('Odd number of labels. Last label will be ignored.'); % ラベル数が奇数の場合、警告を表示し、最後のラベルを無視
                    nLabels = nLabels - 1; % ラベル数を偶数にする
                end

                % ペア数の計算
                nPairs = floor(nLabels / 2); % ペア数を計算
                fs = obj.params.device.sampleRate; % サンプリングレートを取得

                if strcmpi(obj.params.signal.epoch.storageType, 'cell')
                    % 【Cell形式の場合】
                    % セル配列形式での初期化
                    obj.epochs = cell(nPairs, 1); % エポック格納用セル配列を初期化
                    % processedLabelは値のみ
                    obj.epochLabels = zeros(nPairs, 1); % エポックラベル格納用配列を初期化
                    epochTimes = zeros(nPairs, 2); % エポック時間格納用配列を初期化

                    for pair = 1:nPairs
                        oddIdx = 2*pair - 1; % 奇数番目のラベルインデックス
                        evenIdx = 2*pair; % 偶数番目のラベルインデックス

                        startSample = labels(oddIdx).sample; % エポック開始サンプル
                        endSample = labels(evenIdx).sample; % エポック終了サンプル

                        obj.epochs{pair} = data(:, startSample:endSample); % エポックデータを抽出してセル配列に格納
                        obj.epochLabels(pair) = labels(oddIdx).value; % ラベル値を格納 (奇数番目のラベルの値を採用)
                        epochTimes(pair,:) = [startSample/fs, endSample/fs]; % エポックの開始・終了時間を記録
                    end
                else
                    % 【Array形式の場合】
                    % 配列形式での初期化
                    firstStart = labels(1).sample; % 最初のペアの開始サンプルを取得 (配列サイズ決定のため)
                    lastEnd = labels(2).sample; % 最初のペアの終了サンプルを取得 (配列サイズ決定のため)
                    samplesPerEpoch = lastEnd - firstStart + 1; % 1エポックあたりのサンプル数を計算

                    obj.epochs = zeros(size(data, 1), samplesPerEpoch, nPairs); % エポック格納用配列を初期化
                    obj.epochLabels = zeros(nPairs, 1); % エポックラベル格納用配列を初期化
                    epochTimes = zeros(nPairs, 2); % エポック時間格納用配列を初期化

                    for pair = 1:nPairs
                        oddIdx = 2*pair - 1; % 奇数番目のラベルインデックス
                        evenIdx = 2*pair; % 偶数番目のラベルインデックス

                        startSample = labels(oddIdx).sample; % エポック開始サンプル
                        endSample = labels(evenIdx).sample; % エポック終了サンプル

                        obj.epochs(:, :, pair) = data(:, startSample:endSample); % エポックデータを抽出して配列に格納
                        obj.epochLabels(pair) = labels(oddIdx).value; % ラベル値を格納 (奇数番目のラベルの値を採用)
                        epochTimes(pair,:) = [startSample/fs, endSample/fs]; % エポックの開始・終了時間を記録
                    end
                end

                % 処理情報の更新
                obj.processingInfo.epoching.epochTimes = epochTimes; % エポック時間を記録
                obj.processingInfo.epoching.method = 'odd-even'; % エポック分割方法を 'odd-even' に設定
                obj.processingInfo.epoching.totalEpochs = nPairs; % 総エポック数を記録

            catch ME
                error('Odd-even epoching failed: %s', ME.message); % 奇数-偶数ペアベースのエポック分割中にエラーが発生した場合、エラーメッセージを表示
            end
        end

        function epochingCell(obj, nTrials, nSteps, fs, samplesPerEpoch, stepSize, labels, data,startOffset)
            % epochingCellメソッド：時間窓ベースのエポック分割をCell形式で実行する (非視覚タスク用)
            % Input:
            %   nTrials: トライアル数
            %   nSteps: 各トライアルから抽出するステップ数
            %   fs: サンプリングレート
            %   samplesPerEpoch: 1エポックあたりのサンプル数
            %   stepSize: ステップサイズ (サンプル数)
            %   labels: ラベル情報 (構造体配列)
            %   data: エポック分割対象のデータ (チャンネル x サンプル)

            obj.epochs = cell(nTrials * nSteps, 1); % エポック格納用セル配列を初期化 (最大エポック数で初期化)
            obj.epochLabels = zeros(nTrials * nSteps, 1); % エポックラベル格納用配列を初期化 (最大エポック数で初期化)
            epochTimes = zeros(nTrials * nSteps, 2); % エポック時間格納用配列を初期化 (最大エポック数で初期化)

            epochIndex = 1; % エポックインデックスを初期化
            for trial = 1:nTrials
                startSample = labels(trial).sample + startOffset; % トライアル開始サンプルを取得

                for step = 1:nSteps
                    epochStartSample = startSample + (step-1) * stepSize; % エポック開始サンプルを計算 (ステップごとにずらす)
                    epochEndSample = epochStartSample + samplesPerEpoch - 1; % エポック終了サンプルを計算

                    if epochEndSample <= size(data, 2) % エポックがデータ範囲内かチェック
                        obj.epochs{epochIndex} = data(:, epochStartSample:epochEndSample); % エポックデータを抽出してセル配列に格納
                        obj.epochLabels(epochIndex) = labels(trial).value; % ラベル値を格納
                        epochTimes(epochIndex,:) = [epochStartSample/fs, epochEndSample/fs]; % エポックの開始・終了時間を記録
                        epochIndex = epochIndex + 1; % エポックインデックスをインクリメント
                    end
                end
            end

            % 未使用部分の削除 (初期化時に最大エポック数で確保しているため、実際に抽出されたエポック数に合わせて配列をトリム)
            obj.epochs = obj.epochs(1:epochIndex-1);
            obj.epochLabels = obj.epochLabels(1:epochIndex-1);
            obj.processingInfo.epoching.epochTimes = epochTimes(1:epochIndex-1,:);
        end

        function epochingArray(obj, nTrials, nSteps, nChannels, fs, samplesPerEpoch, stepSize, labels, data,startOffset)
            % epochingArrayメソッド：時間窓ベースのエポック分割をArray形式で実行する (非視覚タスク用)
            % Input:
            %   nTrials: トライアル数
            %   nSteps: 各トライアルから抽出するステップ数
            %   nChannels: チャンネル数
            %   fs: サンプリングレート
            %   samplesPerEpoch: 1エポックあたりのサンプル数
            %   stepSize: ステップサイズ (サンプル数)
            %   labels: ラベル情報 (構造体配列)
            %   data: エポック分割対象のデータ (チャンネル x サンプル)

            maxEpochs = nTrials * nSteps; % 最大エポック数を計算
            obj.epochs = zeros(nChannels, samplesPerEpoch, maxEpochs); % エポック格納用配列を初期化 (最大エポック数で初期化)
            obj.epochLabels = zeros(maxEpochs, 1); % エポックラベル格納用配列を初期化 (最大エポック数で初期化)
            epochTimes = zeros(maxEpochs, 2); % エポック時間格納用配列を初期化 (最大エポック数で初期化)

            epochIndex = 1; % エポックインデックスを初期化
            for trial = 1:nTrials
                startSample = labels(trial).sample - startOffset; % トライアル開始サンプルを取得

                for step = 1:nSteps
                    epochStartSample = startSample + (step-1) * stepSize; % エポック開始サンプルを計算 (ステップごとにずらす)
                    epochEndSample = epochStartSample + samplesPerEpoch - 1; % エポック終了サンプルを計算

                    if epochEndSample <= size(data, 2) % エポックがデータ範囲内かチェック
                        obj.epochs(:, :, epochIndex) = data(:, epochStartSample:epochEndSample); % エポックデータを抽出して配列に格納
                        obj.epochLabels(epochIndex) = labels(trial).value; % ラベル値を格納
                        epochTimes(epochIndex,:) = [epochStartSample/fs, epochEndSample/fs]; % エポックの開始・終了時間を記録
                        epochIndex = epochIndex + 1; % エポックインデックスをインクリメント
                    end
                end
            end

            % 未使用部分の削除 (初期化時に最大エポック数で確保しているため、実際に抽出されたエポック数に合わせて配列をトリム)
            obj.epochs = obj.epochs(:, :, 1:epochIndex-1);
            obj.epochLabels = obj.epochLabels(1:epochIndex-1);
            obj.processingInfo.epoching.epochTimes = epochTimes(1:epochIndex-1,:);
        end

        function updateEpochingInfo(obj, analysisWindow, stimulusWindow, overlapRatio, ...
                samplesPerEpoch, stepSize, nSteps)
            % updateEpochingInfoメソッド：エポック分割処理情報を更新する (非視覚タスク用)
            % Input:
            %   analysisWindow: 分析窓長 (秒)
            %   stimulusWindow: 刺激提示窓長 (秒)
            %   overlapRatio: オーバーラップ率
            %   samplesPerEpoch: 1エポックあたりのサンプル数
            %   stepSize: ステップサイズ (サンプル数)
            %   nSteps: ステップ数

            obj.processingInfo.epoching.method = 'time'; % エポック分割方法を 'time' に設定
            obj.processingInfo.epoching.analysisWindow = analysisWindow; % 分析窓長を記録
            obj.processingInfo.epoching.stimulusWindow = stimulusWindow; % 刺激提示窓長を記録
            obj.processingInfo.epoching.overlapRatio = overlapRatio; % オーバーラップ率を記録
            obj.processingInfo.epoching.samplesPerEpoch = samplesPerEpoch; % 1エポックあたりサンプル数を記録
            obj.processingInfo.epoching.stepSize = stepSize; % ステップサイズを記録
            obj.processingInfo.epoching.nSteps = nSteps; % ステップ数を記録
            obj.processingInfo.epoching.storageType = obj.params.signal.epoch.storageType; % エポックデータ保存形式を記録
            obj.processingInfo.epoching.totalEpochs = length(obj.epochLabels); % 総エポック数を記録 (epochLabelsの長さから算出)
        end

        %% --- 視覚タスク用のエポック抽出（Cell形式） ---
        function epochingCellVisual(obj, nTrials, fs, samplesPerEpoch, stepSize, nStepsObs, nStepsImg, labels, data)
            % epochingCellVisualメソッド：視覚タスク用の時間窓ベースのエポック分割をCell形式で実行する
            % Input:
            %   nTrials: トライアル数
            %   fs: サンプリングレート
            %   samplesPerEpoch: 1エポックあたりのサンプル数
            %   stepSize: ステップサイズ (サンプル数)
            %   nStepsObs: 観察期間のステップ数
            %   nStepsImg: イメージ期間のステップ数
            %   labels: ラベル情報 (構造体配列)
            %   data: エポック分割対象のデータ (チャンネル x サンプル)

            % taskTypes の設定に基づき抽出するか判定
            selectedTaskTypes = obj.params.signal.epoch.visual.taskTypes; % 抽出対象タスクタイプを取得
            extractObservation = any(strcmpi(selectedTaskTypes, 'observation')); % 観察期間を抽出するかどうか
            extractImagery     = any(strcmpi(selectedTaskTypes, 'imagery')); % イメージ期間を抽出するかどうか

            totalEpochs = nTrials * (nStepsObs + nStepsImg); % 最大エポック数を計算
            obj.epochs = cell(totalEpochs, 1); % エポック格納用セル配列を初期化 (最大エポック数で初期化)
            % processedLabelは value のみ（数値や文字列）
            obj.epochLabels = zeros(totalEpochs, 1); % エポックラベル格納用配列を初期化 (最大エポック数で初期化)
            epochTimes = zeros(totalEpochs, 2); % エポック時間格納用配列を初期化 (最大エポック数で初期化)

            observationDuration = obj.params.signal.epoch.visual.observationDuration; % 観察期間 (秒) を取得
            signalDuration = obj.params.signal.epoch.visual.signalDuration; % 合図期間 (秒) を取得
            imageryDuration = obj.params.signal.epoch.visual.imageryDuration; % イメージ期間 (秒) を取得

            epochIndex = 1; % エポックインデックスを初期化
            for trial = 1:nTrials
                trialStart = labels(trial).sample; % トライアル開始サンプルを取得
                trialValue = labels(trial).value; % ラベル値を取得

                % 【観察期間】(抽出対象の場合のみ)
                if extractObservation
                    obsStart = trialStart; % 観察期間開始サンプル
                    obsEnd = trialStart + round(observationDuration * fs) - 1; % 観察期間終了サンプル
                    for step = 1:nStepsObs
                        currentStart = obsStart + (step-1)*stepSize; % 現在のステップの開始サンプル
                        currentEnd = currentStart + samplesPerEpoch - 1; % 現在のステップの終了サンプル
                        if currentEnd <= obsEnd % エポックが観察期間内かチェック
                            obj.epochs{epochIndex} = data(:, currentStart:currentEnd); % エポックデータを抽出してセル配列に格納
                            % processedLabel は値のみ
                            obj.epochLabels(epochIndex) = trialValue; % ラベル値を格納
                            epochTimes(epochIndex,:) = [currentStart/fs, currentEnd/fs]; % エポックの開始・終了時間を記録
                            epochIndex = epochIndex + 1; % エポックインデックスをインクリメント
                        end
                    end
                end

                % 【イメージ期間】(抽出対象の場合のみ、合図期間はスキップ)
                if extractImagery
                    imgStart = trialStart + round((observationDuration + signalDuration) * fs); % イメージ期間開始サンプル (観察期間 + 合図期間 の後)
                    imgEnd = imgStart + round(imageryDuration * fs) - 1; % イメージ期間終了サンプル
                    for step = 1:nStepsImg
                        currentStart = imgStart + (step-1)*stepSize; % 現在のステップの開始サンプル
                        currentEnd = currentStart + samplesPerEpoch - 1; % 現在のステップの終了サンプル
                        if currentEnd <= imgEnd % エポックがイメージ期間内かチェック
                            obj.epochs{epochIndex} = data(:, currentStart:currentEnd); % エポックデータを抽出してセル配列に格納
                            obj.epochLabels(epochIndex) = trialValue; % ラベル値を格納
                            epochTimes(epochIndex,:) = [currentStart/fs, currentEnd/fs]; % エポックの開始・終了時間を記録
                            epochIndex = epochIndex + 1; % エポックインデックスをインクリメント
                        end
                    end
                end
            end

            % 未使用部分の削除 (初期化時に最大エポック数で確保しているため、実際に抽出されたエポック数に合わせて配列をトリム)
            obj.epochs = obj.epochs(1:epochIndex-1);
            obj.epochLabels = obj.epochLabels(1:epochIndex-1);
            obj.processingInfo.epoching.epochTimes = epochTimes(1:epochIndex-1,:);
        end

        %% --- 視覚タスク用のエポック抽出（Array形式） ---
        function epochingArrayVisual(obj, nTrials, fs, samplesPerEpoch, stepSize, nStepsObs, nStepsImg, nChannels, labels, data)
            % epochingArrayVisualメソッド：視覚タスク用の時間窓ベースのエポック分割をArray形式で実行する
            % Input:
            %   nTrials: トライアル数
            %   fs: サンプリングレート
            %   samplesPerEpoch: 1エポックあたりのサンプル数
            %   stepSize: ステップサイズ (サンプル数)
            %   nStepsObs: 観察期間のステップ数
            %   nStepsImg: イメージ期間のステップ数
            %   nChannels: チャンネル数
            %   labels: ラベル情報 (構造体配列)
            %   data: エポック分割対象のデータ (チャンネル x サンプル)

            % taskTypes の設定に基づき抽出するか判定
            selectedTaskTypes = obj.params.signal.epoch.visual.taskTypes; % 抽出対象タスクタイプを取得
            extractObservation = any(strcmpi(selectedTaskTypes, 'observation')); % 観察期間を抽出するかどうか
            extractImagery     = any(strcmpi(selectedTaskTypes, 'imagery')); % イメージ期間を抽出するかどうか

            maxEpochs = nTrials * (nStepsObs + nStepsImg); % 最大エポック数を計算
            obj.epochs = zeros(nChannels, samplesPerEpoch, maxEpochs); % エポック格納用配列を初期化 (最大エポック数で初期化)
            % processedLabelは value のみ
            epochLabelsTmp = zeros(maxEpochs, 1); % 一時的なエポックラベル格納用配列 (トリム処理のため)
            epochTimes = zeros(maxEpochs, 2); % エポック時間格納用配列を初期化 (最大エポック数で初期化)

            observationDuration = obj.params.signal.epoch.visual.observationDuration; % 観察期間 (秒) を取得
            signalDuration = obj.params.signal.epoch.visual.signalDuration; % 合図期間 (秒) を取得
            imageryDuration = obj.params.signal.epoch.visual.imageryDuration; % イメージ期間 (秒) を取得

            epochIndex = 1; % エポックインデックスを初期化
            for trial = 1:nTrials
                trialStart = labels(trial).sample; % トライアル開始サンプルを取得
                trialValue = labels(trial).value; % ラベル値を取得

                % 【観察期間】(抽出対象の場合のみ)
                if extractObservation
                    obsStart = trialStart; % 観察期間開始サンプル
                    obsEnd = trialStart + round(observationDuration * fs) - 1; % 観察期間終了サンプル
                    for step = 1:nStepsObs
                        currentStart = obsStart + (step-1)*stepSize; % 現在のステップの開始サンプル
                        currentEnd = currentStart + samplesPerEpoch - 1; % 現在のステップの終了サンプル
                        if currentEnd <= obsEnd % エポックが観察期間内かチェック
                            obj.epochs(:, :, epochIndex) = data(:, currentStart:currentEnd); % エポックデータを抽出して配列に格納
                            epochLabelsTmp(epochIndex) = trialValue; % ラベル値を一時配列に格納
                            epochTimes(epochIndex,:) = [currentStart/fs, currentEnd/fs]; % エポックの開始・終了時間を記録
                            epochIndex = epochIndex + 1; % エポックインデックスをインクリメント
                        end
                    end
                end

                % 【イメージ期間】(抽出対象の場合のみ)
                if extractImagery
                    imgStart = trialStart + round((observationDuration + signalDuration) * fs); % イメージ期間開始サンプル (観察期間 + 合図期間 の後)
                    imgEnd = imgStart + round(imageryDuration * fs) - 1; % イメージ期間終了サンプル
                    for step = 1:nStepsImg
                        currentStart = imgStart + (step-1)*stepSize; % 現在のステップの開始サンプル
                        currentEnd = currentStart + samplesPerEpoch - 1; % 現在のステップの終了サンプル
                        if currentEnd <= imgEnd % エポックがイメージ期間内かチェック
                            obj.epochs(:, :, epochIndex) = data(:, currentStart:currentEnd); % エポックデータを抽出して配列に格納
                            epochLabelsTmp(epochIndex) = trialValue; % ラベル値を一時配列に格納
                            epochTimes(epochIndex,:) = [currentStart/fs, currentEnd/fs]; % エポックの開始・終了時間を記録
                            epochIndex = epochIndex + 1; % エポックインデックスをインクリメント
                        end
                    end
                end
            end

            % 未使用部分の削除 (初期化時に最大エポック数で確保しているため、実際に抽出されたエポック数に合わせて配列をトリム)
            obj.epochs = obj.epochs(:, :, 1:epochIndex-1);
            epochLabelsTmp = epochLabelsTmp(1:epochIndex-1); % 一時ラベル配列をトリム
            obj.epochLabels = epochLabelsTmp; % トリム後のラベル配列をepochLabelsプロパティに代入
            obj.processingInfo.epoching.epochTimes = epochTimes(1:epochIndex-1,:);
        end

    end
end