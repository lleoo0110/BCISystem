%% EEGAcquisitionManager
% 脳波(EEG)データのリアルタイム取得と処理を管理するクラス
% 
% このクラスは以下の主要な機能を提供:
% - LSLを介したEEGデータのストリーミング取得
% - リアルタイムの信号処理とフィルタリング
% - 特徴量抽出と分類処理
% - データの保存と管理
% - GUIによる可視化
%
% 使用例:
%   params = template(); % 設定パラメータの取得
%   manager = EEGAcquisitionManager(params); % インスタンス作成
%   manager.start(); % 計測開始
%
% 詳細仕様:
% - サンプリングレート: デバイスに依存
% - バッファサイズ: 設定可能
% - 保存形式: .mat形式

classdef EEGAcquisitionManager < handle
    properties (Access = private)
        % システムマネージャー群
        lslManager       
        udpManager      
        guiController   
        dataManager     
        dataLoader      
        
        % 前処理用コンポーネント群
        artifactRemover    
        baselineCorrector  
        downSampler       
        firFilter         
        iirFilter         
        notchFilter       
        
        % 特徴抽出コンポーネント群
        powerExtractor     
        faaExtractor      
        abRatioExtractor  
        cspExtractor      
        emotionExtractor  
        
        % 分類器コンポーネント群
        svmClassifier      
        ecocClassifier     
        cnnClassifier      
        lstmClassifier     
        hybridClassifier   
        
        % データ管理・設定関連
        params              
        rawData            
        labels             
        emgData            
        emgLabels          
        classifiers        
        results            
        threshold   
        
        % システム状態管理フラグ
        isRunning           
        isPaused            
        currentTotalSamples 
        isDestroying        
        
        % データバッファ管理
        dataBuffer         
        emgBuffer         
        emgBufferSize     
        bufferSize        
        processingWindow  
        slidingStep       
        
        % 統一タイマー管理
        acquisitionTimer   
        masterStartTimer
        
        % ファイル管理
        tempDataFiles      
        fileIndex         
        lastSaveTimestamp  % 最後の保存時刻（秒）
        lastSavedFilePath 
        latestResults     
        
        % サンプル管理カウンタの精密化
        totalSampleCount       % 総取得サンプル数
        lastResetSampleCount   % 最終リセット時のサンプル数
        emgSampleCount        % EMG総サンプル数
        
        % データレート監視
        actualSampleRate      % 実際のサンプリングレート
    end

    %% Public Methods - システム制御用メソッド群
    methods (Access = public)
        % コンストラクタ - システムの初期化と設定
        function obj = EEGAcquisitionManager(params)
            obj.params = params;
            obj.isRunning = false;
            obj.isPaused = false;
            obj.isDestroying = false;
            obj.threshold = params.classifier.threshold;
            
            % マスタータイマーの初期化
            obj.masterStartTimer = [];
            obj.actualSampleRate = params.device.sampleRate;
            
            obj.initializeDataBuffers();
            obj.initializeResults();
            obj.initializeManagers();
            
            obj.tempDataFiles = {};
            obj.fileIndex = 1;
            
            obj.totalSampleCount = 0;
            obj.lastResetSampleCount = 0;
            obj.emgSampleCount = 0;
        end
        
        %% デストラクタ - システムリソースの解放
        function delete(obj)
            try
                % オブジェクト破棄フラグを設定
                obj.isDestroying = true;
                
                % タイマーの停止と削除処理
                if ~isempty(obj.acquisitionTimer)
                    if isa(obj.acquisitionTimer, 'timer')
                        if isvalid(obj.acquisitionTimer)
                            % 実行中のタイマーを安全に停止
                            if strcmp(obj.acquisitionTimer.Running, 'on')
                                stop(obj.acquisitionTimer);
                                % タイマーの完全停止を待機
                                while strcmp(obj.acquisitionTimer.Running, 'on')
                                    pause(0.1);
                                end
                            end
                            % タイマーオブジェクトの削除
                            delete(obj.acquisitionTimer);
                        end
                    end
                    obj.acquisitionTimer = [];
                end

                % GUI関連リソースの解放
                if ~isempty(obj.guiController)
                    obj.guiController.closeAllWindows();
                    obj.guiController = [];
                end

                % 通信マネージャーの解放
                if ~isempty(obj.udpManager)
                    delete(obj.udpManager);
                    obj.udpManager = [];
                end
                if ~isempty(obj.lslManager)
                    delete(obj.lslManager);
                    obj.lslManager = [];
                end

            catch ME
                % エラー発生時のログ記録
                warning(ME.identifier, '%s', ME.message);
                disp(getReport(ME, 'extended'));
            end
        end
        
        %% データ収集開始メソッド
        function start(obj)
            if ~obj.isRunning
                try
                    % マスタータイマー開始時刻の設定
                    obj.masterStartTimer = tic;
                    fprintf('=== タイマー開始: %.6f ===\n', toc(obj.masterStartTimer));
                    
                    % 他のコンポーネントにマスター時刻を設定
                    obj.udpManager.setMasterStartTimer(obj.masterStartTimer);
                    obj.guiController.setMasterStartTimer(obj.masterStartTimer);
                    
                    % タイマーを初期化して計測開始
                    obj.setupTimers();
                    
                    % システム状態の初期化
                    obj.isRunning = true;
                    obj.isPaused = false;
                    
                    % データバッファの初期化
                    obj.rawData = [];
                    obj.labels = [];
                    obj.tempDataFiles = {};
                    obj.fileIndex = 1;
                    
                    % GUI状態の更新
                    obj.guiController.updateStatus('Recording');

                    % 保存先を指定
                    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                    defaultFileName = sprintf('eeg_recording_%s.mat', timestamp);
                    
                    [saveName, saveDir] = uiputfile('*.mat', '記録データの保存先を選択してください', defaultFileName);
                    
                    if isequal(saveName, 0)
                        fprintf('保存先が選択されませんでした。処理を中止します。\n');
                        return;
                    end
                    
                    obj.lastSavedFilePath = fullfile(saveDir, saveName);
                    fprintf('記録データは次の場所に保存されます: %s\n', obj.lastSavedFilePath);
                    
                catch ME
                    fprintf('処理の開始に失敗しました: %s\n', ME.message);
                    fprintf('スタックトレース:\n%s\n', getReport(ME, 'extended'));
                end
            end
        end
        
        %% データ収集停止メソッド
        function stop(obj)
            if obj.isRunning
                try
                    % システム状態フラグの更新
                    obj.isRunning = false;
                    obj.isPaused = false;

                    % タイマーの安全な停止処理
                    if ~isempty(obj.acquisitionTimer)
                        if strcmp(obj.acquisitionTimer.Running, 'on')
                            stop(obj.acquisitionTimer);
                            % タイマー停止の完了を待機
                            while strcmp(obj.acquisitionTimer.Running, 'on')
                                pause(0.1);
                            end
                        end
                        delete(obj.acquisitionTimer);
                        obj.acquisitionTimer = [];
                    end

                    % 最終データの保存処理
                    if ~isempty(obj.rawData)
                        obj.saveTemporaryData();
                    end
                    obj.mergeAndSaveData();

                    % タイマー停止完了の確認待機
                    pause(0.2);

                    % GUIの終了処理
                    if ~isempty(obj.guiController)
                        obj.guiController.closeAllWindows();
                    end

                    % 全リソースの解放
                    delete(obj);

                catch ME
                    % エラー発生時の後処理
                    warning(ME.identifier, '%s', ME.message);
                    try
                        % エラー発生時でもリソース解放を試行
                        if ~isempty(obj.acquisitionTimer)
                            delete(obj.acquisitionTimer);
                        end
                        if ~isempty(obj.guiController)
                            obj.guiController.closeAllWindows();
                        end
                        delete(obj);
                    catch
                        % クリーンアップ中のエラーは無視
                    end
                end
            end
        end
        
        %% 一時停止メソッド
        function pause(obj)
            if obj.isRunning && ~obj.isPaused
                obj.isPaused = true;
                obj.guiController.updateStatus('Paused');
            end
        end
        
        %% 再開メソッド
        function resume(obj)
            if obj.isRunning && obj.isPaused
                obj.isPaused = false;
                obj.guiController.updateStatus('Recording');
            end
        end
        
        %% メインのデータ取得処理メソッド
        function acquireData(obj)
            if ~isvalid(obj)
                return;
            end

            try
                if obj.isRunning && ~obj.isPaused
                    % 現在時刻の正確な取得
                    currentTime = toc(obj.masterStartTimer);
                    
                    % LSLからのデータ取得
                    [eegData, emgData] = obj.lslManager.getData();

                    if ~isempty(eegData)
                        % EEGデータの蓄積
                        obj.rawData = [obj.rawData, eegData];
                        
                        % EMGデータの処理（有効な場合）
                        if obj.params.acquisition.emg.enable && ~isempty(emgData)
                            obj.emgData = [obj.emgData, emgData];
                        end

                        % データバッファの更新
                        obj.updateDataBuffer(eegData);

                        % トリガー処理の精密化
                        trigger = obj.udpManager.receiveTrigger();
                        if ~isempty(trigger)
                            % サンプル数の正確な計算
                            currentEEGSamples = obj.calculateCurrentSample('eeg', currentTime);
                            
                            eegTrigger = trigger;
                            eegTrigger.sample = currentEEGSamples;
                            eegTrigger.preciseTime = currentTime; % より正確な時刻
                            obj.labels = [obj.labels; eegTrigger];

                            % EMG用トリガー処理（EMG有効時）
                            if obj.params.acquisition.emg.enable
                                currentEMGSamples = obj.calculateCurrentSample('emg', currentTime);
                                emgTrigger = trigger;
                                emgTrigger.sample = currentEMGSamples;
                                emgTrigger.preciseTime = currentTime;
                                obj.emgLabels = [obj.emgLabels; emgTrigger];
                            end

                            % 検証用デバッグ情報
                            obj.validateTriggerTiming(trigger, currentTime, currentEEGSamples);
                        end

                        % 保存間隔チェック
                        currentTime = toc(obj.masterStartTimer);
                        if (currentTime - obj.lastSaveTimestamp) >= obj.params.acquisition.save.saveInterval
                            obj.saveTemporaryData();
                            obj.lastSaveTimestamp = currentTime; % 更新
                        end
                        
                        % GUI更新処理
                        if any([obj.params.gui.display.visualization.enable.rawData, ...
                                obj.params.gui.display.visualization.enable.emgData, ...
                                obj.params.gui.display.visualization.enable.spectrum, ...
                                obj.params.gui.display.visualization.enable.ersp])
                            obj.processGUI();
                        end
                    end
                end
            catch ME
                obj.handleAcquisitionError(ME, currentTime);
            end
        end
    end
    
    %% Private Methods - 内部処理用メソッド群
    methods (Access = private)
        %% システムマネージャーの初期化
        % すべての処理コンポーネントを生成し、初期設定を行う
        function initializeManagers(obj)
            try
                % 基本マネージャーの初期化
                obj.lslManager = LSLManager(obj.params);
                obj.udpManager = UDPManager(obj.params);  % マスター時刻は後で設定
                obj.dataManager = DataManager(obj.params);
                obj.dataLoader = DataLoader(obj.params);
                obj.guiController = GUIControllerManager(obj.params);
                
                % 前処理コンポーネントの初期化
                obj.artifactRemover = ArtifactRemover(obj.params);
                obj.baselineCorrector = BaselineCorrector(obj.params);
                obj.downSampler = DownSampler(obj.params);
                obj.firFilter = FIRFilterDesigner(obj.params);
                obj.iirFilter = IIRFilterDesigner(obj.params);
                obj.notchFilter = NotchFilterDesigner(obj.params);
                
                % 特徴抽出コンポーネントの初期化
                obj.powerExtractor = PowerExtractor(obj.params);
                obj.faaExtractor = FAAExtractor(obj.params);
                obj.abRatioExtractor = ABRatioExtractor(obj.params);
                obj.cspExtractor = CSPExtractor(obj.params);
                obj.emotionExtractor = EmotionExtractor(obj.params);
                
                % 分類器コンポーネントの初期化
                obj.svmClassifier = SVMClassifier(obj.params);
                obj.ecocClassifier = ECOCClassifier(obj.params);
                obj.cnnClassifier = CNNClassifier(obj.params);
                obj.lstmClassifier = LSTMClassifier(obj.params);
                obj.hybridClassifier = HybridClassifier(obj.params);
                
                obj.classifiers = struct(...
                    'svm', [], ...
                    'ecoc', [], ...
                    'cnn', [], ...
                    'lstm', [] ...
                );
                
                % GUIコールバックの設定
                obj.setupGUICallbacks();
                
                % オンラインモード時の初期化
                if strcmpi(obj.params.acquisition.mode, 'online')
                    obj.initializeOnline();
                end
            catch ME
                error('Failed to initialize managers: %s', ME.message);
            end
        end
        
        %% 解析結果構造体の初期化
        % 各種解析結果を格納するための構造体を準備
        function initializeResults(obj)
            obj.results = struct(...
                'power', [], ...     % パワースペクトル解析結果
                'faa', [], ...      % FAA解析結果
                'abRatio', [], ...  % α/β比解析結果
                'emotion', [], ...  % 感情解析結果
                'predict', [] ...  % 分類予測結果
            );
        end
        
        %% GUIコールバックの設定
        % GUIイベントに対する処理関数を登録
        function setupGUICallbacks(obj)
            callbacks = struct(...
                'onStart', @() obj.start(), ...     % 開始ボタン
                'onStop', @() obj.stop(), ...       % 停止ボタン
                'onPause', @() obj.pause(), ...     % 一時停止ボタン
                'onResume', @() obj.resume(), ...   % 再開ボタン
                'onLabel', @(value) obj.handleLabel(value), ...           % ラベル設定
                'onParamChange', @(param, value) obj.updateParameter(param, value)); % パラメータ更新
            
            obj.guiController.setCallbacks(callbacks);
        end
        
        %% タイマーの設定
        % データ取得用タイマーの初期化と設定
        function setupTimers(obj)
            try
                % 既存タイマーのクリーンアップ
                if ~isempty(obj.acquisitionTimer)
                    if isa(obj.acquisitionTimer, 'timer') && isvalid(obj.acquisitionTimer)
                        if strcmp(obj.acquisitionTimer.Running, 'on')
                            stop(obj.acquisitionTimer);
                        end
                        delete(obj.acquisitionTimer);
                    end
                    obj.acquisitionTimer = [];
                end
        
                % 新規タイマーの設定 - StartFcnをfalseに設定して自動スタートを防止
                updateInterval = 0.15;  % 更新間隔（秒）
                obj.acquisitionTimer = timer(...
                    'ExecutionMode', 'fixedRate', ...  % 固定レート実行
                    'Period', updateInterval, ...      % 実行間隔
                    'TimerFcn', @(src, event) obj.safeAcquireData(), ...  % メイン処理関数
                    'ErrorFcn', @(src, event) obj.handleTimerError(event)); % エラー処理関数

                fprintf('=== データ収集を開始します ===\n');
                start(obj.acquisitionTimer);

            catch ME
                warning(ME.identifier, '%s', ME.message);
                disp(getReport(ME, 'extended'));
            end
        end

        %% タイマーエラー処理メソッド
        % タイマー実行中のエラーをハンドリング
        function handleTimerError(obj, event)
            try
                if ~isvalid(obj) || obj.isDestroying
                    return;
                end
                
                % エラー情報の出力
                fprintf('Timer error occurred: %s\n', event.Data.message);
                disp(getReport(event.Data, 'extended'));
                
                % 重大エラー時のシステム停止判定
                if strcmpi(event.Data.identifier, 'MATLAB:nomem') || ...
                   contains(lower(event.Data.message), 'fatal') || ...
                   contains(lower(event.Data.message), 'critical')
                    fprintf('Critical error detected. Stopping acquisition...\n');
                    obj.stop();
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
                disp(getReport(ME, 'extended'));
            end
        end
        
        % 現在サンプル数の正確な計算
        function currentSample = calculateCurrentSample(obj, dataType, currentTime)
            switch lower(dataType)
                case 'eeg'
                    % EEGサンプル数の計算
                    timeBasedSample = round(currentTime * obj.params.device.sampleRate);
                    actualSample = obj.totalSampleCount + size(obj.rawData, 2);
                    
                    % より正確な方を使用
                    if ~isempty(obj.rawData)
                        currentSample = actualSample;
                    else
                        currentSample = timeBasedSample;
                    end
                    
                case 'emg'
                    % EMGサンプル数の計算
                    if obj.params.acquisition.emg.enable
                        timeBasedSample = round(currentTime * obj.params.acquisition.emg.sampleRate);
                        actualSample = obj.emgSampleCount + size(obj.emgData, 2);
                        
                        if ~isempty(obj.emgData)
                            currentSample = actualSample;
                        else
                            currentSample = timeBasedSample;
                        end
                    else
                        currentSample = 0;
                    end
                    
                otherwise
                    error('Unknown data type: %s', dataType);
            end
        end
        
        % トリガータイミング検証
        function validateTriggerTiming(obj, trigger, currentTime, currentSample)
            % タイミングの一致性をチェック
            expectedSample = round(trigger.time/1000 * obj.params.device.sampleRate);
            sampleDiff = abs(currentSample - expectedSample);
            timeDiff = abs(currentTime*1000 - trigger.time);
            
            % 閾値を超えた場合は警告
            if sampleDiff > obj.params.device.sampleRate * 0.1 % 100ms以上のズレ
                fprintf('警告: サンプル数の大きなズレ検出\n');
                fprintf('  計算サンプル: %d, 期待サンプル: %d, 差分: %d\n', ...
                    currentSample, expectedSample, sampleDiff);
            end
            
            if timeDiff > 100 % 100ms以上のズレ
                fprintf('警告: 時刻の大きなズレ検出\n');
                fprintf('  現在時刻: %.3f, トリガー時刻: %.3f, 差分: %.3f ms\n', ...
                    currentTime*1000, trigger.time, timeDiff);
            end
            
            % デバッグ情報の出力
            fprintf('トリガー検証:\n');
            fprintf('  EEG sample: %d (時刻ベース期待値: %d)\n', currentSample, expectedSample);
            fprintf('  時刻差分: %.3f ms\n', timeDiff);
            fprintf('  実測サンプリングレート: %.2f Hz\n', obj.actualSampleRate);
        end
        
        % エラーハンドリングの改善
        function handleAcquisitionError(obj, ME, currentTime)
            errorInfo = struct();
            errorInfo.message = ME.message;
            errorInfo.stack = ME.stack;
            errorInfo.time = datetime('now');
            errorInfo.masterTime = currentTime;
            errorInfo.dataInfo = struct(...
                'rawDataSize', size(obj.rawData), ...
                'bufferSize', size(obj.dataBuffer), ...
                'actualSampleRate', obj.actualSampleRate, ...
                'totalSamples', obj.totalSampleCount);
            if obj.params.acquisition.emg.enable
                errorInfo.dataInfo.emgDataSize = size(obj.emgData);
            end

            % エラーログの保存
            errorLogFile = fullfile(fileparts(obj.lastSavedFilePath), ...
                sprintf('error_log_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
            save(errorLogFile, 'errorInfo');

            fprintf('Data acquisition error at time %.3f: %s\n', currentTime, ME.message);
            disp(getReport(ME, 'extended'));

            if ~isempty(obj.guiController)
                obj.guiController.showError('Data Acquisition Error', ...
                    sprintf('Error at %.3fs: %s\nCheck error log for details.', currentTime, ME.message));
            end

            % 重大エラー時のシステム停止
            if strcmpi(ME.identifier, 'MATLAB:nomem') || ...
               contains(lower(ME.message), 'fatal') || ...
               contains(lower(ME.message), 'critical')
                fprintf('Critical error detected. Stopping acquisition...\n');
                obj.stop();
            end
        end

        %% 安全なデータ取得メソッド
        % オブジェクトの状態を確認しながらデータ取得を実行
        function safeAcquireData(obj)
            try
                % オブジェクトの状態チェック
                if ~isvalid(obj) || obj.isDestroying
                    return;
                end
                if obj.isRunning && ~obj.isPaused
                    obj.acquireData();
                end
            catch ME
                if ~isvalid(obj) || obj.isDestroying
                    return;
                end
                warning(ME.identifier, '%s', ME.message);
                disp(getReport(ME, 'extended'));
            end
        end
        
        %% ラベル処理メソッド
        % トリガー情報の処理とラベル付け
        function handleLabel(obj, trigger)
            if obj.isRunning && ~obj.isPaused
                try
                    % サンプル数の計算
                    currentEEGSamples = obj.totalSampleCount + size(obj.rawData, 2);
                    currentEMGSamples = obj.emgSampleCount + size(obj.emgData, 2);

                    % EEGトリガー情報の設定
                    eegTrigger = trigger;
                    eegTrigger.sample = currentEEGSamples;
                    obj.labels = [obj.labels; eegTrigger];

                    % EMGトリガー情報の設定
                    emgTrigger = trigger;
                    emgTrigger.sample = currentEMGSamples;
                    obj.emgLabels = [obj.emgLabels; emgTrigger];

                    % デバッグ情報の出力
                    fprintf('Label recorded - Button press\n');
                    fprintf('Trigger value: %d\n', trigger.value);
                    fprintf('EEG sample: %d\n', currentEEGSamples);
                    fprintf('EMG sample: %d\n', currentEMGSamples);

                    % GUI状態の更新
                    if ~isempty(obj.guiController)
                        obj.guiController.updateStatus(sprintf('Label recorded: %d', trigger.value));
                    end
                    
                    drawnow;

                catch ME
                    warning(ME.identifier, '%s', ME.message);
                    fprintf('Stack trace:\n');
                    disp(getReport(ME, 'extended'));
                end
            end
        end
        
        %% 一時データ保存メソッド
        % 現在のデータを一時ファイルとして保存
        function saveTemporaryData(obj)
            try
                % 合計サンプル数の更新
                obj.totalSampleCount = obj.totalSampleCount + size(obj.rawData, 2);
                obj.emgSampleCount = obj.emgSampleCount + size(obj.emgData, 2);

                % データ存在チェック
                if isempty(obj.rawData) && isempty(obj.emgData)
                    fprintf('保存する新しいデータがありません\n');
                    return;
                end
        
                % 最小データ長チェック
                minSamples = obj.params.device.sampleRate;
                if size(obj.rawData, 2) < minSamples
                    fprintf('データ長が不十分です (%d < %d samples)\n', ...
                        size(obj.rawData, 2), minSamples);
                    return;
                end
        
                % 保存ディレクトリの決定
                if ~isempty(obj.lastSavedFilePath)
                    % メインの保存先が設定されている場合、そのディレクトリを使用
                    saveDir = fileparts(obj.lastSavedFilePath);
                else
                    % メインの保存先が設定されていない場合は現在のディレクトリを使用
                    saveDir = pwd;
                    fprintf('警告: メインの保存先が指定されていません。現在のディレクトリを使用します: %s\n', saveDir);
                end
        
                % 一時ファイル名の生成
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                tempFilename = sprintf('eeg_temp_%s_%d.mat', timestamp, obj.fileIndex);
                
                % 保存先パスの設定
                tempFilePath = fullfile(saveDir, tempFilename);
        
                % 保存データの構造体作成
                tempData = struct();
        
                % EEGデータの保存準備
                if ~isempty(obj.rawData)
                    tempData.rawData = obj.rawData;
                    tempData.labels = obj.labels;
                    fprintf('保存するEEGデータ: %d サンプル\n', size(obj.rawData, 2));
                end
        
                % EMGデータの保存準備
                if obj.params.acquisition.emg.enable && ~isempty(obj.emgData)
                    tempData.emgData = obj.emgData;
                    tempData.emgLabels = obj.emgLabels;
                    fprintf('保存するEMGデータ: %d サンプル\n', size(obj.emgData, 2));
                end
        
                % メタデータの追加
                tempData.savingInfo = struct(...
                    'timestamp', datetime('now'), ...
                    'totalSamples', obj.totalSampleCount + size(obj.rawData, 2), ...
                    'fileIndex', obj.fileIndex, ...
                    'sampleRate', obj.params.device.sampleRate);
        
                % データの保存実行
                obj.dataManager.saveDataset(tempData, tempFilePath);
                obj.tempDataFiles{end+1} = tempFilePath;
        
                % メモリのクリア
                obj.rawData = [];
                obj.emgData = [];
                obj.labels = [];
                obj.emgLabels = [];
        
                % カウンタの更新
                obj.fileIndex = obj.fileIndex + 1;

                fprintf('一時データを保存しました: %s\n', tempFilePath);
            catch ME
                fprintf('一時データの保存に失敗: %s\n', ME.message);
                fprintf('エラー詳細:\n%s\n', getReport(ME, 'extended'));
            end
        end

        %% 一時ファイルのクリーンアップメソッド
        % 不要になった一時ファイルを削除
        function cleanupTempFiles(obj)
            shouldDeleteTempFiles = false;
            
            % パラメータ構造のチェック
            if isfield(obj.params, 'acquisition') && ...
               isfield(obj.params.acquisition, 'save') && ...
               isfield(obj.params.acquisition.save, 'deleteTempFiles')
                shouldDeleteTempFiles = obj.params.acquisition.save.deleteTempFiles;
            end
            
            fprintf('一時ファイル削除設定: %s\n', mat2str(shouldDeleteTempFiles));
            
            if shouldDeleteTempFiles
                fprintf('一時ファイルを削除します...\n');
                for i = 1:length(obj.tempDataFiles)
                    if exist(obj.tempDataFiles{i}, 'file')
                        delete(obj.tempDataFiles{i});
                        fprintf('  - 一時ファイルを削除しました: %s\n', obj.tempDataFiles{i});
                    end
                end
                fprintf('一時ファイル削除完了: %d ファイル\n', length(obj.tempDataFiles));
            else
                fprintf('一時ファイルは保持されます: %d ファイル\n', length(obj.tempDataFiles));
                % 一時ファイルのリストを表示
                for i = 1:length(obj.tempDataFiles)
                    fprintf('  - %s\n', obj.tempDataFiles{i});
                end
            end
        end
        
        %% エラーログ記録メソッド
        % システムエラー情報の保存
        function logError(obj, ME)
            % エラーログ構造体の作成
            errorLog = struct(...
                'message', ME.message, ...
                'identifier', ME.identifier, ...
                'stack', ME.stack, ...
                'time', datetime('now'), ...
                'dataInfo', struct(...
                'rawDataSize', size(obj.rawData), ...
                'labelsSize', size(obj.labels), ...
                'tempFilesCount', length(obj.tempDataFiles)));
            
            % エラーログファイルの保存
            errorLogFile = fullfile(obj.params.acquisition.save.path, ...
                sprintf('error_log_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
            save(errorLogFile, 'errorLog');
        end
        
        %% 一時ファイルの統合処理メソッド
        % 複数の一時ファイルを1つのデータセットに統合
        function [mergedEEG, mergedEMG, mergedLabels, mergedEMGLabels] = mergeTempFiles(obj)
            try
                % 初期サイズの計算
                totalEEGSamples = 0;
                totalEMGSamples = 0;
                totalLabels = 0;
                totalEMGLabels = 0;

                % 各一時ファイルのサイズを集計
                for i = 1:length(obj.tempDataFiles)
                    if exist(obj.tempDataFiles{i}, 'file')
                        fileInfo = load(obj.tempDataFiles{i});
                        totalEEGSamples = totalEEGSamples + size(fileInfo.rawData, 2);

                        if isfield(fileInfo, 'emgData')
                            totalEMGSamples = totalEMGSamples + size(fileInfo.emgData, 2);
                        end

                        if isfield(fileInfo, 'labels')
                            totalLabels = totalLabels + length(fileInfo.labels);
                        end
                        if isfield(fileInfo, 'emgLabels')
                            totalEMGLabels = totalEMGLabels + length(fileInfo.emgLabels);
                        end
                    end
                end

                % 結果配列の初期化
                mergedEEG = zeros(obj.params.device.channelCount, totalEEGSamples);
                if obj.params.acquisition.emg.enable
                    mergedEMG = zeros(obj.params.acquisition.emg.channels.count, totalEMGSamples);
                else
                    mergedEMG = [];
                end

                % ラベル配列の初期化
                emptyStruct = struct('value', [], 'time', [], 'sample', []);
                mergedLabels = repmat(emptyStruct, totalLabels, 1);
                mergedEMGLabels = repmat(emptyStruct, totalEMGLabels, 1);

                % データの統合処理
                currentEEGSample = 1;
                currentEMGSample = 1;
                labelIndex = 1;
                emgLabelIndex = 1;

                % 各一時ファイルからデータを統合
                for i = 1:length(obj.tempDataFiles)
                    if exist(obj.tempDataFiles{i}, 'file')
                        fileData = load(obj.tempDataFiles{i});

                        % EEGデータの統合
                        eegCount = size(fileData.rawData, 2);
                        mergedEEG(:, currentEEGSample:currentEEGSample+eegCount-1) = fileData.rawData;

                        % EMGデータの統合
                        if obj.params.acquisition.emg.enable && isfield(fileData, 'emgData')
                            emgCount = size(fileData.emgData, 2);
                            mergedEMG(:, currentEMGSample:currentEMGSample+emgCount-1) = fileData.emgData;
                        end

                        % ラベルの統合
                        if isfield(fileData, 'labels')
                            for j = 1:length(fileData.labels)
                                mergedLabels(labelIndex) = fileData.labels(j);
                                labelIndex = labelIndex + 1;
                            end
                        end

                        % EMGラベルの統合
                        if obj.params.acquisition.emg.enable && isfield(fileData, 'emgLabels')
                            for j = 1:length(fileData.emgLabels)
                                mergedEMGLabels(emgLabelIndex) = fileData.emgLabels(j);
                                emgLabelIndex = emgLabelIndex + 1;
                            end
                        end

                        % インデックスの更新
                        currentEEGSample = currentEEGSample + eegCount;
                        if obj.params.acquisition.emg.enable && isfield(fileData, 'emgData')
                            currentEMGSample = currentEMGSample + emgCount;
                        end
                    end
                end

                % 未使用部分の削除
                mergedLabels = mergedLabels(1:labelIndex-1);
                if obj.params.acquisition.emg.enable
                    mergedEMGLabels = mergedEMGLabels(1:emgLabelIndex-1);
                else
                    mergedEMGLabels = [];
                end

                % 統合結果の表示
                fprintf('Data merged successfully:\n');
                fprintf('  EEG samples: %d\n', size(mergedEEG, 2));
                fprintf('  EMG samples: %d\n', size(mergedEMG, 2));
                fprintf('  EEG labels: %d\n', length(mergedLabels));
                fprintf('  EMG labels: %d\n', length(mergedEMGLabels));

            catch ME
                error('Data merging failed: %s', ME.message);
            end
        end
        
        %% 最終データの保存処理メソッド
        % 全データの統合と最終保存を実行
        function mergeAndSaveData(obj)
            try
                fprintf('データファイルを統合中...\n');
        
                % 一時ファイルからデータを統合
                [mergedEEG, mergedEMG, mergedLabels, mergedEMGLabels] = obj.mergeTempFiles();
        
                % 保存データ構造体の作成
                saveData = struct();
                saveData.params = obj.params;         % パラメータ設定
                saveData.rawData = mergedEEG;         % EEGデータ
                saveData.labels = mergedLabels;       % ラベル情報
                saveData.results = obj.results;       % 解析結果
        
                % EMGデータの保存（有効な場合のみ）
                if obj.params.acquisition.emg.enable
                    saveData.emgData = mergedEMG;
                    saveData.emgLabels = mergedEMGLabels;
                end
                
                % 既に指定済みの保存先に保存
                if ~isempty(obj.lastSavedFilePath)
                    % データの最終保存
                    obj.dataManager.saveDataset(saveData, obj.lastSavedFilePath);
                    fprintf('データを保存しました: %s\n', obj.lastSavedFilePath);
                else
                    % 保存先が指定されていない場合は改めて選択
                    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                    defaultFileName = sprintf('eeg_recording_%s.mat', timestamp);
                    
                    [saveName, saveDir] = uiputfile('*.mat', '記録データの保存先を選択してください', defaultFileName);
                    
                    if saveName == 0
                        fprintf('保存がキャンセルされました。データは保存されませんでした。\n');
                        return;
                    end
                    
                    saveFilePath = fullfile(saveDir, saveName);
                    obj.dataManager.saveDataset(saveData, saveFilePath);
                    fprintf('データを保存しました: %s\n', saveFilePath);
                end
        
                obj.cleanupTempFiles();
        
            catch ME
                error('最終結果の保存に失敗: %s', ME.message);
            end
        end
        
        %% オンラインモード初期化メソッド
        % オンライン処理用の学習済みモデルとパラメータを設定
        function initializeOnline(obj)
            try
                fprintf('\n=== オンラインモード初期化 ===\n');
                fprintf('学習済みモデルファイルを選択してください。\n');
                
                % DataLoaderを使用してデータを読み込み
                [loadedData, fileInfo] = obj.dataLoader.loadDataBrowser();
        
                % データの検証と準備
                if iscell(loadedData)
                    if isempty(loadedData)
                        error('有効なデータが読み込まれませんでした');
                    end
                    
                    % 複数ファイルが選択された場合は最初のファイルのみ使用
                    if length(loadedData) > 1
                        error('オンラインモードでは複数読み込みに対応できません');
                    end
                    
                    loadedData = loadedData{1}; % 最初のファイルを使用
                end
        
                % デバイス設定の整合性チェック
                obj.dataLoader.validateDeviceConfig(loadedData);
        
                % 分類器の設定と検証
                if ~isfield(loadedData, 'classifier')
                    error('読み込んだデータに分類器情報が見つかりません');
                end
                
                % アクティブな分類器の確認
                activeClassifier = obj.params.classifier.activeClassifier;
                if ~isfield(loadedData.classifier, activeClassifier)
                    error('選択された分類器 "%s" が読み込んだデータに存在しません', ...
                        upper(activeClassifier));
                end
        
                % 分類器モデルの検証
                if ~isfield(loadedData.classifier.(activeClassifier), 'model')
                    error('%s モデルが読み込んだデータに存在しません', ...
                        upper(activeClassifier));
                end
                
                % 分類器構造体を保存
                obj.classifiers = loadedData.classifier;
        
                % 分類器構造体からCSPフィルタを取得
                if isfield(loadedData.classifier.(activeClassifier), 'cspFilters')
                    fprintf('CSPフィルタを読み込みました\n');
                    obj.results.csp = struct('filters', loadedData.classifier.(activeClassifier).cspFilters);
                    
                    if isfield(loadedData.classifier.(activeClassifier), 'cspParameters')
                        obj.results.csp.parameters = loadedData.classifier.(activeClassifier).cspParameters;
                    end
                end
        
                fprintf('オンライン処理の初期化が完了しました: %s 分類器を使用\n', ...
                    upper(activeClassifier));
                    
                % 読み込んだファイル情報を表示
                fprintf('読み込んだモデルファイル: %s\n', fileInfo.filenames{1});
        
            catch ME
                obj.guiController.showError('オンライン処理の初期化に失敗しました', ME.message);
                obj.stop();
                rethrow(ME);
            end
        end

        %% チャンネル情報表示メソッド
        % デバイスとデータのチャンネル設定を表示
        function displayChannelInfo(~, params, data)
            fprintf('\nチャンネル情報:\n');
            fprintf('デバイス設定: %s (%dチャンネル)\n', ...
                params.device.name, params.device.channelCount);
            fprintf('設定チャンネル: %s\n', strjoin(params.device.channels, ', '));
            fprintf('データチャンネル数: %d\n', size(data.rawData, 1));

            if isfield(data, 'channelNames')
                fprintf('データチャンネル: %s\n', strjoin(data.channelNames, ', '));
            end
            fprintf('\n');
        end
        
        %% データバッファの初期化メソッド
        % 信号処理に使用する各種バッファを準備
        function initializeDataBuffers(obj)
            % 処理ウィンドウサイズの計算
            obj.processingWindow = round(obj.params.signal.window.epochDuration * obj.params.device.sampleRate);
            obj.slidingStep = round(obj.params.signal.window.updateBuffer * obj.params.device.sampleRate);
            obj.bufferSize = round(obj.params.signal.window.bufferSize * obj.params.device.sampleRate);
        
            % メインEEGバッファの初期化
            obj.dataBuffer = zeros(obj.params.device.channelCount, 0);
        
            % EMGバッファの初期化（EMG有効時のみ）
            if obj.params.acquisition.emg.enable
                obj.emgBuffer = zeros(obj.params.acquisition.emg.channels.count, 0);
            end
        end
        
        %% データバッファ更新メソッド
        % 新しいデータの追加と古いデータの削除を管理
        function updateDataBuffer(obj, eegData)
            try
                % オブジェクトの状態チェック
                if ~isvalid(obj) || obj.isDestroying || isempty(eegData)
                    return;
                end

                % バッファへのデータ追加
                obj.dataBuffer = [obj.dataBuffer, eegData];

                % バッファサイズが上限に達した場合の処理
                if size(obj.dataBuffer, 2) >= obj.bufferSize
                    % オンラインモードの場合の処理
                    if strcmpi(obj.params.acquisition.mode, 'online')
                        obj.processOnline();
                    end

                    % バッファの更新（古いデータの削除）
                    obj.dataBuffer = obj.dataBuffer(:, obj.slidingStep+1:end);
                end
            catch ME
                if ~isvalid(obj) || obj.isDestroying
                    return;
                end
                warning(ME.identifier, '%s', ME.message);
                fprintf('Error in updateDataBuffer: %s\n', getReport(ME, 'extended'));
            end
        end
        
        %% オンライン信号処理メソッド
        % リアルタイムでの信号処理とパターン認識を実行
        function processOnline(obj)
            try
                % 現在のサンプル数を更新
                obj.currentTotalSamples = obj.totalSampleCount + size(obj.rawData, 2);

                % 前処理の実行
                preprocessedSegment = obj.preprocessSignal();
                if isempty(preprocessedSegment)
                    return;
                end

                % 解析ウィンドウの抽出
                analysisSegment = preprocessedSegment(:, end-obj.processingWindow+1:end);

                % 各種特徴量の抽出
                obj.processPowerFeatures(analysisSegment);
                obj.processFAAFeatures(analysisSegment);
                obj.processABRatioFeatures(analysisSegment);
                obj.processEmotionFeatures(analysisSegment);
                
                % 分類処理の実行
                [label, score] = obj.processClassification(analysisSegment);

                % 最新の解析結果を構造体として保存
                obj.latestResults = struct(...
                    'prediction', struct(...
                        'label', label, ...
                        'score', score ...
                    ), ...
                    'features', struct(...
                        'faa', obj.getLatestFeature(obj.results.faa), ...
                        'abRatio', obj.getLatestFeature(obj.results.abRatio), ...
                        'emotion', obj.getLatestFeature(obj.results.emotion) ...
                    ) ...
                );

                % 結果のUDP送信
                obj.sendResults(obj.latestResults);

            catch ME
                obj.handleError(ME);
            end
        end
        
        %% 前処理メソッド
        % 信号の前処理（フィルタリング、ノイズ除去等）を実行
        function preprocessedSegment = preprocessSignal(obj)
            try
                preprocessedSegment = [];
                if obj.params.signal.enable && ~isempty(obj.dataBuffer)
                    data = obj.dataBuffer;

                    % ダウンサンプリング処理
                    if obj.params.signal.preprocessing.downsample.enable
                        [data, ~] = obj.downSampler.downsample(...
                            data, obj.params.signal.preprocessing.downsample.targetRate);
                    end

                    % フィルタリング処理
                    if obj.params.signal.preprocessing.filter.notch.enable
                        [data, ~] = obj.notchFilter.designAndApplyFilter(data);
                    end

                    if obj.params.signal.preprocessing.filter.fir.enable
                        [data, ~] = obj.firFilter.designAndApplyFilter(data);
                    end

                    if obj.params.signal.preprocessing.filter.iir.enable
                        [data, ~] = obj.iirFilter.designAndApplyFilter(data);
                    end

                    % アーティファクト除去処理
                    if obj.params.signal.preprocessing.artifact.enable
                        [data, ~] = obj.artifactRemover.removeArtifacts(data, 'all');
                    end

                    % ベースライン補正処理
                    if obj.params.signal.preprocessing.baseline.enable
                        [data, ~] = obj.baselineCorrector.correctBaseline(...
                            data, obj.params.signal.preprocessing.baseline.method);
                    end
                    
                    preprocessedSegment = data;
                end
            catch ME
                error('Preprocessing failed: %s', ME.message);
            end
        end
        
        %% パワースペクトル特徴量抽出メソッド
        % 各周波数帯域のパワー値を計算
        function processPowerFeatures(obj, preprocessedSegment)
            if ~obj.params.feature.power.enable || isempty(preprocessedSegment)
                return;
            end
        
            % 周波数帯域の設定を取得
            bandNames = obj.params.feature.power.bands.names;
            if iscell(bandNames{1})
                bandNames = bandNames{1};
            end
        
            % 各周波数帯域のパワーを計算
            bandPowers = struct();
            for i = 1:length(bandNames)
                bandName = bandNames{i};
                freqRange = obj.params.feature.power.bands.(bandName);
                bandPowers.(bandName) = obj.powerExtractor.calculatePower(preprocessedSegment, freqRange);
            end
            
            % 結果の保存
            currentTime = toc(obj.masterStartTimer)*1000;
            newPowerResult = struct(...
                'bands', bandPowers, ...
                'time', currentTime, ...
                'sample', obj.currentTotalSamples);
        
            if isempty(obj.results.power)
                obj.results.power = newPowerResult;
            else
                obj.results.power(end+1) = newPowerResult;
            end
        end
        
        %% FAA（前頭部α波非対称性）特徴量抽出メソッド
        % 左右前頭部のα波パワーの非対称性を計算
        function processFAAFeatures(obj, preprocessedSegment)
            if ~obj.params.feature.faa.enable || isempty(preprocessedSegment)
                return;
            end
        
            % FAA値の計算
            faaResults = obj.faaExtractor.calculateFAA(preprocessedSegment);
            
            if ~isempty(faaResults)
                if iscell(faaResults)
                    faaResult = faaResults{1};
                else
                    faaResult = faaResults;
                end
                
                % 結果の保存
                currentTime = toc(obj.masterStartTimer)*1000;
                newFAAResult = struct(...
                    'faa', faaResult.faa, ...
                    'arousal', faaResult.pleasureState, ...
                    'time', currentTime, ...
                    'sample', obj.currentTotalSamples);
        
                if isempty(obj.results.faa)
                    obj.results.faa = newFAAResult;
                else
                    obj.results.faa(end+1) = newFAAResult;
                end
            end
        end
        
        %% α/β比特徴量抽出メソッド
        % α波とβ波のパワー比を計算
        function processABRatioFeatures(obj, preprocessedSegment)
            if ~obj.params.feature.abRatio.enable || isempty(preprocessedSegment)
                return;
            end

            % α/β比の計算
            [abRatio, arousalState] = obj.abRatioExtractor.calculateABRatio(preprocessedSegment);
            
            % 結果の保存
            currentTime = toc(obj.masterStartTimer)*1000;
            newABRatioResult = struct(...
                'ratio', abRatio, ...
                'state', arousalState, ...
                'time', currentTime, ...
                'sample', obj.currentTotalSamples);

            if isempty(obj.results.abRatio)
                obj.results.abRatio = newABRatioResult;
            else
                obj.results.abRatio(end+1) = newABRatioResult;
            end
        end
        
        %% 感情特徴量抽出メソッド
        % EEGパターンから感情状態を推定
        function processEmotionFeatures(obj, preprocessedSegment)
            if ~obj.params.feature.emotion.enable || isempty(preprocessedSegment)
                return;
            end

            % 感情状態の分類
            emotionResults = obj.emotionExtractor.classifyEmotion(preprocessedSegment);
            if iscell(emotionResults)
                emotionResult = emotionResults{1};
            else
                emotionResult = emotionResults;
            end
            
            % 結果の保存
            currentTime = toc(obj.masterStartTimer)*1000;
            newEmotionResult = struct(...
                'state', emotionResult.state, ...
                'coordinates', emotionResult.coordinates, ...
                'emotionCoords', emotionResult.emotionCoords, ...
                'time', currentTime, ...
                'sample', obj.currentTotalSamples);

            if isempty(obj.results.emotion)
                obj.results.emotion = newEmotionResult;
            else
                obj.results.emotion(end+1) = newEmotionResult;
            end
        end
        
        %% 分類処理実行メソッド
        % 特徴量から状態を分類
        function [label, score] = processClassification(obj, currentData)
            label = [];
            score = [];
        
            try
                switch obj.params.classifier.activeClassifier
                    case 'svm'
                        % SVM分類の実行
                        [label, score] = obj.svmClassifier.predictOnline( currentData, obj.classifiers.svm);
        
                    case 'ecoc'
                        % ECOC分類の実行
                        [label, score] = obj.ecocClassifier.predictOnline(currentData, obj.classifiers.ecoc);
        
                    case 'cnn'
                        % CNN分類の実行
                        [label, score] = obj.cnnClassifier.predictOnline(currentData, obj.classifiers.cnn);
        
                    case 'lstm'
                        % LSTM分類の実行
                        [label, score] = obj.lstmClassifier.predictOnline(currentData, obj.classifiers.lstm);
        
                    case 'hybrid'
                        % Hybrid分類の実行
                        [label, score] = obj.hybridClassifier.predictOnline(currentData, obj.classifiers.hybrid);
                end

                % 超簡易的な実装
                if score(1) >= obj.threshold
                    label = 2;
                else
                    label = 3;
                end
        
                % 予測結果の保存
                if ~isempty(label)
                    currentTime = toc(obj.masterStartTimer) * 1000;
                    newPredictResult = struct(...
                        'label', label, ...
                        'score', score, ...
                        'time', currentTime ...
                    );
        
                    % 結果の蓄積
                    if isempty(obj.results.predict)
                        obj.results.predict = newPredictResult;
                    else
                        obj.results.predict(end+1) = newPredictResult;
                    end
                end
            catch ME
                fprintf('Error in processClassification: %s\n', ME.message);
                warning(ME.identifier, '%s', ME.message);
            end
        end

        %% UDP結果送信メソッド
        % 解析結果をUDPで外部に送信
        function sendResults(obj, udpData)
            if ~isempty(udpData)
                obj.udpManager.sendTrigger(udpData);
            end
        end
        
        % エラー処理メソッド
        % システムエラーのログ記録と表示
        function handleError(~, ME)
            warning(ME.identifier, 'Error in processOnline: %s\n', ME.message);
            fprintf('Error stack:\n');
            for k = 1:length(ME.stack)
                fprintf('File: %s, Line: %d, Function: %s\n', ...
                    ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
            end
        end
        
        %% GUI更新メソッド
        % リアルタイムデータの可視化を更新
        function processGUI(obj)
            try
                % 表示データの準備
                displayData = struct();
                
                % バンドパスフィルタの設計(1-45Hz)
                fs = obj.params.device.sampleRate;
                [b, a] = butter(4, [1 45]/(fs/2), 'bandpass');

                % データのフィルタリング
                filteredData = zeros(size(obj.rawData));
                for ch = 1:size(obj.rawData, 1)
                    filteredData(ch,:) = filtfilt(b, a, obj.rawData(ch,:));
                end

                % EEGデータの表示準備
                if obj.params.gui.display.visualization.enable.rawData && ~isempty(obj.rawData)
                    displaySeconds = obj.params.gui.display.visualization.scale.displaySeconds;
                    displaySamples = round(displaySeconds * obj.params.device.sampleRate);

                    if size(obj.rawData, 2) > displaySamples
                        displayData.rawData = filteredData(:, end-displaySamples+1:end);
                    else
                        displayData.rawData = filteredData;
                    end
                end
                
                % EMGデータの表示準備
                if obj.params.acquisition.emg.enable && ...
                   obj.params.gui.display.visualization.enable.emgData && ...
                   ~isempty(obj.emgData)
                    displaySeconds = obj.params.gui.display.visualization.scale.displaySeconds;
                    displaySamples = round(displaySeconds * obj.params.device.sampleRate);

                    if size(obj.emgData, 2) > displaySamples
                        displayData.emgData = obj.emgData(:, end-displaySamples+1:end);
                    else
                        displayData.emgData = obj.emgData;
                    end
                end
                
                % スペクトル表示の準備
                if obj.params.gui.display.visualization.enable.spectrum && ...
                   isfield(displayData, 'rawData') && ...
                   ~isempty(displayData.rawData)

                    % スペクトル計算
                    [pxx, f] = obj.powerExtractor.calculateSpectrum(filteredData);

                    % 表示範囲の制限
                    freqRange = obj.params.gui.display.visualization.scale.freq;
                    freqIdx = (f >= freqRange(1) & f <= freqRange(2));

                    % スペクトルデータの保存
                    displayData.spectrum = struct(...
                        'pxx', pxx(freqIdx), ...
                        'f', f(freqIdx));
                end

                % ERSP表示の準備
                if obj.params.gui.display.visualization.enable.ersp && ...
                   isfield(displayData, 'rawData') && ...
                   ~isempty(displayData.rawData)

                    % データ長のチェック
                    minSamples = 2 * obj.params.gui.display.visualization.ersp.numFreqs;
                    if size(displayData.rawData, 2) >= minSamples
                        [ersp, times, freqs] = obj.powerExtractor.calculateERSP(displayData.rawData);
                        if ~isempty(ersp) && ~isempty(times) && ~isempty(freqs)
                            % 表示範囲の制限
                            timeRange = obj.params.gui.display.visualization.ersp.time;
                            freqRange = obj.params.gui.display.visualization.ersp.freqRange;

                            timeIdx = (times >= timeRange(1) & times <= timeRange(2));
                            freqIdx = (freqs >= freqRange(1) & freqs <= freqRange(2));

                            % ERSPデータの保存
                            displayData.ersp = struct(...
                                'ersp', ersp(freqIdx, timeIdx), ...
                                'times', times(timeIdx), ...
                                'freqs', freqs(freqIdx));
                        end
                    end
                end

                % GUI表示の更新
                if ~isempty(fieldnames(displayData))
                    obj.guiController.updateDisplayData(displayData);
                end

            catch ME
                warning(ME.identifier, '%s', ME.message);
                fprintf('Error in processGUI: %s\n', getReport(ME, 'extended'));
            end
        end
        
        %% パラメータ更新メソッド
        % システムパラメータの動的更新
        function updateParameter(obj, paramName, value)
            try
                switch lower(paramName)
                    case 'window size'
                        obj.params.signal.window.epochDuration = value;
                        obj.initializeDataBuffers();
                    case 'filter range'
                        obj.params.signal.filter.fir.frequency = value;
                    case 'threshold'
                        obj.threshold = value;
                        fprintf('Threshold updated: %.3f\n', value);
                end      
            catch ME
                obj.guiController.showError('Parameter Update Error', ME.message);
            end
        end
        
        %% 感情座標取得メソッド
        % 感情状態を4次元座標に変換
        function coords = getEmotionCoordinates(~, emotionState)
            % 感情状態と座標のマッピング
            emotionMap = containers.Map(...
                {'安静', '興奮', '喜び', '快適', 'リラックス', '眠気', '憂鬱', '不快', '緊張'}, ...
                {[0 0 0 0], [100 0 0 100], [100 0 0 0], [100 100 0 0], ...
                [0 100 0 0], [0 100 100 0], [0 0 100 0], [0 0 100 100], [0 0 0 100]});
            
            % 座標の取得
            if emotionMap.isKey(emotionState)
                coords = emotionMap(emotionState);
            else
                coords = [0 0 0 0];  % デフォルト値（安静状態）
                warning('Emotion state "%s" not found. Using default coordinates.', emotionState);
            end
        end
        
        %% 最新特徴量取得メソッド
        % 各種特徴量の最新値を取得
        function latest = getLatestFeature(~, data)
            if isempty(data)
                latest = [];
                return;
            end

            if isstruct(data)
                % 構造体配列の場合
                latest = data(end);
            elseif isnumeric(data)
                % 数値配列の場合
                latest = data(end,:);
            else
                latest = [];
            end
        end
        
        %% 配列フォーマットメソッド
        % 数値配列の小数点以下桁数を制限
        function formattedArray = formatArray(data, precision)
            if nargin < 2
                precision = 2;  % デフォルトは小数点以下2桁
            end

            % 配列の各要素をフォーマット
            formattedArray = arrayfun(@(x) str2double(sprintf('%.*f', precision, x)), data);
        end
    end
end