classdef LSLManager < handle
    % LSLManager - Lab Streaming Layer通信を管理するクラス
    %
    % このクラスは以下の機能を提供します：
    % - EEGおよびEMGデータストリームの初期化と管理
    % - リアルタイムデータの取得
    % - シミュレーションモードのサポート
    % - 接続タイムアウト処理
    
    properties (Access = public)
        eegInlet    % EEGデータ用のLSLインレット
        emgInlet    % EMGデータ用のLSLインレット
    end
    
    properties (Access = private)
        lib         % LSLライブラリインスタンス
        params      % 設定パラメータ
        lastTimestamp   % 最後のデータ取得時刻
        simulationStartTime  % シミュレーション開始時刻
        
        % EMGストリーム用のパラメータ
        emgStreamInfo   % EMGストリーム設定情報

        % LSL接続タイムアウト設定
        connectionTimeout = 10  % タイムアウト時間(秒)
    end
    
    methods (Access = public)
        function obj = LSLManager(params)
            % コンストラクタ：LSLManagerの初期化
            obj.validateParams(params);
            obj.params = params;
            obj.lastTimestamp = 0;
            obj.simulationStartTime = tic;

            % EMGストリーム情報の設定
            obj.emgStreamInfo = obj.params.acquisition.emg.lsl;

            try
                obj.initializeInlets();
            catch ME
                obj.handleInitializationError(ME);
            end
        end

        function delete(obj)
            % デストラクタ：リソースの解放
            obj.cleanupResources();
        end
        
        function [eegData, emgData, timestamp] = getData(obj)
            % データの取得：実データまたはシミュレーションデータを取得
            %
            % 出力:
            %   eegData: EEGデータ配列 [チャンネル数 x サンプル数]
            %   emgData: EMGデータ配列 [チャンネル数 x サンプル数]
            %   timestamp: データのタイムスタンプ
            
            if obj.params.lsl.simulate.enable
                [eegData, emgData, timestamp] = obj.getSimulatedData();
            else
                [eegData, emgData, timestamp] = obj.getLSLData();
            end

            % EMG無効の場合は空配列を返す
            if ~obj.params.acquisition.emg.enable
                emgData = [];
            end
        end
    end
    
    methods (Access = private)
        function validateParams(~, params)
            % パラメータの検証
            required_fields = {'lsl', 'device'};
            if ~all(isfield(params, required_fields))
                error('必要なパラメータフィールドが不足しています');
            end
        end
        
        function initializeInlets(obj)
            % LSLインレットの初期化
            try
                obj.eegInlet = [];
                obj.emgInlet = [];
                obj.lib = [];

                if obj.params.lsl.simulate.enable
                    [obj.eegInlet, obj.emgInlet] = obj.initializeSimulator();
                else
                    [obj.eegInlet, obj.emgInlet, obj.lib] = obj.initializeLSL();
                end
            catch ME
                rethrow(ME);
            end
        end
        
        function [eegInlet, emgInlet, lib] = initializeLSL(obj)
            % LSLストリームの初期化
            try
                % LSLライブラリのロード
                lib = obj.loadLSLLibrary();

                % EEGストリームの解決（タイムアウト付き）
                fprintf('\nEEGストリーム解決中...\n');
                startTime = tic;
                eegResult = [];
                
                % タイムアウトまでストリームの検索を試みる
                while isempty(eegResult) && toc(startTime) < obj.connectionTimeout
                    eegResult = lsl_resolve_byprop(lib, 'name', obj.params.device.lsl.streamName);
                    if isempty(eegResult)
                        pause(0.5);
                        fprintf('.');
                    end
                end
                fprintf('\n');

                % タイムアウトチェック
                if isempty(eegResult)
                    error('LSLManager:ConnectionTimeout', ...
                        'EEGストリームの接続がタイムアウトしました（%d秒）\nデバイスの接続を確認してください', ...
                        obj.connectionTimeout);
                end

                eegInlet = lsl_inlet(eegResult{1});
                obj.displayStreamInfo(eegInlet, 'EEG');

                % EMGストリームの初期化（EMGが有効な場合のみ）
                emgInlet = [];
                if obj.params.acquisition.emg.enable
                    fprintf('\nEMGストリーム解決中...\n');
                    startTime = tic;
                    emgResult = [];
                    
                    while isempty(emgResult) && toc(startTime) < obj.connectionTimeout
                        emgResult = lsl_resolve_byprop(lib, 'name', obj.emgStreamInfo.streamName);
                        if isempty(emgResult)
                            pause(0.5);
                            fprintf('.');
                        end
                    end
                    fprintf('\n');

                    % EMGタイムアウトチェック
                    if isempty(emgResult)
                        error('LSLManager:ConnectionTimeout', ...
                            'EMGストリームの接続がタイムアウトしました（%d秒）\nデバイスの接続を確認してください', ...
                            obj.connectionTimeout);
                    end

                    emgInlet = lsl_inlet(emgResult{1});
                    obj.displayStreamInfo(emgInlet, 'EMG');
                end

            catch ME
                % エラーメッセージの詳細化
                if contains(ME.identifier, 'ConnectionTimeout')
                    rethrow(ME);
                else
                    error('LSLManager:InitializationError', ...
                        'LSL初期化エラー: %s\nスタックトレース:\n%s', ...
                        ME.message, getReport(ME, 'extended'));
                end
            end
        end
        
        function lib = loadLSLLibrary(~)
            % LSLライブラリのロード
            try
                % まず、環境変数のパスを試す
                lib = lsl_loadlib(env_translatepath('dependencies:/liblsl-Matlab/bin'));
            catch
                % デフォルトのパスを試す
                lib = lsl_loadlib();
            end
        end
        
        function [eegInlet, emgInlet] = initializeSimulator(obj)
            % シミュレーションモードの初期化
            eegInlet = struct(...
                'type', 'simulator', ...
                'sampleRate', obj.params.device.sampleRate, ...
                'channelCount', obj.params.device.channelCount, ...
                'getData', @() obj.getSimulatedData());
            
            emgInlet = struct(...
                'type', 'simulator', ...
                'sampleRate', obj.params.device.sampleRate, ...
                'channelCount', 2, ...
                'getData', @() obj.getSimulatedData());
            
            fprintf('シミュレーションモードで初期化しました (EEG + EMG)\n');
            obj.displaySimulatorInfo();
        end
        
        function [eegData, emgData, timestamp] = getLSLData(obj)
            % 実際のLSLストリームからデータを取得
            try
                % 戻り値の初期化
                eegData = [];
                emgData = [];
                timestamp = [];

                % EEGデータの取得
                [eegChunk, eegTimestamps] = obj.eegInlet.pull_chunk();

                % EEGデータの処理
                if ~isempty(eegChunk)
                    if ~isempty(obj.params.device.channelNum)
                        eegData = eegChunk(obj.params.device.channelNum, :);
                    else
                        eegData = eegChunk;
                    end

                    if ~isempty(eegTimestamps)
                        timestamp = eegTimestamps(end);
                    end
                end

                % EMGデータの取得（EMGが有効な場合のみ）
                if obj.params.acquisition.emg.enable && ~isempty(obj.emgInlet)
                    [emgChunk, emgTimestamps] = obj.emgInlet.pull_chunk();

                    if ~isempty(emgChunk)
                        if ~isempty(obj.params.acquisition.emg.channels.channelNum)
                            emgData = emgChunk(obj.params.acquisition.emg.channels.channelNum, :);
                        else
                            emgData = emgChunk;
                        end

                        if ~isempty(emgTimestamps)
                            if isempty(timestamp)
                                timestamp = emgTimestamps(end);
                            else
                                timestamp = max(timestamp, emgTimestamps(end));
                            end
                        end
                    end
                end

            catch ME
                fprintf('LSLデータ取得エラー: %s\n', ME.message);
                fprintf('エラー詳細:\n%s\n', getReport(ME, 'extended'));
                eegData = [];
                emgData = [];
                timestamp = [];
            end
        end
        
        function [eegData, emgData, timestamp] = getSimulatedData(obj)
            % シミュレーションデータの生成
            try
                % 実際の経過時間に基づいてサンプル数を計算
                currentTime = toc(obj.simulationStartTime);
                elapsedTime = currentTime - obj.lastTimestamp;

                % EEGとEMGのサンプル数を計算
                eegNumSamples = ceil(elapsedTime * obj.params.device.sampleRate);
                emgNumSamples = ceil(elapsedTime * obj.params.acquisition.emg.sampleRate);

                if eegNumSamples > 0 && emgNumSamples > 0
                    % 時間軸の生成
                    eegT = (obj.lastTimestamp:(1/obj.params.device.sampleRate):(currentTime));
                    eegT = eegT(1:min(eegNumSamples, length(eegT)));

                    emgT = (obj.lastTimestamp:(1/obj.params.acquisition.emg.sampleRate):(currentTime));
                    emgT = emgT(1:min(emgNumSamples, length(emgT)));

                    % データの生成
                    eegBaseSignal = obj.generateBaseSignal(eegT);
                    eegData = obj.generateChannelData(eegBaseSignal, length(eegT));

                    emgBaseSignal = obj.generateEMGSignal(emgT);
                    emgData = obj.generateEMGChannelData(emgBaseSignal, length(emgT));

                    obj.lastTimestamp = currentTime;
                    timestamp = currentTime;
                else
                    eegData = [];
                    emgData = [];
                    timestamp = [];
                end

            catch ME
                fprintf('シミュレーションデータ生成エラー: %s\n', ME.message);
                eegData = [];
                emgData = [];
                timestamp = [];
            end
        end

        function handleInitializationError(obj, ME)
           % 初期化エラーの処理
           obj.cleanupResources();
           % ConnectionTimeoutの場合は処理を停止
           if contains(ME.identifier, 'ConnectionTimeout')
               error('LSLManager:Fatal', 'LSL接続エラーのため処理を停止します\n%s', ME.message);
           else
               warning(ME.identifier, '%s', ME.message);
           end
        end
        
        function cleanupResources(obj)
            % リソースのクリーンアップ
            try
                % EEG Inlet の削除
                if ~isempty(obj.eegInlet)
                    if isobject(obj.eegInlet) && isvalid(obj.eegInlet)
                        delete(obj.eegInlet);
                    end
                end

                % EMG Inlet の削除
                if ~isempty(obj.emgInlet)
                    if isobject(obj.emgInlet) && isvalid(obj.emgInlet)
                        delete(obj.emgInlet);
                    end
                end

                % ライブラリのリセット
                obj.eegInlet = [];
                obj.emgInlet = [];
                obj.lib = [];
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end

        function displayStreamInfo(~, inlet, streamType)
            % ストリーム情報の表示
            try
                inf = inlet.info();
                fprintf('\n%sストリーム情報:\n', streamType);
                fprintf('名前: %s\n', inf.name());
                fprintf('タイプ: %s\n', inf.type());
                fprintf('チャンネル数: %d\n', inf.channel_count());
                fprintf('サンプリングレート: %d Hz\n', inf.nominal_srate());
                fprintf('ソースID: %s\n', inf.source_id());
                
                fprintf('\nチャンネル情報:\n');
                ch = inf.desc().child('channels').child('channel');
                for k = 1:inf.channel_count()
                    fprintf('  チャンネル %d: %s\n', k, ch.child_value('label'));
                    ch = ch.next_sibling();
                end
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
        
        function displaySimulatorInfo(obj)
            % シミュレータ情報の表示
            fprintf('\nシミュレータ情報:\n');
            fprintf('EEGチャンネル数: %d\n', obj.params.device.channelCount);
            fprintf('EMGチャンネル数: 2\n');
            fprintf('サンプリングレート: %d Hz\n', obj.params.device.sampleRate);
            fprintf('シミュレーション信号設定:\n');
            fprintf('  アルファ波: %d Hz (振幅: %d)\n', ...
                obj.params.lsl.simulate.signal.alpha.freq, ...
                obj.params.lsl.simulate.signal.alpha.amplitude);
            fprintf('  ベータ波: %d Hz (振幅: %d)\n', ...
                                obj.params.lsl.simulate.signal.beta.freq, ...
                obj.params.lsl.simulate.signal.beta.amplitude);
            fprintf('  EMGバースト: 20-80 Hz\n');
            fprintf('  ノイズ振幅: 2\n\n');
        end

        function baseSignal = generateBaseSignal(obj, t)
            % 基本的な脳波信号の生成
            % アルファ波とベータ波の重ね合わせにノイズを加える
            
            % アルファ波成分の生成
            alphaComponent = obj.generateWaveComponent(t, ...
                obj.params.lsl.simulate.signal.alpha.freq, ...
                obj.params.lsl.simulate.signal.alpha.amplitude);
            
            % ベータ波成分の生成
            betaComponent = obj.generateWaveComponent(t, ...
                obj.params.lsl.simulate.signal.beta.freq, ...
                obj.params.lsl.simulate.signal.beta.amplitude);
            
            % バックグラウンドノイズの生成
            noiseAmplitude = 2;
            backgroundNoise = obj.generateGaussianNoise(length(t)) * noiseAmplitude;
            
            % 信号の組み合わせ
            baseSignal = alphaComponent + betaComponent + backgroundNoise;
            
            % 信号の検証と制限
            baseSignal = obj.validateSignal(baseSignal);
        end
        
        function wave = generateWaveComponent(~, t, freq, amplitude)
            % 特定の周波数成分の生成
            % ランダムな位相を持つ正弦波を生成
            phase = 2 * pi * rand();
            wave = amplitude * sin(2*pi*freq*t + phase);
            
            % 無効な値をチェック
            wave(~isfinite(wave)) = 0;
        end
        
        function noise = generateGaussianNoise(~, n)
            % ガウシアンノイズの生成と正規化
            noise = randn(1, n);
            noise = noise - mean(noise);
            noise = noise / (std(noise) + eps);
        end
        
        function signal = validateSignal(~, signal)
            % 信号の検証と制限
            % NaNやInfを除去し、振幅を制限する
            signal(~isfinite(signal)) = 0;
            maxAmplitude = 100; % μV
            signal = max(min(signal, maxAmplitude), -maxAmplitude);
        end
        
        function data = generateChannelData(obj, baseSignal, numSamples)
            % 各チャンネルのデータ生成
            data = zeros(obj.params.device.channelCount, numSamples);
            noise = 2 * randn(1, numSamples);
            
            % 各チャンネルにノイズを加えた信号を設定
            for ch = 1:obj.params.device.channelCount
                data(ch,:) = baseSignal + noise;
            end
        end
        
        function signal = generateEMGSignal(obj, t)
            % EMG信号の生成
            try
                t = t(:)';  % ベクトルの方向を統一

                % ナイキスト周波数の計算
                nyquistFreq = obj.params.acquisition.emg.sampleRate / 2;

                % EMG信号の主要周波数成分を設定
                freqs = [20, 50, 80, 120, 150];  % 特徴的な周波数
                amps = [2, 1.5, 1, 0.8, 0.5];    % 各周波数の振幅

                % 基本信号の生成
                signal = zeros(size(t));
                for i = 1:length(freqs)
                    if freqs(i) < nyquistFreq
                        signal = signal + amps(i) * sin(2*pi*freqs(i)*t);
                    end
                end

                % 高周波ノイズの追加
                noiseAmplitude = 0.5;
                noise = obj.generateEMGNoise(length(t), obj.params.acquisition.emg.sampleRate);
                signal = signal + noiseAmplitude * noise;

                % バースト特性の追加
                burstFreq = 2; % バースト周波数（Hz）
                burstEnvelope = 0.5 * (1 + sin(2*pi*burstFreq*t));
                signal = signal .* burstEnvelope;

                % 振幅の正規化と制限
                maxAmp = 100;
                signal = signal / max(abs(signal)) * maxAmp;
                signal = min(max(signal, -maxAmp), maxAmp);

                % DCオフセットの除去
                signal = signal - mean(signal);

                % 無効な値の除去
                signal(isnan(signal) | isinf(signal)) = 0;

            catch ME
                warning('EMGSignal:GenerationFailed', 'EMG信号生成エラー: %s', ME.message);
                signal = zeros(size(t));
            end
        end

        function data = generateEMGChannelData(obj, baseSignal, numSamples)
            % EMGチャンネルデータの生成
            try
                numChannels = obj.params.acquisition.emg.channels.count;
                data = zeros(numChannels, numSamples);

                % 各チャンネルの生成
                for ch = 1:numChannels
                    % チャンネル固有のノイズを追加
                    channelNoise = obj.generateEMGNoise(numSamples, obj.params.acquisition.emg.sampleRate) * 0.2;

                    % 位相シフトの適用
                    phaseShift = 2 * pi * rand();
                    shiftedSignal = circshift(baseSignal, round(phaseShift * numSamples / (2*pi)));

                    % 信号の組み合わせ
                    data(ch,:) = shiftedSignal + channelNoise;

                    % チャンネル間の振幅変動を追加
                    scaleFactor = 0.9 + 0.2 * rand();
                    data(ch,:) = data(ch,:) * scaleFactor;
                end

                % 全体の振幅を正規化
                maxAmp = 100;
                data = data / max(abs(data(:))) * maxAmp;

            catch ME
                warning('EMGChannel:GenerationFailed', 'EMGチャンネルデータ生成エラー: %s', ME.message);
                data = zeros(numChannels, numSamples);
            end
        end

        function noise = generateEMGNoise(~, numSamples, ~)
            % EMG用のノイズ生成
            try
                numSamples = round(numSamples);
                nfft = 2^nextpow2(numSamples);
                f = (1:floor(nfft/2))';

                % ピンクノイズ特性の生成
                amplitude = 1./sqrt(f + eps);

                % 位相のランダム化
                phase = 2*pi*rand(length(f), 1);
                s = amplitude .* exp(1i*phase);
                s = [0; s; flipud(conj(s(1:end-1)))];

                % 時間領域への変換
                fullNoise = real(ifft(s));
                noise = fullNoise(1:numSamples);

                % 正規化
                noise = noise - mean(noise);
                noise = noise / (std(noise) + eps);
                noise = noise(:)';  % 行ベクトルに変換

            catch ME
                warning('EMGNoise:GenerationFailed', 'EMGノイズ生成エラー: %s', ME.message);
                noise = randn(1, numSamples);
            end

            % 無効な値のチェック
            if any(isnan(noise)) || any(isinf(noise))
                warning('EMGNoise:InvalidValues', 'ノイズ生成で無効な値を検出');
                noise = randn(1, numSamples);
            end
        end
    end
end