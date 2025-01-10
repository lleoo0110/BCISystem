classdef LSLManager < handle
    properties (Access = public)
        inlet
    end
    
    properties (Access = private)
        lib
        params
        lastTimestamp
    end
    
    methods (Access = public)
        function obj = LSLManager(params)
            obj.validateParams(params);
            obj.params = params;
            obj.lastTimestamp = 0;
            
            try
                obj.initializeInlet();
            catch ME
                obj.handleInitializationError(ME);
            end
        end
        
        function delete(obj)
            obj.cleanupResources();
        end
        
        function [data, timestamp] = getData(obj)
            if obj.params.lsl.simulate.enable
                [data, timestamp] = obj.getSimulatedData();
            else
                [data, timestamp] = obj.getLSLData();
            end
        end
    end
    
    methods (Access = private)
        function validateParams(~, params)
            required_fields = {'lsl', 'device'};
            if ~all(isfield(params, required_fields))
                error('Missing required parameter fields');
            end
        end
        
        function initializeInlet(obj)
            try
                obj.inlet = [];
                obj.lib = [];
                
                if obj.params.lsl.simulate.enable
                    obj.inlet = obj.initializeSimulator();
                else
                    [obj.inlet, obj.lib] = obj.initializeLSL();
                end
            catch ME
                rethrow(ME);
            end
        end
        
        function handleInitializationError(obj, ME)
            warning(ME.identifier, '%s', ME.message);
            obj.cleanupResources();
        end
        
        function cleanupResources(obj)
            if ~isempty(obj.inlet) && ~isstruct(obj.inlet)
                delete(obj.inlet);
            end
            obj.inlet = [];
            obj.lib = [];
        end
        
        function [data, timestamp] = getLSLData(obj)
            try
                % LSLのinletが正しく初期化されているか確認
                if isempty(obj.inlet) || ~isobject(obj.inlet)
                    error('LSL inlet is not properly initialized');
                end

                % データのpull_chunk実行
                [chunk, timestamps] = obj.inlet.pull_chunk();
                
                % データが空の場合の処理
                if isempty(chunk)
                    data = [];
                    timestamp = [];
                    return;
                end

                % チャンネル選択
                if ~isempty(obj.params.device.channelNum)
                    data = chunk(obj.params.device.channelNum, :);
                else
                    data = chunk;
                end
                
                % タイムスタンプの設定
                if ~isempty(timestamps)
                    timestamp = timestamps(end);
                else
                    timestamp = [];
                end

            catch ME
                fprintf('LSL data acquisition error: %s\n', ME.message);
                data = [];
                timestamp = [];
            end
        end
        
        function [data, timestamp] = getSimulatedData(obj)
            try
                numSamples = 1;
                t = obj.lastTimestamp + (1:numSamples)/obj.params.device.sampleRate;
                obj.lastTimestamp = t(end);
                
                % ベース信号の生成
                baseSignal = obj.generateBaseSignal(t);
                
                % チャンネルデータの生成
                data = obj.generateChannelData(baseSignal, numSamples);
                timestamp = t(end);
                
            catch ME
                fprintf('Simulation data generation error: %s\n', ME.message);
                data = [];
                timestamp = [];
            end
        end
        
        function baseSignal = generateBaseSignal(obj, t)
            % ベースコンポーネントを生成
            alphaComponent = obj.generateWaveComponent(t, ...
                obj.params.lsl.simulate.signal.alpha.freq, ...
                obj.params.lsl.simulate.signal.alpha.amplitude);
            
            betaComponent = obj.generateWaveComponent(t, ...
                obj.params.lsl.simulate.signal.beta.freq, ...
                obj.params.lsl.simulate.signal.beta.amplitude);
            
            % バックグラウンドノイズの発生
            noiseAmplitude = 2;
            backgroundNoise = obj.generateGaussianNoise(length(t)) * noiseAmplitude;
            
            % 信号と検証の組み合わせ
            baseSignal = alphaComponent + betaComponent + backgroundNoise;
            
            % 出力が有限であり合理的な範囲内にあることを確認
            baseSignal = obj.validateSignal(baseSignal);
        end
        
        function wave = generateWaveComponent(~, t, freq, amplitude)
            % Generate wave with controlled phase
            phase = 2 * pi * rand();
            wave = amplitude * sin(2*pi*freq*t + phase);
            
            % Ensure wave is finite
            wave(~isfinite(wave)) = 0;
        end
        
        function noise = generateGaussianNoise(~, n)
            % 振幅を制御した単純なガウスノイズを発生させる
            noise = randn(1, n);
            
            % 正規化ノイズ
            noise = noise - mean(noise);
            noise = noise / (std(noise) + eps);
        end
        
        function signal = validateSignal(~, signal)
            % 有限でない値をゼロに置き換える
            signal(~isfinite(signal)) = 0;
            
            % 極端な値をクリップして不安定さを防ぐ
            maxAmplitude = 100; % μV
            signal = max(min(signal, maxAmplitude), -maxAmplitude);
        end
        
        function data = generateChannelData(obj, baseSignal, numSamples)
            data = zeros(obj.params.device.channelCount, numSamples);
            noise = 2 * randn(1, numSamples);
            
            for ch = 1:obj.params.device.channelCount
                data(ch,:) = baseSignal + noise;
            end
        end
        
        function simulator = initializeSimulator(obj)
            simulator = struct();
            simulator.type = 'simulator';
            simulator.sampleRate = obj.params.device.sampleRate;
            simulator.channelCount = obj.params.device.channelCount;
            simulator.getData = @() obj.getSimulatedData();
            
            fprintf('シミュレーションモードで初期化しました\n');
            obj.displaySimulatorInfo();
        end
        
        function [inlet, lib] = initializeLSL(obj)
            try
                % LSLライブラリのロード
                lib = obj.loadLSLLibrary();
                
                % ストリームの解決
                streamName = obj.params.device.lsl.streamName;
                result = obj.resolveStream(lib, streamName);
                
                fprintf('インレットを開いています...\n');
                inlet = lsl_inlet(result{1});
                
                % ストリーム情報の表示
                obj.displayStreamInfo(inlet);
                
            catch ME
                error('LSL initialization failed: %s', ME.message);
            end
        end
        
        function lib = loadLSLLibrary(~)
            try
                % まず、環境変数のパスを試す
                lib = lsl_loadlib(env_translatepath('dependencies:/liblsl-Matlab/bin'));
            catch
                % デフォルトのパスを試す
                lib = lsl_loadlib();
            end
        end
        
        function result = resolveStream(~, lib, streamName)
            fprintf('ストリーム解決中: %s\n', streamName);
            result = {};
            timeout = 10;
            startTime = tic;
            
            while isempty(result) && toc(startTime) < timeout
                result = lsl_resolve_byprop(lib, 'name', streamName);
                if isempty(result)
                    pause(0.5);
                    fprintf('.');
                end
            end
            fprintf('\n');
            
            if isempty(result)
                error('ストリームが見つかりませんでした: %s', streamName);
            end
        end
        
        function displayStreamInfo(~, inlet)
            try
                inf = inlet.info();
                fprintf('\nストリーム情報:\n');
                fprintf('名前: %s\n', inf.name());
                fprintf('タイプ: %s\n', inf.type());
                fprintf('チャンネル数: %d\n', inf.channel_count());
                fprintf('サンプリングレート: %d Hz\n', inf.nominal_srate());
                fprintf('ソースID: %s\n', inf.source_id());
                
                % チャンネル情報の表示
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
            fprintf('\nシミュレータ情報:\n');
            fprintf('サンプリングレート: %d Hz\n', obj.params.device.sampleRate);
            fprintf('チャンネル数: %d\n', obj.params.device.channelCount);
            fprintf('シミュレーション信号設定:\n');
            fprintf('  アルファ波: %d Hz (振幅: %d)\n', ...
                obj.params.lsl.simulate.signal.alpha.freq, ...
                obj.params.lsl.simulate.signal.alpha.amplitude);
            fprintf('  ベータ波: %d Hz (振幅: %d)\n', ...
                obj.params.lsl.simulate.signal.beta.freq, ...
                obj.params.lsl.simulate.signal.beta.amplitude);
            fprintf('  ノイズ振幅: 2\n\n');
        end
    end
end