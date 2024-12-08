classdef LSLManager < handle
    properties (Access = public)
        inlet
    end
    
    properties (Access = private)
        lib
        params
    end
    
    methods
        function obj = LSLManager(params)
            obj.params = params;
            try
                obj.inlet = [];
                obj.lib = [];
                if params.lsl.simulate.enable
                    obj.inlet = obj.initializeSimulator();
                    obj.displaySimulatorInfo();
                else
                    [obj.inlet, obj.lib] = obj.initializeLSL();
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
                if ~isempty(obj.inlet)
                    delete(obj.inlet);
                end
                return;
            end
        end
        
        function delete(obj)
            % デストラクタ - リソースの解放
            if ~isempty(obj.inlet) && ~isstruct(obj.inlet)  % LSLインレットの場合のみ
                delete(obj.inlet);
            end
        end
        
        function [data, timestamp] = getData(obj)
            if obj.params.lsl.simulate.enable
                [data, timestamp] = obj.inlet.getData();
            else
                [chunk, timestamps] = obj.inlet.pull_chunk();
                
                if ~isempty(chunk)
                    % チャンネル選択が必要な場合
                    if ~isempty(obj.params.device.channelNum)
                        data = chunk(obj.params.device.channelNum, :);
                    else
                        data = chunk;
                    end
                    timestamp = timestamps(end);
                else
                    data = [];
                    timestamp = [];
                end
            end
        end
    end
    
    methods (Access = private)
        function simulator = initializeSimulator(obj)
            simulator = struct();
            simulator.type = 'simulator';
            simulator.sampleRate = obj.params.device.sampleRate;
            simulator.channelCount = obj.params.device.channelCount;
            simulator.lastTimestamp = 0;
            
            % チャンクサイズ
            % simulator.chunkSize = round(simulator.sampleRate/24);
            simulator.chunkSize = 1;
            % チャンク間のインターバル
            % simulator.interval = round(1/6);
            
            simulator.getData = @() generateSimulatedData(obj, simulator);
            
            fprintf('シミュレーションモードで初期化しました\n');
            obj.displaySimulatorInfo();
        end
        
        function [data, timestamp] = generateSimulatedData(obj, simulator)
            % 固定チャンクサイズでデータを生成
            numSamples = simulator.chunkSize;
            
            % 時間軸の生成（1チャンク分）
            t = simulator.lastTimestamp + (1:numSamples)/simulator.sampleRate;
            simulator.lastTimestamp = t(end);
            
            % 信号生成
            alpha = obj.params.lsl.simulate.signal.alpha.freq;
            alphaAmplitude = obj.params.lsl.simulate.signal.alpha.amplitude;
            beta = obj.params.lsl.simulate.signal.beta.freq;
            betaAmplitude = obj.params.lsl.simulate.signal.beta.amplitude;
            
            baseSignal = alphaAmplitude * sin(2*pi*alpha*t) + ...
                betaAmplitude * sin(2*pi*beta*t);
            
            % 各チャンネルのデータ生成
            data = zeros(simulator.channelCount, numSamples);
            for ch = 1:simulator.channelCount
                data(ch,:) = baseSignal + 2*randn(1, numSamples);
            end
            
            timestamp = t(end);
            
            % チャンク間のインターバルを確保
            % pause(simulator.interval);
        end
        
        function [inlet, lib] = initializeLSL(obj)
            % LSLの初期化
            try
                lib = lsl_loadlib(env_translatepath('dependencies:/liblsl-Matlab/bin'));
            catch
                lib = lsl_loadlib();
            end
            
            % ストリーム名の設定
            streamName = obj.params.device.streamName;
            
            % ストリーム解決（タイムアウト付き）
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
            
            % インレットの開設
            fprintf('インレットを開いています...\n');
            inlet = lsl_inlet(result{1});
            
            obj.displayStreamInfo(inlet);
        end
        
        function displayStreamInfo(~, inlet)
            % ストリーム情報の表示
            try
                inf = inlet.info();
                fprintf('\nストリーム情報:\n');
                fprintf('XML メタデータ:\n%s\n', inf.as_xml());
                
                try
                    manufacturer = inf.desc().child_value('manufacturer');
                    fprintf('製造元: %s\n', manufacturer);
                catch ME
                    warning(ME.identifier, '%s', ME.message);
                    fprintf('製造元情報は利用できません\n');
                end
                
                fprintf('チャンネルラベル:\n');
                try
                    ch = inf.desc().child('channels').child('channel');
                    for k = 1:inf.channel_count()
                        fprintf('  %s\n', ch.child_value('label'));
                        ch = ch.next_sibling();
                    end
                catch ME
                    warning(ME.identifier, '%s', ME.message);
                    fprintf('チャンネル情報の取得に失敗しました\n');
                end
                fprintf('\n');
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