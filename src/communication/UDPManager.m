classdef UDPManager < handle
    properties (Access = private)
        receiveSocket  % udpport オブジェクト（受信用）
        sendSocket     % udpport オブジェクト（送信用）
        params
        startTime
    end
    
    methods (Access = public)
        function obj = UDPManager(params)
            obj.params = params;
            obj.startTime = tic;  % 計測開始時間の初期化
            obj.initializeUDP();
        end
        
        function trigger = receiveTrigger(obj)
            trigger = [];
            try
                if obj.receiveSocket.NumDatagramsAvailable > 0
                    % 利用可能なバイト数を取得して受信
                    receivedData = read(obj.receiveSocket,  'uint8');
                    
                    % 受信データを UTF-8 文字列に変換（行列の向きを調整）
                    receivedStr = native2unicode(receivedData', 'UTF-8');
                    receivedStr = strtrim(receivedStr);
                    
                    % デバッグ用出力
                    fprintf('Received raw data: %s\n', receivedStr);
                    
                    if ~isempty(receivedStr)
                        % トリガーマッピングから対応する数値を検索
                        mappings = obj.params.udp.receive.triggers.mappings;
                        fields = fieldnames(mappings);
                        triggerValue = obj.params.udp.receive.triggers.defaultValue;
                        
                        for i = 1:length(fields)
                            if strcmp(mappings.(fields{i}).text, receivedStr)
                                triggerValue = mappings.(fields{i}).value;
                                break;
                            end
                        end
                        
                        currentTime = toc(obj.startTime);
                        trigger = struct(...
                            'value', triggerValue, ...
                            'time', uint64(currentTime * 1000), ...
                            'sample', [] ...
                        );
                        
                        % デバッグ用の出力
                        fprintf('Mapped trigger: %s -> %d\n', receivedStr, triggerValue);
                    end
                    
                    drawnow;
                end
            catch ME
                warning(ME.identifier, 'UDP receive error: %s', ME.message);
            end
        end
        
        function sendTrigger(obj, trigger)
            try
                fprintf('----------------------------------------\n');
                if isstruct(trigger)
                    % 構造体を JSON に変換して送信
                    jsonStr = jsonencode(trigger);
                    
                    if strlength(jsonStr) > obj.params.udp.send.bufferSize
                        warning('JSON data size exceeds UDP buffer size!');
                        fprintf('JSON content preview: %.100s...\n', jsonStr);
                    end
                    
                    % 文字列を uint8 配列に変換して送信
                    data = uint8(jsonStr);
                    write(obj.sendSocket, data, 'uint8', obj.params.udp.send.address, obj.params.udp.send.port);
                    fprintf('UDP Sent: %s\n', jsonStr);
                    
                elseif iscategorical(trigger)
                    % categorical 型データの送信
                    triggerValue = double(trigger);
                    bytes = typecast(int32(triggerValue), 'uint8');
                    write(obj.sendSocket, bytes, 'uint8', obj.params.udp.send.address, obj.params.udp.send.port);
                    
                elseif isnumeric(trigger)
                    if mod(trigger, 1) == 0
                        bytes = typecast(int32(trigger), 'uint8');
                    else
                        bytes = typecast(single(trigger), 'uint8');
                    end
                    write(obj.sendSocket, bytes, 'uint8', obj.params.udp.send.address, obj.params.udp.send.port);
                else
                    error('Unsupported trigger data type: %s', class(trigger));
                end
                
                fprintf('----------------------------------------\n');
                
            catch ME
                warning(ME.identifier, 'UDP send error: %s', ME.message);
                fprintf('ERROR: Failed to send trigger: %s\n', ME.message);
                % エラー発生時の詳細情報
                fprintf('Error details:\n');
                fprintf('  Data type: %s\n', class(trigger));
                if isstruct(trigger)
                    fprintf('  Struct fields: %s\n', strjoin(fieldnames(trigger), ', '));
                end
                fprintf('  Stack trace:\n');
                for i = 1:length(ME.stack)
                    fprintf('    File: %s, Line: %d, Function: %s\n', ...
                        ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
                end
            end
        end
        
        function updateReceiveAddress(obj, address, port)
            % 受信アドレスの動的更新（udpport ではバインドするローカルIP の指定はできないため port のみ再生成）
            try
                if nargin < 3
                    port = obj.params.udp.receive.port;
                end
                
                % 既存のソケットを解放
                if ~isempty(obj.receiveSocket)
                    obj.receiveSocket = [];
                end
                
                % 新しい受信用 udpport の作成
                obj.receiveSocket = udpport('datagram', ...
                    'LocalPort', port, ...
                    'ReceiveBufferSize', obj.params.udp.receive.bufferSize);
                fprintf('INFO: Receive socket updated to port: %d\n', port);
                
            catch ME
                warning(ME.identifier, 'Failed to update receive address: %s', ME.message);
            end
        end
        
        function updateSendAddress(obj, address, port)
            % 送信アドレスの動的更新（udpport の送信時は送信先アドレス・ポートを write() の引数で指定）
            try
                if nargin < 3
                    port = obj.params.udp.send.port;
                end
                
                % 既存のソケットを解放
                if ~isempty(obj.sendSocket)
                    obj.sendSocket = [];
                end
                
                % 新しい送信用 udpport の作成（送信用は特にローカルポートをバインドする必要はありません）
                obj.sendSocket = udpport('datagram');
                fprintf('INFO: Send socket updated to address: %s, port: %d\n', address, port);
                
            catch ME
                warning(ME.identifier, 'Failed to update send address: %s', ME.message);
            end
        end
        
        function delete(obj)
            try
                if ~isempty(obj.receiveSocket)
                    obj.receiveSocket = [];
                end
                if ~isempty(obj.sendSocket)
                    obj.sendSocket = [];
                end
            catch ME
                warning(ME.identifier, 'UDP cleanup error: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function initializeUDP(obj)
            try
                % UDP の初期化
                % 受信用ソケットの設定
                receiveAddress = obj.params.udp.receive.address;
                obj.receiveSocket = udpport('datagram', ...
                    'LocalPort', obj.params.udp.receive.port);
                fprintf('INFO: Receive socket initialized at %s:%d\n', ...
                    receiveAddress, obj.params.udp.receive.port);
                
                % 送信用ソケットの設定
                sendAddress = obj.params.udp.send.address;
                obj.sendSocket = udpport('datagram');
                fprintf('INFO: Send socket initialized for remote address %s:%d\n', ...
                    sendAddress, obj.params.udp.send.port);
                
            catch ME
                warning(ME.identifier, 'UDP initialization error: %s', ME.message);
                obj.cleanup();
                rethrow(ME);
            end
        end
        
        function logTriggerInfo(~, trigger)
            fprintf('=== Trigger Information ===\n');
            fprintf('Data type: %s\n', class(trigger));
            
            if isstruct(trigger)
                fprintf('Structure fields:\n');
                fields = fieldnames(trigger);
                for i = 1:length(fields)
                    fprintf('  %s: %s\n', fields{i}, class(trigger.(fields{i})));
                end
            elseif iscategorical(trigger)
                fprintf('Categories: %s\n', strjoin(string(categories(trigger)), ', '));
                fprintf('Value: %s (numeric: %d)\n', char(trigger), double(trigger));
            elseif isnumeric(trigger)
                fprintf('Value: %d\n', trigger);
            elseif ischar(trigger) || isstring(trigger)
                fprintf('Value: %s\n', char(trigger));
            end
            
            fprintf('========================\n');
        end
        
        function cleanup(obj)
            if ~isempty(obj.receiveSocket)
                obj.receiveSocket = [];
            end
            if ~isempty(obj.sendSocket)
                obj.sendSocket = [];
            end
        end
    end
end