classdef UDPManager < handle
    properties (Access = private)
        receiveSocket
        sendSocket
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
                if obj.receiveSocket.BytesAvailable > 0
                    % データを受信
                    receivedData = fread(obj.receiveSocket, obj.receiveSocket.BytesAvailable, 'uint8');
                    % UTF8でデコード
                    receivedStr = native2unicode(receivedData', 'UTF-8');
                    % 文字列の前後の空白を削除
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
                end
            catch ME
                warning(ME.identifier, 'UDP receive error: %s', ME.message);
            end
        end
        
        function sendTrigger(obj, trigger)
            try
                if ischar(trigger) || isstring(trigger)
                    % 文字列データの送信
                    fprintf(obj.sendSocket, char(trigger));
                    fprintf('Sent text message: %s\n', char(trigger));

                elseif iscategorical(trigger)
                    % categorical型データの送信
                    triggerValue = double(trigger);
                    bytes = typecast(int32(triggerValue), 'uint8');
                    fwrite(obj.sendSocket, bytes, 'uint8');
                    fprintf('Sent categorical trigger value: %d\n', triggerValue);

                elseif isnumeric(trigger)
                    % 数値データの送信
                    if mod(trigger, 1) == 0
                        % 整数値
                        bytes = typecast(int32(trigger), 'uint8');
                        dataType = 'integer';
                    else
                        % 浮動小数点値
                        bytes = typecast(single(trigger), 'uint8');
                        dataType = 'float';
                    end

                    % データ送信
                    fwrite(obj.sendSocket, bytes, 'uint8');
                    fprintf('Sent %s trigger: %d\n', dataType, trigger);

                else
                    % その他のデータ型の場合はエラー
                    error('Unsupported trigger data type: %s', class(trigger));
                end

                % 送信確認のログ
                fprintf('INFO: Trigger sent successfully\n');

            catch ME
                warning(ME.identifier, 'UDP send error: %s', ME.message);
                % エラーの詳細をログに記録
                fprintf('ERROR: Failed to send trigger: %s\n', ME.message);
            end
        end
        
        function updateReceiveAddress(obj, address, port)
            % 受信アドレスの動的更新
            try
                if nargin < 3
                    port = obj.params.udp.receive.port;
                end
                
                % 既存のソケットをクリーンアップ
                if ~isempty(obj.receiveSocket)
                    fclose(obj.receiveSocket);
                    delete(obj.receiveSocket);
                end
                
                % 新しいソケットの作成
                obj.receiveSocket = udp(address, ...
                    'LocalPort', port, ...
                    'InputBufferSize', obj.params.udp.receive.bufferSize);
                fopen(obj.receiveSocket);
                fprintf('INFO: Receive socket updated to address: %s, port: %d\n', address, port);
                
            catch ME
                warning(ME.identifier, 'Failed to update receive address: %s', ME.message);
            end
        end
        
        function updateSendAddress(obj, address, port)
            % 送信アドレスの動的更新
            try
                if nargin < 3
                    port = obj.params.udp.send.port;
                end
                
                % 既存のソケットをクリーンアップ
                if ~isempty(obj.sendSocket)
                    fclose(obj.sendSocket);
                    delete(obj.sendSocket);
                end
                
                % 新しいソケットの作成
                obj.sendSocket = udp(address, ...
                    'RemotePort', port, ...
                    'OutputBufferSize', obj.params.udp.send.bufferSize);
                fopen(obj.sendSocket);
                fprintf('INFO: Send socket updated to address: %s, port: %d\n', address, port);
                
            catch ME
                warning(ME.identifier, 'Failed to update send address: %s', ME.message);
            end
        end
        
        function delete(obj)
            try
                if ~isempty(obj.receiveSocket)
                    fclose(obj.receiveSocket);
                    delete(obj.receiveSocket);
                end
                if ~isempty(obj.sendSocket)
                    fclose(obj.sendSocket);
                    delete(obj.sendSocket);
                end
            catch ME
                warning(ME.identifier, 'UDP cleanup error: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function initializeUDP(obj)
            try
                % UDPの初期化
                instrreset;
                
                % 受信用ソケットの設定
                receiveAddress = obj.params.udp.receive.address;
                obj.receiveSocket = udp(receiveAddress, ...
                    'LocalPort', obj.params.udp.receive.port, ...
                    'InputBufferSize', obj.params.udp.receive.bufferSize);
                fopen(obj.receiveSocket);
                fprintf('INFO: Receive socket initialized at %s:%d\n', ...
                    receiveAddress, obj.params.udp.receive.port);
                
                % 送信用ソケットの設定
                sendAddress = obj.params.udp.send.address;
                obj.sendSocket = udp(sendAddress, ...
                    'RemotePort', obj.params.udp.send.port, ...
                    'OutputBufferSize', obj.params.udp.send.bufferSize);
                fopen(obj.sendSocket);
                fprintf('INFO: Send socket initialized at %s:%d\n', ...
                    sendAddress, obj.params.udp.send.port);
                
            catch ME
                warning(ME.identifier, 'UDP initialization error: %s', ME.message);
                obj.cleanup();
                rethrow(ME);
            end
        end
        
        function cleanup(obj)
            if ~isempty(obj.receiveSocket)
                fclose(obj.receiveSocket);
                delete(obj.receiveSocket);
            end
            if ~isempty(obj.sendSocket)
                fclose(obj.sendSocket);
                delete(obj.sendSocket);
            end
        end
    end
end