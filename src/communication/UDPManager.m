classdef UDPManager < handle
    properties (Access = private)
        receiveSocket
        sendSocket
        params
        startTime
        remoteAddress
        remotePort
    end
    
    methods (Access = public)
        function obj = UDPManager(params)
            obj.params = params;
            obj.startTime = tic;
            obj.remoteAddress = params.udp.send.address;
            obj.remotePort = params.udp.send.port;
            obj.initializeUDP();
        end
        
        function trigger = receiveTrigger(obj)
            trigger = [];
            try
                if obj.receiveSocket.NumDatagramsAvailable > 0
                    datagram = read(obj.receiveSocket, 1, "uint8"); 
                    receivedData = datagram.Data;
                    
                    if isempty(receivedData)
                        return;
                    end
                    
                    try
                        receivedStr = native2unicode(receivedData', 'UTF-8'); 
                        receivedStr = strtrim(receivedStr);
                        receivedStr = receivedStr(:)'; % 形を横ベクトルに統一
                        
                        if isempty(receivedStr)
                            return;
                        end

                        mappings = obj.params.udp.receive.triggers.mappings;
                        fields = fieldnames(mappings);
                        triggerValue = obj.params.udp.receive.triggers.defaultValue;

                        for i = 1:length(fields)
                            mappingText = mappings.(fields{i}).text(:)'; % マッピングデータも横ベクトル化
        
                            if isequal(mappingText, receivedStr)
                                triggerValue = mappings.(fields{i}).value;
                                break;
                            end
                        end
                        
                        trigger = struct('value', triggerValue, 'time', uint64(toc(obj.startTime) * 1000), 'sample', []);
                        fprintf('Trigger received - Value: %d\n', triggerValue);
                    catch ME
                        warning(ME.identifier, '%s', ME.message);
                        disp(getReport(ME, 'extended'));
                    end
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
                disp(getReport(ME, 'extended'));
            end
        end

        function sendTrigger(obj, trigger)
            try
                if isstruct(trigger)
                    jsonStr = jsonencode(trigger);
                    if strlength(jsonStr) > obj.params.udp.send.bufferSize
                        warning('JSON data size exceeds UDP buffer size!');
                    end
                    data = uint8(jsonStr);
                else
                    data = uint8(num2str(double(trigger)));
                end
                
                write(obj.sendSocket, data, "uint8", obj.remoteAddress, obj.remotePort);
                fprintf('UDP received - Value: %d\n', trigger);
            catch ME
                warning(ME.identifier, '%s', ME.message);
                disp(getReport(ME, 'extended'));
            end
        end
        
        function updateReceiveAddress(obj, address, port)
            try
                if nargin < 3
                    port = obj.params.udp.receive.port;
                end
                if ~isempty(obj.receiveSocket) && isvalid(obj.receiveSocket)
                    delete(obj.receiveSocket);
                end
                obj.receiveSocket = udpport("datagram", "LocalHost", address, "LocalPort", port, "EnablePortSharing", true);
            catch ME
                warning(ME.identifier, '%s', ME.message);
                disp(getReport(ME, 'extended'));
            end
        end
        
        function updateSendAddress(obj, address, port)
            try
                if nargin < 3
                    port = obj.params.udp.send.port;
                end
                obj.remoteAddress = address;
                obj.remotePort = port;
            catch ME
                warning(ME.identifier, '%s', ME.message);
                disp(getReport(ME, 'extended'));
            end
        end
        
        function delete(obj)
            try
                if ~isempty(obj.receiveSocket) && isvalid(obj.receiveSocket)
                    flush(obj.receiveSocket); % 受信バッファをクリア
                    delete(obj.receiveSocket);
                    obj.receiveSocket = [];
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
            
            try
                if ~isempty(obj.sendSocket) && isvalid(obj.sendSocket)
                    delete(obj.sendSocket);
                    obj.sendSocket = [];
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function initializeUDP(obj)
            try
                if isempty(obj.receiveSocket) || ~isvalid(obj.receiveSocket)
                    obj.receiveSocket = udpport("datagram", ...
                        "LocalHost", obj.params.udp.receive.address, ...
                        "LocalPort", obj.params.udp.receive.port, ...
                        "EnablePortSharing", true);
                end
        
                if isempty(obj.sendSocket) || ~isvalid(obj.sendSocket)
                    obj.sendSocket = udpport("datagram");
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
                obj.cleanup();
                rethrow(ME);
            end
        end
        
        function cleanup(obj)
            try
                if ~isempty(obj.receiveSocket) && isvalid(obj.receiveSocket)
                    flush(obj.receiveSocket);
                    delete(obj.receiveSocket);
                    obj.receiveSocket = [];
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        
            try
                if ~isempty(obj.sendSocket) && isvalid(obj.sendSocket)
                    delete(obj.sendSocket);
                    obj.sendSocket = [];
                end
            catch ME
                warning(ME.identifier, '%s', ME.message);
            end
        end
    end
end