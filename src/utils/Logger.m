classdef Logger < handle
    properties (Access = private)
        logFile
        logLevel
    end
    
    methods (Access = public)
        function obj = Logger(savePath)
            logFileName = fullfile(savePath, sprintf('log_%s.txt', ...
                datestr(now, 'yyyymmdd_HHMMSS')));
            obj.logFile = fopen(logFileName, 'w');
            obj.logLevel = 1; % 1: INFO, 2: WARNING, 3: ERROR
        end
        
        function delete(obj)
            if ~isempty(obj.logFile) && obj.logFile ~= -1
                fclose(obj.logFile);
            end
        end
        
        function info(obj, message, varargin)
            if obj.logLevel <= 1
                obj.writeLog('INFO', message, varargin{:});
            end
        end
        
        function warning(obj, message, varargin)
            if obj.logLevel <= 2
                obj.writeLog('WARNING', message, varargin{:});
            end
        end
        
        function error(obj, message, varargin)
            if obj.logLevel <= 3
                obj.writeLog('ERROR', message, varargin{:});
            end
        end
    end
    
    methods (Access = private)
        function writeLog(obj, level, message, varargin)
            timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
            formattedMessage = sprintf(message, varargin{:});
            logMessage = sprintf('%s [%s] %s\n', timestamp, level, formattedMessage);
            
            % ファイルに書き込み
            if ~isempty(obj.logFile) && obj.logFile ~= -1
                fprintf(obj.logFile, logMessage);
            end
            
            % コンソールにも出力
            fprintf(logMessage);
        end
    end
end