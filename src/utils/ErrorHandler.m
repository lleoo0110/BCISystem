classdef ErrorHandler < handle
    properties (Access = private)
        logger          % ロガー
        lastError       % 最後のエラー
        errorCount     % エラーカウント
    end
    
    properties (Constant)
        MAX_ERROR_COUNT = 5;  % 最大エラー数
    end
    
    methods (Access = public)
        function obj = ErrorHandler(logger)
            obj.logger = logger;
            obj.errorCount = 0;
            obj.lastError = [];
        end
        
        function handleError(obj, ME, severity)
            % エラーの処理
            if nargin < 3
                severity = 'error';  % デフォルトの重要度
            end
            
            try
                % エラーカウントの更新
                obj.errorCount = obj.errorCount + 1;
                obj.lastError = ME;
                
                % エラー情報のログ記録
                obj.logError(ME, severity);
                
                % エラーカウントのチェック
                if obj.errorCount >= obj.MAX_ERROR_COUNT
                    obj.handleCriticalError(ME);
                end
                
            catch InnerME
                warning('Error in error handler: %s', InnerME.message);
            end
        end
        
        function clearErrors(obj)
            % エラー状態のリセット
            obj.errorCount = 0;
            obj.lastError = [];
        end
        
        function status = getErrorStatus(obj)
            % エラー状態の取得
            status = struct(...
                'hasError', ~isempty(obj.lastError), ...
                'errorCount', obj.errorCount, ...
                'lastError', obj.lastError);
        end
    end
    
    methods (Access = private)
        function logError(obj, ME, severity)
            % エラー情報のログ記録
            if isempty(obj.logger)
                warning('Logger not available for error handling');
                return;
            end
            
            % スタックトレースの取得
            stack = getReport(ME, 'extended', 'hyperlinks', 'off');
            
            % エラーメッセージの構築
            errorMsg = sprintf('Error occurred: %s\nStack trace:\n%s', ...
                ME.message, stack);
            
            % ログレベルに応じた記録
            switch lower(severity)
                case 'warning'
                    obj.logger.warning(errorMsg);
                case 'error'
                    obj.logger.error(errorMsg);
                case 'critical'
                    obj.logger.error('CRITICAL: %s', errorMsg);
                otherwise
                    obj.logger.error(errorMsg);
            end
        end
        
        function handleCriticalError(obj, ME)
            % 重大エラーの処理
            criticalMsg = sprintf(...
                'Maximum error count reached (%d). Last error: %s', ...
                obj.MAX_ERROR_COUNT, ME.message);
            
            obj.logger.error('CRITICAL: %s', criticalMsg);
            error('CriticalError:MaxErrorCount', criticalMsg);
        end
    end
end