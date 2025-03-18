classdef DataManager < handle
    properties (Access = private)
        params      % 設定パラメータ
    end
    
    methods (Access = public)
        function obj = DataManager(params)
            obj.params = params;
        end
        
        function savedFilePath = saveDataset(~, data, filePath)
            try
                % タイムスタンプの生成
                timestamp = datetime('now');
                
                % ファイル名が指定されていない場合はエラー
                if nargin < 3 || isempty(filePath)
                    error('DataManager:NoFilePath', '保存先ファイルパスが指定されていません');
                end
                
                % 保存ディレクトリの確認と作成
                savePath = fileparts(filePath);
                if ~isempty(savePath) && ~exist(savePath, 'dir')
                    mkdir(savePath);
                end

                % データの保存（すべてのフィールドを保存）
                saveStruct = data;
                
                % タイムスタンプの追加
                if ~isfield(saveStruct, 'timestamp')
                    saveStruct.timestamp = timestamp;
                end
                
                % データが空かどうか確認
                if isempty(fieldnames(saveStruct))
                    warning('DataManager:NoData', '保存するデータがありません');
                    savedFilePath = '';
                    return;
                end

                % データの保存
                save(filePath, '-struct', 'saveStruct', '-v7.3');

                % 保存情報の表示
                fprintf('データを保存しました: %s\n', filePath);
                fprintf('保存されたフィールド数: %d\n', length(fieldnames(saveStruct)));

                % 保存したファイルパスを返す
                savedFilePath = filePath;

            catch ME
                error('DataManager:SaveError', 'データ保存に失敗: %s', ME.message);
            end
        end
        
        function data = loadDataset(~, filePath)
            try
                % ファイルの存在確認
                if ~exist(filePath, 'file')
                    error('ファイルが見つかりません: %s', filePath);
                end

                % データの読み込み（すべてのフィールドを読み込み）
                data = load(filePath);
                
                % 読み込み情報の表示
                fprintf('データを読み込みました: %s\n', filePath);

            catch ME
                error('DataManager:LoadError', 'データ読み込みに失敗: %s', ME.message);
            end
        end
        
        function info = getDatasetInfo(~, filePath)
            try
                % データの読み込み
                data = load(filePath);
                
                % 情報構造体の作成
                info = struct();
                
                % 基本情報
                info.filename = filePath;
                info.fileSize = dir(filePath).bytes;
                info.savedFields = fieldnames(data);
                
                if isfield(data, 'timestamp')
                    info.timestamp = data.timestamp;
                end
                
                % 各フィールドの情報
                info.fieldInfo = struct();
                fields = fieldnames(data);
                
                for i = 1:length(fields)
                    field = fields{i};
                    currentData = data.(field);
                    
                    if isnumeric(currentData) || islogical(currentData)
                        info.fieldInfo.(field) = struct(...
                            'size', size(currentData), ...
                            'class', class(currentData), ...
                            'bytes', whos('currentData').bytes);
                            
                    elseif isstruct(currentData)
                        info.fieldInfo.(field) = struct(...
                            'type', 'struct', ...
                            'fields', fieldnames(currentData), ...
                            'bytes', whos('currentData').bytes);
                            
                    elseif iscell(currentData)
                        info.fieldInfo.(field) = struct(...
                            'type', 'cell', ...
                            'size', size(currentData), ...
                            'bytes', whos('currentData').bytes);
                    end
                end
                
                % メモリ使用量の合計
                info.totalBytes = sum(structfun(@(x) x.bytes, info.fieldInfo));
                
            catch ME
                error('DataManager:InfoError', 'データセット情報の取得に失敗: %s', ME.message);
            end
        end
    end
end