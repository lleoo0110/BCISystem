classdef DataManager < handle
    properties (Access = private)
        params      % 設定パラメータ
        dataStore   % データ保存用構造体
    end
    
    methods (Access = public)
        function obj = DataManager(params)
            obj.params = params;
            obj.dataStore = struct();
        end
        
        function savedFilenamePath = saveDataset(obj, data)
            try
                % タイムスタンプの生成
                timestamp = datetime('now');

                % ファイル名の生成
                filename = obj.generateFilename(timestamp);

                % データストアの初期化
                obj.dataStore = struct();

                % 設定に基づいてデータを保存
                if ~obj.params.acquisition.save.enable
                    error('DataManager:SaveDisabled', 'Save functionality is disabled in parameters');
                end

                fields = fieldnames(obj.params.acquisition.save.fields);
                for i = 1:length(fields)
                    field = fields{i};
                    % 設定で有効かつ，特別なフィールドでない場合のみチェック
                    if obj.params.acquisition.save.fields.(field) && ...
                       ~strcmp(field, 'timestamp') && ...
                       ~strcmp(field, 'params')
                        if isfield(data, field)
                            obj.dataStore.(field) = data.(field);
                        else
                            warning('DataManager:MissingField', ...
                                'Field "%s" is enabled for saving but not provided in data', field);
                        end
                    end
                end

                % パラメータの追加
                if obj.params.acquisition.save.fields.params
                    obj.dataStore.params = obj.params;
                end

                % 保存するデータがあるか確認
                if isempty(fieldnames(obj.dataStore))
                    warning('DataManager:NoData', 'No data to save');
                    return;
                end

                % 保存ディレクトリの確認と作成
                savePath = fileparts(filename);
                if ~exist(savePath, 'dir')
                    mkdir(savePath);
                end

                % データの保存
                saveStruct = obj.dataStore;
                save(filename, '-struct', 'saveStruct');

                % 保存情報の表示
                fprintf('Dataset saved successfully to: %s\n', filename);
                fprintf('Saved fields: \n');
                savedFields = fieldnames(obj.dataStore);
                for i = 1:length(savedFields)
                    fprintf('  - %s\n', savedFields{i});
                end
                
                % 保存したファイルパスを返す
                savedFilenamePath = filename;

            catch ME
                error('DataManager:SaveError', 'Failed to save dataset: %s', ME.message);
            end
        end
        
        function data = loadDataset(obj, filenamePath)
            try
                % ファイルの存在確認
                if ~exist(filenamePath, 'file')
                    error('File not found: %s', filenamePath);
                end

                % データの読み込み
                loadedData = load(filenamePath);

                % 返却データの準備
                data = struct();

                % 設定に基づいてデータを読み込み
                fields = fieldnames(obj.params.acquisition.load.fields);
                loadedFields = fieldnames(loadedData);

                for i = 1:length(fields)
                    field = fields{i};
                    if obj.params.acquisition.load.fields.(field)  % ロードが有効な場合
                        if ismember(field, loadedFields)
                            data.(field) = loadedData.(field);
                            fprintf('Loaded field: %s\n', field);
                        else
                            warning('DataManager:LoadField', ...
                                'Field "%s" is enabled for loading but not found in file', field);
                        end
                    else
                        if ismember(field, loadedFields)
                            fprintf('Field "%s" is available but not enabled for loading\n', field);
                        end
                    end
                end

                % 読み込んだデータが空かどうかの確認
                if isempty(fieldnames(data))
                    warning('DataManager:NoData', 'No data was loaded');
                else
                    fprintf('\nSuccessfully loaded %d fields from: %s\n', ...
                        length(fieldnames(data)), filenamePath);
                end

            catch ME
                error('DataManager:LoadError', 'Failed to load dataset: %s', ME.message);
            end
        end
        
        function info = getDatasetInfo(~, filename)
            try
                data = load(filename);
                info = struct();
                
                % 基本情報
                info.filename = filename;
                info.fileSize = dir(filename).bytes;
                info.savedFields = fieldnames(data);
                if isfield(data, 'timestamp')
                    info.timestamp = data.timestamp;
                end
                
                % 各フィールドの情報
                fields = fieldnames(data);
                for i = 1:length(fields)
                    field = fields{i};
                    if isnumeric(data.(field)) || islogical(data.(field))
                        info.fieldInfo.(field) = struct(...
                            'size', size(data.(field)), ...
                            'class', class(data.(field)) ...
                        );
                    elseif isstruct(data.(field))
                        info.fieldInfo.(field) = struct(...
                            'type', 'struct', ...
                            'fields', fieldnames(data.(field)) ...
                        );
                    end
                end
                
            catch ME
                error('DataManager:InfoError', 'Failed to get dataset info: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function filename = generateFilename(obj, timestamp)
            % 基本ファイル名の生成
            baseFilename = sprintf('%s_%s', ...
                obj.params.acquisition.save.name, ...
                datestr(timestamp, 'yyyymmdd_HHMMSS'));
            
            % パスの結合
            filename = fullfile(obj.params.acquisition.save.path, ...
                sprintf('%s.mat', baseFilename));
            
            % ファイルが既に存在する場合の処理
            counter = 1;
            while exist(filename, 'file')
                filename = fullfile(obj.params.acquisition.save.path, ...
                    sprintf('%s_%d.mat', baseFilename, counter));
                counter = counter + 1;
            end
        end
    end
end