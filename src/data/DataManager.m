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
        
        function savedFilePath = saveDataset(obj, data, filePath)
            try
                % タイムスタンプの生成
                timestamp = datetime('now');
                
                % ファイル名が指定されていない場合は生成
                if nargin < 3 || isempty(filePath)
                    filePath = obj.generateFilename(timestamp);
                end
                
                % データストアの初期化
                obj.dataStore = struct();
                
                % データの保存準備
                if isfield(obj.params, 'acquisition') && ...
                   isfield(obj.params.acquisition, 'save') && ...
                   isfield(obj.params.acquisition.save, 'fields')
                    % パラメータに基づく保存フィールドの設定
                    saveFields = obj.params.acquisition.save.fields;
                else
                    % デフォルトの保存フィールド
                    saveFields = DataManager.getDefaultSaveFields();
                end
                
                % フィールドの保存処理
                fields = fieldnames(saveFields);
                for i = 1:length(fields)
                    field = fields{i};
                    if saveFields.(field) && ...
                       ~strcmp(field, 'timestamp') && ...
                       ~strcmp(field, 'params')
                        if isfield(data, field)
                            obj.dataStore.(field) = data.(field);
                        else
                            warning('DataManager:MissingField', ...
                                'フィールド "%s" は有効ですが、データに存在しません', field);
                        end
                    end
                end

                % パラメータの追加
                if isfield(saveFields, 'params') && saveFields.params
                    obj.dataStore.params = obj.params;
                end

                % 保存するデータがあるか確認
                if isempty(fieldnames(obj.dataStore))
                    warning('DataManager:NoData', '保存するデータがありません');
                    savedFilePath = '';
                    return;
                end

                % 保存ディレクトリの確認と作成
                savePath = fileparts(filePath);
                if ~isempty(savePath) && ~exist(savePath, 'dir')
                    mkdir(savePath);
                end

                % データの保存
                saveStruct = obj.dataStore;
                save(filePath, '-struct', 'saveStruct', '-v7.3');

                % 保存情報の表示
                fprintf('データを保存しました: %s\n', filePath);
                fprintf('保存されたフィールド:\n');
                savedFields = fieldnames(obj.dataStore);
                for i = 1:length(savedFields)
                    fprintf('  - %s\n', savedFields{i});
                end

                % 保存したファイルパスを返す
                savedFilePath = filePath;

            catch ME
                error('DataManager:SaveError', 'データ保存に失敗: %s', ME.message);
            end
        end
        
        function data = loadDataset(obj, filePath)
            try
                % ファイルの存在確認
                if ~exist(filePath, 'file')
                    error('ファイルが見つかりません: %s', filePath);
                end

                % データの読み込み
                loadedData = load(filePath);

                % 返却データの準備
                data = struct();

                % 読み込みフィールドの設定
                if isfield(obj.params, 'acquisition') && ...
                   isfield(obj.params.acquisition, 'load') && ...
                   isfield(obj.params.acquisition.load, 'fields')
                    % パラメータに基づく読み込みフィールドの設定
                    loadFields = obj.params.acquisition.load.fields;
                else
                    % デフォルトの読み込みフィールド
                    loadFields = DataManager.getDefaultLoadFields();
                end

                % フィールドの読み込み処理
                fields = fieldnames(loadFields);
                loadedFields = fieldnames(loadedData);

                for i = 1:length(fields)
                    field = fields{i};
                    if loadFields.(field)  % 読み込みが有効な場合
                        if ismember(field, loadedFields)
                            data.(field) = loadedData.(field);
                            fprintf('読み込んだフィールド: %s\n', field);
                        else
                            warning('DataManager:LoadField', ...
                                'フィールド "%s" は有効ですが、ファイルに存在しません', field);
                        end
                    else
                        if ismember(field, loadedFields)
                            fprintf('フィールド "%s" は利用可能ですが、読み込みが無効です\n', field);
                        end
                    end
                end

                % 読み込んだデータが空かどうかの確認
                if isempty(fieldnames(data))
                    warning('DataManager:NoData', 'データが読み込まれませんでした');
                else
                    fprintf('\n%d個のフィールドを読み込みました: %s\n', ...
                        length(fieldnames(data)), filePath);
                end

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
    
    methods (Access = private)
        function filename = generateFilename(obj, timestamp)
            % 基本ファイル名の生成
            if isfield(obj.params, 'acquisition') && ...
               isfield(obj.params.acquisition, 'save')
                baseName = obj.params.acquisition.save.name;
                savePath = obj.params.acquisition.save.path;
            else
                baseName = 'dataset';
                savePath = './data';
            end
            
            baseFilename = sprintf('%s_%s', ...
                baseName, ...
                datestr(timestamp, 'yyyymmdd_HHMMSS'));
            
            % パスの結合
            filename = fullfile(savePath, sprintf('%s.mat', baseFilename));
            
            % ファイルが既に存在する場合の処理
            counter = 1;
            while exist(filename, 'file')
                filename = fullfile(savePath, ...
                    sprintf('%s_%d.mat', baseFilename, counter));
                counter = counter + 1;
            end
        end
    end
    
    methods (Static, Access = private)
        function fields = getDefaultSaveFields()
            % デフォルトの保存フィールド
            fields = struct(...
                'params', true, ...         % 設定情報
                'rawData', true, ...        % 生脳波データ
                'labels', true, ...         % イベントマーカー
                'processedData', true, ...  % 処理済みデータ
                'processedLabel', true, ... % 処理済みラベル
                'processingInfo', true, ... % 処理情報
                'classifier', true, ...     % 分類器
                'results', true ...         % 解析結果
            );
        end
        
        function fields = getDefaultLoadFields()
            % デフォルトの読み込みフィールド
            fields = struct(...
                'params', true, ...         % 設定情報
                'rawData', true, ...        % 生脳波データ
                'labels', true, ...         % イベントマーカー
                'processedData', true, ...  % 処理済みデータ
                'processedLabel', true, ... % 処理済みラベル
                'processingInfo', true, ... % 処理情報
                'classifier', true, ...     % 分類器
                'results', true ...         % 解析結果
            );
        end
    end
end