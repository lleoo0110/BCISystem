classdef DataLoader < handle
    properties
        params      % 設定パラメータ
        dataManager % データ管理オブジェクト
    end
    
    methods
        function obj = DataLoader(params)
            % 設定パラメータを保存し、データマネージャを初期化
            obj.params = params;
            obj.dataManager = DataManager(params);
        end
        
        function [loadedData, fileInfo] = loadDataBrowser(obj)
            try
                % ファイル選択ダイアログの表示
                [filenames, filepath] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, ...
                    'データファイルを選択', 'MultiSelect', 'on');
                
                if isequal(filenames, 0)
                    loadedData = {};
                    fileInfo = [];
                    return;
                end
                
                % 単一ファイル選択の場合はセル配列に変換
                if ~iscell(filenames)
                    filenames = {filenames};
                end
                
                % ファイル情報の構造体を作成
                fileInfo = struct();
                fileInfo.filenames = filenames;
                fileInfo.filepath = filepath;
                fileInfo.fullpaths = cellfun(@(f) fullfile(filepath, f), ...
                    filenames, 'UniformOutput', false);
                
                % データの読み込み
                loadedData = cell(1, length(filenames));
                
                for i = 1:length(filenames)
                    try
                        fullpath = fullfile(filepath, filenames{i});
                        currentData = obj.dataManager.loadDataset(fullpath);
                        
                        % 基本的な検証のみ実施
                        if ~isfield(currentData, 'rawData')
                            warning('DataLoader:NoRawData', 'ファイル %s に rawData がありません', filenames{i});
                        end
                        
                        loadedData{i} = currentData;
                        fprintf('読み込み完了: %s\n', filenames{i});
                    catch ME
                        warning('DataLoader:LoadError', 'ファイル %s の読み込みエラー: %s', filenames{i}, ME.message);
                        loadedData{i} = [];
                    end
                end
                
            catch ME
                error('DataLoader:LoadError', 'データ読み込みエラー: %s', ME.message);
            end
        end

        function [loadedData, fileInfo] = loadData(obj, fullpath)
            try
                % ファイルパスからファイル名とディレクトリを取得
                [filepath, filename, ext] = fileparts(fullpath);
                filenameWithExt = [filename, ext];

                if isempty(filenameWithExt)
                    loadedData = {};
                    fileInfo = [];
                    warning('DataLoader:InvalidPath', '指定されたパスが不正です: %s', fullpath);
                    return;
                end

                % ファイル情報の構造体を作成
                fileInfo = struct();
                fileInfo.filenames = {filenameWithExt};
                fileInfo.filepath = filepath;
                fileInfo.fullpaths = {fullpath};

                % データの読み込み
                loadedData = cell(1, 1); % 単一ファイルなのでサイズは1

                try
                    currentData = obj.dataManager.loadDataset(fullpath);

                    % 基本的な検証のみ実施
                    if ~isfield(currentData, 'rawData')
                        warning('DataLoader:NoRawData', 'ファイル %s に rawData がありません', filenameWithExt);
                    end

                    loadedData{1} = currentData;
                    fprintf('読み込み完了: %s\n', filenameWithExt);

                catch ME
                    warning('DataLoader:LoadError', 'ファイル %s の読み込みエラー: %s', filenameWithExt, ME.message);
                    loadedData{1} = [];
                end

            catch ME
                error('DataLoader:LoadError', 'データ読み込みエラー: %s', ME.message);
            end
        end
        
        function success = saveData(obj, data)
            try
                % 保存先の選択
                [filename, pathname] = uiputfile('*.mat', '保存先を選択してください');
                
                if isequal(filename, 0)
                    success = false;
                    return;
                end
                
                % データの保存
                fullpath = fullfile(pathname, filename);
                obj.dataManager.saveDataset(data, fullpath);
                
                fprintf('データが保存されました: %s\n', fullpath);
                success = true;
                
            catch ME
                error('DataLoader:SaveError', 'データ保存エラー: %s', ME.message);
            end
        end
        
        function validateDeviceConfig(obj, data)
            if size(data.rawData, 1) ~= obj.params.device.channelCount
                error('DataLoader:ChannelMismatch', ...
                    'データのチャンネル数(%d)がデバイス設定(%d)と一致しません。\n設定ファイルのデバイス種別を確認してください。', ...
                    size(data.rawData, 1), obj.params.device.channelCount);
            end

            if isfield(data, 'channelNames') && length(data.channelNames) ~= length(obj.params.device.channels)
                warning('DataLoader:ChannelNameMismatch', ...
                    'チャンネル名の数がデバイス設定と異なります。\n期待値: %s\n実際: %s', ...
                    strjoin(obj.params.device.channels, ', '), ...
                    strjoin(data.channelNames, ', '));
            end
        end
    end
end