classdef DataLoader
    methods (Static)
        function loadedData = loadDataBrowserWithPrompt(purpose, options)
            % 目的に応じたデータ読み込みを行う
            % 
            % 入力:
            %   purpose - 読み込みの目的 ('analysis', 'online', 'normalization' など)
            %   options - 読み込みオプション (省略可)
            %
            % 出力:
            %   loadedData - 読み込まれたデータ
            
            try
                % デフォルトオプションの設定
                if nargin < 2
                    options = struct();
                end
                
                % 目的に応じたダイアログ設定
                dialogSettings = DataLoader.getDialogSettings(purpose);
                
                % ファイル選択ダイアログの表示
                [filenames, filepath] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, ...
                    dialogSettings.title, 'MultiSelect', 'on');
                
                if isequal(filenames, 0)
                    loadedData = {};
                    return;
                end
                
                % 単一ファイル選択の場合はセル配列に変換
                if ~iscell(filenames)
                    filenames = {filenames};
                end
                
                % 読み込みモードの選択（複数ファイルの場合のみ）
                if length(filenames) > 1
                    loadMode = DataLoader.selectLoadMode();
                    if isempty(loadMode)
                        loadedData = {};
                        return;
                    end
                else
                    loadMode = 'individual';
                end
                
                % データの読み込みと検証
                switch loadMode
                    case 'batch'
                        loadedData = DataLoader.loadBatchData(filepath, filenames, purpose, options);
                    case 'individual'
                        loadedData = DataLoader.loadIndividualData(filepath, filenames, purpose, options);
                end
                
            catch ME
                error('DataLoader:LoadError', 'データ読み込みエラー: %s', ME.message);
            end
        end
        
        function success = saveData(data, purpose, options)
            % データの保存を行う
            % 
            % 入力:
            %   data    - 保存するデータ
            %   purpose - 保存の目的
            %   options - 保存オプション
            %
            % 出力:
            %   success - 保存の成功/失敗
            
            try
                % デフォルトオプションの設定
                if nargin < 3
                    options = struct();
                end
                
                % 保存モードの選択
                saveMode = DataLoader.selectSaveMode();
                if isempty(saveMode)
                    success = false;
                    return;
                end
                
                % 保存先の選択
                switch saveMode
                    case 'batch'
                        outputPath = uigetdir('', '保存先フォルダを選択してください');
                        if isequal(outputPath, 0)
                            success = false;
                            return;
                        end
                        success = DataLoader.saveBatchData(data, outputPath, purpose, options);
                        
                    case 'individual'
                        success = DataLoader.saveIndividualData(data, purpose, options);
                end
                
            catch ME
                error('DataLoader:SaveError', 'データ保存エラー: %s', ME.message);
            end
        end
        
        function [validatedData, info] = validateData(data, purpose)
            % データの検証を行う
            % 
            % 入力:
            %   data    - 検証するデータ
            %   purpose - 検証の目的
            %
            % 出力:
            %   validatedData - 検証済みデータ
            %   info         - 検証情報
            
            try
                % 目的に応じた必須フィールドの定義
                requiredFields = DataLoader.getRequiredFields(purpose);
                
                % フィールドの存在チェック
                missingFields = DataLoader.checkRequiredFields(data, requiredFields);
                if ~isempty(missingFields)
                    error('DataLoader:ValidationError', ...
                        '必須フィールドが不足しています: %s', ...
                        strjoin(missingFields, ', '));
                end
                
                % データ形式の検証
                validatedData = DataLoader.validateDataFormat(data, purpose);
                
                % 検証情報の生成
                info = struct(...
                    'timestamp', datetime('now'), ...
                    'purpose', purpose, ...
                    'validationPassed', true, ...
                    'dataSize', DataLoader.getDataSize(validatedData));
                
            catch ME
                error('DataLoader:ValidationError', 'データ検証エラー: %s', ME.message);
            end
        end
        
        function validateDeviceConfig(data, params)
            % デバイス設定とデータの整合性チェック
            if size(data.rawData, 1) ~= params.device.channelCount
                error('DataLoader:ChannelMismatch', ...
                    'データのチャンネル数(%d)がデバイス設定(%d)と一致しません。\n設定ファイルのデバイス種別を確認してください。', ...
                    size(data.rawData, 1), params.device.channelCount);
            end

            % チャンネル名の検証（存在する場合）
            if isfield(data, 'channelNames') && length(data.channelNames) ~= length(params.device.channels)
                warning('DataLoader:ChannelNameMismatch', ...
                    'チャンネル名の数がデバイス設定と異なります。\n期待値: %s\n実際: %s', ...
                    strjoin(params.device.channels, ', '), ...
                    strjoin(data.channelNames, ', '));
            end
        end
    end
    
    methods (Static, Access = private)
        function dialogSettings = getDialogSettings(purpose)
            % 目的に応じたダイアログ設定を返す
            dialogSettings = struct();
            
            switch purpose
                case 'analysis'
                    dialogSettings.title = '解析用データファイルの選択';
                case 'online'
                    dialogSettings.title = 'オンライン処理用データファイルの選択';
                otherwise
                    dialogSettings.title = 'データファイルの選択';
            end
        end
        
        function mode = selectLoadMode()
            % 読み込みモードの選択
            choice = questdlg('読み込みモードを選択してください:', ...
                '読み込みモード選択', ...
                'データ統合', '個別読み込み', 'キャンセル', ...
                'データ統合');
            
            switch choice
                case 'データ統合'
                    mode = 'batch';
                case '個別読み込み'
                    mode = 'individual';
                otherwise
                    mode = '';
            end
        end
        
        function mode = selectSaveMode()
            % 保存モードの選択
            choice = questdlg('保存モードを選択してください:', ...
                '保存モード選択', ...
                '一括保存', '個別保存', 'キャンセル', ...
                '個別保存');
            
            switch choice
                case '一括保存'
                    mode = 'batch';
                case '個別保存'
                    mode = 'individual';
                otherwise
                    mode = '';
            end
        end
        
        function loadedData = loadBatchData(filepath, filenames, purpose, options)
            % 複数ファイルの統合読み込み
            try
                fprintf('データ統合読み込みを開始...\n');
                
                % DataManagerを使用してデータを読み込み
                dataManager = DataManager(options);
                loadedData = cell(1, length(filenames));
                
                % 各ファイルの読み込みと検証
                for i = 1:length(filenames)
                    fullpath = fullfile(filepath, filenames{i});
                    currentData = dataManager.loadDataset(fullpath);
                    [currentData, ~] = DataLoader.validateData(currentData, purpose);
                    loadedData{i} = currentData;
                    fprintf('読み込み完了 (%d/%d): %s\n', i, length(filenames), filenames{i});
                end
                
            catch ME
                error('DataLoader:BatchLoadError', 'バッチ読み込みエラー: %s', ME.message);
            end
        end
        
        function loadedData = loadIndividualData(filepath, filenames, purpose, options)
            % 個別ファイルの読み込み
            try
                % DataManagerを使用してデータを読み込み
                dataManager = DataManager(options);
                loadedData = cell(1, length(filenames));
                
                for i = 1:length(filenames)
                    fullpath = fullfile(filepath, filenames{i});
                    currentData = dataManager.loadDataset(fullpath);
                    [currentData, ~] = DataLoader.validateData(currentData, purpose);
                    loadedData{i} = currentData;
                    fprintf('読み込み完了: %s\n', filenames{i});
                end
                
            catch ME
                error('DataLoader:IndividualLoadError', '個別読み込みエラー: %s', ME.message);
            end
        end
        
        function success = saveBatchData(data, outputPath, purpose, options)
            % バッチ保存
            try
                % DataManagerを使用してデータを保存
                dataManager = DataManager(options);
                
                % ファイル名の生成
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                filename = sprintf('%s_results_%s.mat', purpose, timestamp);
                fullpath = fullfile(outputPath, filename);
                
                % データの保存
                dataManager.saveDataset(data, fullpath);
                fprintf('一括保存完了: %s\n', fullpath);
                success = true;
                
            catch ME
                error('DataLoader:BatchSaveError', 'バッチ保存エラー: %s', ME.message);
            end
        end
        
        function success = saveIndividualData(data, purpose, options)
            % 個別保存
            try
                % DataManagerを使用してデータを保存
                dataManager = DataManager(options);
                
                % 保存先の選択
                [filename, pathname] = uiputfile('*.mat', ...
                    sprintf('%s - 保存先を選択してください', purpose));
                
                if isequal(filename, 0)
                    success = false;
                    return;
                end
                
                % データの保存
                fullpath = fullfile(pathname, filename);
                dataManager.saveDataset(data, fullpath);
                fprintf('個別保存完了: %s\n', fullpath);
                success = true;
                
            catch ME
                error('DataLoader:IndividualSaveError', '個別保存エラー: %s', ME.message);
            end
        end
        
        function missingFields = checkRequiredFields(data, requiredFields)
            % 必須フィールドのチェック
            missingFields = requiredFields(~isfield(data, requiredFields));
        end
        
        function requiredFields = getRequiredFields(purpose)
            % 目的に応じた必須フィールドの定義
            switch purpose
                case 'analysis'
                    requiredFields = {'rawData', 'labels'};
                case 'online'
                    requiredFields = {'processingInfo', 'classifier', 'results'};
                otherwise
                    requiredFields = {'rawData'};
            end
        end
        
        function validatedData = validateDataFormat(data, purpose)
            % データ形式の検証
            validatedData = data;
            
            switch purpose
                case 'analysis'
                    % 解析用データの検証
                    if ~isfield(data, 'rawData') || ~isfield(data, 'labels')
                        error('DataLoader:ValidationError', '解析に必要なフィールドが不足しています');
                    end
                    
                case 'online'
                    % オンライン処理用データの検証
                    if ~isfield(data, 'classifier') || ~isfield(data, 'processingInfo')
                        error('DataLoader:ValidationError', 'オンライン処理に必要なフィールドが不足しています');
                    end
                    
                case 'normalization'
                    % 正規化用データの検証
                    if ~isfield(data, 'rawData')
                        error('DataLoader:ValidationError', '正規化に必要なフィールドが不足しています');
                    end
            end
        end
        
        function sizeInfo = getDataSize(data)
            % データサイズ情報の取得
            sizeInfo = struct();
            
            if isfield(data, 'rawData')
                sizeInfo.rawData = size(data.rawData);
            end
            
            if isfield(data, 'labels')
                sizeInfo.labels = length(data.labels);
            end
            
            if isfield(data, 'processedData')
                sizeInfo.processedData = size(data.processedData);
            end
        end
    end
end