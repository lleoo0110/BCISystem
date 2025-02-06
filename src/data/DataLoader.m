classdef DataLoader
    methods (Static)
        function [loadedData, fileInfo] = loadDataBrowserWithPrompt(purpose, options)
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
                    fileInfo = [];
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
                        fileInfo = [];
                        return;
                    end
                else
                    loadMode = 'individual';
                end
                
                % ファイル情報の構造体を作成
                fileInfo = struct();
                fileInfo.filenames = filenames;
                fileInfo.filepath = filepath;
                fileInfo.fullpaths = cellfun(@(f) fullfile(filepath, f), ...
                    filenames, 'UniformOutput', false);
                fileInfo.loadMode = loadMode;
                
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
            if size(data.rawData, 1) ~= params.device.channelCount
                error('DataLoader:ChannelMismatch', ...
                    'データのチャンネル数(%d)がデバイス設定(%d)と一致しません。\n設定ファイルのデバイス種別を確認してください。', ...
                    size(data.rawData, 1), params.device.channelCount);
            end

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
            try
                fprintf('データ統合読み込みを開始...\n');
                dataManager = DataManager(options);
                
                % 統合データの初期化
                integratedData = struct();
                integratedData.rawData = [];
                integratedData.labels = struct('value', {}, 'time', {}, 'sample', {});
                
                % チャンネル数の一貫性チェック用
                channelCount = [];
                totalSamples = 0;

                % 各ファイルの読み込みと統合
                for i = 1:length(filenames)
                    fprintf('ファイル %d/%d を読み込み中: %s\n', i, length(filenames), filenames{i});
                    
                    % 現在のファイルを読み込み
                    fullpath = fullfile(filepath, filenames{i});
                    currentData = dataManager.loadDataset(fullpath);
                    [currentData, ~] = DataLoader.validateData(currentData, purpose);
                    
                    % チャンネル数の確認
                    if isempty(channelCount)
                        channelCount = size(currentData.rawData, 1);
                    elseif size(currentData.rawData, 1) ~= channelCount
                        error('チャンネル数が一致しません: ファイル %s', filenames{i});
                    end
                    
                    % rawDataの統合
                    if isfield(currentData, 'rawData') && ~isempty(currentData.rawData)
                        integratedData.rawData = [integratedData.rawData, currentData.rawData];
                        
                        % ラベルの統合
                        if isfield(currentData, 'labels') && ~isempty(currentData.labels)
                            for j = 1:length(currentData.labels)
                                newLabel = currentData.labels(j);
                                if ~isempty(newLabel.sample)
                                    newLabel.sample = newLabel.sample + totalSamples;
                                end
                                integratedData.labels(end+1) = newLabel;
                            end
                        end
                        
                        % その他のフィールドの統合
                        otherFields = setdiff(fieldnames(currentData), {'rawData', 'labels'});
                        for j = 1:length(otherFields)
                            fieldName = otherFields{j};
                            if ~isfield(integratedData, fieldName)
                                integratedData.(fieldName) = currentData.(fieldName);
                            end
                        end
                        
                        % totalSamplesの更新
                        totalSamples = totalSamples + size(currentData.rawData, 2);
                    end
                    
                    fprintf('ファイル %d の統合完了\n', i);
                end
                
                % 統合結果の表示
                fprintf('\n統合結果:\n');
                fprintf('チャンネル数: %d\n', channelCount);
                fprintf('総サンプル数: %d\n', size(integratedData.rawData, 2));
                fprintf('総ラベル数: %d\n', length(integratedData.labels));
                
                % 統合データをセル配列の最初の要素として返す
                loadedData = {integratedData};
                
            catch ME
                error('DataLoader:BatchLoadError', 'バッチ読み込みエラー: %s', ME.message);
            end
        end

        function loadedData = loadIndividualData(filepath, filenames, purpose, options)
            try
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
            try
                dataManager = DataManager(options);
                
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                filename = sprintf('%s_results_%s.mat', purpose, timestamp);
                fullpath = fullfile(outputPath, filename);
                
                dataManager.saveDataset(data, fullpath);
                fprintf('一括保存完了: %s\n', fullpath);
                success = true;
                
            catch ME
                error('DataLoader:BatchSaveError', 'バッチ保存エラー: %s', ME.message);
            end
        end
        
        function success = saveIndividualData(data, purpose, options)
            try
                dataManager = DataManager(options);
                
                [filename, pathname] = uiputfile('*.mat', ...
                    sprintf('%s - 保存先を選択してください', purpose));
                
                if isequal(filename, 0)
                    success = false;
                    return;
                end
                
                fullpath = fullfile(pathname, filename);
                dataManager.saveDataset(data, fullpath);
                fprintf('個別保存完了: %s\n', fullpath);
                success = true;
                
            catch ME
                error('DataLoader:IndividualSaveError', '個別保存エラー: %s', ME.message);
            end
        end
        
        function missingFields = checkRequiredFields(data, requiredFields)
            missingFields = requiredFields(~isfield(data, requiredFields));
        end
        
        function requiredFields = getRequiredFields(purpose)
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
            validatedData = data;
            
            switch purpose
                case 'analysis'
                    if ~isfield(data, 'rawData') || isempty(data.rawData)
                        error('DataLoader:ValidationError', '解析用データが空です');
                    end
                    if ~isfield(data, 'labels')
                        validatedData.labels = struct('value', {}, 'time', {}, 'sample', {});
                    end
                    
                case 'online'
                    if ~isfield(data, 'classifier') || ~isfield(data, 'processingInfo')
                        error('DataLoader:ValidationError', 'オンライン処理に必要なフィールドが不足しています');
                    end
                    
                case 'normalization'
                    if ~isfield(data, 'rawData')
                        error('DataLoader:ValidationError', '正規化に必要なフィールドが不足しています');
                    end
            end
        end
        
        function sizeInfo = getDataSize(data)
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