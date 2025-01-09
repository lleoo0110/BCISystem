classdef DataLoader
    methods (Static)
        function loadedData = loadDataBrowserWithPrompt(purpose)
            try
                % ダイアログタイトルの設定
                switch purpose
                    case 'normalization'
                        dialogTitle = '正規化パラメータ計算用のデータを選択';
                        promptMessage = '正規化パラメータ計算';
                    case 'csp'
                        dialogTitle = 'CSPフィルタ用のデータを選択';
                        promptMessage = 'CSPフィルタ計算';
                    case 'baseline'
                        dialogTitle = 'ベースライン用のデータを選択';
                        promptMessage = 'ベースライン計算';
                    case 'classifier'
                        dialogTitle = '分類器用のデータを選択';
                        promptMessage = '分類器データ';
                    case 'concatenate'
                        dialogTitle = '結合するデータファイルを選択';
                        promptMessage = 'データ結合';
                    case 'online'
                        dialogTitle = 'オンライン処理用の学習済みモデルを選択';
                        promptMessage = 'オンライン処理';
                    otherwise
                        dialogTitle = 'データファイルを選択';
                        promptMessage = 'データ読み込み';
                end

                choice = questdlg(sprintf('%s用のデータを読み込みますか？', promptMessage), ...
                    'データ読み込みの確認', 'はい', 'いいえ', 'はい');

                if strcmp(choice, 'はい')
                    [filenames, pathname] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, ...
                        dialogTitle, 'MultiSelect', 'on');
                    
                    if isequal(filenames, 0)
                        error('ファイルが選択されませんでした．');
                    end
                    
                    % 単一ファイル選択時はセル配列に変換
                    if ~iscell(filenames)
                        filenames = {filenames};
                    end
                    
                    if length(filenames) == 1
                        % 単一ファイルの場合は直接読み込み
                        fullpath = fullfile(pathname, filenames{1});
                        loadedData = load(fullpath);
                        fprintf('%s用のデータを読み込みました: %s\n', purpose, fullpath);
                    else
                        % 複数ファイルの場合は結合
                        loadedData = DataLoader.concatenateFiles(pathname, filenames);
                        fprintf('%d個のファイルを結合しました\n', length(filenames));
                    end
                    
                    % データの検証
                    DataLoader.validateLoadedData(loadedData, purpose);
                else
                    error('ユーザーによってキャンセルされました．');
                end
                
            catch ME
                errordlg(sprintf('データ読み込みエラー: %s', ME.message), 'エラー');
                rethrow(ME);
            end
        end
        
        function validateLoadedData(loadedData, purpose)
            % 読み込んだデータの検証
            switch purpose
                case 'normalization'
                    if ~isfield(loadedData, 'rawData') && ~isfield(loadedData, 'processedData')
                        error('正規化用の有効なデータが見つかりません．');
                    end
                case {'csp', 'classifier', 'concatenate'}
                    if ~isfield(loadedData, 'rawData')
                        error('分類用の有効なデータが見つかりません．');
                    end
                    if ~isfield(loadedData, 'labels')
                        error('ラベルデータが見つかりません．');
                    end
                case 'baseline'
                    if ~isfield(loadedData, 'processedData')
                        error('ERD計算用の処理済みデータが見つかりません．');
                    end
                case 'online'
                    required_fields = {'processingInfo', 'classifier', 'results'};
                    for i = 1:length(required_fields)
                        if ~isfield(loadedData, required_fields{i})
                            error('オンライン処理に必要な %s が見つかりません．', required_fields{i});
                        end
                    end
            end
        end
        
        function concatenatedData = concatenateFiles(pathname, filenames)
            try
                % 初期化
                totalSamples = 0;
                totalLabels = 0;
                channelCount = [];
                fileData = cell(length(filenames), 1);
                
                % 1回目のパス：サイズの計算とデータの整合性チェック
                fprintf('データファイルの解析中...\n');
                for i = 1:length(filenames)
                    fullpath = fullfile(pathname, filenames{i});
                    data = load(fullpath);
                    
                    % データの検証
                    if ~isfield(data, 'rawData') || ~isfield(data, 'labels')
                        error('ファイル %s に必要なフィールドがありません', filenames{i});
                    end
                    
                    % チャンネル数の一貫性チェック
                    if isempty(channelCount)
                        channelCount = size(data.rawData, 1);
                    elseif channelCount ~= size(data.rawData, 1)
                        error('ファイル %s のチャンネル数が一致しません（期待値: %d, 実際: %d）', ...
                            filenames{i}, channelCount, size(data.rawData, 1));
                    end
                    
                    fileData{i} = data;
                    totalSamples = totalSamples + size(data.rawData, 2);
                    if ~isempty(data.labels)
                        totalLabels = totalLabels + length(data.labels);
                    end
                    
                    % サンプリングレートの確認（params構造体がある場合）
                    if isfield(data, 'params') && i > 1
                        if data.params.device.sampleRate ~= fileData{1}.params.device.sampleRate
                            warning('ファイル %s のサンプリングレートが異なります（%d Hz vs %d Hz）', ...
                                filenames{i}, data.params.device.sampleRate, ...
                                fileData{1}.params.device.sampleRate);
                        end
                    end
                end
                
                % 結合データの初期化
                mergedRawData = zeros(channelCount, totalSamples);
                mergedLabels = struct('value', {}, 'sample', {});
                
                % 2回目のパス：データの結合
                fprintf('データの結合を開始...\n');
                currentSample = 1;
                labelIndex = 1;
                
                for i = 1:length(fileData)
                    data = fileData{i};
                    numSamples = size(data.rawData, 2);
                    
                    % rawDataの結合
                    mergedRawData(:, currentSample:currentSample+numSamples-1) = data.rawData;
                    
                    % ラベルの結合と調整
                    if ~isempty(data.labels)
                        for j = 1:length(data.labels)
                            mergedLabels(labelIndex).value = data.labels(j).value;
                            % サンプル番号を調整
                            mergedLabels(labelIndex).sample = data.labels(j).sample + (currentSample - 1);
                            labelIndex = labelIndex + 1;
                        end
                    end
                    
                    currentSample = currentSample + numSamples;
                    fprintf('ファイル %d/%d を処理しました\n', i, length(fileData));
                end
                
                % パラメータ情報の継承（最初のファイルから）
                concatenatedData = struct();
                if isfield(fileData{1}, 'params')
                    concatenatedData.params = fileData{1}.params;
                end
                concatenatedData.rawData = mergedRawData;
                concatenatedData.labels = mergedLabels;
                
                % 結合情報の追加
                concatenatedData.concatenationInfo = struct(...
                    'numberOfFiles', length(filenames), ...
                    'filenames', {filenames}, ...
                    'totalSamples', totalSamples, ...
                    'totalLabels', totalLabels, ...
                    'channelCount', channelCount, ...
                    'concatenationDate', datetime('now') ...
                );
                
                % 結果の表示
                DataLoader.displayConcatenationSummary(concatenatedData);
                
            catch ME
                fprintf('エラーが発生しました: %s\n', ME.message);
                fprintf('スタックトレース:\n');
                for k = 1:length(ME.stack)
                    fprintf('  File: %s, Line: %d, Function: %s\n', ...
                        ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
                end
                rethrow(ME);
            end
        end
        
        function displayConcatenationSummary(data)
            if ~isfield(data, 'concatenationInfo')
                return;
            end
            
            fprintf('\n=== データ結合サマリー ===\n');
            fprintf('結合ファイル数: %d\n', data.concatenationInfo.numberOfFiles);
            fprintf('結合ファイル:\n');
            for i = 1:length(data.concatenationInfo.filenames)
                fprintf('  %d. %s\n', i, data.concatenationInfo.filenames{i});
            end
            fprintf('総サンプル数: %d\n', data.concatenationInfo.totalSamples);
            fprintf('総ラベル数: %d\n', data.concatenationInfo.totalLabels);
            fprintf('チャンネル数: %d\n', data.concatenationInfo.channelCount);
            fprintf('結合日時: %s\n', char(data.concatenationInfo.concatenationDate));
            
            % ラベルの分布を表示
            fprintf('\nラベルの分布:\n');
            labelValues = [data.labels.value];
            uniqueLabels = unique(labelValues);
            for i = 1:length(uniqueLabels)
                count = sum(labelValues == uniqueLabels(i));
                fprintf('ラベル %d: %d個 (%.1f%%)\n', ...
                    uniqueLabels(i), count, (count/length(labelValues))*100);
            end
            fprintf('=====================\n\n');
        end
    end
end