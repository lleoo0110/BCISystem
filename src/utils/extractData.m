function [extractedData, extractedLabels] = extractData(varargin)
%% extractData - EEGデータファイルからの情報抽出と処理ツール
%
% このツールはMAT形式のEEGデータファイルからデータを抽出・加工し、特定の条件に基づいた
% データセットを作成します。複数のデータ操作機能を提供し、解析前のデータ準備を効率化します。
%
% 主な機能:
%   - 特定クラスのみの抽出
%   - 特定サンプル範囲の抽出
%   - クラス番号の変更（リマッピング）
%   - 特定チャンネルのみの抽出
%   - クラス分布のバランス化
%   - 複数ファイルの統合
%   - 処理結果の保存
%
% 使用例:
%   1. 基本的な使用法（UIでファイル選択）
%      [data, labels] = extractData();
%
%   2. 特定のクラスのみ抽出
%      [data, labels] = extractData('Classes', [1 2], 'SaveExtracted', true);
%
%   3. 特定のサンプル範囲を抽出
%      [data, labels] = extractData('Samples', 1:100, 'SaveExtracted', true);
%
%   4. クラス番号の変更（クラス4をクラス3に変更）
%      [data, labels] = extractData('ClassMap', [4, 3]);
%
%   5. 複数クラス番号の同時変更（クラス4→3、クラス2→1）
%      [data, labels] = extractData('ClassMap', [4, 3; 2, 1], 'SaveExtracted', true);
%
%   6. クラス抽出と番号変更の組み合わせ
%      [data, labels] = extractData('Classes', [1 2 4], 'ClassMap', [4, 3], 'SaveExtracted', true);
%
%   7. クラス数を均衡にして抽出
%      [data, labels] = extractData('SaveExtracted', true, 'BalanceClasses', true);
%
%   8. 特定のチャンネルのみ抽出
%      [data, labels] = extractData('Channels', [1, 2, 6, 8, 27, 29], 'SaveExtracted', true);
%
%   9. 複数ファイルを統合して処理
%      [data, labels] = extractData('MergeFiles', true, 'SaveExtracted', true);
%
%   10. 複数ファイルを一括保存で処理
%      [data, labels] = extractData('SaveExtracted', true, 'BatchSave', true);
%
% 入力パラメータ:
%   'Filename'         - ファイル名指定（省略時はUI選択）
%   'Classes'          - 抽出するクラスの配列 [1 2 3]
%   'Samples'          - 抽出するサンプルインデックスの配列 [1:100]
%   'Channels'         - 抽出するチャンネルインデックスの配列 [1 4 6]
%   'ClassMap'         - クラス番号変換マップ [元の値, 新しい値] のN×2行列
%   'BalanceClasses'   - クラス分布均衡化フラグ (true/false)
%   'MergeFiles'       - 複数ファイル統合フラグ (true/false)
%   'ValidateOnly'     - ファイル検証のみ実行 (true/false)
%   'SaveExtracted'    - 抽出結果保存フラグ (true/false)
%   'BatchSave'        - 一括保存フラグ (true/false)、複数ファイル選択時に有効
%   'OutputPath'       - 出力先パス (指定時はダイアログ表示なし)
%
% 出力:
%   extractedData      - 抽出されたデータ
%   extractedLabels    - 抽出されたラベル
%
% 注意事項:
%   - データファイルには最低限 'rawData'と'labels'フィールドが必要です
%   - 'rawData'は [channels x timepoints] の形式であることを想定しています
%   - 'labels'は少なくとも'value'と'sample'フィールドを持つ構造体の配列であることを想定しています
%
% Author: LLEOO
% Version: 2.2

    % 入力パーサーの初期化
    p = inputParser;
    addParameter(p, 'Filename', '', @ischar);
    addParameter(p, 'Classes', [], @isnumeric);
    addParameter(p, 'Samples', [], @isnumeric);
    addParameter(p, 'Channels', [], @isnumeric);
    addParameter(p, 'ValidateOnly', false, @islogical);
    addParameter(p, 'SaveExtracted', false, @islogical);
    addParameter(p, 'BatchSave', false, @islogical); % 新機能: 一括保存
    addParameter(p, 'OutputPath', '', @ischar);
    addParameter(p, 'ClassMap', [], @(x) isempty(x) || (isnumeric(x) && size(x,2)==2));
    addParameter(p, 'BalanceClasses', false, @islogical);
    addParameter(p, 'MergeFiles', false, @islogical); % 新機能: ファイル統合
    parse(p, varargin{:});
    
    options = p.Results;

    try
        % ファイル選択
        [filenames, filepath] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, ...
            'データファイルを選択', 'MultiSelect', 'on');
        
        if isequal(filenames, 0)
            error('ファイルが選択されませんでした。');
        end

        % 単一ファイル選択時はセル配列に変換
        if ~iscell(filenames)
            filenames = {filenames};
        end

        fprintf('\n=== データ処理開始 ===\n');
        fprintf('選択されたファイル数: %d\n\n', length(filenames));
        
        % ファイル統合処理 (MergeFilesパラメータが設定されている場合のみ)
        if options.MergeFiles && length(filenames) > 1
            fprintf('ファイル統合モードが有効です。選択された%d個のファイルを統合します。\n', length(filenames));
            [mergedData, mergedLabels] = mergeFiles(filepath, filenames);
            
            % 統合データの処理
            [processedData, processedLabels] = processData(mergedData, mergedLabels, options);
            
            % 結果の保存
            if options.SaveExtracted
                % デフォルトのファイル名を生成
                defaultFileName = generateDefaultFileName('merged_data', options);
                
                % 保存先を指定するダイアログを表示
                [saveFileName, savePath] = uiputfile('*.mat', '統合データの保存先を選択', defaultFileName);
                
                if saveFileName ~= 0
                    saveFilePath = fullfile(savePath, saveFileName);
                    saveProcessedData(processedData, processedLabels, saveFilePath, options, 'MergedFiles', {filenames});
                    fprintf('\n統合データを保存しました: %s\n', saveFilePath);
                else
                    fprintf('\n保存がキャンセルされました。\n');
                end
            end
            
            extractedData = processedData;
            extractedLabels = processedLabels;
            
        else
            % 個別ファイル処理モード
            % 結果格納用の配列
            extractedData = cell(1, length(filenames));
            extractedLabels = cell(1, length(filenames));
            
            % 一括保存モードの設定
            batchSavePath = '';
            if options.SaveExtracted && options.BatchSave && length(filenames) > 1
                % 一括保存用の保存先フォルダを選択
                batchSavePath = uigetdir(filepath, '一括保存のための保存先フォルダを選択してください');
                if isequal(batchSavePath, 0)
                    fprintf('一括保存がキャンセルされました。個別保存モードに切り替えます。\n');
                    options.BatchSave = false;
                else
                    fprintf('一括保存モード: 処理結果を %s に保存します。\n', batchSavePath);
                end
            end

            % 各ファイルの処理
            for i = 1:length(filenames)
                try
                    fullpath = fullfile(filepath, filenames{i});
                    fprintf('処理中 (%d/%d): %s\n', i, length(filenames), filenames{i});

                    % データの読み込み
                    data = load(fullpath);
                    validateFields(data);
                    
                    % データの処理
                    [currentData, currentLabels] = processData(data.rawData, data.labels, options);

                    % 結果の保存
                    if ~isempty(currentData)
                        extractedData{i} = currentData;
                        extractedLabels{i} = currentLabels;

                        % 保存処理
                        if options.SaveExtracted
                            % ファイル名のベース部分を取得
                            [~, baseName, ~] = fileparts(filenames{i});
                            
                            % 処理内容を反映したデフォルトファイル名を生成
                            defaultFileName = generateDefaultFileName(baseName, options);
                            
                            if options.BatchSave && ~isempty(batchSavePath)
                                % 一括保存モード: フォルダは選択済み、ファイル名は自動生成
                                saveFilePath = fullfile(batchSavePath, defaultFileName);
                                saveProcessedData(currentData, currentLabels, saveFilePath, options, 'SourceFile', fullpath);
                                fprintf('\n処理済みデータを一括保存しました: %s\n', saveFilePath);
                            else
                                % 個別保存モード: 毎回ダイアログ表示
                                [saveFileName, savePath] = uiputfile('*.mat', ...
                                    sprintf('処理済みデータの保存先を選択 (%d/%d)', i, length(filenames)), ...
                                    defaultFileName);
                                
                                if saveFileName ~= 0
                                    saveFilePath = fullfile(savePath, saveFileName);
                                    saveProcessedData(currentData, currentLabels, saveFilePath, options, 'SourceFile', fullpath);
                                    fprintf('\n処理済みデータを保存しました: %s\n', saveFilePath);
                                else
                                    fprintf('\n保存がキャンセルされました。\n');
                                end
                            end
                        end
                    end

                catch ME
                    fprintf('ファイル %s の処理中にエラー: %s\n', filenames{i}, ME.message);
                    fprintf('スタックトレース:\n%s\n', getReport(ME, 'extended'));
                    continue;
                end
            end
        end

        fprintf('\n=== データ処理完了 ===\n');

    catch ME
        fprintf('エラーが発生しました: %s\n', ME.message);
        fprintf('スタックトレース:\n%s\n', getReport(ME, 'extended'));
        extractedData = {};
        extractedLabels = {};
    end
end

%% ファイル統合関数
function [mergedData, mergedLabels] = mergeFiles(filepath, filenames)
    fprintf('\n=== ファイル統合処理開始 ===\n');
    
    % 統合データの初期化
    mergedData = [];
    mergedLabels = struct('value', {}, 'time', {}, 'sample', {});
    
    % チャンネル数の一貫性チェック用
    channelCount = [];
    totalSamples = 0;
    
    % 各ファイルの読み込みと統合
    for i = 1:length(filenames)
        fprintf('ファイル %d/%d を統合中: %s\n', i, length(filenames), filenames{i});
        
        try
            % 現在のファイルを読み込み
            fullpath = fullfile(filepath, filenames{i});
            currentData = load(fullpath);
            validateFields(currentData);
            
            % チャンネル数の確認
            if isempty(channelCount)
                channelCount = size(currentData.rawData, 1);
            elseif size(currentData.rawData, 1) ~= channelCount
                error('チャンネル数が一致しません: ファイル %s は %d チャンネルですが、先行ファイルは %d チャンネルです', ...
                    filenames{i}, size(currentData.rawData, 1), channelCount);
            end
            
            % rawDataの統合
            mergedData = [mergedData, currentData.rawData];
            
            % ラベルの統合
            if ~isempty(currentData.labels)
                for j = 1:length(currentData.labels)
                    newLabel = currentData.labels(j);
                    if isfield(newLabel, 'sample') && ~isempty(newLabel.sample)
                        newLabel.sample = newLabel.sample + totalSamples;
                    end
                    mergedLabels(end+1) = newLabel;
                end
            end
            
            % 総サンプル数の更新
            totalSamples = totalSamples + size(currentData.rawData, 2);
            fprintf('  - ファイル統合済み: 累計サンプル数 %d\n', totalSamples);
            
        catch ME
            fprintf('  - ファイル %s の統合中にエラー: %s\n', filenames{i}, ME.message);
            fprintf('    スタックトレース:\n%s\n', getReport(ME, 'extended'));
            continue;
        end
    end
    
    % 統合結果の表示
    fprintf('\nファイル統合結果:\n');
    fprintf('  - 総ファイル数: %d\n', length(filenames));
    fprintf('  - チャンネル数: %d\n', channelCount);
    fprintf('  - 総サンプル数: %d\n', size(mergedData, 2));
    fprintf('  - 総ラベル数: %d\n', length(mergedLabels));
    
    fprintf('=== ファイル統合処理完了 ===\n\n');
end

%% データ前処理関数
function [processedData, processedLabels] = processData(data, labels, options)
    % データの基本情報を表示
    displayFileInfo(data, labels);

    % 検証モードの場合はここで終了
    if options.ValidateOnly
        processedData = [];
        processedLabels = [];
        return;
    end
    
    % 処理初期値の設定
    processedData = data;
    processedLabels = labels;

    % クラスによる抽出
    if ~isempty(options.Classes)
        [processedData, processedLabels] = extractByClass(processedData, processedLabels, options.Classes);
    end

    % サンプルによる抽出
    if ~isempty(options.Samples) && ~isempty(processedData)
        [processedData, processedLabels] = extractBySamples(processedData, processedLabels, options.Samples);
    end

    % クラス番号の変換
    if ~isempty(options.ClassMap)
        processedLabels = remapClasses(processedLabels, options.ClassMap);
        fprintf('\nクラス番号の変換を実行しました\n');
        displayClassMapping(options.ClassMap);
    end
    
    % チャンネルの抽出
    if ~isempty(options.Channels)
        if max(options.Channels) > size(processedData, 1) || min(options.Channels) < 1
            error('指定されたチャンネルインデックスがデータサイズを超えています。');
        end
        processedData = processedData(options.Channels, :);
        fprintf('\nチャンネルの抽出を実行しました: チャンネル [%s]\n', num2str(options.Channels));
    end

    % クラス分布の均衡化
    if options.BalanceClasses && ~isempty(processedLabels)
        [processedData, processedLabels] = balanceClasses(processedData, processedLabels);
        fprintf('\n各クラスのラベル数を均衡化しました。（rawData は変更せずコピー）\n');
    end

    % 抽出結果の表示
    displayExtractionResults(processedData, processedLabels);
end

%% デフォルトファイル名生成関数
function defaultFileName = generateDefaultFileName(baseName, options)
    % 処理内容を反映したサフィックスを生成
    suffix = '';
    
    if ~isempty(options.Classes)
        suffix = [suffix sprintf('_classes%s', strjoin(string(options.Classes), '-'))];
    end
    
    if ~isempty(options.Samples)
        suffix = [suffix sprintf('_samples%d-%d', min(options.Samples), max(options.Samples))];
    end
    
    if ~isempty(options.Channels)
        suffix = [suffix sprintf('_channels%s', strjoin(string(options.Channels), '-'))];
    end
    
    if ~isempty(options.ClassMap)
        suffix = [suffix '_remapped'];
    end
    
    if options.BalanceClasses
        suffix = [suffix '_balanced'];
    end
    
    if options.MergeFiles && strcmp(baseName, 'merged_data')
        suffix = [suffix '_merged'];
    end
    
    if isempty(suffix)
        suffix = '_extracted';
    end
    
    % タイムスタンプを追加
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    
    % 最終的なファイル名を構築
    defaultFileName = sprintf('%s%s_%s.mat', baseName, suffix, timestamp);
end

%% 処理済みデータ保存関数
function saveProcessedData(data, labels, saveFilePath, options, varargin)
    % 追加情報の処理
    extraInfo = struct();
    for i = 1:2:length(varargin)
        extraInfo.(varargin{i}) = varargin{i+1};
    end
    
    % 保存データの構築
    saveData = struct(...
        'rawData', data, ...
        'labels', labels, ...
        'extractionInfo', struct(...
            'extractedClasses', options.Classes, ...
            'extractedSamples', options.Samples, ...
            'extractedChannels', options.Channels, ...
            'classMap', options.ClassMap, ...
            'balanceClasses', options.BalanceClasses, ...
            'mergeFiles', options.MergeFiles, ...
            'timestamp', datetime('now'), ...
            'extraInfo', extraInfo ...
        ) ...
    );

    % ファイルに保存
    save(saveFilePath, '-struct', 'saveData');
end

%% フィールド検証関数
function validateFields(data)
    required_fields = {'rawData', 'labels'};
    missing_fields = required_fields(~isfield(data, required_fields));
    
    if ~isempty(missing_fields)
        error('必須フィールドが見つかりません: %s', strjoin(missing_fields, ', '));
    end
end

%% クラスによるデータ抽出関数
function [extractedData, extractedLabels] = extractByClass(data, labels, targetClasses)
    if isempty(targetClasses)
        extractedData = data;
        extractedLabels = labels;
        return;
    end

    validIndices = false(length(labels), 1);
    for i = 1:length(labels)
        if ismember(labels(i).value, targetClasses)
            validIndices(i) = true;
        end
    end

    if any(validIndices)
        extractedLabels = labels(validIndices);
        extractedData = data;
    else
        warning('指定されたクラスのデータが見つかりませんでした。');
        extractedData = data;
        extractedLabels = [];
    end
end

%% サンプルによるデータ抽出関数
function [extractedData, extractedLabels] = extractBySamples(data, labels, targetSamples)
    validSamples = targetSamples(targetSamples <= length(labels));
    
    if ~isempty(validSamples)
        extractedLabels = labels(validSamples);
        sampleIndices = [extractedLabels.sample];
        extractedData = data(:, sampleIndices);
    else
        warning('指定されたサンプル範囲が無効です。');
        extractedData = [];
        extractedLabels = [];
    end
end

%% クラスのリマッピング関数
function labels = remapClasses(labels, classMap)
    numChanges = zeros(size(classMap, 1), 1);
    
    for i = 1:length(labels)
        for j = 1:size(classMap, 1)
            if labels(i).value == classMap(j, 1)
                labels(i).value = classMap(j, 2);
                numChanges(j) = numChanges(j) + 1;
                break;
            end
        end
    end
    
    fprintf('\n変換されたラベル数:\n');
    for j = 1:size(classMap, 1)
        fprintf('クラス %d → %d: %d 個\n', ...
            classMap(j, 1), classMap(j, 2), numChanges(j));
    end
end

%% クラス分布均衡化関数
function [balancedData, balancedLabels] = balanceClasses(data, labels)
    % rawData はそのままコピー
    balancedData = data;
    classValues = [labels.value];
    uniqueClasses = unique(classValues);
    
    % 各クラスのサンプル数を計算
    counts = zeros(1, length(uniqueClasses));
    for k = 1:length(uniqueClasses)
        indices = find(classValues == uniqueClasses(k));
        counts(k) = length(indices);
    end
    
    % 最小クラスのサンプル数を取得
    minCount = min(counts);
    
    % 各クラスからランダムに minCount 個サンプルを選択
    allSelected = [];
    for k = 1:length(uniqueClasses)
        indices = find(classValues == uniqueClasses(k));
        perm = randperm(length(indices));
        selected = indices(perm(1:minCount));
        allSelected = [allSelected; selected]; % インデックスを縦方向に保存
    end
    
    % 選択されたインデックスを使用して縦方向に構造体を構築
    balancedLabels = labels(allSelected);
    % reshape を使用して確実に N*1 構造体にする
    balancedLabels = reshape(balancedLabels, [], 1);
    
    fprintf('\nクラス均衡化結果:\n');
    fprintf('  - 各クラスのサンプル数: %d\n', minCount);
    fprintf('  - 均衡化後の総サンプル数: %d\n', length(balancedLabels));
end

%% クラスマッピング表示関数
function displayClassMapping(classMap)
    fprintf('クラス番号の変換マップ:\n');
    for i = 1:size(classMap, 1)
        fprintf('  クラス %d → %d\n', classMap(i, 1), classMap(i, 2));
    end
end

%% ファイル情報表示関数
function displayFileInfo(data, labels)
    fprintf('\nデータ基本情報:\n');
    fprintf('データサイズ: [%d, %d]\n', size(data));
    fprintf('ラベル数: %d\n', length(labels));
    
    uniqueClasses = unique([labels.value]);
    fprintf('\n元のクラス分布:\n');
    for i = 1:length(uniqueClasses)
        classCount = sum([labels.value] == uniqueClasses(i));
        fprintf('クラス %d: %d サンプル (%.1f%%)\n', uniqueClasses(i), classCount, ...
            (classCount/length(labels))*100);
    end
end

%% 抽出結果表示関数
function displayExtractionResults(data, labels)
    if ~isempty(data)
        fprintf('\n抽出結果:\n');
        fprintf('抽出されたデータサイズ: [%d, %d]\n', size(data));
        fprintf('抽出されたラベル数: %d\n', length(labels));

        if ~isempty(labels)
            uniqueClasses = unique([labels.value]);
            fprintf('\n抽出後のクラス分布:\n');
            for i = 1:length(uniqueClasses)
                classCount = sum([labels.value] == uniqueClasses(i));
                fprintf('クラス %d: %d サンプル (%.1f%%)\n', uniqueClasses(i), classCount, ...
                    (classCount/length(labels))*100);
            end
        end
    end
end