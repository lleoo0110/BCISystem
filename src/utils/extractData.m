function [extractedData, extractedLabels] = extractData(varargin)
    %% extractData の使用例
    % パスの設定に注意→パスを通していないと使用できない
    %
    % よく使う設定例:
    % [data, labels] = extractData('Classes', [1 2 4], 'ClassMap', [4, 3], 'SaveExtracted', true);
    %
    % 1. 基本的な使用法（UIでファイル選択）
    % [data, labels] = extractData();
    %
    % 2. 特定のクラスのみ抽出
    % [data, labels] = extractData('Classes', [1 2], 'SaveExtracted', true);
    %
    % 3. 特定のサンプル範囲を抽出
    % [data, labels] = extractData('Samples', 1:100, 'SaveExtracted', true);
    %
    % 4. クラス番号の変更（クラス4をクラス3に変更）
    % [data, labels] = extractData('ClassMap', [4, 3]);
    %
    % 5. 複数クラスの同時変更（クラス4→3、クラス2→1）
    % [data, labels] = extractData('ClassMap', [4, 3; 2, 1], 'SaveExtracted', true);
    %
    % 6. クラス抽出と番号変更の組み合わせ
    % [data, labels] = extractData('Classes', [1 2 4], 'ClassMap', [4, 3], 'SaveExtracted', true);
    %
    % 7. クラス数を均衡にして抽出
    % [data, labels] = extractData('SaveExtracted', true, 'BalanceClasses', true);
    %
    % 8. 特定のチャンネルのみ抽出
    % [data, labels] = extractData('Channels', [1, 2, 6, 8, 27, 29], 'SaveExtracted', true);
    
    % 入力パーサーの初期化
    p = inputParser;
    addParameter(p, 'Filename', '', @ischar);
    addParameter(p, 'Classes', [], @isnumeric);
    addParameter(p, 'Samples', [], @isnumeric);
    addParameter(p, 'Channels', [], @isnumeric);  % <-- 追加：抽出するチャンネルのインデックス
    addParameter(p, 'ValidateOnly', false, @islogical);
    addParameter(p, 'SaveExtracted', false, @islogical);
    addParameter(p, 'SaveMode', '', @(x) isempty(x) || ismember(x, {'individual', 'batch'}));
    addParameter(p, 'OutputPath', '', @ischar);
    addParameter(p, 'ClassMap', [], @(x) isempty(x) || (isnumeric(x) && size(x,2)==2));
    % BalanceClasses オプションを追加
    addParameter(p, 'BalanceClasses', false, @islogical);
    parse(p, varargin{:});

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

        fprintf('\n=== データ抽出開始 ===\n');
        fprintf('選択されたファイル数: %d\n\n', length(filenames));

        % 保存モードの確認（SaveExtracted が true の場合）
        saveMode = p.Results.SaveMode;
        if isempty(saveMode) && p.Results.SaveExtracted
            if length(filenames) == 1
                % ファイルが1つの場合は、直接「保存先フォルダ選択」ウィンドウを表示
                saveMode = 'batch';  % 一括保存モードとして扱い、保存先を指定させる
                outputPath = uigetdir(filepath, '保存先フォルダを選択してください');
                if isequal(outputPath, 0)
                    error('保存先フォルダが選択されませんでした。');
                end
            else
                % 複数ファイルの場合は通常の questdlg を使用
                saveMode = questdlg('保存モードを選択してください:', ...
                    '保存モードの選択', '一括保存', '個別保存', 'キャンセル', '個別保存');
                if strcmp(saveMode, 'キャンセル')
                    fprintf('処理がキャンセルされました。\n');
                    return;
                elseif strcmp(saveMode, '個別保存')
                    saveMode = 'individual';
                else
                    saveMode = 'batch';
                end
            end
        end

        % 複数ファイルの場合でバッチ保存なら保存先を選択
        outputPath = '';
        if strcmp(saveMode, 'batch') && p.Results.SaveExtracted && length(filenames) > 1
            outputPath = uigetdir(filepath, '保存先フォルダを選択してください');
            if isequal(outputPath, 0)
                error('保存先フォルダが選択されませんでした。');
            end
        end

        % 結果格納用の配列
        extractedData = cell(1, length(filenames));
        extractedLabels = cell(1, length(filenames));

        % 各ファイルの処理
        for i = 1:length(filenames)
            try
                fullpath = fullfile(filepath, filenames{i});
                fprintf('処理中 (%d/%d): %s\n', i, length(filenames), filenames{i});

                % データの処理と抽出
                [currentData, currentLabels] = processFile(fullpath, p.Results);

                % BalanceClasses が指定されている場合、ラベルのみを均衡化（rawData はそのままコピー）
                if p.Results.BalanceClasses && ~isempty(currentLabels)
                    [currentData, currentLabels] = balanceClasses(currentData, currentLabels);
                    fprintf('\n各クラスのラベル数を均衡化しました。（rawData は変更せずコピー）\n');
                    displayExtractionResults(currentData, currentLabels);
                end

                % 結果の保存
                if ~isempty(currentData)
                    extractedData{i} = currentData;
                    extractedLabels{i} = currentLabels;

                    % 保存処理
                    if p.Results.SaveExtracted
                        if strcmp(saveMode, 'individual')
                            saveExtractedFile(currentData, currentLabels, fullpath, p.Results);
                        elseif strcmp(saveMode, 'batch')
                            batchSaveFile(currentData, currentLabels, fullpath, outputPath, p.Results);
                        end
                    end
                end

            catch ME
                fprintf('ファイル %s の処理中にエラー: %s\n', filenames{i}, ME.message);
                continue;
            end
        end

        fprintf('\n=== データ抽出完了 ===\n');

    catch ME
        fprintf('エラーが発生しました: %s\n', ME.message);
        fprintf('スタックトレース:\n%s\n', getReport(ME, 'extended'));
        extractedData = {};
        extractedLabels = {};
    end
end

function saveExtractedFile(data, labels, sourcePath, params)
    [folder, name, ~] = fileparts(sourcePath);
    
    % サフィックスの生成
    suffix = '';
    if ~isempty(params.Classes)
        suffix = [suffix sprintf('_classes%s', strjoin(string(params.Classes), '-'))];
    end
    if ~isempty(params.Samples)
        suffix = [suffix sprintf('_samples%d-%d', min(params.Samples), max(params.Samples))];
    end
    if ~isempty(params.Channels)
        suffix = [suffix sprintf('_channels%s', strjoin(string(params.Channels), '-'))];
    end
    if ~isempty(params.ClassMap)
        suffix = [suffix '_remapped'];
    end
    if isempty(suffix)
        suffix = '_extracted';
    end

    saveFilename = fullfile(folder, [name suffix '.mat']);

    % 保存データの構築
    saveData = struct(...
        'rawData', data, ...
        'labels', labels, ...
        'extractionInfo', struct(...
            'sourceFile', sourcePath, ...
            'extractedClasses', params.Classes, ...
            'extractedSamples', params.Samples, ...
            'extractedChannels', params.Channels, ...
            'classMap', params.ClassMap, ...
            'timestamp', datetime('now') ...
        ) ...
    );

    save(saveFilename, '-struct', 'saveData');
    fprintf('\n抽出データを個別保存しました: %s\n', saveFilename);
end

function batchSaveFile(data, labels, sourcePath, outputPath, params)
    [~, name, ~] = fileparts(sourcePath);
    
    % サフィックスの生成
    suffix = '';
    if ~isempty(params.Classes)
        suffix = [suffix sprintf('_classes%s', strjoin(string(params.Classes), '-'))];
    end
    if ~isempty(params.Samples)
        suffix = [suffix sprintf('_samples%d-%d', min(params.Samples), max(params.Samples))];
    end
    if ~isempty(params.Channels)
        suffix = [suffix sprintf('_channels%s', strjoin(string(params.Channels), '-'))];
    end
    if ~isempty(params.ClassMap)
        suffix = [suffix '_remapped'];
    end
    if isempty(suffix)
        suffix = '_extracted';
    end

    saveFilename = fullfile(outputPath, [name suffix '.mat']);

    % 保存データの構築
    saveData = struct(...
        'rawData', data, ...
        'labels', labels, ...
        'extractionInfo', struct(...
            'sourceFile', sourcePath, ...
            'extractedClasses', params.Classes, ...
            'extractedSamples', params.Samples, ...
            'extractedChannels', params.Channels, ...
            'classMap', params.ClassMap, ...
            'timestamp', datetime('now') ...
        ) ...
    );

    save(saveFilename, '-struct', 'saveData');
    fprintf('\n抽出データを一括保存しました: %s\n', saveFilename);
end

function validateFields(data)
    required_fields = {'rawData', 'labels'};
    missing_fields = required_fields(~isfield(data, required_fields));
    
    if ~isempty(missing_fields)
        error('必須フィールドが見つかりません: %s', strjoin(missing_fields, ', '));
    end
end

function [processedData, processedLabels] = processFile(fullpath, params)
    % データの読み込み
    data = load(fullpath);
    validateFields(data);

    % rawData はそのままコピー
    processedData = data.rawData;
    processedLabels = data.labels;

    % データの基本情報を表示
    displayFileInfo(fullpath, processedData, processedLabels);

    % 検証モードの場合はここで終了
    if params.ValidateOnly
        processedData = [];
        processedLabels = [];
        return;
    end

    % クラスによる抽出が指定されている場合のみ実行
    if ~isempty(params.Classes)
        [processedData, processedLabels] = extractByClass(processedData, processedLabels, params.Classes);
    end

    % サンプルによる抽出が指定されている場合のみ実行
    if ~isempty(params.Samples) && ~isempty(processedData)
        [processedData, processedLabels] = extractBySamples(processedData, processedLabels, params.Samples);
    end

    % クラス番号の変換が指定されている場合のみ実行
    if ~isempty(params.ClassMap)
        processedLabels = remapClasses(processedLabels, params.ClassMap);
        fprintf('\nクラス番号の変換を実行しました\n');
        displayClassMapping(params.ClassMap);
    end
    
    % 追加: 指定されたチャンネルのみ抽出（1試行のデータは Ch x timepoints）
    if ~isempty(params.Channels)
        if max(params.Channels) > size(processedData, 1) || min(params.Channels) < 1
            error('指定されたチャンネルインデックスがデータサイズを超えています。');
        end
        processedData = processedData(params.Channels, :);
        fprintf('\nチャンネルの抽出を実行しました: チャンネル [%s]\n', num2str(params.Channels));
    end

    % 抽出結果の表示
    displayExtractionResults(processedData, processedLabels);
end

function [balancedData, balancedLabels] = balanceClasses(data, labels)
    % balanceClasses: ラベルのみを各クラスの最小数に合わせてダウンサンプリングし、
    % rawData はそのままコピーする
    balancedData = data; % rawData は変更せずコピー
    classValues = [labels.value];
    uniqueClasses = unique(classValues);
    % 各クラスのインデックスを取得
    balancedLabels = struct([]); % 空の構造体として初期化
    
    for k = 1:length(uniqueClasses)
        indices = find(classValues == uniqueClasses(k));
        % 各クラス内のサンプル数
        counts(k) = length(indices); %#ok<AGROW>
    end
    
    minCount = min(counts);
    allSelected = [];
    
    % 各クラスからランダムに minCount 個ずつ選択
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
end

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

function displayClassMapping(classMap)
    fprintf('クラス番号の変換マップ:\n');
    for i = 1:size(classMap, 1)
        fprintf('  クラス %d → %d\n', classMap(i, 1), classMap(i, 2));
    end
end

function displayFileInfo(fullpath, data, labels)
    fprintf('\nデータファイル: %s\n', fullpath);
    fprintf('データサイズ: [%d, %d]\n', size(data));
    fprintf('ラベル数: %d\n', length(labels));
    
    uniqueClasses = unique([labels.value]);
    fprintf('\n元のクラス分布:\n');
    for i = 1:length(uniqueClasses)
        classCount = sum([labels.value] == uniqueClasses(i));
        fprintf('クラス %d: %d サンプル\n', uniqueClasses(i), classCount);
    end
end

function displayExtractionResults(data, labels)
    if ~isempty(data)
        fprintf('\n抽出結果:\n');
        fprintf('抽出されたデータサイズ: [%d, %d]\n', size(data));
        fprintf('抽出されたラベル数: %d\n', length(labels));

        uniqueClasses = unique([labels.value]);
        fprintf('\n抽出後のクラス分布:\n');
        for i = 1:length(uniqueClasses)
            classCount = sum([labels.value] == uniqueClasses(i));
            fprintf('クラス %d: %d サンプル\n', uniqueClasses(i), classCount);
        end
    end
end
