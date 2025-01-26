function [extractedData, extractedLabels] = extractData(varargin)
    %% extractData の使用例
    % パスの設定に注意→パスを通していないと使用できない
    
   % よく使う設定
   % [data, labels] = extractData('Classes', [1 2 4], 'ClassMap', [4, 3], 'SaveExtracted', true);
    
    % 1. 基本的な使用法（UIでファイル選択）
    % [data, labels] = extractData();
    
    % 2. 特定のクラスのみ抽出
    % [data, labels] = extractData('Classes', [1 2], 'SaveExtracted', true);
    
    % 3. 特定のサンプル範囲を抽出
    % [data, labels] = extractData('Samples', 1:100, 'SaveExtracted', true);
    
    % 4. クラス番号の変更（クラス4をクラス3に変更）
    % [data, labels] = extractData('ClassMap', [4, 3]);
    
    % 5. 複数クラスの同時変更（クラス4→3、クラス2→1）
    % [data, labels] = extractData('ClassMap', [4, 3; 2, 1], 'SaveExtracted', true);
    
    % 6. クラス抽出と番号変更の組み合わせ
    % [data, labels] = extractData('Classes', [1 2 4], 'ClassMap', [4, 3], 'SaveExtracted', true);

    % 7. 保存モードの選択（一括保存モード）
    % [data, labels] = extractData('SaveExtracted', true, 'SaveMode', 'batch');
    
    % 8. 保存モードの選択（個別保存モード）
    % [data, labels] = extractData('SaveExtracted', true, 'SaveMode', 'individual');

    % 入力パーサーの初期化
    p = inputParser;
    addParameter(p, 'Filename', '', @ischar);
    addParameter(p, 'Classes', [], @isnumeric);
    addParameter(p, 'Samples', [], @isnumeric);
    addParameter(p, 'ValidateOnly', false, @islogical);
    addParameter(p, 'SaveExtracted', false, @islogical);
    addParameter(p, 'SaveMode', '', @(x) isempty(x) || ismember(x, {'individual', 'batch'})); % 保存モードの変更
    addParameter(p, 'OutputPath', '', @ischar); % バッチモード時の保存先指定
    addParameter(p, 'ClassMap', [], @(x) isempty(x) || (isnumeric(x) && size(x,2)==2));
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

        % 保存モードの確認
        saveMode = p.Results.SaveMode;
        if isempty(saveMode)
            saveMode = questdlg('保存モードを選択してください:', ...
                '保存モードの選択', '個別保存', '一括保存', 'キャンセル', '個別保存');

            if strcmp(saveMode, 'キャンセル')
                fprintf('処理がキャンセルされました。\n');
                return;
            elseif strcmp(saveMode, '個別保存')
                saveMode = 'individual';
            else
                saveMode = 'batch';
            end
        end

        outputPath = '';
        if strcmp(saveMode, 'batch') && p.Results.SaveExtracted
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

function [currentData, currentLabels] = processFile(fullpath, params)
    % データの読み込み
    data = load(fullpath);
    validateFields(data);

    % データとラベルの初期化（rawDataは直接代入）
    currentData = data.rawData;
    currentLabels = data.labels;

    % データの基本情報を表示
    displayFileInfo(fullpath, currentData, currentLabels);

    % 検証モードの場合はここで終了
    if params.ValidateOnly
        currentData = [];
        currentLabels = [];
        return;
    end

    % クラスによる抽出が指定されている場合のみ実行
    if ~isempty(params.Classes)
        [currentData, currentLabels] = extractByClass(currentData, currentLabels, params.Classes);
    end

    % サンプルによる抽出が指定されている場合のみ実行
    if ~isempty(params.Samples) && ~isempty(currentData)
        [currentData, currentLabels] = extractBySamples(currentData, currentLabels, params.Samples);
    end

    % クラス番号の変換が指定されている場合のみ実行
    if ~isempty(params.ClassMap)
        currentLabels = remapClasses(currentLabels, params.ClassMap);
        fprintf('\nクラス番号の変換を実行しました\n');
        displayClassMapping(params.ClassMap);
    end

    % 抽出結果の表示
    displayExtractionResults(currentData, currentLabels);
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
    
    % ラベルの変換処理
    for i = 1:length(labels)
        for j = 1:size(classMap, 1)
            if labels(i).value == classMap(j, 1)
                labels(i).value = classMap(j, 2);
                numChanges(j) = numChanges(j) + 1;
                break;
            end
        end
    end
    
    % 変換結果の表示
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
    
    % クラスの分布を表示
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