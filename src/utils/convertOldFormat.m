function convertOldFormat()
    try
        [filenames, filepath] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, ...
            '変換するデータファイルを選択', 'MultiSelect', 'on');
        
        if isequal(filenames, 0)
            error('ファイルが選択されませんでした。');
        end

        % 単一ファイル選択時はセル配列に変換
        if ~iscell(filenames)
            filenames = {filenames};
        end

        fprintf('\n=== 変換処理開始 ===\n');
        fprintf('選択されたファイル数: %d\n\n', length(filenames));

        % 保存モードの選択
        saveMode = questdlg('保存モードを選択してください:', ...
            '保存モードの選択', ...
            '個別保存', '一括保存', 'キャンセル', '個別保存');

        if strcmp(saveMode, 'キャンセル')
            fprintf('処理がキャンセルされました。\n');
            return;
        end

        outputPath = '';
        if strcmp(saveMode, '一括保存')
            outputPath = uigetdir(filepath, '保存先フォルダを選択してください');
            if isequal(outputPath, 0)
                error('保存先フォルダが選択されませんでした。');
            end
        end

        % 各ファイルの処理
        for i = 1:length(filenames)
            try
                fullpath = fullfile(filepath, filenames{i});
                fprintf('処理中 (%d/%d): %s\n', i, length(filenames), filenames{i});

                % データの読み込みと検証
                data = load(fullpath);
                validateFields(data);

                % データ変換
                [convertedData, conversionInfo] = convertData(data, fullpath);

                % 保存処理
                if strcmp(saveMode, '個別保存')
                    [savefile, savepath] = uiputfile('*.mat', ...
                        sprintf('変換後のファイル %d/%d の保存先を選択', i, length(filenames)), ...
                        fullfile(filepath, [filenames{i}(1:end-4) '_converted.mat']));

                    if isequal(savefile, 0)
                        fprintf('  ファイル %s の保存がスキップされました\n', filenames{i});
                        continue;
                    end

                    saveFilename = fullfile(savepath, savefile);
                else
                    saveFilename = fullfile(outputPath, [filenames{i}(1:end-4) '_converted.mat']);
                end

                % 保存の実行
                save(saveFilename, '-struct', 'convertedData');

                % 結果表示
                displayResults(convertedData, conversionInfo, saveFilename);

            catch ME
                fprintf('ファイル %s の処理中にエラー: %s\n', filenames{i}, ME.message);
                continue;
            end
        end

        fprintf('\n=== 変換処理完了 ===\n');

    catch ME
        fprintf('エラーが発生しました: %s\n', ME.message);
        fprintf('スタックトレース:\n%s\n', getReport(ME, 'extended'));
    end
end

function validateFields(data)
    required_fields = {'eegData', 'stimulusStart'};
    missing_fields = required_fields(~isfield(data, required_fields));
    
    if ~isempty(missing_fields)
        error('必須フィールドが見つかりません: %s', strjoin(missing_fields, ', '));
    end
end

function [convertedData, conversionInfo] = convertData(data, fullpath)
    % データ抽出と変換
    rawData = data.eegData(4:17, :);
    
    % ラベル変換
    values = num2cell(data.stimulusStart(:, 1));
    times = num2cell(uint64(data.stimulusStart(:, 2) * 1000));
    samples = num2cell(round(data.stimulusStart(:, 2) * 256));
    
    labels = struct('value', values, 'time', times, 'sample', samples);
    labels = labels(:);  % 確実に列vectorにする

    % 変換情報の作成
    conversionInfo = struct(...
        'sourceFile', fullpath, ...
        'originalSize', size(data.eegData), ...
        'convertedSize', size(rawData), ...
        'timestamp', datetime('now'), ...
        'sampleRate', 256 ...
    );

    % 変換後のデータ構造体
    convertedData = struct(...
        'rawData', rawData, ...
        'labels', labels, ...
        'conversionInfo', conversionInfo ...
    );
end

function displayResults(convertedData, conversionInfo, saveFilename)
    fprintf('  元のデータサイズ: [%d, %d]\n', conversionInfo.originalSize);
    fprintf('  変換後のデータサイズ: [%d, %d]\n', size(convertedData.rawData));
    fprintf('  ラベル数: %d\n', length(convertedData.labels));
    fprintf('  保存先: %s\n\n', saveFilename);
end
