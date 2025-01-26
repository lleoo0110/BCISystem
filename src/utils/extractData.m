function [extractedData, extractedLabels] = extractData(varargin)
    % データ抽出関数
    %
    % 使用例:
    %   [data, labels] = extractData();  % UIでファイル選択
    %   [data, labels] = extractData('Classes', [1 2], 'SaveExtracted', true);
    %   [data, labels] = extractData('Samples', 1:100, 'SaveExtracted', true);
    
    % 入力パーサーの設定
    p = inputParser;
    addParameter(p, 'Filename', '', @ischar);
    addParameter(p, 'Classes', [], @isnumeric);
    addParameter(p, 'Samples', [], @isnumeric);
    addParameter(p, 'ValidateOnly', false, @islogical);
    addParameter(p, 'SaveExtracted', false, @islogical);
    parse(p, varargin{:});

    try
        % ファイル選択UI
        if isempty(p.Results.Filename)
            [filename, filepath] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, 'データファイルを選択');
            if isequal(filename, 0)
                error('ファイルが選択されませんでした。');
            end
            fullpath = fullfile(filepath, filename);
        else
            fullpath = p.Results.Filename;
        end

        % データの読み込み
        data = load(fullpath);
        
        % 必須フィールドの確認
        required_fields = {'rawData', 'labels'};
        for i = 1:length(required_fields)
            if ~isfield(data, required_fields{i})
                error('Required field %s not found in the data file', required_fields{i});
            end
        end

        % データとラベルの初期化
        extractedData = data.rawData;  % rawDataは常に全体を保持
        extractedLabels = data.labels;

        % データの基本情報を表示
        fprintf('\nデータファイル: %s\n', fullpath);
        fprintf('データサイズ: [%d, %d]\n', size(extractedData));
        fprintf('ラベル数: %d\n', length(extractedLabels));

        % 検証モードの場合はここで終了
        if p.Results.ValidateOnly
            fprintf('検証完了\n');
            return;
        end

        % クラスによる抽出（labelsのみ）
        if ~isempty(p.Results.Classes)
            validIndices = false(length(extractedLabels), 1);
            for i = 1:length(extractedLabels)
                if ismember(extractedLabels(i).value, p.Results.Classes)
                    validIndices(i) = true;
                end
            end
            
            if any(validIndices)
                extractedLabels = extractedLabels(validIndices);
            else
                warning('指定されたクラスのデータが見つかりませんでした。');
                extractedLabels = [];
                return;
            end
        end

        % サンプルによる抽出（labelsのみ）
        if ~isempty(p.Results.Samples)
            validSamples = p.Results.Samples;
            validSamples = validSamples(validSamples <= length(extractedLabels));
            if ~isempty(validSamples)
                extractedLabels = extractedLabels(validSamples);
            else
                warning('指定されたサンプル範囲が無効です。');
                extractedLabels = [];
                return;
            end
        end

        % 抽出結果の表示
        fprintf('\n抽出結果:\n');
        fprintf('データサイズ: [%d, %d]\n', size(extractedData));
        fprintf('抽出されたラベル数: %d\n', length(extractedLabels));

        % クラスごとのサンプル数を表示
        if ~isempty(extractedLabels)
            uniqueClasses = unique([extractedLabels.value]);
            fprintf('\nクラスごとのサンプル数:\n');
            for i = 1:length(uniqueClasses)
                classCount = sum([extractedLabels.value] == uniqueClasses(i));
                fprintf('クラス %d: %d サンプル\n', uniqueClasses(i), classCount);
            end
        end

        % 抽出データの保存
        if p.Results.SaveExtracted && ~isempty(extractedLabels)
            % 保存ファイル名の生成
            [filepath, name, ~] = fileparts(fullpath);
            if ~isempty(p.Results.Classes)
                suffix = sprintf('_classes%s', strjoin(string(p.Results.Classes), '-'));
            elseif ~isempty(p.Results.Samples)
                suffix = sprintf('_samples%d-%d', min(p.Results.Samples), max(p.Results.Samples));
            else
                suffix = '_extracted';
            end
            
            saveFilename = fullfile(filepath, [name suffix '.mat']);
            
            % 保存の確認
            choice = questdlg('抽出したデータを保存しますか？', ...
                'データ保存の確認', ...
                'はい', 'いいえ', 'はい');
            
            if strcmp(choice, 'はい')
                % 保存先の選択
                [savefile, savepath] = uiputfile('*.mat', 'データの保存先を選択', saveFilename);
                if ~isequal(savefile, 0)
                    finalSavePath = fullfile(savepath, savefile);
                    
                    % 保存データの構築
                    saveData = struct();
                    saveData.rawData = extractedData;
                    saveData.labels = extractedLabels;
                    saveData.extractionInfo = struct(...
                        'sourceFile', fullpath, ...
                        'extractedClasses', p.Results.Classes, ...
                        'extractedSamples', p.Results.Samples, ...
                        'timestamp', datetime('now') ...
                    );
                    
                    % データの保存
                    save(finalSavePath, '-struct', 'saveData');
                    fprintf('\n抽出データを保存しました: %s\n', finalSavePath);
                end
            end
        end

    catch ME
        fprintf('エラーが発生しました: %s\n', ME.message);
        fprintf('エラーの詳細:\n');
        fprintf('  発生場所: %s\n', ME.stack(1).name);
        fprintf('  行番号: %d\n', ME.stack(1).line);
        rethrow(ME);
    end
end