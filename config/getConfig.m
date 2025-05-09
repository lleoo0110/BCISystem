function params = getConfig(deviceType, varargin)
    % getConfig - デバイス設定とプリセットを読み込み、設定構造体を返す関数
    %
    % この関数は指定されたデバイスタイプとプリセット名に基づき、EEG解析システムの
    % 設定を読み込みます。設定には、デバイス仕様、信号処理パラメータ、特徴抽出方法、
    % 分類器設定などが含まれます。
    %
    % 構文:
    %   params = getConfig(deviceType)
    %   params = getConfig(deviceType, 'preset', presetName)
    %
    % 入力引数:
    %   deviceType - 文字列。デバイスタイプを指定 ('epocx', 'mn8', 'openbci8', 'openbci16', 'epocflex', 'select')
    %   varargin   - 名前-値ペアで指定する追加パラメータ
    %     'preset' - 文字列。使用するプリセット名 (例: 'template', 'character', 'magic')
    %                指定がない場合は 'template' が使用される
    %
    % 出力引数:
    %   params    - 設定パラメータを含む構造体
    %               主な構成要素: 
    %               - info: プリセット情報
    %               - device: デバイス設定
    %               - signal: 信号処理設定
    %               - feature: 特徴抽出設定
    %               - classifier: 分類器設定
    %               - gui: GUI表示設定
    %
    % 例:
    %   % EPOCXデバイスの標準テンプレート設定を読み込む
    %   params = getConfig('epocx');
    %
    %   % OpenBCI 8チャンネルデバイスのキャラクタープリセット設定を読み込む
    %   params = getConfig('openbci8', 'preset', 'character');
    %
    % 詳細:
    %   この関数は入力パラメータを検証し、デバイス設定を読み込んだ後、
    %   指定されたプリセットファイルを動的に評価して設定を構築します。
    %   最後に、設定の整合性チェックを行い、問題があれば警告や
    %   エラーを出力します。

    % 入力パーサーの設定と初期化
    p = inputParser;
    p.addRequired('deviceType', @(x) ischar(x) || isstring(x));
    p.addParameter('preset', '', @(x) ischar(x) || isstring(x));

    % 入力引数のパース実行
    parse(p, deviceType, varargin{:});

    try
        % デバッグ情報の出力開始
        fprintf('=== 設定読み込み ===\n');
        
        % プリセット名の決定（指定がなければテンプレートを使用）
        if isempty(p.Results.preset)
            presetName = 'template';
            fprintf('プリセット指定なし - テンプレート設定を使用\n');
        else
            presetName = p.Results.preset;
            fprintf('指定プリセット: %s\n', presetName);
        end

        % プリセットファイルの完全パスを構築
        currentDir = fileparts(mfilename('fullpath'));
        presetFile = fullfile(currentDir, 'presets', [presetName '.m']);

        % プリセットファイルの存在確認
        if ~exist(presetFile, 'file')
            error('プリセットファイルが見つかりません: %s', presetFile);
        end

        % プリセットファイルを読み込んで実行
        % 1. ファイルのディレクトリをパスに追加
        % 2. プリセット関数を実行して設定を取得
        % 3. 追加したパスを削除（環境をクリーンに保つ）
        [presetDir, presetFuncName] = fileparts(presetFile);
        addpath(presetDir);
        params = feval(presetFuncName);
        rmpath(presetDir);

        % デバイス固有の設定を追加
        params.device = getDeviceConfig(deviceType);

        % プリセット情報の表示（存在する場合）
        if isfield(params, 'info')
            fprintf('\nプリセット情報:\n');
            fprintf('  名前: %s\n', params.info.name);
            fprintf('  説明: %s\n', params.info.description);
            fprintf('  バージョン: %s\n', params.info.version);
            fprintf('  作成者: %s\n', params.info.author);
            fprintf('  作成日: %s\n', params.info.date);
            fprintf('---------------------------\n');
        end

        % デバイス設定情報の表示（存在する場合）
        if isfield(params, 'device')
            fprintf('デバイスタイプ: %s\n', deviceType);
            fprintf('  設定チャンネル数: %d\n', params.device.channelCount);
            if isfield(params.device, 'channels') && ~isempty(params.device.channels)
                % セル配列の場合は strjoin で連結
                if iscell(params.device.channels)
                    channelList = strjoin(params.device.channels, ', ');
                else
                    channelList = num2str(params.device.channels);
                end
                fprintf('  チャンネル配置: %s\n', channelList);
            else
                fprintf('  チャンネル配置: 未定義\n');
            end
            fprintf('  サンプルレート: %d Hz\n', params.device.sampleRate);
        end

        % 設定の整合性チェック
        validateConfig(params, deviceType);
        fprintf('=== 設定読み込み完了 ===\n');

    catch ME
        % エラー情報の詳細表示
        fprintf('\n=== エラー発生 ===\n');
        fprintf('エラーメッセージ: %s\n', ME.message);
        fprintf('エラー位置:\n');
        for i = 1:length(ME.stack)
            fprintf('  File: %s\n  Line: %d\n  Function: %s\n\n', ...
                ME.stack(i).file, ME.stack(i).line, ME.stack(i).name);
        end
        % エラーを再スロー（呼び出し元にエラーを伝播）
        rethrow(ME);
    end
end

function deviceConfig = getDeviceConfig(deviceType)
    % getDeviceConfig - デバイスタイプに基づいた基本設定を返す内部関数
    %
    % 入力引数:
    %   deviceType - デバイスタイプ文字列
    %
    % 出力引数:
    %   deviceConfig - デバイス設定構造体
    %
    % 詳細:
    %   この関数はデバイスタイプに基づいて、適切なチャンネル数、サンプリングレート、
    %   チャンネル配置など、デバイス固有の設定を構造体として返します。
    %   サポートされるデバイス: 'epocx', 'epocflex', 'mn8', 'openbci8', 'openbci16', 'select'
    
    switch lower(deviceType)
        case 'epocx'
            % EPOCX デバイス設定（Emotiv EPOC X）
            deviceConfig = struct(...
                'name', 'EPOCX', ...
                'channelCount', 14, ...
                'channelNum', 4:17, ...  % チャンネル番号範囲（修正：括弧を削除）
                'channels', {{'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'}}, ...
                'sampleRate', 256, ...
                'resolution', 14, ...
                'reference', 'CMS/DRL', ...
                'lsl', struct(...  % Lab Streaming Layer設定
                    'streamName', 'EmotivDataStream-EEG', ...
                    'type', 'EEG', ...
                    'format', 'float32', ...
                    'nominal_srate', 256, ...
                    'source_id', 'emotiv_epocx_1' ...
                ), ...
                'eog', struct(...  % 眼電位設定
                    'channelCount', 2, ...
                    'pairs', struct(...
                        'primary', struct(...
                            'left', 1, ...
                            'right', 14 ...
                        ), ...
                        'secondary', struct(...
                            'left', 2, ...
                            'right', 13 ...
                        ) ...
                    ) ...
                ) ...
            );
        
        case 'epocflex'
            % EPOCFLEX デバイス設定（Emotiv EPOC Flex）
            deviceConfig = struct(...
                'name', 'EPOCFLEX', ...
                'channelCount', 32, ...     % 最大32チャンネル対応
                'channelNum', 4:35, ...     % チャンネル番号範囲（修正：括弧を削除）
                'channels', {{'Cz','Fz','Fp1','F7','F3','FC1','C3','FC5','FT9','T7','CP5','CP1','P3','P7','PO9','O1','Pz','Oz','O2','PO10','P8','P4','CP2','CP6','T8','FT10','FC6','C4','FC2','F4','F8','Fp2'}}, ... % 10-20システムに基づく配置
                'sampleRate', 256, ...      % 256Hzのサンプリングレート
                'resolution', 16, ...       % 16ビット解像度
                'reference', 'CMS/DRL', ... % 共通モードセンス（CMS）と駆動右脚（DRL）を参照
                'lsl', struct(...  % Lab Streaming Layer設定
                    'streamName', 'EmotivDataStream-EEG', ...
                    'type', 'EEG', ...
                    'format', 'float32', ...
                    'nominal_srate', 256, ...
                    'source_id', 'emotiv_epocflex' ...
                ), ...
                'eog', struct(...  % 眼電位設定
                    'channelCount', 2, ...
                    'pairs', struct(...
                        'primary', struct(...
                            'left', 3, ...
                            'right', 14 ...
                        ), ...
                        'secondary', struct(...
                            'left', 32, ...
                            'right', 13 ...
                        ) ...
                    ) ...
                ) ...
            );

        case 'mn8'
            % MN8 デバイス設定（Emotiv MUSE NeuroLink 8）
            deviceConfig = struct(...
                'name', 'MN8', ...
                'channelNum', 4:5, ...  % チャンネル番号範囲（修正：括弧を削除）
                'channelCount', 2, ...
                'channels', {{'T7','T8'}}, ...  % 側頭部チャンネル
                'sampleRate', 128, ...  % 128Hzのサンプリングレート
                'resolution', 8, ...    % 8ビット解像度
                'reference', 'CMS/DRL', ...
                'lsl', struct(...  % Lab Streaming Layer設定
                    'streamName', 'EmotivDataStream-EEG', ...
                    'type', 'EEG', ...
                    'format', 'float32', ...
                    'nominal_srate', 128, ...
                    'source_id', 'emotiv_mn8' ...
                ) ...
            );

        case 'openbci8'
            % OpenBCI 8チャンネルデバイス設定
            deviceConfig = struct(...
                'name', 'OPENBCI8', ...
                'channelNum', 1:8, ...  % チャンネル番号範囲（修正：括弧を削除）
                'channelCount', 8, ...
                'channels', {{'Fp1','Fp2','C3','C4','P7','P8','O1','O2'}}, ...  % 標準的な8チャンネル配置
                'sampleRate', 125, ...  % 125Hzのサンプリングレート
                'resolution', 24, ...   % 24ビット解像度（高精度）
                'reference', 'SRB', ...  % SRB（可変基準電極）
                'lsl', struct(...  % Lab Streaming Layer設定
                    'streamName', 'OpenBCI-EEG', ...
                    'type', 'EEG', ...
                    'format', 'float32', ...
                    'nominal_srate', 125, ...
                    'source_id', 'openbci_8ch' ...
                ) ...
            );

        case 'openbci16'
            % OpenBCI 16チャンネルデバイス設定（Cyton+Daisy）
            deviceConfig = struct(...
                'name', 'OPENBCI16', ...
                'channelNum', 1:16, ...  % チャンネル番号範囲（修正：括弧を削除）
                'channelCount', 16, ...
                'channels', {{'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8'}}, ...  % 標準的な16チャンネル配置
                'sampleRate', 250, ...  % 250Hzのサンプリングレート（Daisyボード使用時）
                'resolution', 24, ...   % 24ビット解像度
                'reference', 'SRB', ...  % SRB（可変基準電極）
                'lsl', struct(...  % Lab Streaming Layer設定
                    'streamName', 'OpenBCI-EEG', ...
                    'type', 'EEG', ...
                    'format', 'float32', ...
                    'nominal_srate', 250, ...
                    'source_id', 'openbci_16ch' ...
                ) ...
            );

        case 'select'
            % SELECT デバイス設定（カスタム選択チャンネル）
            deviceConfig = struct(...
                'name', 'SELECT', ...
                'channelCount', 6, ...     % 選択された6チャンネル
                'channelNum', [1, 2, 6, 8, 27, 29], ...  % 選択されたチャンネル番号（個別指定なので括弧必要）
                'channels',  {{'Cz', 'Fz', 'FC1', 'FC5', 'FC6', 'FC2'}}, ...  % 選択されたチャンネル名
                'sampleRate', 256, ...      % 256Hzのサンプリングレート
                'resolution', 16, ...       % 16ビット解像度
                'reference', 'CMS/DRL' ... % 共通モードセンス（CMS）と駆動右脚（DRL）を参照
            );

        otherwise
            % 未知のデバイスタイプに対するエラー処理
            error('Unknown device type: %s', deviceType);
    end
end

function validateConfig(params, deviceType)
    % validateConfig - 設定の整合性をチェックする内部関数
    %
    % 入力引数:
    %   params - 検証する設定パラメータ構造体
    %   deviceType - デバイスタイプ文字列
    %
    % 詳細:
    %   この関数は、読み込まれた設定パラメータの整合性をチェックします。
    %   必須フィールドの存在確認、デバイス名の一致確認、数値パラメータの
    %   有効性確認などを行い、問題があれば警告またはエラーを出力します。

    % 必須フィールドの確認
    requiredFields = {'device'};
    for i = 1:length(requiredFields)
        if ~isfield(params, requiredFields{i})
            error('必須フィールドが欠落しています: %s', requiredFields{i});
        end
    end

    % デバイス設定の整合性チェック（デバイス名が一致するか）
    if ~strcmpi(params.device.name, getDeviceConfig(deviceType).name)
        warning('デバイス名が一致しません。設定: %s, 期待値: %s', ...
            params.device.name, getDeviceConfig(deviceType).name);
    end

    % プリセット固有の設定が存在する場合の追加チェック
    if isfield(params, 'signal') && isfield(params.signal, 'window')
        % サンプリングレートと時間窓の検証
        if isfield(params.signal.window, 'epochDuration')
            % epochDurationの有効性をチェック - 条件を分けて評価
            if ~isnumeric(params.signal.window.epochDuration)
                warning('epochDurationが数値ではありません');
            elseif params.signal.window.epochDuration <= 0
                warning('epochDurationが正しくありません: %f', params.signal.window.epochDuration);
            end
        elseif isfield(params.signal.window, 'analysis')
            % 旧フィールド名の場合はanalysisをチェック - 条件を分けて評価
            if ~isnumeric(params.signal.window.analysis)
                warning('analysisが数値ではありません');
            elseif params.signal.window.analysis <= 0
                warning('analysisが正しくありません: %f', params.signal.window.analysis);
            end
        else
            warning('epochDurationまたはanalysisが設定されていません');
        end
        
        % timeRangeの検証
        if isfield(params.signal.window, 'timeRange')
            % timeRangeのフォーマットをチェック
            if iscell(params.signal.window.timeRange)
                % 複数時間範囲の場合（セル配列）
                for i = 1:length(params.signal.window.timeRange)
                    range = params.signal.window.timeRange{i};
                    if ~isnumeric(range)
                        warning('timeRange[%d]が数値ではありません', i);
                    elseif length(range) ~= 2
                        warning('timeRange[%d]のサイズが正しくありません', i);
                    elseif range(1) >= range(2)
                        warning('timeRange[%d]の値が正しくありません: [%s]', i, mat2str(range));
                    end
                end
            elseif isnumeric(params.signal.window.timeRange)
                % 単一時間範囲の場合（数値配列）
                if size(params.signal.window.timeRange, 2) ~= 2
                    warning('timeRangeの形式が正しくありません: %s', mat2str(params.signal.window.timeRange));
                elseif any(params.signal.window.timeRange(:,1) >= params.signal.window.timeRange(:,2))
                    warning('timeRangeの値が正しくありません: %s', mat2str(params.signal.window.timeRange));
                end
            else
                warning('timeRangeの形式が正しくありません');
            end
        else
            warning('timeRangeが設定されていません');
        end

        % チャンネル数の整合性チェック
        if isfield(params.device, 'channels') && isfield(params.device, 'channelCount')
            if length(params.device.channels) ~= params.device.channelCount
                warning('チャンネル数の設定に不整合があります: 実際=%d, 設定=%d', ...
                    length(params.device.channels), params.device.channelCount);
            end
        end
    end
end