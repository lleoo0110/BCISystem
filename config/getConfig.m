function params = getConfig(deviceType, varargin)
    p = inputParser;
    addRequired(p, 'deviceType', @(x) ischar(x) || isstring(x));
    addParameter(p, 'preset', 'default/template', @(x) ischar(x) || isstring(x));
    parse(p, deviceType, varargin{:});
    
    try
        % プリセットパスの解析
        [presetCategory, presetName] = parsePresetPath(p.Results.preset);
        
        % 基本テンプレートの読み込み
        template = loadPreset('default/template');
        
        % デバイス設定の取得
        if ~isfield(template.device, lower(deviceType))
            error('Unknown device type: %s', deviceType);
        end
        deviceConfig = template.device.(lower(deviceType));
        
        % パラメータの構築
        params = template;
        params.device = deviceConfig;
        
        % カスタムプリセットの適用
        if ~strcmp(p.Results.preset, 'default/template')
            customPreset = loadPreset(p.Results.preset);
            params = mergeStructs(params, customPreset);
        end
        
    catch ME
        error('Configuration error: %s', ME.message);
    end
end

function [category, name] = parsePresetPath(presetPath)
    parts = strsplit(presetPath, '/');
    if length(parts) == 1
        category = 'custom';
        name = parts{1};
    else
        category = parts{1};
        name = parts{2};
    end
end

function preset = loadPreset(presetPath)
    try
        % 現在のファイルのディレクトリからプリセットのパスを取得
        currentDir = fileparts(mfilename('fullpath'));
        presetDir = fullfile(currentDir, 'presets');
        
        % プリセットファイルの完全パスを構築
        parts = strsplit(presetPath, '/');
        if length(parts) == 1
            fullPath = fullfile(presetDir, 'custom', [parts{1} '_preset.m']);
        else
            fullPath = fullfile(presetDir, parts{1}, [parts{2} '_preset.m']);
        end
        
        if ~exist(fullPath, 'file')
            error('Preset file not found: %s', fullPath);
        end
        
        % プリセットの読み込み
        [presetDir, presetName] = fileparts(fullPath);
        addpath(presetDir);
        preset = feval(presetName);
        rmpath(presetDir);
        
    catch ME
        error('Failed to load preset %s: %s', presetPath, ME.message);
    end
end

function result = mergeStructs(orig, new)
    % 構造体の結合を行う関数
    % orig: オリジナルの構造体
    % new: 新しい構造体（これの値が優先される）
    
    % 入力チェック
    if ~isstruct(orig) || ~isstruct(new)
        error('両方の入力が構造体である必要があります');
    end
    
    % 元の構造体をコピー
    result = orig;
    fields = fieldnames(new);
    
    % 各フィールドに対して再帰的に処理
    for i = 1:length(fields)
        field = fields{i};
        if isfield(orig, field)
            % フィールドが両方の構造体に存在する場合
            if isstruct(new.(field)) && isstruct(orig.(field))
                % 両方が構造体の場合は再帰的にマージ
                result.(field) = mergeStructs(orig.(field), new.(field));
            else
                % 構造体でない場合は新しい値で上書き
                result.(field) = new.(field);
            end
        else
            % フィールドが存在しない場合は新規追加
            result.(field) = new.(field);
        end
    end
end