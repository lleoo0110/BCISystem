function params = getConfig(deviceType, varargin)
    % 使用例：
    % params = getConfig('epocx');     % デフォルト設定で初期化
    % params = getConfig('epocx', 'preset', 'motor_imagery');   % プリセット設定で初期化
    % deviceType: 'epocx', 'mn8', 'openbci8', 'openbci16' から選択
    % preset: 'default', 'motor_imagery', 'ssvep', 'p300' から選択
    
    %% === 入力パーサー設定 ===
    p = inputParser;
    addRequired(p, 'deviceType', @(x) ischar(x) || isstring(x));
    addParameter(p, 'preset', 'default', @(x) ischar(x) || isstring(x));
    parse(p, deviceType, varargin{:});
    
    %% === 保存&読み込みパラメータ ===
    acquisition_params = struct(...
        'mode', 'offline', ...       % モード選択: 'offline'（解析用）または 'online'（リアルタイム処理用）
        'save', struct(...
            'enable', true, ...     % データ保存の有効/無効
            'name', 'test', ...     % 保存時のファイル名プレフィックス
            'path', './Experiment Data', ...    % データ保存先ディレクトリ
            'saveInterval', 60, ...     % 一時保存を行う間隔（秒）．大きすぎると負荷が高くなる
            'fields', struct(...     % 保存する項目の選択（true/false）
                'params', true, ...             % 設定情報の保存
                'rawData', true, ...            % 生脳波データの保存
                'labels', true, ...             % イベントマーカー/トリガー情報の保存
                'processedData', true, ...      % 前処理済みデータの保存
                'processedLabel', true, ...     % 処理済みラベル情報の保存
                'processingInfo', true, ...     % 処理情報
                'cspFilters', true, ...         % CSPフィルタの保存（学習後）
                'cspFeatures', true, ...        % CSP特徴量の保存
                'svmClassifier', true, ...       % 学習済みSVM分類器の保存
                'results', true ...            % 解析結果の保存
            ) ...
        ), ...
        'load', struct(...        % データ読み込み設定
            'enable', true, ...   % データ読み込みの有効/無効
            'filename', '', ...     % 読み込むファイル名（空文字の場合はブラウザで選択）
            'path', '', ...             % 読み込むファイルのパス（空文字の場合はブラウザで選択）
            'fields', struct(...  % 読み込む項目の選択（true/false）
                'params', true, ...          % 設定情報の読み込み
                'rawData', true, ...         % 生脳波データの読み込み
                'labels', true, ...          % イベントマーカー/トリガー情報の読み込み
                'processedData', true, ...      % 前処理済みデータの読み込み
                'processedLabel', true, ...     % 処理済みラベル情報の読み込み
                'processingInfo', true, ...     % 処理情報
                'cspFilters', true, ...         % CSPフィルタの読み込み
                'cspFeatures', true, ...        % CSP特徴量の読み込み
                'svmClassifier', true, ...      % 学習済みSVM分類器の読み込み
                'results', true ...            % 解析結果の読み込み
            ) ...
        ) ...
    );

    %% === LSLパラメータ ===
    lsl_params = struct(...
        'stream', struct(...
            'name', '', ...            % ストリーム名：デバイス設定から自動設定される
            'libraryPath', '', ...     % LSLライブラリのパス：実行ディレクトリから自動設定
            'type', 'EEG' ...         % ストリームタイプ：'EEG'固定．変更不要
        ), ...
        'simulate', struct(...        % シミュレーションモード設定
            'enable', true, ...       % シミュレーションの有効/無効：実機がない場合はtrue
            'signal', struct(...      % シミュレーション信号の設定
                'alpha', struct(...   % アルファ波（8-13Hz）のシミュレーション設定
                    'freq', 10, ...   % 中心周波数：8-13の間で設定（Hz）
                    'amplitude', 20 ... % 振幅：大きいほど信号が強くなる（推奨：5-20）
                ), ...
                'beta', struct(...      % ベータ波（13-30Hz）のシミュレーション設定
                    'freq', 20, ...     % 中心周波数：13-30の間で設定（Hz）
                    'amplitude', 10 ...   % 振幅：大きいほど信号が強くなる（推奨：3-15）
                ) ...
            ) ...
        ) ...
    );

    %% === UDP設定パラメータ ===
    % トリガーのマッピング設定（ラベルと数値の対応付け）
    trigger_mappings = {
        '安静', 1;         % クラス1：安静状態（基準状態）
        '炎魔法', 2;       % クラス2：タスク状態，必要に応じて追加可能
    };

    % マッピング構造体の動的生成（通常は変更不要）
    mapping_struct = struct();
    for i = 1:size(trigger_mappings, 1)
        field_name = sprintf('trigger%d', i);
        mapping_struct.(field_name) = struct(...
            'text', trigger_mappings{i,1}, ...
            'value', trigger_mappings{i,2} ...
        );
    end

    udp_params = struct(...
        'receive', struct(...
            'enable', true, ...         % UDP受信の有効/無効
            'port', 12345, ...          % 受信ポート番号：Unity等の送信側と合わせる
            'address', '127.0.0.1', ...     % 受信アドレス：ローカルホストの場合は変更不要
            'bufferSize', 1024, ...     % 受信バッファサイズ：通常は変更不要
            'encoding', 'UTF-8', ...     % 文字エンコーディング：通常は変更不要
            'triggers', struct(...      % トリガー設定
                'enabled', true, ...    % トリガー処理の有効/無効
                'mappings', mapping_struct, ...     % 上で設定したマッピングが自動設定
                'defaultValue', 0 ...   % トリガーなしの場合のデフォルト値
            ) ...
        ), ...
        'send', struct(...
            'enabled', true, ...       % UDP送信の有効/無効
            'port', 54321, ...         % 送信ポート番号：受信側と合わせる
            'address', '127.0.0.1', ... % 送信先アドレス：別PCの場合はIPアドレスを指定
            'bufferSize', 1024, ...    % 送信バッファサイズ：通常は変更不要
            'encoding', 'UTF-8' ...    % 文字エンコーディング：通常は変更不要
        ) ...
    );

    %% === 信号処理パラメータ ===
    signal_params = struct(...
        'enable', true, ...           % 信号処理の有効/無効：通常はtrue
        'window', struct(...          % 解析窓の設定
            'analysis', 2.0, ...      % 解析窓の長さ（秒）：ERDは2.0秒，MIは1.0秒程度
            'stimulus', 5.0, ...      % 刺激提示時間（秒）：実験プロトコルに合わせて設定
            'bufferSize', 15, ...     % データバッファのサイズ（秒）：メモリ使用量に影響
            'updateBuffer', 1, ...    % バッファの更新間隔（秒）：小さいほど処理負荷増
            'step', [], ...           % 解析窓のシフト幅：自動計算（変更不要）
            'updateInterval', [] ...  % 更新間隔：自動計算（変更不要）
        ), ...
        'epoch', struct(...           % エポック分割の設定
            'storageType', 'array', ... % データ形式：'array'または'cell'を選択
            'method', 'time', ...         % エポック化方法：'time'または'odd-even'（注意：welch.windowLengthよりも小さいエポックはエラーが出る）
            'overlap', 0.25, ...      % オーバーラップ率：0-1の値（0.25推奨）
            'baseline', [] ...        % ベースライン期間：自動設定（変更不要）
        ), ...
        'augmentation', struct(...    % データ拡張の設定
            'enabled', false, ...     % データ拡張の有効/無効
            'numAugmentations', 4, ... % 1エポックあたりの拡張数：2-8程度
            'maxShiftRatio', 0.1, ... % 最大シフト率：0-0.5の値（0.1推奨）
            'noiseLevel', 0.05 ...    % ノイズレベル：0-0.2の値（0.05推奨）
        ), ...
        'normalize', struct(...       % 正規化の設定
            'enabled', true, ...      % 正規化の有効/無効：計測データ全体に対して正規化
            'type', 'all', ...        % 正規化の種類：'all'（全体）or 'epoch'（エポックごと）
            'method', 'zscore' ...    % 正規化方法：'zscore'，'minmax'，'robust'から選択
        ), ...
        'frequency', struct(...       % 周波数解析の設定
            'min', 1, ...             % 解析する最小周波数（Hz）：通常1Hz
            'max', 50, ...            % 解析する最大周波数（Hz）：通常50Hz
            'bands', struct(...       % 周波数帯域の定義
                'delta', [1 4], ...   % デルタ波帯域（Hz）
                'theta', [4 8], ...   % シータ波帯域（Hz）
                'alpha', [8 13], ...  % アルファ波帯域（Hz）
                'beta',  [13 30], ... % ベータ波帯域（Hz）
                'gamma', [30 50] ...  % ガンマ波帯域（Hz）
            ) ...
        ), ...
        'filter', struct(...          % フィルタリングの設定
            'notch', struct(...       % ノッチフィルタ（商用周波数除去）
                'enabled', false, ... % ノッチフィルタの有効/無効
                'frequency', [50 60], ... % 除去する周波数：日本は50Hz，米国は60Hz
                'bandwidth', 2 ...    % フィルタの帯域幅（Hz）：通常2Hz
            ), ...
            'fir', struct(...         % FIRフィルタの設定
                'enabled', true, ...  % FIRフィルタの有効/無効：通常はtrue
                'scaledPassband', true, ... % パスバンドのスケーリング：true推奨
                'filterOrder', 1024, ... % フィルタ次数：大きいほど急峻だが遅延増加
                'designMethod', 'window', ... % 設計方法：'window'または'ls'
                'windowType', 'hamming', ... % 窓関数：'hamming'，'hann'，'blackman'等
                'passbandRipple', 1, ... % パスバンドリップル（dB）：1-3程度
                'stopbandAttenuation', 60 ... % ストップバンド減衰量（dB）：40-80程度
            ) ...
        ) ...
    );

    %% === 特徴抽出用パラメータ ===
    feature_params = struct(...
        'power', struct(...           % パワー値計算の設定
            'enable', true, ...       % パワー値計算の有効/無効
            'method', 'welch', ...    % パワー計算方法：'fft','welch','filter','wavelet','hilbert','bandpower'
            'normalize', true, ...    % パワー値の正規化：z-score正規化（標準化）
            'fft', struct(...         % FFT解析設定
                'windowType', 'hamming', ... % 窓関数：'hamming','hann','blackman'等
                'nfft', 512 ...       % FFTポイント数：2のべき乗（512推奨）
            ), ...
            'welch', struct(...       % Welch法設定（推奨）
                'windowType', 'hamming', ... % 窓関数：'hamming','hann','blackman'等
                'windowLength', 256, ... % 窓長：2のべき乗（256推奨）
                'overlap', 0.5, ...   % オーバーラップ率：0.5-0.75推奨
                'nfft', 512, ...      % FFTポイント数：2のべき乗
                'freqResolution', 0.5, ... % 周波数分解能（Hz）
                'segmentNum', 8 ...    % セグメント数：4-16程度
            ), ...
            'filter', struct(...      % フィルタリング設定
                'type', 'butter', ... % フィルタタイプ：'butter','cheby1','ellip'等
                'order', 4 ...        % フィルタ次数：4-8程度
            ), ...
            'wavelet', struct(...     % ウェーブレット解析設定
                'type', 'morl', ...   % ウェーブレットタイプ：'morl','mexh','haar'等
                'scaleNum', 128 ...   % スケール数：64-256程度
            ), ...
            'hilbert', struct(...     % ヒルベルト変換設定
                'filterOrder', 4 ...  % フィルタ次数：4-8程度
            ), ...
            'bands', struct(...       % 周波数帯域の設定
                'names', {{'delta', 'theta', 'alpha', 'beta', 'gamma'}}, ... % 解析する帯域名
                'delta', [0.5 4], ... % デルタ波帯域（Hz）
                'theta', [4 8], ...   % シータ波帯域（Hz）
                'alpha', [8 13], ...  % アルファ波帯域（Hz）
                'beta',  [13 30], ... % ベータ波帯域（Hz）
                'gamma', [30 45] ...  % ガンマ波帯域（Hz）
            ) ...
        ), ...
        'faa', struct(...            % 前頭部アルファ非対称性の設定
            'enable', true ...       % FAA特徴量の有効/無効
        ), ...
        'emotion', struct(...
            'enable', true, ...              % 感情分類の有効/無効
            'channels', struct(...           % 使用するチャンネル設定
                'left', [1, 3], ...          % 左前頭葉チャンネル
                'right', [14, 12] ...        % 右前頭葉チャンネル
            ), ...
            'thresholds', struct(...         % 閾値設定
                'abRatio', 1.0, ...          % α/β比の閾値
                'centerRegion', 0.3, ...     % 中心領域の判定閾値
                'faa', 0.5 ...               % FAA判定の閾値
            ), ...
            'labels', struct(...
                'states', {{'興奮', '喜び', '快適', 'リラックス', ...
                           '眠気', '憂鬱', '不快', '緊張', '安静'}}, ... % セル配列として定義
                'neutral', '安静' ...
            ), ...
            'coordinates', struct(...        % 座標変換設定
                'normalizeMethod', 'tanh', ...% 正規化方法
                'scaling', 1.0 ...           % スケーリング係数
            ) ...
        ), ...
        'erd', struct(...            % 事象関連脱同期の設定
            'enable', false ...      % ERD特徴量の有効/無効
        ), ...
        'csp', struct(...            % 共通空間パターンの設定
            'enable', true, ...     % CSP特徴量の有効/無効
            'storageType', 'array', ... % データ保存形式：'array'または'cell'
            'patterns', 7, ...       % 使用するパターン数：全チャンネル数以下
            'regularization', 0.05 ... % 正則化パラメータ：0.01-0.1程度
        ) ...
    );

    %% === 特徴分類用パラメータ ===
    classifier_params = struct(...
        'svm', struct(...            % SVMの設定
            'enable', true, ...     % SVM分類器の有効/無効
            'type', 'ecoc', ...       % 分類器タイプ：'svm' or ecoc
            'kernel', 'rbf', ...     % カーネル関数：'rbf','linear','polynomial'等
            'optimize', true, ...    % ハイパーパラメータ最適化：trueを推奨
            'probability', true, ... % 確率推定の有効/無効：trueを推奨
            'threshold', struct(...  % 閾値関連の設定を集約
                'rest', 0.5, ...     % デフォルトの安静状態閾値
                'useOptimal', true, ...  % 最適閾値を使用するかどうか
                'optimal', [], ...    % クロスバリデーションで得られる最適閾値
                'range', [0.1:0.05:0.9] ...  % 閾値探索の範囲
            ), ...
            'hyperparameters', struct(... % ハイパーパラメータ設定
                'optimizer', 'gridsearch', ... % 最適化手法：'gridsearch'または'bayesian'
                'boxConstraint', [0.1, 1, 10, 100], ... % Cパラメータの探索範囲
                'kernelScale', [0.1, 1, 10, 100] ... % γパラメータの探索範囲
            ) ...
        ), ...
        'evaluation', struct(...     % 評価設定
            'enable', true, ...      % 評価機能の有効/無効
            'method', 'kfold', ...   % 評価方法：'kfold'または'holdout'
            'kfold', 5, ...          % k分割交差検証のk値：5または10推奨
            'holdoutRatio', 0.2, ... % ホールドアウト法の検証データ比率
            'metrics', struct(...    % 評価指標の設定
                'accuracy', true, ... % 正解率の計算：true/false
                'precision', true, ... % 適合率の計算：true/false
                'recall', true, ...   % 再現率の計算：true/false
                'f1score', true, ... % F1スコアの計算：true/false
                'auc', true, ...     % AUC値の計算：true/false
                'confusion', true ... % 混同行列の計算：true/false
            ), ...
            'visualization', struct(... % 可視化設定
                'enable', true, ...  % 可視化機能の有効/無効
                'confusionMatrix', true, ... % 混同行列の表示
                'roc', true, ...     % ROC曲線の表示
                'learningCurve', true, ... % 学習曲線の表示
                'featureImportance', true ... % 特徴量重要度の表示
            ) ...
        ) ...
    );

    %% === ディスプレイ用パラメータ ===
    gui_params = struct(...
        'display', struct(...
            'visualization', struct(...
                'refreshRate', 0.2, ...     % 表示更新レート（秒）：0.1-0.5程度
                'enable', struct(...        % 表示項目の有効/無効設定
                    'rawData', true, ...    % 生データの表示：true/false
                    'processedData', true, ... % 処理済みデータの表示：true/false
                    'spectrum', true, ...   % スペクトル表示：true/false
                    'ersp', true ...       % 事象関連スペクトル表示：true/false
                ), ...
                'scale', struct(...        % 表示スケールの設定
                    'auto', true, ...      % 自動スケーリング：trueを推奨
                    'raw', [4300 4400], ... % 生データの表示範囲（μV）
                    'processed', [-50 50], ... % 処理済みデータの表示範囲（μV）
                    'freq', [0 50], ...    % 周波数表示範囲（Hz）
                    'power', [0.01 100], ... % パワー表示範囲（μV²/Hz）
                    'displaySeconds', 5 ... % 時系列データの表示時間幅（秒）
                ), ...
                'showBands', true, ...     % 周波数帯域の表示：true/false
                'ersp', struct(...         % ERSP表示の詳細設定
                    'scale', [-10 10], ...   % ERSPの表示範囲（dB）
                    'time', [-1 2], ...    % 時間範囲（秒）：刺激前後
                    'baseline', [-0.5 0], ... % ベースライン期間（秒）
                    'freqRange', [1 50], ... % 周波数範囲（Hz）
                    'numFreqs', 50, ...    % 周波数分割数：30-100程度
                    'method', 'wavelet', ... % 解析手法：'wavelet'または'stft'
                    'colormap', struct(...  % カラーマップ設定
                        'type', 'jet', ... % タイプ：'jet','parula','hot','cool'等
                        'reverse', false, ... % カラーマップの反転：true/false
                        'limit', 'sym', ... % 制限方法：'sym'（対称）または'abs'（絶対値）
                        'background', 'white' ... % 背景色：'white'または'black'
                    ) ...
                ) ...
            ) ...
        ), ...
        'slider', struct(...
            'enable', false, ...        % スライダーの有効/無効
            'position', [800 400 400 200], ... % [x y width height]
            'defaultValue', 1, ...     % デフォルト値
            'minValue', 1, ...         % 最小値
            'maxValue', 9, ...         % 最大値
            'steps', 9, ...            % ステップ数
            'title', 'Label Slider' ...     % ウィンドウタイトル
        ) ...
    );

    %% === パラメータの統合 ===
    params = struct();
    params.acquisition = acquisition_params;  % データ収集パラメータ
    params.lsl = lsl_params;                 % LSL関連パラメータ
    params.udp = udp_params;                 % UDP通信パラメータ
    params.signal = signal_params;           % 信号処理パラメータ
    params.feature = feature_params;         % 特徴抽出パラメータ
    params.classifier = classifier_params;    % 分類器パラメータ
    params.gui = gui_params;                 % GUI表示パラメータ
    params.device = configureDevice(deviceType);  % デバイス固有パラメータ

    % プリセットの適用（指定された場合）
    if ~strcmp(p.Results.preset, 'default')
        try
            preset = loadBuiltinPreset(p.Results.preset);  % プリセット読み込み
            params = mergeStructs(params, preset);  % プリセットとデフォルト値の結合
        catch ME
            error('プリセット "%s" の読み込みに失敗: %s', p.Results.preset, ME.message);
        end
    end
end

%% === デバイス設定用補助関数 ===
function deviceConfig = configureDevice(deviceType)
    % デバイスごとの設定
    switch lower(deviceType)
        case 'epocx'  % Emotiv EPOC X
            deviceConfig = struct(...
                'name', 'EPOCX', ...
                'channelCount', 14, ...      % チャンネル数
                'channelNum', [4:17], ...    % チャンネル番号
                'channels', {{'AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'}}, ... % 電極配置
                'sampleRate', 256, ...       % サンプリングレート（Hz）
                'resolution', 14, ...        % 解像度（ビット）
                'reference', 'CMS/DRL', ...  % リファレンス方式
                'streamName', 'EmotivDataStream-EEG' ... % LSLストリーム名
            );
        case 'mn8'    % EMOTIV MN8
            deviceConfig = struct(...
                'name', 'MN8', ...
                'channelNum', [4:5], ...    % チャンネル番号
                'channelCount', 2, ...
                'channels', {{'T7','T8'}}, ...
                'sampleRate', 128, ...
                'resolution', 8, ...
                'reference', 'CMS/DRL', ...
                'streamName', 'EmotivDataStream-EEG' ...
            );
        case 'openbci8'   % OpenBCI 8ch
            deviceConfig = struct(...
                'name', 'OpenBCI8', ...
                'channelCount', 8, ...
                'channels', {{'Fp1','Fp2','C3','C4','P7','P8','O1','O2'}}, ...
                'sampleRate', 250, ...
                'resolution', 24, ...
                'reference', 'SRB', ...
                'streamName', 'OpenBCIDataStream' ...
            );
        case 'openbci16'  % OpenBCI 16ch
            deviceConfig = struct(...
                'name', 'OpenBCI16', ...
                'channelCount', 8, ...
                'channels', {{'Fp1','Fp2','C3','C4','P7','P8','O1','O2'}}, ... % 16ch用に更新予定
                'sampleRate', 125, ...
                'resolution', 24, ...
                'reference', 'SRB', ...
                'streamName', 'OpenBCIDataStream' ...
            );
        otherwise
            error('未知のデバイスタイプ: %s', deviceType);
    end
end

%% === 補助関数 ===
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
        try
            if isfield(orig, field)
                if isstruct(new.(field)) && isstruct(orig.(field))
                    % 両方が構造体の場合は再帰的にマージ
                    result.(field) = mergeStructs(orig.(field), new.(field));
                elseif iscell(new.(field)) && iscell(orig.(field))
                    % 両方がセル配列の場合は新しい値で上書き
                    result.(field) = new.(field);
                elseif isnumeric(new.(field)) && isnumeric(orig.(field))
                    % 両方が数値の場合は新しい値で上書き
                    result.(field) = new.(field);
                else
                    % その他の場合も新しい値で上書き
                    result.(field) = new.(field);
                end
            else
                % フィールドが存在しない場合は新規追加
                result.(field) = new.(field);
            end
        catch ME
            warning('フィールド %s の結合時にエラー: %s', field, ME.message);
        end
    end
end

function preset = loadBuiltinPreset(presetName)
    % 内蔵プリセットを読み込む関数
    % presetName: プリセット名（例：'motor_imagery', 'ssvep'）
    
    % 現在のファイルのディレクトリからプリセットディレクトリのパスを取得
    currentDir = fileparts(mfilename('fullpath'));
    presetDir = fullfile(currentDir, 'presets');
    
    % プリセットファイルの完全パスを構築
    presetPath = fullfile(presetDir, [presetName '_preset.m']);
    
    % プリセットファイルの存在確認
    if ~exist(presetPath, 'file')
        error('プリセットファイル "%s_preset.m" がpresetディレクトリに存在しません', presetName);
    end
    
    % プリセットを読み込む
    preset = loadPresetFromFile(presetPath);
end

function preset = loadPresetFromFile(presetPath)
    % プリセットファイルから設定を読み込む関数
    % presetPath: プリセットファイルの完全パス
    
    try
        % ファイルパスの検証
        if ~exist(presetPath, 'file')
            error('プリセットファイルが見つかりません: %s', presetPath);
        end
        
        % パスからファイル名を取得して関数として実行
        [~, funcName, ~] = fileparts(presetPath);
        
        % 一時的にプリセットのディレクトリをパスに追加
        presetDir = fileparts(presetPath);
        addpath(presetDir);
        
        try
            % 関数ハンドルを作成して実行
            fh = str2func(funcName);
            preset = fh();
        catch ME
            rmpath(presetDir);
            rethrow(ME);
        end
        
        % パスから削除
        rmpath(presetDir);
        
        % プリセットの検証
        validatePreset(preset);
        
    catch ME
        error('プリセットファイルの読み込みに失敗: %s\nエラー: %s', presetPath, ME.message);
    end
end

function validatePreset(preset)
    % プリセットの内容を検証する関数
    % preset: 検証するプリセット構造体
    
    % 必須フィールドの定義
    requiredFields = {'signal', 'feature', 'classifier'};
    
    % プリセットが構造体であることを確認
    if ~isstruct(preset)
        error('プリセットは構造体である必要があります');
    end
    
    % 必須フィールドの存在確認
    for i = 1:length(requiredFields)
        if ~isfield(preset, requiredFields{i})
            error('無効なプリセット構造体: フィールド ''%s'' が見つかりません', requiredFields{i});
        end
        if ~isstruct(preset.(requiredFields{i}))
            error('フィールド ''%s'' は構造体である必要があります', requiredFields{i});
        end
    end
end