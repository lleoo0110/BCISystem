function preset = template()
    %% === プリセット情報 ===
    preset_info = struct(...
        'name', 'template', ...
        'description', 'Default template preset', ...
        'version', '1.0', ...
        'author', 'Your Name', ...
        'date', '2025-01-01' ...
    );

    %% === トリガーマッピング設定 ===
    % トリガー値とクラスラベルの対応付け
    % 形式: {'状態名', トリガー値}
    trigger_mappings = {
        '安静', 1;         % クラス1: 安静状態 (ベースライン)
        'タスクA', 2;      % クラス2: タスク状態 (例: 運動イメージ)
        'タスクB', 3;      % クラス3: タスク状態 (例: 計算タスク)
    };

    % トリガーマッピング構造体の生成
    mapping_struct = struct();
    for i = 1:size(trigger_mappings, 1)
        field_name = sprintf('trigger%d', i);
        mapping_struct.(field_name) = struct(...
            'text', trigger_mappings{i,1}, ...  % トリガーテキスト
            'value', trigger_mappings{i,2} ...   % トリガー値 (1-255)
        );
    end
    
    %% === LSL設定 ===
    % Lab Streaming Layer通信の設定
    lsl = struct(...
        'simulate', struct(...           % シミュレーションモード設定
            'enable', true, ...          % true/false: シミュレーション有効/無効
            'signal', struct(...         % シミュレーション信号の設定
                'alpha', struct(...      % α波シミュレーション
                    'freq', 10, ...      % 周波数 (8-13 Hz)
                    'amplitude', 20 ...   % 振幅 (μV)
                ), ...
                'beta', struct(...       % β波シミュレーション
                    'freq', 20, ...      % 周波数 (13-30 Hz)
                    'amplitude', 10 ...   % 振幅 (μV)
                ) ...
            ) ...
        ) ...
    );

    %% === データ収集設定 ===
    % データ収集に関する基本設定
    acquisition = struct(...
        'mode', 'online', ...           % モード: 'online'/'offline'
        'emg', struct(...               % EMG計測の設定
            'enable', false, ...        % true/false: EMG計測有効/無効
            'channels', struct(...      % EMGチャンネル設定
                'channelNum', (1:2), ...   % チャンネル番号 (配列)
                'names', {{'EMG1', 'EMG2'}}, ... % チャンネル名
                'count', 2 ...          % チャンネル数 (1-8)
            ), ...
            'sampleRate', 250, ...      % サンプリングレート
            'lsl', struct(...           % EMG用LSL設定
                'streamName', 'OpenBCI-EMG', ... % ストリーム名
                'type', 'EMG', ...      % データタイプ: 'EMG'/'EEG'
                'format', 'float32', ... % データ形式: 'float32'/'double64'
                'nominal_srate', 250, ... % 公称サンプリングレート (Hz)
                'source_id', 'emg_device' ... % ソースID
            ) ...
        ), ...
        'eog', struct(...               % EOG計測の設定
            'enable', false, ...        % true/false: EOG計測有効/無効
            'thresholds', struct(...    % EOG検出閾値
                'left', -30, ...        % 左方向閾値 (-100～0 μV)
                'right', 30, ...        % 右方向閾値 (0～100 μV)
                'duration', 0.2, ...    % 最小持続時間 (0.1-1.0 秒)
                'blink', 100 ...        % 瞬き検出閾値 (50-200 μV)
            ), ...
            'filter', struct(...        % EOGフィルタ設定
                'bandpass', struct(...  % バンドパスフィルタ
                    'enable', true, ... % true/false: フィルタ有効/無効
                    'low', 0.1, ...    % 低域カットオフ (0.1-1 Hz)
                    'high', 15 ...     % 高域カットオフ (10-30 Hz)
                ), ...
                'order', 4 ...         % フィルタ次数 (2/4/6/8)
            ), ...
            'baseline', 0.5, ...       % ベースライン期間 (0.1-2.0 秒)
            'calibration', struct(...  % キャリブレーション設定
                'enable', false, ...   % true/false: キャリブレーション有効/無効
                'autoStart', false, ... % true/false: 自動開始
                'duration', 5, ...     % キャリブレーション時間 (3-10 秒)
                'trials', 3, ...       % 試行回数 (1-10)
                'confidenceThreshold', 0.6 ... % 信頼度閾値 (0.5-0.9)
            ) ...
        ), ...
        'save', struct(...             % データ保存設定
            'enable', true, ...        % true/false: 保存機能有効/無効
            'name', 'result', ...      % 保存ファイル名のプレフィックス
            'path', './Experiment Data', ... % 保存先ディレクトリ
            'saveInterval', 60, ...    % 一時保存間隔 (30-300 秒)
            'fields', struct(...       % 保存項目の選択
                'params', true, ...    % true/false: パラメータ保存
                'rawData', true, ...   % true/false: 生データ保存
                'emgData', true, ...   % true/false: EMGデータ保存
                'labels', true, ...    % true/false: ラベル保存
                'emgLabels', true, ... % true/false: EMGラベル保存
                'processedData', true, ... % true/false: 処理済みデータ保存
                'processedLabel', true, ... % true/false: 処理済みラベル保存
                'processingInfo', true, ... % true/false: 処理情報保存
                'classifier', true, ... % true/false: 分類器保存
                'results', true ...     % true/false: 結果保存
            ) ...
        ), ...
        'load', struct(...            % データ読み込み設定
            'enable', true, ...       % true/false: 読み込み機能有効/無効
            'filename', '', ...       % 読み込みファイル名 (空文字で選択ダイアログ)
            'path', '', ...          % 読み込みパス (空文字で選択ダイアログ)
            'fields', struct(...      % 読み込み項目の選択
                'params', true, ...   % true/false: パラメータ読み込み
                'rawData', true, ...  % true/false: 生データ読み込み
                'emgData', true, ...  % true/false: EMGデータ読み込み
                'labels', true, ...   % true/false: ラベル読み込み
                'emgLabels', true, ... % true/false: EMGラベル読み込み
                'processedData', true, ... % true/false: 処理済みデータ読み込み
                'processedLabel', true, ... % true/false: 処理済みラベル読み込み
                'processingInfo', true, ... % true/false: 処理情報読み込み
                'classifier', true, ... % true/false: 分類器読み込み
                'results', true ...    % true/false: 結果読み込み
            ) ...
        ) ...
    );

    %% === 通信設定 ===
    % UDP通信の設定
    udp = struct(...
        'receive', struct(...          % UDP受信設定
            'enable', true, ...        % true/false: UDP受信有効/無効
            'port', 12345, ...         % 受信ポート番号 (1024-65535)
            'address', '127.0.0.1', ... % 受信アドレス (IPv4形式)
            'bufferSize', 8192, ...    % バッファサイズ (1024-65535 bytes)
            'encoding', 'UTF-8', ...   % エンコーディング (UTF-8/ASCII/SJIS)
            'triggers', struct(...     % トリガー設定
                'enabled', true, ...   % true/false: トリガー処理有効/無効
                'mappings', mapping_struct, ... % トリガーマッピング
                'defaultValue', 0 ...  % デフォルトトリガー値 (0-255)
            ) ...
        ), ...
        'send', struct(...           % UDP送信設定
            'enabled', true, ...     % true/false: UDP送信有効/無効
            'port', 54321, ...       % 送信ポート番号 (1024-65535)
            'address', '127.0.0.1', ... % 送信先アドレス (IPv4形式)
            'bufferSize', 8192, ...  % バッファサイズ (1024-65535 bytes)
            'encoding', 'UTF-8' ...  % エンコーディング (UTF-8/ASCII/SJIS)
        ) ...
    );

    %% === 信号処理設定 ===
    % 信号処理パラメータの設定
    signal = struct(...
        'enable', true, ...           % true/false: 信号処理有効/無効
        'window', struct(...          % 解析窓の設定
            'analysis', 2.0, ...      % 解析窓長 (0.5-10.0 秒)
            'stimulus', 5.0, ...      % 刺激提示時間 (1.0-30.0 秒)
            'bufferSize', 15, ...     % バッファサイズ (5-30 秒)
            'updateBuffer', 1, ...    % バッファ更新間隔 (0.1-2.0 秒)
            'step', [], ...           % 解析窓シフト幅 (自動計算)
            'updateInterval', [] ...  % 更新間隔 (自動計算)
        ), ...
        'epoch', struct(...           % エポック化設定
            'method', 'time', ...     % 方法: 'time'/'odd-even'
            'storageType', 'array', ... % 保存形式: 'array'/'cell'
            'overlap', 0.25, ...      % オーバーラップ率 (0-0.9)
            'visual', struct(...      % 視覚タスク設定
                'enable', false, ...  % true/false: 視覚タスク有効/無効
                'taskTypes', {{'observation', 'imagery'}}, ... % タスク種類: 'observation'/'imagery'
                'observationDuration', 5.0, ... % 観察時間 (2.0-10.0 秒)
                'signalDuration', 1.0, ...     % 合図時間 (0.5-2.0 秒)
                'imageryDuration', 5.0 ...     % イメージ時間 (2.0-10.0 秒)
            ) ...
        ), ...
        'frequency', struct(...       % 周波数解析設定
            'min', 8, ...             % 最小周波数 (0.1-100 Hz)
            'max', 30, ...            % 最大周波数 (1-200 Hz)
            'bands', struct(...       % 周波数帯域定義
                'delta', [1 4], ...   % デルタ波帯域 (0.5-4 Hz)
                'theta', [4 8], ...   % シータ波帯域 (4-8 Hz)
                'alpha', [8 13], ...  % アルファ波帯域 (8-13 Hz)
                'beta',  [13 30], ... % ベータ波帯域 (13-30 Hz)
                'gamma', [30 50] ...  % ガンマ波帯域 (30-100 Hz)
            ) ...
        ), ...
        'preprocessing', struct(...   % 前処理設定
            'artifact', struct(...    % アーティファクト除去
                'enable', false, ...  % true/false: アーティファクト除去有効/無効
                'method', 'all', ...  % 方法: 'all'/'eog'/'emg'/'baseline'/'threshold'
                'thresholds', struct(...
                    'eog', 100, ...    % EOG閾値 (50-200 μV)
                    'emg', 100, ...    % EMG閾値 (50-200 μV)
                    'amplitude', 150 ... % 振幅閾値 (100-300 μV)
                ), ...
                'windowSize', 1.0 ...  % 解析窓サイズ (0.5-2.0 秒)
            ), ...
            'baseline', struct(...    % ベースライン補正
                'enable', false, ...  % true/false: ベースライン補正有効/無効
                'method', 'interval', ... % 方法: 'interval'/'trend'/'dc'/'moving'
                'windowSize', 1.0, ... % 窓サイズ (0.5-5.0 秒)
                'overlap', 0.5 ...     % オーバーラップ率 (0-0.9)
            ), ...
            'downsample', struct(...  % ダウンサンプリング設定
                'enable', false, ...  % true/false: ダウンサンプリング有効/無効
                'targetRate', 128, ... % 目標サンプリングレート (64/128/256 Hz)
                'filterOrder', 30 ...  % フィルタ次数 (10-50)
            ), ...
            'filter', struct(...      % フィルタリング設定
                'notch', struct(...   % ノッチフィルタ
                    'enable', false, ... % true/false: ノッチフィルタ有効/無効
                    'frequency', [50 60], ... % 除去周波数 ([50]/[60]/[50 60] Hz)
                    'bandwidth', 2 ...  % 帯域幅 (1-5 Hz)
                ), ...
                'fir', struct(...     % FIRフィルタ
                    'enable', true, ... % true/false: FIRフィルタ有効/無効
                    'scaledPassband', true, ... % true/false: パスバンドスケーリング
                    'filterOrder', 1024, ... % フィルタ次数 (128-2048)
                    'designMethod', 'window', ... % 設計法: 'window'/'kaiser'/'equiripple'
                    'windowType', 'hamming', ... % 窓関数: 'hamming'/'hann'/'blackman'
                    'passbandRipple', 1, ... % パスバンドリップル (0.5-3 dB)
                    'stopbandAttenuation', 60 ... % 阻止域減衰量 (40-80 dB)
                ) ...
            ), ...
            'normalize', struct(...    % 正規化設定
                'enable', true, ...    % true/false: 正規化有効/無効
                'type', 'all', ...     % 正規化範囲: 'all'/'epoch'
                'method', 'zscore' ... % 正規化方法: 'zscore'/'minmax'/'robust'
            ), ...
            'augmentation', struct(... % データ拡張設定
                'enable', false, ...   % true/false: データ拡張有効/無効
                'augmentationRatio', 7, ... % 拡張比率 (2-10)
                'combinationLimit', 3, ... % 最大手法数 (1-5)
                'methods', struct(...   % 拡張手法設定
                    'noise', struct(...  % ノイズ付加
                        'enable', true, ... % true/false: ノイズ付加有効/無効
                        'types', {{'gaussian', 'pink'}}, ... % 'gaussian'/'pink'/'white'
                        'variance', 0.01, ... % 分散 (0.001-0.1)
                        'probability', 0.5 ... % 適用確率 (0-1)
                    ), ...
                    'scaling', struct(... % スケーリング
                        'enable', true, ... % true/false: スケーリング有効/無効
                        'range', [0.9 1.1], ... % スケール範囲 (0.5-1.5)
                        'probability', 0.3 ... % 適用確率 (0-1)
                    ), ...
                    'timeshift', struct(... % 時間シフト
                        'enable', true, ... % true/false: 時間シフト有効/無効
                        'maxShift', 0.1, ... % 最大シフト量 (0.05-0.5 秒)
                        'probability', 0.3 ... % 適用確率 (0-1)
                    ), ...
                    'mirror', struct(...    % 反転
                        'enable', false, ... % true/false: 反転有効/無効
                        'probability', 0.2 ... % 適用確率 (0-1)
                    ), ...
                    'channelSwap', struct(... % チャンネル入れ替え
                        'enable', false, ... % true/false: チャンネル入替有効/無効
                        'pairs', {{{'F3','F4'}, {'C3','C4'}, {'P3','P4'}}}, ... % 入替ペア
                        'probability', 0.2 ... % 適用確率 (0-1)
                    ) ...
                ) ...
            ) ...
        ) ...
    );

    %% === 特徴抽出設定 ===
    % 特徴量抽出に関する設定
    feature = struct(...
        'power', struct(...            % パワースペクトル解析設定
            'enable', false, ...       % true/false: パワー解析有効/無効
            'method', 'welch', ...     % 解析方法: 'welch'/'filter'
            'normalize', struct(...    % パワー正規化設定
                'enable', false, ...   % true/false: 正規化有効/無効
                'methods', {{'relative', 'log'}} ... % 'relative'/'log'/'zscore'/'db'
            ), ...
            'welch', struct(...        % Welch法パラメータ
                'windowType', 'hamming', ... % 窓関数: 'hamming'/'hann'/'blackman'
                'windowLength', 256, ... % 窓長 (128-1024 サンプル)
                'overlap', 0.5, ...    % オーバーラップ率 (0-0.9)
                'nfft', 512, ...       % FFTポイント数 (256-2048)
                'freqResolution', 0.5, ... % 周波数分解能 (0.1-2.0 Hz)
                'segmentNum', 8 ...    % 平均化セグメント数 (4-16)
            ), ...
            'filter', struct(...       % フィルタリング法パラメータ
                'type', 'butter', ...  % フィルタタイプ: 'butter'/'cheby1'/'ellip'
                'order', 4 ...         % フィルタ次数 (2-8)
            ), ...
            'bands', struct(...        % 周波数帯域設定
                'names', {{'delta', 'theta', 'alpha', 'beta', 'gamma'}}, ...
                'delta', [0.5 4], ...  % デルタ波帯域 (0.5-4 Hz)
                'theta', [4 8], ...    % シータ波帯域 (4-8 Hz)
                'alpha', [8 13], ...   % アルファ波帯域 (8-13 Hz)
                'beta',  [13 30], ...  % ベータ波帯域 (13-30 Hz)
                'gamma', [30 45] ...   % ガンマ波帯域 (30-100 Hz)
            ) ...
        ), ...
        'faa', struct(...             % FAA特徴抽出設定
            'enable', false, ...      % true/false: FAA解析有効/無効
            'channels', struct(...    % チャンネル設定
                'left', [1], ...      % 左前頭部チャンネル (配列)
                'right', [14] ...     % 右前頭部チャンネル (配列)
            ), ...
            'threshold', 0 ...        % FAA判定閾値 (-1-1)
        ), ...
        'abRatio', struct(...         % α/β比設定
            'enable', false, ...      % true/false: α/β比解析有効/無効
            'channels', struct(...    % チャンネル設定
                'left', [1], ...      % 左前頭部チャンネル (配列)
                'right', [14] ...     % 右前頭部チャンネル (配列)
            ), ...
            'threshold', 1.0 ...      % 判定閾値 (0.5-2.0)
        ), ...
        'emotion', struct(...         % 感情特徴抽出設定
            'enable', false, ...      % true/false: 感情解析有効/無効
            'channels', struct(...    % チャンネル設定
                'left', [1], ...      % 左前頭部チャンネル (配列)
                'right', [14] ...     % 右前頭部チャンネル (配列)
            ), ...
            'threshold', 0.3, ...     % 判定閾値 (0.1-0.5)
            'labels', struct(...      % 感情ラベル設定
                'states', {{'興奮', '喜び', '快適', 'リラックス', ...
                           '眠気', '憂鬱', '不快', '緊張', '安静'}}, ... % 感情状態
                'neutral', '安静' ...  % 中立状態
            ), ...
            'coordinates', struct(... % 座標変換設定
                'normalizeMethod', 'tanh', ... % 正規化方法: 'tanh'/'sigmoid'/'linear'
                'scaling', 1.0 ...    % スケーリング係数 (0.1-2.0)
            ) ...
        ), ...
        'csp', struct(...            % CSP特徴抽出設定
            'enable', false, ...      % true/false: CSP解析有効/無効
            'patterns', 7, ...        % パターン数 (3-10)
            'regularization', 0.05 ... % 正則化パラメータ (0.01-0.1)
        ) ...
    );

    %% === 分類器設定 ===
    % 分類器のパラメータ設定
    num_classes = size(trigger_mappings, 1);  % クラス数を動的設定
    
    classifier = struct(...
        'activeClassifier', 'lstm', ... % 使用分類器: 'svm'/'ecoc'/'cnn'/'lstm'/'hybrid'
        'svm', struct(...              % SVMの設定
            'enable', false, ...       % true/false: SVM有効/無効
            'optimize', true, ...      % true/false: パラメータ最適化有効/無効
            'probability', true, ...   % true/false: 確率出力有効/無効
            'kernel', 'rbf', ...       % カーネル関数: 'linear'/'rbf'/'polynomial'
            'threshold', struct(...    % 閾値設定
                'rest', 0.5, ...       % 安静状態閾値 (0-1)
                'useOptimal', true, ... % true/false: 最適閾値使用
                'optimal', [], ...      % 最適閾値 (自動設定)
                'range', [0.1:0.05:0.9] ... % 閾値探索範囲
            ), ...
            'hyperparameters', struct(... % ハイパーパラメータ
                'optimizer', 'gridsearch', ... % 最適化法: 'gridsearch'/'bayesopt'
                'boxConstraint', [0.1, 1, 10, 100], ... % Cパラメータ候補
                'kernelScale', [0.1, 1, 10, 100] ... % カーネルスケール候補
            ) ...
        ), ...
        'ecoc', struct(...            % ECOC設定
            'enable', false, ...      % true/false: ECOC有効/無効
            'optimize', false, ...    % true/false: パラメータ最適化有効/無効
            'probability', true, ...  % true/false: 確率出力有効/無効
            'kernel', 'rbf', ...      % カーネル関数: 'linear'/'rbf'/'polynomial'
            'coding', 'onevsall', ... % コーディング: 'onevsall'/'allpairs'
            'learners', 'svm', ...    % 基本学習器: 'svm'/'tree'
            'hyperparameters', struct(... % ハイパーパラメータ
                'optimizer', 'gridsearch', ... % 最適化法: 'gridsearch'/'bayesopt'
                'boxConstraint', [0.1, 1, 10, 100], ... % Cパラメータ候補
                'kernelScale', [0.1, 1, 10, 100] ... % カーネルスケール候補
            ) ...
        ), ...
        'cnn', struct(...             % CNN設定
            'enable', false, ...      % true/false: CNN有効/無効
            'gpu', true, ...          % true/false: GPU使用有効/無効
            'optimize', true, ...     % true/false: パラメータ最適化有効/無効
            'architecture', struct(... % ネットワークアーキテクチャ
                'numClasses', num_classes, ... % クラス数 (自動設定)
                'convLayers', struct(... % 畳み込み層設定
                    'conv1', struct('size', [3 3], 'filters', 32, 'stride', 1, 'padding', 'same'), ...
                    'conv2', struct('size', [3 3], 'filters', 64, 'stride', 1, 'padding', 'same'), ...
                    'conv3', struct('size', [3 3], 'filters', 128, 'stride', 1, 'padding', 'same') ...
                ), ...
                'poolLayers', struct(... % プーリング層設定
                    'pool1', struct('size', 2, 'stride', 2), ... % size: 1-4, stride: 1-4
                    'pool2', struct('size', 2, 'stride', 2), ...
                    'pool3', struct('size', 2, 'stride', 2) ...
                ), ...
                'dropoutLayers', struct(... % ドロップアウト層設定
                    'dropout1', 0.3, ... % ドロップアウト率 (0-0.8)
                    'dropout2', 0.4, ...
                    'dropout3', 0.5 ...
                ), ...
                'batchNorm', true, ... % true/false: バッチ正規化有効/無効
                'fullyConnected', [128 64] ... % 全結合層ユニット数 (配列)
            ), ...
            'training', struct(...    % 学習設定
                'optimizer', struct(... % オプティマイザ設定
                    'type', 'adam', ... % タイプ: 'adam'/'sgdm'/'rmsprop'
                    'learningRate', 0.001, ... % 学習率 (0.0001-0.01)
                    'beta1', 0.9, ...   % 一次モーメント係数 (0.8-0.99)
                    'beta2', 0.999, ... % 二次モーメント係数 (0.9-0.9999)
                    'epsilon', 1e-8 ... % 数値安定化係数 (1e-8-1e-4)
                ), ...
                'maxEpochs', 100, ... % 最大エポック数 (10-1000)
                'miniBatchSize', 128, ... % ミニバッチサイズ (8-512)
                'shuffle', 'every-epoch', ... % シャッフル: 'never'/'once'/'every-epoch'
                'validation', struct(... % 検証設定
                    'enable', false, ... % true/false: 検証有効/無効
                    'frequency', 10, ... % 検証頻度 (エポック)
                    'patience', 20, ... % 早期終了の待機回数
                    'holdout', 0.2, ... % ホールドアウト比率 (0.1-0.3)
                    'kfold', 5 ...     % 交差検証分割数 (3-10)
                ) ...
            ), ...
            'optimization', struct(... % 最適化設定
                'searchSpace', struct(... % パラメータ探索範囲
                    'learningRate', [0.0005, 0.005], ... % 学習率範囲
                    'miniBatchSize', [64, 256], ...      % バッチサイズ範囲
                    'kernelSize', {[3,3], [5,5]}, ...    % カーネルサイズ候補
                    'numFilters', [16, 64], ...          % フィルタ数範囲
                    'dropoutRate', [0.3, 0.7], ...       % ドロップアウト率範囲
                    'fcUnits', [64, 256] ...             % 全結合層ユニット数範囲
                ) ...
            ) ...
        ), ...
        'lstm', struct(...            % LSTM設定
            'enable', false, ...      % true/false: LSTM有効/無効
            'gpu', false, ...         % true/false: GPU使用有効/無効
            'optimize', false, ...    % true/false: パラメータ最適化有効/無効
            'architecture', struct(... % ネットワークアーキテクチャ
                'numClasses', num_classes, ... % クラス数 (自動設定)
                'sequenceInputLayer', struct(... % 入力層設定
                    'inputSize', [], ...      % 入力サイズ (自動設定)
                    'sequenceLength', [], ... % シーケンス長 (自動設定)
                    'normalization', 'none' ... % 正規化: 'none'/'zscore'/'minmax'
                ), ...
                'lstmLayers', struct(...      % LSTM層設定
                    'lstm1', struct('numHiddenUnits', 128, 'OutputMode', 'last'), ...
                    'lstm2', struct('numHiddenUnits', 64, 'OutputMode', 'last'), ...
                    'lstm3', struct('numHiddenUnits', 32, 'OutputMode', 'last') ...
                ), ...
                'dropoutLayers', struct(...   % ドロップアウト層設定
                    'dropout1', 0.3, ... % ドロップアウト率 (0-0.8)
                    'dropout2', 0.4, ...
                    'dropout3', 0.5 ...
                ), ...
                'batchNorm', true, ... % true/false: バッチ正規化有効/無効
                'fullyConnected', [128 64] ... % 全結合層ユニット数 (配列)
            ), ...
            'training', struct(...    % 学習設定
                'optimizer', struct(... % オプティマイザ設定
                    'type', 'adam', ... % タイプ: 'adam'/'sgdm'/'rmsprop'
                    'learningRate', 0.001, ... % 学習率 (0.0001-0.01)
                    'beta1', 0.9, ...   % 一次モーメント係数 (0.8-0.99)
                    'beta2', 0.999, ... % 二次モーメント係数 (0.9-0.9999)
                    'epsilon', 1e-8, ... % 数値安定化係数 (1e-8-1e-4)
                    'gradientThreshold', 1 ... % 勾配クリッピング閾値 (0.1-10)
                ), ...
                'maxEpochs', 10, ...  % 最大エポック数 (5-100)
                'miniBatchSize', 32, ... % ミニバッチサイズ (8-128)
                'shuffle', 'every-epoch', ... % シャッフル: 'never'/'once'/'every-epoch'
                'validation', struct(... % 検証設定
                    'enable', false, ... % true/false: 検証有効/無効
                    'frequency', 10, ... % 検証頻度 (エポック)
                    'patience', 20, ... % 早期終了の待機回数
                    'holdout', 0.2, ... % ホールドアウト比率 (0.1-0.3)
                    'kfold', 5 ...     % 交差検証分割数 (3-10)
                ) ...
            ), ...
            'optimization', struct(... % 最適化設定
                'searchSpace', struct(... % パラメータ探索範囲
                    'learningRate', [0.0001, 0.01], ... % 学習率範囲
                    'miniBatchSize', [16, 64], ...      % バッチサイズ範囲
                    'numHiddenUnits', [32, 256], ...    % 隠れユニット数範囲
                    'numLayers', [1, 3], ...            % LSTM層数範囲
                    'dropoutRate', [0.2, 0.7], ...      % ドロップアウト率範囲
                    'fcUnits', [32, 256] ...            % 全結合層ユニット数範囲
                ) ...
            ) ...
        ), ...
        'hybrid', struct(...          % ハイブリッドモデル設定
            'enable', true, ...       % true/false: ハイブリッドモデル有効/無効
            'gpu', true, ...          % true/false: GPU使用有効/無効
            'optimize', true, ...     % true/false: パラメータ最適化有効/無効
            'evaluation', struct(...  % 評価設定
                'kfold', 5 ...       % 交差検証分割数 (3-10)
            ), ...
            'overfitThreshold', 5, ... % 過学習判定閾値 (%) (1-10)
            'cnn', struct(...        % CNN部分の設定
                'inputSize', [32, 32, 3], ... % 入力サイズ [高さ,幅,チャンネル]
                'filterSize', 3, ...  % フィルタサイズ (3-7)
                'numFilters', 16, ... % フィルタ数 (8-64)
                'poolSize', 2, ...    % プーリングサイズ (2-4)
                'poolStride', 2, ...  % プーリングストライド (1-4)
                'numClasses', num_classes ... % クラス数 (自動設定)
            ), ...
            'lstm', struct(...       % LSTM部分の設定
                'enable', true, ...   % true/false: LSTM有効/無効
                'gpu', true, ...      % true/false: GPU使用有効/無効
                'optimize', false, ... % true/false: パラメータ最適化有効/無効
                'architecture', struct(... % LSTMアーキテクチャ
                    'numClasses', num_classes, ...
                    'sequenceInputLayer', struct(...
                        'inputSize', [], ...      % 入力サイズ (自動設定)
                        'sequenceLength', [], ... % シーケンス長 (自動設定)
                        'normalization', 'none' ... % 正規化方法
                    ), ...
                    'lstmLayers', struct(...    % LSTM層設定
                        'lstm1', struct('numHiddenUnits', 256, 'OutputMode', 'last'), ...
                        'lstm2', struct('numHiddenUnits', 256, 'OutputMode', 'last'), ...
                        'lstm3', struct('numHiddenUnits', 256, 'OutputMode', 'last'), ...
                        'lstm4', struct('numHiddenUnits', 64, 'OutputMode', 'last') ...
                    ), ...
                    'dropoutLayers', struct(... % ドロップアウト層設定
                        'dropout1', 0.3, ... % ドロップアウト率 (0-0.8)
                        'dropout2', 0.4, ...
                        'dropout3', 0.4, ...
                        'dropout4', 0.5 ...
                    ), ...
                    'batchNorm', true, ... % true/false: バッチ正規化有効/無効
                    'fullyConnected', [128 64] ... % 全結合層ユニット数 (配列)
                ) ...
            ), ...
            'fc', struct(...         % 全結合層設定
                'numUnits', 50 ...   % ユニット数 (32-256)
            ), ...
            'training', struct(...   % 学習設定
                'optimizerType', 'adam', ... % オプティマイザ: 'adam'/'sgdm'/'rmsprop'
                'learningRate', 0.001, ... % 学習率 (0.0001-0.01)
                'maxEpochs', 20, ...      % 最大エポック数 (10-100)
                'miniBatchSize', 32, ...  % バッチサイズ (16-128)
                'validationFrequency', 30, ... % 検証頻度 (10-50)
                'validationPatience', 5, ... % 検証パティエンス (3-10)
                'validation', struct(... % 検証設定
                    'enable', false, ... % true/false: 検証有効/無効
                    'frequency', 10, ... % 検証頻度 (エポック)
                    'patience', 20, ...  % 早期終了の待機回数
                    'holdout', 0.2, ... % ホールドアウト比率 (0.1-0.3)
                    'kfold', 5 ...      % 交差検証分割数 (3-10)
                ) ...
            ), ...
            'optimization', struct(... % 最適化設定
                'searchSpace', struct(... % パラメータ探索範囲
                    'learningRate', [0.0001, 0.01], ... % 学習率範囲
                    'miniBatchSize', [16, 64], ...      % バッチサイズ範囲
                    'cnnFilters', [8, 32], ...          % CNNフィルタ数範囲
                    'filterSize', [3, 5], ...           % フィルタサイズ範囲
                    'poolSize', [2, 3], ...             % プーリングサイズ範囲
                    'numLstmLayers', [2, 4], ...        % LSTM層数範囲
                    'lstmUnits', [64, 256], ...         % LSTMユニット数範囲
                    'dropoutRate', [0.2, 0.5], ...      % ドロップアウト率範囲
                    'fcUnits', [32, 128] ...            % 全結合層ユニット数範囲
                ) ...
            ) ...
        ), ...
        'evaluation', struct(...        % 評価設定
            'enable', true, ...         % true/false: 評価機能有効/無効
            'method', 'kfold', ...      % 評価方法: 'kfold'/'holdout'/'bootstrap'
            'kfold', 5, ...             % 分割数 (3-10)
            'metrics', struct(...        % 評価指標設定
                'accuracy', true, ...    % true/false: 正解率評価
                'precision', true, ...   % true/false: 適合率評価
                'recall', true, ...      % true/false: 再現率評価
                'f1score', true, ...     % true/false: F1スコア評価
                'auc', true, ...         % true/false: AUC評価
                'confusion', true ...    % true/false: 混同行列評価
            ), ...
            'visualization', struct(...   % 可視化設定
                'enable', true, ...      % true/false: 可視化機能有効/無効
                'confusionMatrix', true, ... % true/false: 混同行列表示
                'roc', true, ...            % true/false: ROC曲線表示
                'learningCurve', true, ...  % true/false: 学習曲線表示
                'featureImportance', true ... % true/false: 特徴量重要度表示
            ) ...
        ) ...
    );

    %% === GUI表示設定 ===
    % GUI表示に関する設定
    gui = struct(...
        'display', struct(...          % 表示設定
            'visualization', struct(... % 可視化設定
                'refreshRate', 0.2, ... % 表示更新レート (0.1-1.0 秒)
                'enable', struct(...    % 表示項目の有効/無効
                    'rawData', true, ... % true/false: 生データ表示
                    'emgData', true, ... % true/false: EMGデータ表示
                    'spectrum', true, ... % true/false: スペクトル表示
                    'ersp', true ...     % true/false: ERSP表示
                ), ...
                'channels', struct(...  % チャンネル表示設定
                    'eeg', struct(...   % EEGチャンネル
                        'display', [1 2 3], ... % 表示チャンネル番号
                        'names', {{'AF3','F7','F3'}} ... % チャンネル名
                    ), ...
                    'emg', struct(...   % EMGチャンネル
                        'display', [1 2], ... % 表示チャンネル番号
                        'names', {{'EMG1','EMG2'}} ... % チャンネル名
                    ) ...
                ), ...
                'scale', struct(...     % スケール設定
                    'auto', true, ...   % true/false: 自動スケーリング
                    'raw', [-100 100], ... % 生データ表示範囲 (μV)
                    'emg', [-1000 1000], ... % EMG表示範囲 (μV)
                    'freq', [0 50], ... % 周波数表示範囲 (Hz)
                    'power', [0.01 100], ... % パワー表示範囲 (μV²/Hz)
                    'displaySeconds', 5 ... % 表示時間幅 (1-30 秒)
                ), ...
                'showBands', true, ... % true/false: 周波数帯域表示
                'ersp', struct(...     % ERSP表示設定
                    'scale', [0 100], ... % ERSP表示範囲 (dB)
                    'time', [0 5], ...    % 時間範囲 (秒)
                    'baseline', [-1 0], ... % ベースライン期間 (秒)
                    'freqRange', [1 50], ... % 周波数範囲 (Hz)
                    'numFreqs', 50, ...     % 周波数分割数 (20-100)
                    'method', 'wavelet', ... % 解析手法: 'wavelet'/'stft'
                    'colormap', struct(...   % カラーマップ設定
                        'type', 'jet', ...   % タイプ: 'jet'/'parula'/'viridis'
                        'reverse', false, ... % true/false: 反転
                        'limit', 'sym', ...  % 制限方法: 'sym'/'pos'/'neg'
                        'background', 'white' ... % 背景色: 'white'/'black'
                    ) ...
                ) ...
            ) ...
        ), ...
        'slider', struct(...          % スライダー設定
            'enable', false, ...      % true/false: スライダー有効/無効
            'position', [800 400 400 200], ... % 位置とサイズ [x y width height]
            'defaultValue', 1, ...    % デフォルト値 (1-9)
            'minValue', 1, ...        % 最小値 (整数)
            'maxValue', 9, ...        % 最大値 (整数)
            'steps', 9, ...           % ステップ数 (2-20)
            'title', 'Label Slider' ... % ウィンドウタイトル
        ) ...
    );

    %% === プリセット構造体の構築 ===
    % 設定をプリセット構造体にまとめる
    preset = struct();
    preset.info = preset_info;           % プリセット情報
    preset.acquisition = acquisition;     % データ収集設定
    preset.lsl = lsl;                    % LSL設定
    preset.udp = udp;                    % UDP設定
    preset.signal = signal;              % 信号処理設定
    preset.feature = feature;            % 特徴抽出設定
    preset.classifier = classifier;       % 分類器設定
    preset.gui = gui;                    % GUI設定
end