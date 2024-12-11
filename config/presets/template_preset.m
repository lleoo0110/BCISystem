
function preset = template_preset()
    % トリガーマッピングの設定
    trigger_mappings = {
        '安静', 1;         % クラス1：安静状態（基準状態）
        '炎魔法', 2;       % クラス2：タスク状態，必要に応じて追加可能
    };

    % マッピング構造体の生成（通常は変更不要）
    mapping_struct = struct();
    for i = 1:size(trigger_mappings, 1)
        field_name = sprintf('trigger%d', i);
        mapping_struct.(field_name) = struct(...
            'text', trigger_mappings{i,1}, ...
            'value', trigger_mappings{i,2} ...
        );
    end

    preset = struct(...
        'acquisition', struct(...
            'mode', 'offline', ...      % モード選択: 'offline'（解析用）または 'online'（リアルタイム処理用）
            'save', struct(...
                'enable', true, ...     % データ保存の有効/無効：通常はtrue
                'name', 'test', ...     % 保存時のファイル名プレフィックス：実験内容を反映した名前を推奨
                'path', './Experiment Data', ...    % データ保存先ディレクトリ：絶対パスまたは相対パス
                'saveInterval', 60, ...  % 一時保存を行う間隔（秒）：30-120秒程度，大きすぎると負荷増大
                'fields', struct(...     % 各データの保存有無を設定
                    'params', true, ...             % 設定情報：通常はtrue
                    'rawData', true, ...            % 生脳波データ：通常はtrue
                    'labels', true, ...             % イベントマーカー：通常はtrue
                    'processedData', true, ...      % 前処理済みデータ：必要に応じて
                    'processedLabel', true, ...     % 処理済みラベル：必要に応じて
                    'processingInfo', true, ...     % 処理パラメータ情報：デバッグ用
                    'cspFilters', true, ...         % CSPフィルタ：オンライン処理で使用
                    'cspFeatures', true, ...        % CSP特徴量：分析用
                    'svmClassifier', true, ...      % 学習済みSVM：オンライン処理で使用
                    'results', true ...             % 解析結果：評価用
                ) ...
            ), ...
            'load', struct(...        % データ読み込み設定
                'enable', true, ...   % データ読み込みの有効/無効：オンライン処理時はtrue
                'filename', '', ...   % 空文字の場合はファイル選択ダイアログを表示
                'path', '', ...       % 空文字の場合はファイル選択ダイアログを表示
                'fields', struct(...  % 各データの読み込み有無を設定（save.fieldsと同様）
                    'params', true, ...
                    'rawData', true, ...
                    'labels', true, ...
                    'processedData', true, ...
                    'processedLabel', true, ...
                    'processingInfo', true, ...
                    'cspFilters', true, ...
                    'cspFeatures', true, ...
                    'svmClassifier', true, ...
                    'results', true ...
                ) ...
            ) ...
        ), ...
        'signal', struct(...
            'enable', true, ...           % 信号処理の有効/無効：通常はtrue
            'window', struct(...          % 解析窓の設定
                'analysis', 2.0, ...      % 解析窓の長さ（秒）：ERDは2.0秒，MIは1.0秒程度
                'stimulus', 5.0, ...      % 刺激提示時間（秒）：実験プロトコルに合わせて設定
                'bufferSize', 15, ...     % データバッファのサイズ（秒）：10-20秒程度．大きいとメモリ使用量増
                'updateBuffer', 0.5, ...  % バッファの更新間隔（秒）：0.1-1.0秒．小さいほど処理負荷増
                'step', [], ...           % 解析窓のシフト幅：自動計算（変更不要）
                'updateInterval', [] ...  % 更新間隔：自動計算（変更不要）
            ), ...
            'epoch', struct(...           % エポック分割の設定
                'storageType', 'array', ... % データ形式：'array'（行列）または'cell'（セル配列）
                'overlap', 0.25, ...      % オーバーラップ率：0-1の値．0.25-0.5程度を推奨
                'baseline', [] ...        % ベースライン期間：自動設定（変更不要）
            ), ...
            'augmentation', struct(...    % データ拡張の設定
                'enabled', false, ...     % データ拡張の有効/無効：データ量が少ない場合はtrue
                'numAugmentations', 4, ... % 1エポックあたりの拡張数：2-8程度
                'maxShiftRatio', 0.1, ... % 最大シフト率：0-0.5の値．0.1-0.2程度を推奨
                'noiseLevel', 0.05 ...    % ノイズレベル：0-0.2の値．0.05-0.1程度を推奨
            ), ...
            'normalize', struct(...       % 正規化の設定
                'enabled', true, ...      % 正規化の有効/無効：通常はtrue
                'type', 'all', ...        % 正規化の種類：'all'（全体）or 'epoch'（エポックごと）
                'method', 'robust' ...    % 正規化方法：'zscore'，'minmax'，'robust'から選択
            ), ...
            'frequency', struct(...       % 周波数解析の設定
                'min', 1, ...             % 解析する最小周波数（Hz）：通常1Hz
                'max', 50, ...            % 解析する最大周波数（Hz）：通常50Hz
                'bands', struct(...       % 周波数帯域の定義
                    'delta', [1 4], ...   % デルタ波帯域（Hz）：1-4Hz
                    'theta', [4 8], ...   % シータ波帯域（Hz）：4-8Hz
                    'alpha', [8 13], ...  % アルファ波帯域（Hz）：8-13Hz
                    'beta',  [13 30], ... % ベータ波帯域（Hz）：13-30Hz
                    'gamma', [30 50] ...  % ガンマ波帯域（Hz）：30-50Hz
                ) ...
            ), ...
            'filter', struct(...          % フィルタリングの設定
                'notch', struct(...       % ノッチフィルタ（商用周波数除去）
                    'enabled', false, ... % ノッチフィルタの有効/無効：必要に応じて
                    'frequency', [50 60], ... % 除去する周波数（Hz）：日本は50Hz，米国は60Hz
                    'bandwidth', 2 ...    % フィルタの帯域幅（Hz）：1-3Hz程度
                ), ...
                'fir', struct(...         % FIRフィルタの設定
                    'enabled', true, ...  % FIRフィルタの有効/無効：通常はtrue
                    'scaledPassband', true, ... % パスバンドのスケーリング：true推奨
                    'filterOrder', 1024, ... % フィルタ次数：大きいほど急峻だが遅延増加（512-2048）
                    'designMethod', 'window', ... % 設計方法：'window'または'ls'（最小二乗）
                    'windowType', 'hamming', ... % 窓関数：'hamming'，'hann'，'blackman'等
                    'passbandRipple', 1, ... % パスバンドリップル（dB）：1-3程度
                    'stopbandAttenuation', 60 ... % ストップバンド減衰量（dB）：40-80程度
                ) ...
            ) ...
        ), ...
        'feature', struct(...
            'power', struct(...           % パワー値計算の設定
                'enable', true, ...       % パワー値計算の有効/無効：特徴量として使用する場合true
                'method', 'welch', ...    % パワー計算方法：'welch'（推奨），'fft'，'filter'，'wavelet'，'hilbert'
                'normalize', true, ...    % パワー値の正規化：z-score正規化を推奨
                'fft', struct(...         % FFT解析設定
                    'windowType', 'hamming', ... % 窓関数：'hamming'（推奨），'hann'，'blackman'等
                    'nfft', 512 ...       % FFTポイント数：2のべき乗（512または1024推奨）
                ), ...
                'welch', struct(...       % Welch法設定（推奨）
                    'windowType', 'hamming', ... % 窓関数：'hamming'（推奨），'hann'，'blackman'等
                    'windowLength', 256, ... % 窓長：2のべき乗（128-512程度）
                    'overlap', 0.5, ...   % オーバーラップ率：0.5-0.75推奨
                    'nfft', 512, ...      % FFTポイント数：windowLength以上の2のべき乗
                    'freqResolution', 0.5, ... % 周波数分解能（Hz）：0.1-1.0程度
                    'segmentNum', 8 ...    % セグメント数：4-16程度
                ), ...
                'filter', struct(...      % フィルタリング設定
                    'type', 'butter', ... % フィルタタイプ：'butter'，'cheby1'，'ellip'
                    'order', 4 ...        % フィルタ次数：4-8程度．高いほど急峻
                ), ...
                'wavelet', struct(...     % ウェーブレット解析設定
                    'type', 'morl', ...   % ウェーブレットタイプ：'morl'，'mexh'，'haar'等
                    'scaleNum', 128 ...   % スケール数：64-256程度
                ), ...
                'hilbert', struct(...     % ヒルベルト変換設定
                    'filterOrder', 4 ...  % フィルタ次数：4-8程度
                ), ...
                'bands', struct(...       % 周波数帯域の設定
                    'names', {{'delta', 'theta', 'alpha', 'beta', 'gamma'}}, ... % 解析する帯域名
                    'delta', [0.5 4], ... % デルタ波帯域（Hz）：0.5-4Hz
                    'theta', [4 8], ...   % シータ波帯域（Hz）：4-8Hz
                    'alpha', [8 13], ...  % アルファ波帯域（Hz）：8-13Hz
                    'beta',  [13 30], ... % ベータ波帯域（Hz）：13-30Hz
                    'gamma', [30 45] ...  % ガンマ波帯域（Hz）：30-45Hz
                ) ...
            ), ...
            'faa', struct(...            % 前頭部アルファ非対称性の設定
                'enable', true ...       % FAA特徴量の有効/無効：感情分析時はtrue
            ), ...
            'emotion', struct(...        % 感情分析の設定
                'enable', true, ...              % 感情分類の有効/無効
                'channels', struct(...           % 使用するチャンネル設定
                    'left', [1, 3], ...         % 左前頭葉チャンネル：電極配置に応じて設定
                    'right', [14, 12] ...       % 右前頭葉チャンネル：電極配置に応じて設定
                ), ...
                'thresholds', struct(...        % 閾値設定
                    'abRatio', 1.0, ...         % α/β比の閾値：0.5-2.0程度
                    'centerRegion', 0.3, ...    % 中心領域の判定閾値：0.2-0.4程度
                    'faa', 0.5 ...              % FAA判定の閾値：0.3-0.7程度
                ), ...
                'labels', struct(...
                    'states', {{'興奮', '喜び', '快適', 'リラックス', ...
                               '眠気', '憂鬱', '不快', '緊張', '安静'}}, ... % 感情状態のラベル
                    'neutral', '安静' ...       % 中立状態のラベル
                ), ...
                'coordinates', struct(...       % 座標変換設定
                    'normalizeMethod', 'tanh', ... % 正規化方法：'tanh'，'sigmoid'等
                    'scaling', 1.0 ...          % スケーリング係数：0.5-2.0程度
                ) ...
            ), ...
            'erd', struct(...            % 事象関連脱同期の設定
                'enable', false ...      % ERD特徴量の有効/無効：運動想起時はtrue
            ), ...
            'csp', struct(...            % 共通空間パターンの設定
                'enable', true, ...      % CSP特徴量の有効/無効：分類時はtrue
                'storageType', 'array', ... % データ保存形式：'array'または'cell'
                'patterns', 7, ...       % 使用するパターン数：チャンネル数の半分程度
                'regularization', 0.05 ... % 正則化パラメータ：0.01-0.1程度
            ) ...
        ), ...
        'classifier', struct(...
            'svm', struct(...            % SVMの設定
                'enable', true, ...      % SVM分類器の有効/無効：分類時はtrue
                'type', 'svm', ...       % 分類器タイプ：'svm'（2クラス）または'ecoc'（多クラス）
                'kernel', 'rbf', ...     % カーネル関数：'rbf'（推奨），'linear'，'polynomial'
                'optimize', true, ...    % ハイパーパラメータ最適化：精度重視の場合true
                'probability', true, ...  % 確率推定の有効/無効：閾値調整時はtrue
                'threshold', struct(...   % 閾値関連の設定
                    'rest', 0.5, ...     % デフォルトの安静状態閾値：0.3-0.7程度
                    'useOptimal', true, ... % 最適閾値を使用するか：通常はtrue
                    'optimal', [], ...    % クロスバリデーションで自動設定される
                    'range', [0.1:0.05:0.9] ... % 閾値探索の範囲：細かい刻みで探索
                ), ...
                'hyperparameters', struct(... % ハイパーパラメータ設定
                    'optimizer', 'gridsearch', ... % 最適化手法：'gridsearch'または'bayesian'
                    'boxConstraint', [0.1, 1, 10, 100], ... % Cパラメータの探索範囲
                    'kernelScale', [0.1, 1, 10, 100] ... % γパラメータの探索範囲
                ) ...
            ), ...
            'evaluation', struct(...     % 評価設定
                'enable', true, ...      % 評価機能の有効/無効：学習時はtrue
                'method', 'kfold', ...   % 評価方法：'kfold'（推奨）または'holdout'
                'kfold', 5, ...          % 分割数：5または10を推奨
                'holdoutRatio', 0.2, ... % ホールドアウト法の検証データ比率：0.2-0.3程度
                'metrics', struct(...    % 評価指標の設定
                    'accuracy', true, ... % 正解率：基本指標としてtrue
                    'precision', true, ... % 適合率：クラス別性能評価用
                    'recall', true, ...   % 再現率：クラス別性能評価用
                    'f1score', true, ... % F1スコア：バランスの取れた評価指標
                    'auc', true, ...     % AUC：2クラス分類の総合評価指標
                    'confusion', true ... % 混同行列：詳細な性能分析用
                ), ...
                'visualization', struct(... % 可視化設定
                    'enable', true, ...    % 可視化機能の有効/無効
                    'confusionMatrix', true, ... % 混同行列の表示：分類性能の詳細確認
                    'roc', true, ...       % ROC曲線：2クラス分類の性能評価
                    'learningCurve', true, ... % 学習曲線：過学習の確認
                    'featureImportance', true ... % 特徴量重要度：特徴選択用
                ) ...
            ) ...
        ), ...
        'gui', struct(...
            'display', struct(...
                'visualization', struct(...
                    'refreshRate', 0.2, ...     % 表示更新レート（秒）：0.1-0.5程度
                    'enable', struct(...        % 表示項目の有効/無効設定
                        'rawData', true, ...    % 生データの表示：信号確認用
                        'processedData', true, ... % 処理済みデータ：処理結果確認用
                        'spectrum', true, ...   % スペクトル表示：周波数分析用
                        'ersp', true ...       % 事象関連スペクトル：時間周波数分析用
                    ), ...
                    'scale', struct(...        % 表示スケールの設定
                        'auto', true, ...      % 自動スケーリング：通常はtrue
                        'raw', [4300 4400], ... % 生データの表示範囲（μV）
                        'processed', [-50 50], ... % 処理済みデータの表示範囲（μV）
                        'freq', [0 50], ...    % 周波数表示範囲（Hz）
                        'power', [0.01 100], ... % パワー表示範囲（μV²/Hz）
                        'displaySeconds', 5 ... % 時系列データの表示時間幅（秒）
                    ), ...
                    'showBands', true, ...     % 周波数帯域の表示：帯域の可視化
                    'ersp', struct(...         % ERSP表示の詳細設定
                        'scale', [-10 10], ...   % ERSPの表示範囲（dB）
                        'time', [-1 2], ...    % 時間範囲（秒）：刺激前後
                        'baseline', [-0.5 0], ... % ベースライン期間（秒）
                        'freqRange', [1 50], ... % 周波数範囲（Hz）
                        'numFreqs', 50, ...    % 周波数分割数：30-100程度
                        'method', 'wavelet', ... % 解析手法：'wavelet'または'stft'
                        'colormap', struct(...  % カラーマップ設定
                            'type', 'jet', ... % タイプ：'jet'，'parula'，'hot'，'cool'等
                            'reverse', false, ... % カラーマップの反転：必要に応じて
                            'limit', 'sym', ... % 制限方法：'sym'（対称）または'abs'（絶対値）
                            'background', 'white' ... % 背景色：'white'または'black'
                        ) ...
                    ) ...
                ) ...
            ) ...
        ) ...
    );
end
