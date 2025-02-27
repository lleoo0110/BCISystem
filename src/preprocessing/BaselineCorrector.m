classdef BaselineCorrector < handle
    properties (Access = private)
        params              % 設定パラメータを格納する構造体

        % 処理パラメータ (共通) - どの補正手法でも共通して使用するパラメータ
        applyToChannels     % 適用チャネル (空の場合は全チャネルに適用)。チャネル番号のベクトルで指定
        windowSize        % 補正窓サイズ (移動平均、区間平均の自動区間分割で使用)。サンプル数で指定
        overlap          % オーバーラップ率 (区間平均の自動区間分割で使用)。0から1の間の値で指定

        % 処理パラメータ (区間平均補正) - 区間平均補正 ('interval' method) で使用するパラメータ
        intervalType       % 区間タイプ ('auto', 'prepost', 'custom')
                           %   'auto': 自動区間分割 (windowSize, overlap を使用)
                           %   'prepost': イベント前後の区間 (preBaselineDuration, postBaselineDuration を使用)
                           %   'custom': カスタム区間 (baselineIntervals を使用)
        preBaselineDuration % プレベースライン区間長 (intervalType='prepost' の場合に使用)。イベント開始前のベースライン区間の長さ (秒単位)
        postBaselineDuration % ポストベースライン区間長 (intervalType='prepost' の場合に使用)。イベント終了後のベースライン区間の長さ (秒単位)
        baselineIntervals   % カスタムベースライン区間 (intervalType='custom' の場合に使用)。ベースライン区間をサンプル数で指定する Nx2 行列。
                           %   各行が1つの区間を表し、[開始サンプル, 終了サンプル] の形式

        % 処理パラメータ (トレンド除去補正) - トレンド除去補正 ('trend' method) で使用するパラメータ
        trendType          % トレンド除去タイプ ('polynomial', 'linear')
                           %   'polynomial': 多項式トレンド除去 (polynomialOrder 次数を使用)
                           %   'linear': 線形トレンド除去 (1次多項式)
        polynomialOrder    % 多項式次数 (trendType='polynomial' の場合に使用)。正の整数で指定

        % 結果保存用
        correctionInfo    % 補正情報を格納する構造体。補正後に情報が記録される
    end

    methods (Access = public)
        function obj = BaselineCorrector(params)
            % BaselineCorrector - コンストラクタ
            %   BaselineCorrector クラスのインスタンスを作成し、パラメータを初期化する
            %
            % 入力:
            %   params: 設定パラメータを格納した構造体。以下のフィールドを持つ構造体を想定:
            %       params.signal.preprocessing.baseline: ベースライン補正に関するパラメータ
            %       params.device.sampleRate: サンプリングレート
            %       params.device.channelCount: チャネル数

            obj.params = params;
            obj.initializeParameters(); % パラメータを初期化
        end

        function [correctedData, correctionInfo] = correctBaseline(obj, data, method, varargin)
            % correctBaseline - ベースライン補正を実行するメインメソッド
            % [correctedData, correctionInfo] = correctBaseline(obj, data, method, varargin)
            %
            % 入力:
            %   data:           補正対象データ (チャネル x 時間)。数値行列
            %   method:         補正方法 ('interval', 'trend', 'dc', 'moving' のいずれか)
            %   varargin:       可変長引数。method='interval' かつ intervalType='prepost' の場合、イベント開始/終了サンプルを渡す
            %                   例: correctBaseline(obj, data, 'interval', eventStartSample, eventEndSample);
            %
            % 出力:
            %   correctedData:  補正後データ (チャネル x 時間)。入力データと同じサイズの数値行列
            %   correctionInfo: 補正情報 (構造体)。実行された補正に関する情報を含む

            try
                % データの検証
                obj.validateData(data); % 入力データの形式とパラメータの妥当性を検証

                % 適用チャネルの決定 (空の場合は全チャネル)
                channelsToCorrect = 1:size(data, 1); % デフォルトは全チャネル
                if ~isempty(obj.applyToChannels)
                    channelsToCorrect = obj.applyToChannels; % applyToChannels が指定されていれば、指定されたチャネルのみ補正対象とする
                end

                % 補正方法に応じた処理
                switch lower(method) % 補正方法を小文字に変換して比較 (case-insensitive)
                    case 'interval'
                        % 区間平均補正
                        if strcmpi(obj.intervalType, 'prepost') % intervalType が 'prepost' の場合
                            if length(varargin) ~= 2 % 可変長引数が2つ (eventStartSample, eventEndSample) であるか確認
                                error('BaselineCorrector:InvalidArgument', ...
                                      'For intervalType="prepost", eventStartSample and eventEndSample must be provided.'); % エラーメッセージ
                            end
                            eventStartSample = varargin{1}; % イベント開始サンプル
                            eventEndSample = varargin{2};   % イベント終了サンプル
                            [correctedData, correctionInfo] = obj.intervalCorrection(data, channelsToCorrect, eventStartSample, eventEndSample); % 区間平均補正を実行 (prepost モード)
                        else
                            [correctedData, correctionInfo] = obj.intervalCorrection(data, channelsToCorrect); % 区間平均補正を実行 (auto または custom モード)
                        end
                    case 'trend'
                        % トレンド除去補正
                        [correctedData, correctionInfo] = obj.trendRemoval(data, channelsToCorrect); % トレンド除去補正を実行
                    case 'dc'
                        % DC除去補正
                        [correctedData, correctionInfo] = obj.dcRemoval(data, channelsToCorrect); % DC除去補正を実行
                    case 'moving'
                        % 移動平均補正
                        [correctedData, correctionInfo] = obj.movingAverageCorrection(data, channelsToCorrect); % 移動平均補正を実行
                    otherwise
                        % 不明な補正方法が指定された場合
                        error('BaselineCorrector:UnknownMethod', 'Unknown correction method: %s', method); % エラーメッセージ
                end

                obj.correctionInfo = correctionInfo; % 補正情報をオブジェクトのプロパティに保存

            catch ME % try-catch ブロックでエラーを捕捉
                error('BaselineCorrector:CorrectionError', 'Correction failed: %s', ME.message); % エラーメッセージを表示し、エラーを再throw
            end
        end

        function info = getCorrectionInfo(obj)
            % getCorrectionInfo - 補正情報を取得するメソッド
            % info = getCorrectionInfo(obj)
            %
            % 出力:
            %   info: 補正情報 (構造体)。correctionInfo プロパティの内容を返す

            info = obj.correctionInfo; % correctionInfo プロパティの値を返す
        end
    end

    methods (Access = private)
        function initializeParameters(obj)
            % initializeParameters - パラメータ構造体から設定を読み込み、プロパティを初期化するプライベートメソッド
            % initializeParameters(obj)
            %
            % 内部処理:
            %   params プロパティに格納された設定パラメータを読み込み、各プロパティに値を設定する。
            %   パラメータが存在しない場合は、デフォルト値を設定する。

            baseline_params = obj.params.signal.preprocessing.baseline; % 設定パラメータ構造体からベースライン補正に関するパラメータを抽出

            % 共通設定
            if isfield(baseline_params, 'applyToChannels') % 'applyToChannels' フィールドが存在するか確認
                obj.applyToChannels = baseline_params.applyToChannels; % 設定値があればそれを適用
            else
                obj.applyToChannels = []; % デフォルト値: 全チャネル (空配列で表現)
            end
            obj.windowSize = round(baseline_params.windowSize * obj.params.device.sampleRate); % 窓サイズ (秒 -> サンプル数に変換して丸め)
            obj.overlap = baseline_params.overlap; % オーバーラップ率

            % 区間平均設定
            if isfield(baseline_params, 'intervalType') % 'intervalType' フィールドが存在するか確認
                obj.intervalType = baseline_params.intervalType; % 設定値があればそれを適用
            else
                obj.intervalType = 'auto'; % デフォルト値: 'auto' (自動区間分割)
            end
            if isfield(baseline_params, 'preBaselineDuration') % 'preBaselineDuration' フィールドが存在するか確認
                obj.preBaselineDuration = round(baseline_params.preBaselineDuration * obj.params.device.sampleRate); % プレベースライン区間長 (秒 -> サンプル数に変換して丸め)
            else
                obj.preBaselineDuration = round(0.5 * obj.params.device.sampleRate); % デフォルト値: 0.5秒 (サンプル数)
            end
            if isfield(baseline_params, 'postBaselineDuration') % 'postBaselineDuration' フィールドが存在するか確認
                obj.postBaselineDuration = round(baseline_params.postBaselineDuration * obj.params.device.sampleRate); % ポストベースライン区間長 (秒 -> サンプル数に変換して丸め)
            else
                obj.postBaselineDuration = round(0.5 * obj.params.device.sampleRate); % デフォルト値: 0.5秒 (サンプル数)
            end
            if isfield(baseline_params, 'baselineIntervals') % 'baselineIntervals' フィールドが存在するか確認
                obj.baselineIntervals = round(baseline_params.baselineIntervals * obj.params.device.sampleRate); % カスタムベースライン区間 (時間 -> サンプル数に変換して丸め)
            else
                obj.baselineIntervals = []; % デフォルト値: [] (カスタム区間なし)
            end


            % トレンド除去設定
            if isfield(baseline_params, 'trendType') % 'trendType' フィールドが存在するか確認
                obj.trendType = baseline_params.trendType; % 設定値があればそれを適用
            else
                obj.trendType = 'polynomial'; % デフォルト値: 'polynomial' (多項式トレンド除去)
            end
            if isfield(baseline_params, 'polynomialOrder') % 'polynomialOrder' フィールドが存在するか確認
                obj.polynomialOrder = baseline_params.polynomialOrder; % 設定値があればそれを適用
            else
                obj.polynomialOrder = 3; % デフォルト値: 3 (3次多項式)
            end
        end

        function validateData(obj, data)
            % validateData - 入力データの検証を行うプライベートメソッド
            % validateData(obj, data)
            %
            % 入力:
            %   data: 補正対象データ (チャネル x 時間)。数値行列
            %
            % 内部処理:
            %   入力データの形式、次元数、データ型などを検証し、エラーがあればエラーを発生させる。
            %   パラメータの値の妥当性も検証する。

            validateattributes(data, {'numeric'}, ... % データ型が数値型であることを検証
                {'2d', 'nrows', obj.params.device.channelCount}, ... % 2次元行列であり、行数 (チャネル数) が設定と一致することを検証
                'BaselineCorrector', 'data'); % エラーメッセージのクラス名と変数名

            % パラメータ検証
            validatestring(obj.intervalType, {'auto', 'prepost', 'custom'}, 'BaselineCorrector', 'intervalType'); % intervalType が指定されたいずれかの文字列であることを検証
            validateattributes(obj.windowSize, {'numeric'}, {'scalar', 'positive'}, 'BaselineCorrector', 'windowSize'); % windowSize がスカラ値かつ正の値であることを検証
            validateattributes(obj.overlap, {'numeric'}, {'scalar', '>=', 0, '<', 1}, 'BaselineCorrector', 'overlap'); % overlap がスカラ値かつ 0以上1未満であることを検証
            validatestring(obj.trendType, {'polynomial', 'linear'}, 'BaselineCorrector', 'trendType'); % trendType が指定されたいずれかの文字列であることを検証
            validateattributes(obj.polynomialOrder, {'numeric'}, {'integer', '>=', 1}, 'BaselineCorrector', 'polynomialOrder'); % polynomialOrder が整数かつ 1以上であることを検証

            if strcmp(obj.intervalType, 'prepost') % intervalType が 'prepost' の場合
                validateattributes(obj.preBaselineDuration, {'numeric'}, {'scalar', 'nonnegative'}, 'BaselineCorrector', 'preBaselineDuration'); % preBaselineDuration がスカラ値かつ非負の値であることを検証
                validateattributes(obj.postBaselineDuration, {'numeric'}, {'scalar', 'nonnegative'}, 'BaselineCorrector', 'postBaselineDuration'); % postBaselineDuration がスカラ値かつ非負の値であることを検証
            elseif strcmp(obj.intervalType, 'custom') % intervalType が 'custom' の場合
                validateattributes(obj.baselineIntervals, {'numeric'}, {'2d', 'ncols', 2, 'nonnegative'}, 'BaselineCorrector', 'baselineIntervals'); % baselineIntervals が 2次元行列、列数が2、かつ非負の値であることを検証
                for i = 1:size(obj.baselineIntervals, 1) % 各カスタム区間について検証
                    if obj.baselineIntervals(i, 1) >= obj.baselineIntervals(i, 2) % 開始サンプルが終了サンプル以上の場合
                        error('BaselineCorrector:InvalidParameter', 'baselineIntervals: Start time must be less than end time in interval %d.', i); % エラーメッセージ
                    end
                end
            end
        end

        function [correctedData, correctionInfo] = intervalCorrection(obj, data, channelsToCorrect, varargin)
            % intervalCorrection - 区間平均によるベースライン補正を行うプライベートメソッド
            % [correctedData, correctionInfo] = intervalCorrection(obj, data, channelsToCorrect, varargin)
            %
            % 入力:
            %   data:           補正対象データ (チャネル x 時間)。数値行列
            %   channelsToCorrect: 補正を適用するチャネルのインデックスベクトル
            %   varargin:       可変長引数。intervalType='prepost' の場合、イベント開始/終了サンプルを渡す
            %
            % 出力:
            %   correctedData:  補正後データ (チャネル x 時間)。入力データと同じサイズの数値行列
            %   correctionInfo: 補正情報 (構造体)。補正に関する情報を含む

            correctedData = data; % 出力データを入力データで初期化 (in-place 処理を避けるため)
            correctionInfo = struct('method', 'interval', 'intervalType', obj.intervalType, 'corrections', []); % 補正情報構造体を初期化。補正手法と区間タイプを記録
            corrections = zeros(obj.params.device.channelCount, 1); % チャネルごとの補正量を保存するベクトルを初期化

            for ch = channelsToCorrect % 指定されたチャネルごとに処理
                intervalStart = []; % 区間開始インデックスを格納する変数を初期化

                switch obj.intervalType % 区間タイプに応じて処理を分岐
                    case 'auto'
                        % 自動区間分割 (windowSize, overlapを使用)
                        intervalStart = 1:round((1-obj.overlap)*obj.windowSize):size(data,2)-obj.windowSize+1; % オーバーラップを考慮して区間開始インデックスを計算
                        baselineValues = zeros(length(intervalStart), 1); % 各区間のベースライン値を格納するベクトルを初期化
                        for i = 1:length(intervalStart) % 各区間について処理
                            idx = intervalStart(i):min(intervalStart(i)+obj.windowSize-1, size(data,2)); % 現在の区間のサンプルインデックスを計算 (窓がデータ範囲を超える場合はclip)
                            baselineValues(i) = mean(data(ch, idx)); % 現在の区間の平均値をベースライン値として計算
                        end

                    case 'prepost'
                        % プレ/ポストベースライン区間を使用
                        if length(varargin) ~= 2 % 可変長引数が2つ (eventStartSample, eventEndSample) であるか確認
                            error('BaselineCorrector:InvalidArgument', ...
                                  'For intervalType="prepost", eventStartSample and eventEndSample must be provided.'); % エラーメッセージ
                        end
                        eventStartSample = varargin{1}; % イベント開始サンプル
                        eventEndSample = varargin{2};   % イベント終了サンプル

                        preBaselineStart = max(1, eventStartSample - obj.preBaselineDuration); % プレベースライン区間の開始サンプルを計算 (最小値は1)
                        preBaselineEnd = eventStartSample - 1; % プレベースライン区間の終了サンプル (イベント開始サンプルの1つ前)
                        postBaselineStart = eventEndSample + 1; % ポストベースライン区間の開始サンプル (イベント終了サンプルの1つ後)
                        postBaselineEnd = min(size(data, 2), eventEndSample + obj.postBaselineDuration); % ポストベースライン区間の終了サンプル (最大値はデータ長)

                        intervalStart = {[preBaselineStart:preBaselineEnd], [postBaselineStart:postBaselineEnd]}; % プレ/ポストベースライン区間を cell 配列で定義
                        baselineValues = []; % ベースライン値を格納するベクトルを初期化
                        for i = 1:length(intervalStart) % 各区間 (プレ/ポスト) について処理
                            idx = intervalStart{i}; % 現在の区間のサンプルインデックスを取得
                            if ~isempty(idx) % 区間が空でないか確認 (イベント開始/終了がデータの最初/最後の場合など、区間が空になる可能性がある)
                                baselineValues = [baselineValues, mean(data(ch, idx), 2)]; % 現在の区間の平均値をベースライン値として計算し、ベクトルに追加
                            end
                        end


                    case 'custom'
                        % カスタム区間を使用 (obj.baselineIntervals を使用)
                        intervalStart = {}; % 区間開始インデックスを格納する cell 配列を初期化
                        baselineValues = []; % ベースライン値を格納するベクトルを初期化
                        for i = 1:size(obj.baselineIntervals, 1) % カスタム区間ごとに処理
                            startSample = round(obj.baselineIntervals(i, 1)); % 区間開始サンプル (丸め処理)
                            endSample = round(obj.baselineIntervals(i, 2));   % 区間終了サンプル (丸め処理)
                            interval = [startSample:endSample]; % 現在の区間のサンプルインデックスを計算
                            if ~isempty(interval) % 区間が空でないか確認
                                intervalStart{end+1} = interval; % 区間を cell 配列に追加
                                baselineValues = [baselineValues, mean(data(ch, interval), 2)]; % 現在の区間の平均値をベースライン値として計算し、ベクトルに追加
                            end
                        end

                    otherwise
                        % 不明な区間タイプが指定された場合
                        error('BaselineCorrector:UnknownIntervalType', 'Unknown interval type: %s', obj.intervalType); % エラーメッセージ
                end


                % 平均ベースライン値を使用して補正
                baselineValue = mean(baselineValues); % 計算されたベースライン値の平均値を最終的なベースライン値とする
                correctedData(ch,:) = data(ch,:) - baselineValue; % データからベースライン値を差し引いて補正
                corrections(ch) = baselineValue; % チャネルごとの補正量を記録
            end

            correctionInfo.corrections = corrections; % 補正情報を構造体に格納 (チャネルごとの補正量)
        end

        function [correctedData, correctionInfo] = trendRemoval(obj, data, channelsToCorrect)
            % trendRemoval - トレンド除去によるベースライン補正を行うプライベートメソッド
            % [correctedData, correctionInfo] = trendRemoval(obj, data, channelsToCorrect)
            %
            % 入力:
            %   data:           補正対象データ (チャネル x 時間)。数値行列
            %   channelsToCorrect: 補正を適用するチャネルのインデックスベクトル
            %
            % 出力:
            %   correctedData:  補正後データ (チャネル x 時間)。入力データと同じサイズの数値行列
            %   correctionInfo: 補正情報 (構造体)。補正に関する情報を含む

            correctedData = data; % 出力データを入力データで初期化
            correctionInfo = struct('method', 'trend', 'trendType', obj.trendType, 'polynomials', cell(1, size(data,1))); % 補正情報構造体を初期化。補正手法、トレンドタイプ、多項式係数を保存する cell 配列を用意

            for ch = channelsToCorrect % 指定されたチャネルごとに処理
                x = 1:size(data,2); % 時間軸ベクトル (サンプルインデックス) を作成
                switch obj.trendType % トレンド除去タイプに応じて処理を分岐
                    case 'polynomial'
                        % 多項式フィッティング (obj.polynomialOrder 次数を使用)
                        [p, ~] = polyfit(x, data(ch,:), obj.polynomialOrder); % polyfit 関数で多項式フィッティング。p は多項式係数、~ は不要な出力を無視
                    case 'linear'
                        % 線形フィッティング (1次多項式)
                        [p, ~] = polyfit(x, data(ch,:), 1); % 1次多項式でフィッティング (線形近似)
                    otherwise
                        % 不明なトレンドタイプが指定された場合
                        error('BaselineCorrector:UnknownTrendType', 'Unknown trend type: %s', obj.trendType); % エラーメッセージ
                end
                trend = polyval(p, x); % polyval 関数でフィッティングした多項式に基づいてトレンド曲線を計算

                % トレンド除去
                correctedData(ch,:) = data(ch,:) - trend; % 元のデータからトレンド曲線を差し引いて補正
                correctionInfo.polynomials{ch} = p; % チャネルごとの多項式係数を補正情報に保存
            end
        end

        function [correctedData, correctionInfo] = dcRemoval(obj, data, channelsToCorrect)
            % dcRemoval - DC除去によるベースライン補正を行うプライベートメソッド
            % [correctedData, correctionInfo] = dcRemoval(obj, data, channelsToCorrect)
            %
            % 入力:
            %   data:           補正対象データ (チャネル x 時間)。数値行列
            %   channelsToCorrect: 補正を適用するチャネルのインデックスベクトル
            %
            % 出力:
            %   correctedData:  補正後データ (チャネル x 時間)。入力データと同じサイズの数値行列
            %   correctionInfo: 補正情報 (構造体)。補正に関する情報を含む

            correctedData = data; % 出力データを入力データで初期化
            correctionInfo = struct('method', 'dc', 'means', zeros(size(data,1), 1)); % 補正情報構造体を初期化。補正手法とチャネルごとの平均値を保存するベクトルを用意

            for ch = channelsToCorrect % 指定されたチャネルごとに処理
                dcValue = mean(data(ch,:)); % チャネルデータの平均値 (DC成分) を計算
                correctedData(ch,:) = data(ch,:) - dcValue; % データから DC成分を差し引いて補正
                correctionInfo.means(ch) = dcValue; % チャネルごとの平均値を補正情報に保存
            end
        end

        function [correctedData, correctionInfo] = movingAverageCorrection(obj, data, channelsToCorrect)
            % movingAverageCorrection - 移動平均によるベースライン補正を行うプライベートメソッド
            % [correctedData, correctionInfo] = movingAverageCorrection(obj, data, channelsToCorrect)
            %
            % 入力:
            %   data:           補正対象データ (チャネル x 時間)。数値行列
            %   channelsToCorrect: 補正を適用するチャネルのインデックスベクトル
            %
            % 出力:
            %   correctedData:  補正後データ (チャネル x 時間)。入力データと同じサイズの数値行列
            %   correctionInfo: 補正情報 (構造体)。補正に関する情報を含む

            correctedData = data; % 出力データを入力データで初期化
            correctionInfo = struct('method', 'moving', 'windowSize', obj.windowSize); % 補正情報構造体を初期化。補正手法と窓サイズを記録

            % 移動平均フィルタの設計
            b = ones(1, obj.windowSize) / obj.windowSize; % 移動平均フィルタの分子係数 (窓サイズ分の1のベクトル)
            a = 1; % 移動平均フィルタの分母係数 (1) - IIRフィルタではなくFIRフィルタとして実装

            for ch = channelsToCorrect % 指定されたチャネルごとに処理
                % 移動平均の計算と補正
                baseline = filtfilt(b, a, data(ch,:)); % filtfilt 関数でゼロ位相フィルタリング (順方向と逆方向のフィルタ処理を適用して位相遅れをなくす) を行い、移動平均を計算
                correctedData(ch,:) = data(ch,:) - baseline; % 元のデータから移動平均 (ベースライン) を差し引いて補正
            end
        end
    end
end