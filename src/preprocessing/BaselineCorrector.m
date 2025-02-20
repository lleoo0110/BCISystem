classdef BaselineCorrector < handle
    properties (Access = private)
        params              % 設定パラメータ

        % 処理パラメータ (共通)
        applyToChannels     % 適用チャネル
        windowSize        % 補正窓サイズ (移動平均、区間平均)
        overlap          % オーバーラップ率 (区間平均)

        % 処理パラメータ (区間平均)
        intervalType       % 区間タイプ ('auto', 'prepost', 'custom')
        preBaselineDuration % プレベースライン区間長 (サンプル数)
        postBaselineDuration % ポストベースライン区間長 (サンプル数)
        baselineIntervals   % カスタムベースライン区間 (サンプル数, Nx2行列)

        % 処理パラメータ (トレンド除去)
        trendType          % トレンド除去タイプ ('polynomial', 'linear')
        polynomialOrder    % 多項式次数

        % 結果保存用
        correctionInfo    % 補正情報
    end

    methods (Access = public)
        function obj = BaselineCorrector(params)
            obj.params = params;
            obj.initializeParameters();
        end

        function [correctedData, correctionInfo] = correctBaseline(obj, data, method, varargin)
            % correctBaseline - ベースライン補正を実行
            % [correctedData, correctionInfo] = correctBaseline(obj, data, method, varargin)
            %
            % 入力:
            %   data:           補正対象データ (チャネル x 時間)
            %   method:         補正方法 ('interval', 'trend', 'dc', 'moving')
            %   varargin:       可変長引数 (method='interval' かつ intervalType='prepost' の場合、イベント開始/終了サンプルを渡す)
            %                   例: correctBaseline(obj, data, 'interval', eventStartSample, eventEndSample);
            %
            % 出力:
            %   correctedData:  補正後データ (チャネル x 時間)
            %   correctionInfo: 補正情報 (構造体)

            try
                % データの検証
                obj.validateData(data);

                % 適用チャネルの決定 (空の場合は全チャネル)
                channelsToCorrect = 1:size(data, 1);
                if ~isempty(obj.applyToChannels)
                    channelsToCorrect = obj.applyToChannels;
                end

                % 補正方法に応じた処理
                switch lower(method)
                    case 'interval'
                        if strcmpi(obj.intervalType, 'prepost')
                            if length(varargin) ~= 2
                                error('BaselineCorrector:InvalidArgument', ...
                                      'For intervalType="prepost", eventStartSample and eventEndSample must be provided.');
                            end
                            eventStartSample = varargin{1};
                            eventEndSample = varargin{2};
                            [correctedData, correctionInfo] = obj.intervalCorrection(data, channelsToCorrect, eventStartSample, eventEndSample);
                        else
                            [correctedData, correctionInfo] = obj.intervalCorrection(data, channelsToCorrect);
                        end
                    case 'trend'
                        [correctedData, correctionInfo] = obj.trendRemoval(data, channelsToCorrect);
                    case 'dc'
                        [correctedData, correctionInfo] = obj.dcRemoval(data, channelsToCorrect);
                    case 'moving'
                        [correctedData, correctionInfo] = obj.movingAverageCorrection(data, channelsToCorrect);
                    otherwise
                        error('BaselineCorrector:UnknownMethod', 'Unknown correction method: %s', method);
                end

                obj.correctionInfo = correctionInfo;

            catch ME
                error('BaselineCorrector:CorrectionError', 'Correction failed: %s', ME.message);
            end
        end

        function info = getCorrectionInfo(obj)
            info = obj.correctionInfo;
        end
    end

    methods (Access = private)
        function initializeParameters(obj)
            % initializeParameters - パラメータ構造体から設定を読み込み、プロパティを初期化
            baseline_params = obj.params.signal.preprocessing.baseline;

            % 共通設定
            if isfield(baseline_params, 'applyToChannels')
                obj.applyToChannels = baseline_params.applyToChannels;
            else
                obj.applyToChannels = []; % デフォルト値: 全チャネル
            end
            obj.windowSize = round(baseline_params.windowSize * obj.params.device.sampleRate);
            obj.overlap = baseline_params.overlap;

            % 区間平均設定
            if isfield(baseline_params, 'intervalType')
                obj.intervalType = baseline_params.intervalType;
            else
                obj.intervalType = 'auto'; % デフォルト値: auto
            end
            if isfield(baseline_params, 'preBaselineDuration')
                obj.preBaselineDuration = round(baseline_params.preBaselineDuration * obj.params.device.sampleRate); % 秒 -> サンプル数
            else
                obj.preBaselineDuration = round(0.5 * obj.params.device.sampleRate); % デフォルト値: 0.5秒 (サンプル数)
            end
            if isfield(baseline_params, 'postBaselineDuration')
                obj.postBaselineDuration = round(baseline_params.postBaselineDuration * obj.params.device.sampleRate); % 秒 -> サンプル数
            else
                obj.postBaselineDuration = round(0.5 * obj.params.device.sampleRate); % デフォルト値: 0.5秒 (サンプル数)
            end
            if isfield(baseline_params, 'baselineIntervals')
                obj.baselineIntervals = round(baseline_params.baselineIntervals * obj.params.device.sampleRate); % 時間 -> サンプル (エラー修正)
            else
                obj.baselineIntervals = []; % デフォルト値: []
            end


            % トレンド除去設定
            if isfield(baseline_params, 'trendType')
                obj.trendType = baseline_params.trendType;
            else
                obj.trendType = 'polynomial'; % デフォルト値: polynomial
            end
            if isfield(baseline_params, 'polynomialOrder')
                obj.polynomialOrder = baseline_params.polynomialOrder;
            else
                obj.polynomialOrder = 3; % デフォルト値: 3
            end
        end

        function validateData(obj, data)
            % validateData - 入力データの検証
            validateattributes(data, {'numeric'}, ...
                {'2d', 'nrows', obj.params.device.channelCount}, ...
                'BaselineCorrector', 'data');

            % パラメータ検証
            validatestring(obj.intervalType, {'auto', 'prepost', 'custom'}, 'BaselineCorrector', 'intervalType');
            validateattributes(obj.windowSize, {'numeric'}, {'scalar', 'positive'}, 'BaselineCorrector', 'windowSize');
            validateattributes(obj.overlap, {'numeric'}, {'scalar', '>=', 0, '<', 1}, 'BaselineCorrector', 'overlap');
            validatestring(obj.trendType, {'polynomial', 'linear'}, 'BaselineCorrector', 'trendType');
            validateattributes(obj.polynomialOrder, {'numeric'}, {'integer', '>=', 1}, 'BaselineCorrector', 'polynomialOrder');

            if strcmp(obj.intervalType, 'prepost')
                validateattributes(obj.preBaselineDuration, {'numeric'}, {'scalar', 'nonnegative'}, 'BaselineCorrector', 'preBaselineDuration');
                validateattributes(obj.postBaselineDuration, {'numeric'}, {'scalar', 'nonnegative'}, 'BaselineCorrector', 'postBaselineDuration');
            elseif strcmp(obj.intervalType, 'custom')
                validateattributes(obj.baselineIntervals, {'numeric'}, {'2d', 'ncols', 2, 'nonnegative'}, 'BaselineCorrector', 'baselineIntervals');
                for i = 1:size(obj.baselineIntervals, 1)
                    if obj.baselineIntervals(i, 1) >= obj.baselineIntervals(i, 2)
                        error('BaselineCorrector:InvalidParameter', 'baselineIntervals: Start time must be less than end time in interval %d.', i);
                    end
                end
            end
        end

        function [correctedData, correctionInfo] = intervalCorrection(obj, data, channelsToCorrect, varargin)
            % intervalCorrection - 区間平均によるベースライン補正
            correctedData = data;
            correctionInfo = struct('method', 'interval', 'intervalType', obj.intervalType, 'corrections', []);
            corrections = zeros(obj.params.device.channelCount, 1);

            for ch = channelsToCorrect
                intervalStart = []; % 初期化

                switch obj.intervalType
                    case 'auto'
                        % 自動区間分割 (windowSize, overlapを使用)
                        intervalStart = 1:round((1-obj.overlap)*obj.windowSize):size(data,2)-obj.windowSize+1;
                        baselineValues = zeros(length(intervalStart), 1);
                        for i = 1:length(intervalStart)
                            idx = intervalStart(i):min(intervalStart(i)+obj.windowSize-1, size(data,2));
                            baselineValues(i) = mean(data(ch, idx));
                        end

                    case 'prepost'
                        % プレ/ポストベースライン区間を使用
                        if length(varargin) ~= 2
                            error('BaselineCorrector:InvalidArgument', ...
                                  'For intervalType="prepost", eventStartSample and eventEndSample must be provided.');
                        end
                        eventStartSample = varargin{1};
                        eventEndSample = varargin{2};

                        preBaselineStart = max(1, eventStartSample - obj.preBaselineDuration);
                        preBaselineEnd = eventStartSample - 1;
                        postBaselineStart = eventEndSample + 1;
                        postBaselineEnd = min(size(data, 2), eventEndSample + obj.postBaselineDuration);

                        intervalStart = {[preBaselineStart:preBaselineEnd], [postBaselineStart:postBaselineEnd]}; % 区間を cell 配列で定義
                        baselineValues = [];
                        for i = 1:length(intervalStart)
                            idx = intervalStart{i};
                            if ~isempty(idx)
                                baselineValues = [baselineValues, mean(data(ch, idx), 2)];
                            end
                        end


                    case 'custom'
                        % カスタム区間を使用 (obj.baselineIntervals を使用)
                        intervalStart = {}; % cell 配列で初期化
                        baselineValues = [];
                        for i = 1:size(obj.baselineIntervals, 1)
                            startSample = round(obj.baselineIntervals(i, 1));
                            endSample = round(obj.baselineIntervals(i, 2));
                            interval = [startSample:endSample];
                            if ~isempty(interval) % 区間が空でないか確認
                                intervalStart{end+1} = interval; % 区間を cell 配列に追加
                                baselineValues = [baselineValues, mean(data(ch, interval), 2)];
                            end
                        end

                    otherwise
                        error('BaselineCorrector:UnknownIntervalType', 'Unknown interval type: %s', obj.intervalType);
                end


                % 平均ベースライン値を使用して補正
                baselineValue = mean(baselineValues);
                correctedData(ch,:) = data(ch,:) - baselineValue;
                corrections(ch) = baselineValue;
            end

            correctionInfo.corrections = corrections;
        end

        function [correctedData, correctionInfo] = trendRemoval(obj, data, channelsToCorrect)
            % trendRemoval - トレンド除去によるベースライン補正
            correctedData = data;
            correctionInfo = struct('method', 'trend', 'trendType', obj.trendType, 'polynomials', cell(1, size(data,1)));

            for ch = channelsToCorrect
                x = 1:size(data,2);
                switch obj.trendType
                    case 'polynomial'
                        % 多項式フィッティング (obj.polynomialOrder 次数を使用)
                        [p, ~] = polyfit(x, data(ch,:), obj.polynomialOrder);
                    case 'linear'
                        % 線形フィッティング (1次多項式)
                        [p, ~] = polyfit(x, data(ch,:), 1);
                    otherwise
                        error('BaselineCorrector:UnknownTrendType', 'Unknown trend type: %s', obj.trendType);
                end
                trend = polyval(p, x);

                % トレンド除去
                correctedData(ch,:) = data(ch,:) - trend;
                correctionInfo.polynomials{ch} = p;
            end
        end

        function [correctedData, correctionInfo] = dcRemoval(obj, data, channelsToCorrect)
            % dcRemoval - DC除去によるベースライン補正
            correctedData = data;
            correctionInfo = struct('method', 'dc', 'means', zeros(size(data,1), 1));

            for ch = channelsToCorrect
                dcValue = mean(data(ch,:));
                correctedData(ch,:) = data(ch,:) - dcValue;
                correctionInfo.means(ch) = dcValue;
            end
        end

        function [correctedData, correctionInfo] = movingAverageCorrection(obj, data, channelsToCorrect)
            % movingAverageCorrection - 移動平均によるベースライン補正
            correctedData = data;
            correctionInfo = struct('method', 'moving', 'windowSize', obj.windowSize);

            % 移動平均フィルタの設計
            b = ones(1, obj.windowSize) / obj.windowSize;
            a = 1;

            for ch = channelsToCorrect
                % 移動平均の計算と補正
                baseline = filtfilt(b, a, data(ch,:));
                correctedData(ch,:) = data(ch,:) - baseline;
            end
        end
    end
end