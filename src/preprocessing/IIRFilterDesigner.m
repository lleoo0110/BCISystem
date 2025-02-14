classdef IIRFilterDesigner < handle
    % IIRFilterDesignerクラスは、IIRフィルタの設計と適用を行うクラスである。
    % バターワース、チェビシェフI型、チェビシェフII型、楕円フィルタなどのIIRフィルタを設計できる。
    % フィルタ設計パラメータ（フィルタ次数、カットオフ周波数、設計手法など）を properties として保持し、
    % designAndApplyFilter メソッドでフィルタ設計、適用、特性解析を実行する。
    % フィルタ情報は filterInfo プロパティに格納される。
    % 
    % フィルタ設計と適用
    % iirFilterDesigner = IIRFilterDesigner(params);
    % [filteredData, filterInfo] = iirFilterDesigner.designAndApplyFilter(data);
    % 
    % フィルタ情報の取得
    % info = iirFilterDesigner.getFilterInfo();

    properties (Access = private)
        params % パラメータ構造体
        filterInfo % フィルタ情報構造体

        % フィルタ設計パラメータ
        fmin % 最小通過域周波数
        fmax % 最大通過域周波数
        filterOrder % フィルタ次数
        designMethod % 設計手法 ('butterworth', 'chebyshev1', 'chebyshev2', 'ellip')
        passbandRipple % 通過域リップル (dB) (チェビシェフI型、楕円フィルタ用)
        stopbandAttenuation % ストップバンド減衰量 (dB) (チェビシェフII型、楕円フィルタ用)
        filterType % フィルタタイプ ('bandpass', 'lowpass', 'highpass', 'bandstop')
    end

    methods (Access = public)
        function obj = IIRFilterDesigner(params)
            % IIRFilterDesigner コンストラクタ
            %
            % 入力:
            %   params: パラメータ構造体

            obj.params = params; % パラメータ構造体をオブジェクトに格納
            obj.initializeParameters(); % フィルタパラメータを初期化
        end

        function [filteredData, filterInfo] = designAndApplyFilter(obj, data)
            % フィルタ設計、適用、特性解析を実行する関数
            %
            % 入力:
            %   data: フィルタ処理を行うデータ (チャンネル x 時間)
            %
            % 出力:
            %   filteredData: フィルタ処理後のデータ (チャンネル x 時間)
            %   filterInfo: フィルタ情報構造体

            try
                % データの検証
                obj.validateData(data); % 入力データの検証

                % フィルタの設計
                [num, den] = obj.designFilter(); % IIRフィルタを設計 (分子係数 num, 分母係数 den を取得)

                % フィルタ特性の解析
                filterInfo = obj.analyzeFilter(num, den); % 設計したフィルタの特性を解析

                % フィルタの適用
                filteredData = obj.applyFilter(data, num, den); % データにフィルタを適用

                % フィルタ情報の保存
                filterInfo.filterNumerator = num; % 分子係数をフィルタ情報に保存
                filterInfo.filterDenominator = den; % 分母係数をフィルタ情報に保存
                obj.filterInfo = filterInfo; % フィルタ情報をオブジェクトに保存

            catch ME
                error('IIRFilterDesigner:FilterError', 'フィルタ処理に失敗: %s', ME.message); % エラーが発生した場合、エラーメッセージを表示
            end
        end

        function info = getFilterInfo(obj)
            % フィルタ情報を取得する関数
            %
            % 出力:
            %   info: フィルタ情報構造体

            info = obj.filterInfo; % filterInfo プロパティを返す
        end
    end

    methods (Access = private)
        function initializeParameters(obj)
            % フィルタパラメータを初期化する関数
            % params 構造体からフィルタ設計に必要なパラメータを読み込み、オブジェクトのプロパティに設定する。

            filter_params = obj.params.signal.preprocessing.filter.iir; % IIRフィルタパラメータを params 構造体から取得
            freq_params = obj.params.signal.frequency; % 周波数関連パラメータを params 構造体から取得

            obj.fmin = freq_params.min; % 最小通過域周波数を設定
            obj.fmax = freq_params.max; % 最大通過域周波数を設定
            obj.filterOrder = filter_params.filterOrder; % フィルタ次数を設定
            obj.designMethod = filter_params.designMethod; % 設計手法を設定
            obj.passbandRipple = filter_params.passbandRipple; % 通過域リップルを設定
            obj.stopbandAttenuation = filter_params.stopbandAttenuation; % ストップバンド減衰量を設定
            obj.filterType = filter_params.filterType; % フィルタタイプを設定
        end

        function validateData(obj, data)
            % 入力データを検証する関数
            % 入力データが数値型、2次元配列、かつチャンネル数 (nrows) が params で指定されたチャンネル数と一致するか確認する。

            validateattributes(data, {'numeric'}, ... % データ型が数値型であることを検証
                {'2d', 'nrows', obj.params.device.channelCount}, ... % 2次元配列で、行数が指定チャンネル数と一致することを検証
                'IIRFilterDesigner', 'data'); % エラーメッセージの先頭に表示するクラス名と変数名
        end

        function [num, den] = designFilter(obj)
            % IIRフィルタを設計する関数
            % designMethod プロパティに基づいて、適切な IIR フィルタ設計関数 (butter, cheby1, cheby2, ellip) を用いてフィルタ係数を計算する。
            %
            % 出力:
            %   num: フィルタの分子係数 (numerator coefficients)
            %   den: フィルタの分母係数 (denominator coefficients)

            fs = obj.params.device.sampleRate; % サンプリングレートを取得

            % 正規化周波数を計算
            wn = [obj.fmin, obj.fmax] / (fs/2);
            order = obj.filterOrder; % フィルタ次数を取得

            switch lower(obj.designMethod) % 設計手法 (designMethod) に基づいて処理を分岐
                case 'butterworth' % Butterworthフィルタの場合
                    [num, den] = butter(order, wn, obj.filterType); % butter関数で設計
                case 'chebyshev1' % Chebyshev Type I フィルタの場合
                    rp = obj.passbandRipple; % パスバンドリップルを取得
                    [num, den] = cheby1(order, rp, wn, obj.filterType); % cheby1関数で設計
                case 'chebyshev2' % Chebyshev Type II フィルタの場合
                    rs = obj.stopbandAttenuation; % ストップバンド減衰量を取得
                    [num, den] = cheby2(order, rs, wn, obj.filterType); % cheby2関数で設計
                case 'ellip' % Ellipticフィルタの場合
                    rp = obj.passbandRipple;   % パスバンドリップルを取得
                    rs = obj.stopbandAttenuation;   % ストップバンド減衰量を取得
                    [num, den] = ellip(order, rp, rs, wn, obj.filterType); % ellip関数で設計
                otherwise % 未知の設計手法の場合
                    error('IIRFilterDesigner:UnknownDesignMethod', '未知のIIRフィルタ設計手法: %s', obj.designMethod); % エラーメッセージを表示
            end
        end

        function filterInfo = analyzeFilter(obj, num, den)
            % 設計したIIRフィルタの特性を解析する関数
            % 周波数応答、位相応答、群遅延などを計算し、filterInfo 構造体に格納する。
            % -3dB カットオフ周波数も検出する。
            %
            % 入力:
            %   num: フィルタの分子係数
            %   den: フィルタの分母係数
            %
            % 出力:
            %   filterInfo: フィルタ情報構造体

            fs = obj.params.device.sampleRate; % サンプリングレートを取得

            % 周波数応答を計算
            [h, f] = freqz(num, den, 1024, fs); % freqz関数で周波数応答 (h) と周波数ベクトル (f) を計算

            filterInfo = struct(); % フィルタ情報構造体を初期化
            filterInfo.frequency = f; % 周波数ベクトルを格納
            filterInfo.magnitude = 20 * log10(abs(h)); %  magnitude response (dB) を計算して格納
            filterInfo.phase = angle(h); % 位相応答を計算して格納
            filterInfo.groupDelay = grpdelay(num, den, 1024); % 群遅延を計算して格納

            % -3dBポイントの検出 (IIRフィルタに合わせて調整が必要な場合がある)
            mag_db = filterInfo.magnitude; % magnitude response (dB) を取得
            if strcmpi(obj.filterType, 'bandpass') || strcmpi(obj.filterType, 'bandstop') % バンドパスまたはバンドストップフィルタの場合
                passband_idx = find(f >= obj.fmin & f <= obj.fmax); % 通過域のインデックスを検出
                if ~isempty(passband_idx) % 通過域が存在する場合
                    cutoff_low = find(mag_db(1:passband_idx(1)) >= -3, 1, 'last'); % 低域側 -3dB 周波数を検出
                    cutoff_high = find(mag_db(passband_idx(end):end) >= -3, 1) + passband_idx(end) - 1; % 高域側 -3dB 周波数を検出
                    if isempty(cutoff_low) % 低域側カットオフ周波数が検出されない場合
                        cutoff_low = 1; % 周波数範囲の最初をカットオフ周波数とする
                    end
                    if isempty(cutoff_high) % 高域側カットオフ周波数が検出されない場合
                        cutoff_high = length(f); % 周波数範囲の最後をカットオフ周波数とする
                    end
                    filterInfo.cutoffFrequencies = [f(cutoff_low), f(cutoff_high)]; % カットオフ周波数を格納
                else
                    filterInfo.cutoffFrequencies = [NaN, NaN]; % 通過域がない場合は NaN を格納
                end
            elseif strcmpi(obj.filterType, 'lowpass') % ローパスフィルタの場合
                cutoff_high_idx = find(mag_db >= -3, 1, 'last'); % 高域側 -3dB 周波数を検出
                if isempty(cutoff_high_idx) % カットオフ周波数が検出されない場合
                    cutoff_high_idx = length(f); % 周波数範囲の最後をカットオフ周波数とする
                end
                filterInfo.cutoffFrequencies = [NaN, f(cutoff_high_idx)]; % カットオフ周波数を格納 (低域側は NaN)
            elseif strcmpi(obj.filterType, 'highpass') % ハイパスフィルタの場合
                cutoff_low_idx = find(mag_db >= -3, 1, 'last'); % 低域側 -3dB 周波数を検出
                if isempty(cutoff_low_idx) % カットオフ周波数が検出されない場合
                    cutoff_low_idx = 1; % 周波数範囲の最初をカットオフ周波数とする
                end
                filterInfo.cutoffFrequencies = [f(cutoff_low_idx), NaN]; % カットオフ周波数を格納 (高域側は NaN)
            else % その他のフィルタタイプの場合
                filterInfo.cutoffFrequencies = [NaN, NaN]; % カットオフ周波数は NaN
            end


            filterInfo.params = struct(... % フィルタパラメータを構造体に格納
                'order', obj.filterOrder, ... % フィルタ次数
                'method', obj.designMethod, ... % 設計手法
                'type', obj.filterType, ... % フィルタタイプ
                'fmin', obj.fmin, ... % 最小通過域周波数
                'fmax', obj.fmax, ... % 最大通過域周波数
                'passbandRipple', obj.passbandRipple, ... % 通過域リップル
                'stopbandAttenuation', obj.stopbandAttenuation); % ストップバンド減衰量
        end


        function filteredData = applyFilter(~, data, num, den)
            % データにIIRフィルタを適用する関数
            % 各チャンネルごとに filtfilt 関数を用いてゼロ位相フィルタリングを行う。
            %
            % 入力:
            %   data: フィルタ処理を行うデータ (チャンネル x 時間)
            %   num: フィルタの分子係数
            %   den: フィルタの分母係数
            %
            % 出力:
            %   filteredData: フィルタ処理後のデータ (チャンネル x 時間)

            filteredData = zeros(size(data)); % 出力データ配列を初期化

            % 各チャンネルにフィルタを適用
            for ch = 1:size(data, 1) % チャンネルごとにループ処理
                filteredData(ch, :) = filtfilt(num, den, data(ch, :)); % filtfilt関数でゼロ位相フィルタリングを適用
            end
        end
    end
end