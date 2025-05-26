classdef EEGNormalizer < handle
    % EEGNormalizer - 脳波データの正規化に特化したクラス
    %
    % 生脳波データに対する正規化処理を行う専用クラス。
    % チャンネル単位での正規化により、チャンネル間のスケールを揃え、
    % アーティファクトの影響を軽減します。CSP処理の前段階で使用します。
    %
    % 使用例:
    %   params = getConfig('device_type');
    %   normalizer = EEGNormalizer(params);
    %   [normalizedEEG, normParams] = normalizer.normalize(eegData);
    %   % オンラインで使用する場合
    %   newNormalizedEEG = normalizer.normalizeOnline(newEEGData, normParams);
    
    properties (Access = private)
        params          % システムパラメータ
        normalizeParams % 正規化に使用するパラメータ
    end
    
    methods (Access = public)
        % コンストラクタ
        function obj = EEGNormalizer(params)
            obj.params = params;
            
            % 正規化パラメータの取得
            if isfield(params.classifier, 'normalize')
                % 信号処理の正規化設定を使用
                obj.normalizeParams = params.classifier.normalize;
            else
                % デフォルトのパラメータを設定
                obj.normalizeParams = struct('method', 'zscore', 'enable', true);
                warning('EEG正規化パラメータが見つかりません。デフォルト値を使用します。');
            end
            
            % methodフィールドが存在するか確認
            if ~isfield(obj.normalizeParams, 'method')
                obj.normalizeParams.method = 'zscore'; % デフォルト値
                warning('正規化パラメータに"method"フィールドがありません。デフォルト値"zscore"を設定します。');
            end
        end
        
        % 脳波データの正規化を実行
        function [normalizedEEG, normParams] = normalize(obj, eegData)
            % 脳波データの正規化処理を実行し、正規化パラメータを返す
            %
            % 入力:
            %   eegData - 脳波データ [チャンネル数 × サンプル数] または
            %             エポック化されたデータ [チャンネル数 × サンプル数 × エポック数]
            %
            % 出力:
            %   normalizedEEG - 正規化後の脳波データ
            %   normParams - 正規化に使用したパラメータ（オンライン処理用）
            
            % 入力チェック
            if isempty(eegData)
                normalizedEEG = [];
                normParams = struct('method', obj.normalizeParams.method);
                return;
            end
            
            % データ次元数に応じた処理
            if ndims(eegData) == 3
                % 3次元データ（チャンネル × サンプル × エポック）
                [normalizedEEG, normParams] = obj.normalize3D(eegData);
            elseif iscell(eegData)
                % セル配列
                [normalizedEEG, normParams] = obj.normalizeCell(eegData);
            else
                % 2次元データ（チャンネル × サンプル）
                [normalizedEEG, normParams] = obj.normalize2D(eegData);
            end
        end
        
        % オンライン処理用の正規化
        function normalizedEEG = normalizeOnline(obj, eegData, normParams)
            % 事前計算された正規化パラメータを使用して脳波データを正規化
            %
            % 入力:
            %   eegData - 新しい脳波データ
            %   normParams - normalize()で得られた正規化パラメータ
            %
            % 出力:
            %   normalizedEEG - 正規化後の脳波データ
            
            % 入力チェック
            if isempty(eegData)
                normalizedEEG = [];
                return;
            end
            
            % データ次元数に応じた処理
            if ndims(eegData) == 3
                % 3次元データ
                normalizedEEG = obj.normalizeOnline3D(eegData, normParams);
            elseif iscell(eegData)
                % セル配列
                normalizedEEG = obj.normalizeOnlineCell(eegData, normParams);
            else
                % 2次元データ
                normalizedEEG = obj.normalizeOnline2D(eegData, normParams);
            end
        end
    end
    
    methods (Access = private)
        % 2次元脳波データの正規化
        function [normalizedEEG, normParams] = normalize2D(obj, eegData)
            % 2次元脳波データ（チャンネル × サンプル）を正規化
            
            % 入力サイズの取得
            [channels, samples] = size(eegData);
            
            % データ形式の確認（チャンネル数がサンプル数より少ない）
            if channels > samples && channels > 100
                warning('データのサイズが通常の脳波形式と異なります。転置が必要かもしれません。');
            end
            
            % デバイスのチャンネル数と比較
            if isfield(obj.params, 'device') && isfield(obj.params.device, 'channelCount')
                if channels ~= obj.params.device.channelCount && samples >= obj.params.device.channelCount
                    % チャンネル数が不一致の場合、転置を提案
                    warning(['データのチャンネル数(%d)がデバイス設定のチャンネル数(%d)と一致しません。', ...
                        'データが転置されている可能性があります。'], channels, obj.params.device.channelCount);
                end
            end
            
            % 正規化パラメータの初期化
            normParams = struct('method', obj.normalizeParams.method);
            normParams.dataFormat = 'channels_x_samples';
            normParams.originalSize = [channels, samples];
            
            % 正規化処理の実行
            normalizedEEG = zeros(size(eegData));
            
            switch obj.normalizeParams.method
                case 'zscore'
                    % Z-score正規化（各チャンネルごとに平均0、標準偏差1に）
                    channelMeans = mean(eegData, 2);  % 各チャンネルの平均
                    channelStds = std(eegData, 0, 2); % 各チャンネルの標準偏差
                    
                    % ゼロ除算防止
                    channelStds(channelStds < eps) = 1;
                    
                    % 正規化を適用
                    for ch = 1:channels
                        normalizedEEG(ch, :) = (eegData(ch, :) - channelMeans(ch)) / channelStds(ch);
                    end
                    
                    % パラメータを保存
                    normParams.mean = channelMeans;
                    normParams.std = channelStds;
                    
                case 'minmax'
                    % MinMax正規化（0-1スケーリング）
                    channelMin = min(eegData, [], 2);  % 各チャンネルの最小値
                    channelMax = max(eegData, [], 2);  % 各チャンネルの最大値
                    channelRange = channelMax - channelMin;
                    
                    % ゼロ除算防止
                    channelRange(channelRange < eps) = 1;
                    
                    % 正規化を適用
                    for ch = 1:channels
                        normalizedEEG(ch, :) = (eegData(ch, :) - channelMin(ch)) / channelRange(ch);
                    end
                    
                    % パラメータを保存
                    normParams.min = channelMin;
                    normParams.max = channelMax;
                    
                case 'robust'
                    % Robust正規化（外れ値に強い）
                    channelMedian = median(eegData, 2);  % 各チャンネルの中央値
                    
                    % 絶対偏差の計算
                    madMatrix = zeros(size(eegData));
                    for ch = 1:channels
                        madMatrix(ch, :) = abs(eegData(ch, :) - channelMedian(ch));
                    end
                    
                    channelMAD = median(madMatrix, 2);  % 各チャンネルの絶対偏差中央値
                    
                    % ゼロ除算防止
                    channelMAD(channelMAD < eps) = 1;
                    
                    % 正規化を適用
                    for ch = 1:channels
                        normalizedEEG(ch, :) = (eegData(ch, :) - channelMedian(ch)) / channelMAD(ch);
                    end
                    
                    % パラメータを保存
                    normParams.median = channelMedian;
                    normParams.mad = channelMAD;
                    
                case 'none'
                    % 正規化なし - そのまま返す
                    normalizedEEG = eegData;
                    
                otherwise
                    error('未知の正規化方法: %s', obj.normalizeParams.method);
            end
            
            % 出力サイズの確認
            if any(isnan(normalizedEEG(:))) || any(isinf(normalizedEEG(:)))
                warning('正規化後のデータにNaNまたはInfが含まれています');
            end
        end
        
        % 3次元脳波データの正規化
        function [normalizedEEG, normParams] = normalize3D(obj, eegData)
            % 3次元脳波データ（チャンネル × サンプル × エポック）を正規化
            
            [channels, samples, epochs] = size(eegData);
            normalizedEEG = zeros(size(eegData));
            
            % 正規化パラメータの初期化
            normParams = struct('method', obj.normalizeParams.method);
            normParams.dataFormat = '3d_channels_x_samples_x_epochs';
            normParams.originalSize = [channels, samples, epochs];
            
            % 正規化方法の選択
            if isfield(obj.normalizeParams, 'epochMode') && ...
               strcmpi(obj.normalizeParams.epochMode, 'independent')
                % エポックごとに独立して正規化
                epochParams = cell(epochs, 1);
                
                for e = 1:epochs
                    [normalizedEEG(:,:,e), epochParams{e}] = obj.normalize2D(eegData(:,:,e));
                end
                
                normParams.epochMode = 'independent';
                normParams.epochParams = epochParams;
            else
                % すべてのエポックを一緒に正規化（チャンネルごとのパラメータを共有）
                
                % チャンネルごとの統計量を計算
                switch obj.normalizeParams.method
                    case 'zscore'
                        % チャンネルごとの平均と標準偏差
                        channelMeans = zeros(channels, 1);
                        channelStds = zeros(channels, 1);
                        
                        for ch = 1:channels
                            % すべてのエポックのサンプルを結合して計算
                            allSamples = reshape(eegData(ch,:,:), 1, []);
                            channelMeans(ch) = mean(allSamples);
                            channelStds(ch) = std(allSamples);
                            
                            % ゼロ除算防止
                            if channelStds(ch) < eps
                                channelStds(ch) = 1;
                            end
                            
                            % 正規化を適用
                            for e = 1:epochs
                                normalizedEEG(ch,:,e) = (eegData(ch,:,e) - channelMeans(ch)) / channelStds(ch);
                            end
                        end
                        
                        normParams.mean = channelMeans;
                        normParams.std = channelStds;
                        
                    case 'minmax'
                        % チャンネルごとの最小値と最大値
                        channelMins = zeros(channels, 1);
                        channelMaxs = zeros(channels, 1);
                        
                        for ch = 1:channels
                            allSamples = reshape(eegData(ch,:,:), 1, []);
                            channelMins(ch) = min(allSamples);
                            channelMaxs(ch) = max(allSamples);
                            channelRange = channelMaxs(ch) - channelMins(ch);
                            
                            % ゼロ除算防止
                            if channelRange < eps
                                channelRange = 1;
                            end
                            
                            % 正規化を適用
                            for e = 1:epochs
                                normalizedEEG(ch,:,e) = (eegData(ch,:,e) - channelMins(ch)) / channelRange;
                            end
                        end
                        
                        normParams.min = channelMins;
                        normParams.max = channelMaxs;
                        
                    case 'robust'
                        % チャンネルごとの中央値と絶対偏差中央値
                        channelMedians = zeros(channels, 1);
                        channelMADs = zeros(channels, 1);
                        
                        for ch = 1:channels
                            allSamples = reshape(eegData(ch,:,:), 1, []);
                            channelMedians(ch) = median(allSamples);
                            madValues = abs(allSamples - channelMedians(ch));
                            channelMADs(ch) = median(madValues);
                            
                            % ゼロ除算防止
                            if channelMADs(ch) < eps
                                channelMADs(ch) = 1;
                            end
                            
                            % 正規化を適用
                            for e = 1:epochs
                                normalizedEEG(ch,:,e) = (eegData(ch,:,e) - channelMedians(ch)) / channelMADs(ch);
                            end
                        end
                        
                        normParams.median = channelMedians;
                        normParams.mad = channelMADs;
                        
                    case 'none'
                        % 正規化なし
                        normalizedEEG = eegData;
                end
                
                normParams.epochMode = 'shared';
            end
            
            % 出力サイズの確認
            if any(isnan(normalizedEEG(:))) || any(isinf(normalizedEEG(:)))
                warning('正規化後のデータにNaNまたはInfが含まれています');
            end
        end
        
        % セル配列脳波データの正規化
        function [normalizedEEG, normParams] = normalizeCell(obj, eegData)
            % セル配列形式の脳波データを正規化
            
            numCells = length(eegData);
            normalizedEEG = cell(size(eegData));
            
            % 空のセルをスキップ
            validCellIdx = ~cellfun(@isempty, eegData);
            
            if ~any(validCellIdx)
                % すべてのセルが空の場合
                normParams = struct('method', obj.normalizeParams.method, 'isEmpty', true);
                return;
            end
            
            % 正規化方法の選択
            if isfield(obj.normalizeParams, 'cellMode') && ...
               strcmpi(obj.normalizeParams.cellMode, 'independent')
                % セルごとに独立して正規化
                cellParams = cell(numCells, 1);
                
                for i = 1:numCells
                    if validCellIdx(i)
                        [normalizedEEG{i}, cellParams{i}] = obj.normalize2D(eegData{i});
                    end
                end
                
                normParams = struct('method', obj.normalizeParams.method);
                normParams.cellMode = 'independent';
                normParams.cellParams = cellParams;
            else
                % すべてのセルを一緒に正規化（チャンネルごとのパラメータを共有）
                
                % 最初の有効なセルからチャンネル数を取得
                firstValidIdx = find(validCellIdx, 1);
                [channels, ~] = size(eegData{firstValidIdx});
                
                % チャンネルごとの統計量を計算
                switch obj.normalizeParams.method
                    case 'zscore'
                        % チャンネルごとの平均と標準偏差
                        channelMeans = zeros(channels, 1);
                        channelStds = zeros(channels, 1);
                        allSamples = cell(channels, 1);
                        
                        % すべてのセルからデータを収集
                        for i = 1:numCells
                            if validCellIdx(i)
                                cellData = eegData{i};
                                for ch = 1:channels
                                    allSamples{ch} = [allSamples{ch}, cellData(ch,:)];
                                end
                            end
                        end
                        
                        % 統計量を計算
                        for ch = 1:channels
                            channelMeans(ch) = mean(allSamples{ch});
                            channelStds(ch) = std(allSamples{ch});
                            
                            % ゼロ除算防止
                            if channelStds(ch) < eps
                                channelStds(ch) = 1;
                            end
                        end
                        
                        % 正規化を適用
                        for i = 1:numCells
                            if validCellIdx(i)
                                cellData = eegData{i};
                                normalizedCell = zeros(size(cellData));
                                
                                for ch = 1:channels
                                    normalizedCell(ch,:) = (cellData(ch,:) - channelMeans(ch)) / channelStds(ch);
                                end
                                
                                normalizedEEG{i} = normalizedCell;
                            end
                        end
                        
                        normParams = struct('method', obj.normalizeParams.method);
                        normParams.mean = channelMeans;
                        normParams.std = channelStds;
                        normParams.cellMode = 'shared';
                        
                    case 'minmax'
                        % チャンネルごとの最小値と最大値
                        % （同様に実装...）
                        channelMins = zeros(channels, 1);
                        channelMaxs = zeros(channels, 1);
                        allSamples = cell(channels, 1);
                        
                        % すべてのセルからデータを収集
                        for i = 1:numCells
                            if validCellIdx(i)
                                cellData = eegData{i};
                                for ch = 1:channels
                                    allSamples{ch} = [allSamples{ch}, cellData(ch,:)];
                                end
                            end
                        end
                        
                        % 統計量を計算
                        for ch = 1:channels
                            channelMins(ch) = min(allSamples{ch});
                            channelMaxs(ch) = max(allSamples{ch});
                        end
                        
                        % 正規化を適用
                        for i = 1:numCells
                            if validCellIdx(i)
                                cellData = eegData{i};
                                normalizedCell = zeros(size(cellData));
                                
                                for ch = 1:channels
                                    channelRange = channelMaxs(ch) - channelMins(ch);
                                    if channelRange < eps
                                        channelRange = 1;
                                    end
                                    normalizedCell(ch,:) = (cellData(ch,:) - channelMins(ch)) / channelRange;
                                end
                                
                                normalizedEEG{i} = normalizedCell;
                            end
                        end
                        
                        normParams = struct('method', obj.normalizeParams.method);
                        normParams.min = channelMins;
                        normParams.max = channelMaxs;
                        normParams.cellMode = 'shared';
                        
                    case 'robust'
                        % チャンネルごとの中央値と絶対偏差中央値
                        % （同様に実装...）
                        channelMedians = zeros(channels, 1);
                        channelMADs = zeros(channels, 1);
                        allSamples = cell(channels, 1);
                        
                        % すべてのセルからデータを収集
                        for i = 1:numCells
                            if validCellIdx(i)
                                cellData = eegData{i};
                                for ch = 1:channels
                                    allSamples{ch} = [allSamples{ch}, cellData(ch,:)];
                                end
                            end
                        end
                        
                        % 統計量を計算
                        for ch = 1:channels
                            channelMedians(ch) = median(allSamples{ch});
                            madValues = abs(allSamples{ch} - channelMedians(ch));
                            channelMADs(ch) = median(madValues);
                            
                            % ゼロ除算防止
                            if channelMADs(ch) < eps
                                channelMADs(ch) = 1;
                            end
                        end
                        
                        % 正規化を適用
                        for i = 1:numCells
                            if validCellIdx(i)
                                cellData = eegData{i};
                                normalizedCell = zeros(size(cellData));
                                
                                for ch = 1:channels
                                    normalizedCell(ch,:) = (cellData(ch,:) - channelMedians(ch)) / channelMADs(ch);
                                end
                                
                                normalizedEEG{i} = normalizedCell;
                            end
                        end
                        
                        normParams = struct('method', obj.normalizeParams.method);
                        normParams.median = channelMedians;
                        normParams.mad = channelMADs;
                        normParams.cellMode = 'shared';
                        
                    case 'none'
                        % 正規化なし
                        normalizedEEG = eegData;
                        normParams = struct('method', 'none');
                        normParams.cellMode = 'none';
                end
            end
        end
        
        % オンライン用2次元データの正規化
        function normalizedEEG = normalizeOnline2D(~, eegData, normParams)
            % 保存されたパラメータを使用して2次元脳波データを正規化
            
            [channels, ~] = size(eegData);
            normalizedEEG = zeros(size(eegData));
            
            switch normParams.method
                case 'zscore'
                    % Z-score正規化
                    for ch = 1:channels
                        normalizedEEG(ch,:) = (eegData(ch,:) - normParams.mean(ch)) / normParams.std(ch);
                    end
                    
                case 'minmax'
                    % MinMax正規化
                    for ch = 1:channels
                        normalizedEEG(ch,:) = (eegData(ch,:) - normParams.min(ch)) / (normParams.max(ch) - normParams.min(ch));
                    end
                    
                case 'robust'
                    % Robust正規化
                    for ch = 1:channels
                        normalizedEEG(ch,:) = (eegData(ch,:) - normParams.median(ch)) / normParams.mad(ch);
                    end
                    
                case 'none'
                    % 正規化なし
                    normalizedEEG = eegData;
            end
        end
        
        % オンライン用3次元データの正規化
        function normalizedEEG = normalizeOnline3D(obj, eegData, normParams)
            % 保存されたパラメータを使用して3次元脳波データを正規化
            
            [~, ~, epochs] = size(eegData);
            normalizedEEG = zeros(size(eegData));
            
            if isfield(normParams, 'epochMode') && strcmpi(normParams.epochMode, 'independent')
                % エポックごとに独立したパラメータを使用
                for e = 1:min(epochs, length(normParams.epochParams))
                    normalizedEEG(:,:,e) = obj.normalizeOnline2D(eegData(:,:,e), normParams.epochParams{e});
                end
                
                % 余分なエポックは最初のパラメータで処理
                if epochs > length(normParams.epochParams)
                    for e = length(normParams.epochParams)+1:epochs
                        normalizedEEG(:,:,e) = obj.normalizeOnline2D(eegData(:,:,e), normParams.epochParams{1});
                    end
                end
            else
                % すべてのエポックに同じパラメータを使用
                for e = 1:epochs
                    normalizedEEG(:,:,e) = obj.normalizeOnline2D(eegData(:,:,e), normParams);
                end
            end
        end
        
        % オンライン用セル配列データの正規化
        function normalizedEEG = normalizeOnlineCell(obj, eegData, normParams)
            % 保存されたパラメータを使用してセル配列脳波データを正規化
            
            numCells = length(eegData);
            normalizedEEG = cell(size(eegData));
            
            % 空のセルをスキップ
            validCellIdx = ~cellfun(@isempty, eegData);
            
            if ~any(validCellIdx)
                return;
            end
            
            if isfield(normParams, 'cellMode') && strcmpi(normParams.cellMode, 'independent')
                % セルごとに独立したパラメータを使用
                for i = 1:numCells
                    if validCellIdx(i) && i <= length(normParams.cellParams) && ~isempty(normParams.cellParams{i})
                        normalizedEEG{i} = obj.normalizeOnline2D(eegData{i}, normParams.cellParams{i});
                    elseif validCellIdx(i)
                        % 適切なパラメータがない場合は最初の有効なパラメータを使用
                        firstValidParam = find(~cellfun(@isempty, normParams.cellParams), 1);
                        if ~isempty(firstValidParam)
                            normalizedEEG{i} = obj.normalizeOnline2D(eegData{i}, normParams.cellParams{firstValidParam});
                        else
                            normalizedEEG{i} = eegData{i}; % パラメータがない場合はそのまま
                        end
                    end
                end
            else
                % すべてのセルに同じパラメータを使用
                for i = 1:numCells
                    if validCellIdx(i)
                        normalizedEEG{i} = obj.normalizeOnline2D(eegData{i}, normParams);
                    end
                end
            end
        end
    end
end