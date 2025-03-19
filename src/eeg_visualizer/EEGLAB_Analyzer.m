function EEG = EEGLAB_Analyzer(processedData, labels, params)
    % EEGLAB_Analyzer - GUIを使わずにEEGLABの解析を行う関数
    %
    % 入力:
    %   processedData - 3次元行列 [チャンネル数 x サンプリングポイント数 x 試行回数]
    %   labels - 各試行のラベル情報（試行回数と同じ長さ）
    %   params - 解析パラメータ構造体
    %
    % params構造体に含めるべきフィールド:
    %   .srate - サンプリングレート（Hz）
    %   .times - 時間軸ベクトル（ms）
    %   .chanlocs - チャンネル位置情報
    %   .baseline - ベースライン期間 [開始時間(ms) 終了時間(ms)]
    %   .analysis - 実行する解析のセル配列 {'erp', 'ersp', 'topo'}
    %   .conditions - 条件名のセル配列
    %   .topoTimes - トポプロット表示する時間点（ms）
    %   .freqRange - 周波数範囲 [最小周波数 最大周波数]（Hz）
    %   .outputDir - 出力ディレクトリ
    
    if ~isfield(params, 'analysis') || isempty(params.analysis)
        % デフォルトですべての解析を実行
        error('解析タイプ（params.analysis）が指定されていません');
    end

    % EEGLABが初期化されているか確認し、初期化されていなければスタートする
    if ~exist('ALLEEG', 'var')
        % [ALLEEG, EEG, CURRENTSET] = eeglab("nogui");
        eeglab("nogui");
        close(gcf); % GUIは閉じる
    end
    
    % パラメータのバリデーション
    if ~isfield(params.device, 'sampleRate') || isempty(params.device.sampleRate)
        error('サンプリングレート（params.device.sampleRate）が指定されていません');
    end
    
    if ~isfield(params.signal.window, 'timeRange') || isempty(params.signal.window.timeRange)
        % 時間軸が指定されていない場合は、サンプリングポイントから計算
        disp("params.signal.window.timeRangeがありません");
        params.signal.window.timeRange = linspace(0, size(processedData, 2) / params.device.sampleRate * 1000, size(processedData, 2));
    end
    
    if ~isfield(params, 'chanlocs') || isempty(params.chanlocs)
        warning('チャンネル位置情報（params.chanlocs）が指定されていません。トポプロットは実行できません。');
    end
    
    
    
    if ~isfield(params, 'conditions') || isempty(conditions)
        % ラベルのユニークな値を条件として設定
        
    end
    
    if ~isfield(params.analysis, 'outputDir') || isempty(params.analysis.outputDir)
        params.analysis.outputDir = './eeglab_results';
    end
    
    % 出力ディレクトリの作成
    if ~exist(params.analysis.outputDir, 'dir')
        mkdir(params.analysis.outputDir);
    end
   
    % valueフィールドから条件を抽出
    allValues = arrayfun(@(x) x.value, labels, 'UniformOutput', false);

    % valueの型を確認して適切に処理
    if iscell(allValues)
        % 数値型などの場合は、セル配列を数値配列に変換
        if all(cellfun(@isnumeric, allValues))
            numValues = cell2mat(allValues);
            conditions = unique(numValues);
            isNumeric = true;
            disp("数値型の条件を抽出しました");
            disp(conditions);
        else
            % 混合型の場合はエラーを回避するために文字列に変換
            allValues = cellfun(@num2str, allValues, 'UniformOutput', false);
            conditions = unique(allValues);
            isNumeric = false;
            disp("混合型の条件を抽出しました");
        end
    else
        % 文字列セル配列の場合は直接unique
        conditions = unique(allValues);
        isNumeric = false;
        disp("文字列型の条件を抽出しました");
    end

    % 条件ごとのデータ分割
    conditionData = cell(length(conditions), 1);
    for i = 1:length(conditions)
        % 条件に対応する試行を抽出
        if isNumeric
            % 数値の場合は数値比較
            if iscell(allValues)
                numValues = cell2mat(allValues);
                trials = find(numValues == conditions(i));
            else
                trials = find([labels.value] == conditions(i));
            end
        else
            % 文字列の場合はstrcmp
            trials = find(strcmp(allValues, conditions(i)));
        end
        
        if isempty(trials)
            warning(['条件 "' num2str(conditions(i)) '" に対応する試行がありません']);
            continue;
        end
        conditionData{i} = processedData(:, :, trials);
    end
    
    % EEGLABのデータ構造を作成
    EEG = eeg_emptyset;
    EEG.setname = 'EEGLAB_Analyzer';
    EEG.srate = params.device.sampleRate;
    EEG.times = params.signal.window.timeRange;
    EEG.nbchan = size(processedData, 1);
    EEG.pnts = size(processedData, 2);
    EEG.trials = size(processedData, 3);
    EEG.data = processedData;
    EEG.xmin   = params.signal.window.timeRange{1}(1);      % 開始時間 (秒) - これは仮の値です。必要に応じて調整してください。
    EEG.xmax   = params.signal.window.timeRange{1}(2); % 終了時間 (秒) を計算

        % 5. イベント情報の追加
    %    labels構造体からイベント情報をEEG構造体に追加
    for i = 1:length(labels)
        EEG.event(i).type   = num2str(labels(i).value);   % イベントタイプ
        EEG.event(i).latency = labels(i).sample;  % レイテンシー (サンプル単位)
        EEG.event(i).urevent = i;                  % イベントのID
    end
    
    % チャンネル位置情報があれば追加
    if isfield(params, 'chanlocs') && ~isempty(params.chanlocs)
        EEG.chanlocs = params.chanlocs;
    end

    % 4. チャンネルロケーションのロード
    % epocx_14ch.ced から chanlocs を読み込む
    try
        % ファイルの存在確認
        ced_file = 'epocx_14ch.ced'; % またはフルパス
        if ~exist(ced_file, 'file')
            error('ファイル "%s" が見つかりません。', ced_file);
        end
    
        % readlocs を使用してチャンネルロケーションをロード
        EEG.chanlocs = readlocs(ced_file, 'filetype', 'autodetect');
        fprintf('チャンネルロケーションを %s からロードしました。\n', ced_file);
    
    catch ME
        % ロードに失敗した場合のエラー処理
        fprintf('チャンネルロケーション %s のロードに失敗しました。以下のエラーが発生しました:\n', ced_file);
        disp(ME.message);
    
        % エラー処理: ダミーのチャンネルロケーションを設定
        % チャンネル数に合わせて、適切な処理を行う必要があります
        for i = 1:EEG.nbchan
            EEG.chanlocs(i).labels = ['Ch' num2str(i)]; % チャンネル名をCh1, Ch2... と設定
    
            % ダミーの位置情報を設定 (全て (0, 0, 0) に設定)
            EEG.chanlocs(i).X = 0;
            EEG.chanlocs(i).Y = 0;
            EEG.chanlocs(i).Z = 0;
            EEG.chanlocs(i).theta = NaN; % または適切なデフォルト値
            EEG.chanlocs(i).phi = NaN; % または適切なデフォルト値
        end
    
        EEG = pop_chanedit(EEG, 'rename',  {1:EEG.nbchan,{EEG.chanlocs.labels}} );
        fprintf('ダミーのチャンネルロケーションを設定しました。\n');
    end
    
    if(params.analysis.ersp.enable)
        % analyzeERSP(EEG,conditionData, params,conditions);
    end
    if(params.analysis.erp.enable)
        % analyzeERP(conditionData, params,conditions);
    end
    if(params.analysis.topography.enable)
        analyzeTopoplot(EEG,conditionData, params,conditions);
    end
    disp('EEGLAB解析が完了しました');
end

function analyzeERP(conditionData, params,conditions)
    % ERP（事象関連電位）解析を実行
    % 
    % 入力:
    %   conditionData - 条件ごとのデータセル配列
    %   params - 解析パラメータ構造体
    
    disp('ERP解析を実行中...');
    
    % 出力ディレクトリの設定
    outDir = fullfile(params.analysis.outputDir, 'ERP');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    % 条件ごとの平均ERPを計算
    for i = 1:length(conditionData)
        if isempty(conditionData{i})
            continue;
        end
        
        % 平均ERP計算（チャンネル×時間）
        meanERP = mean(conditionData{i}, 3);
        
        % ベースライン補正（指定されていれば）
        % if isfield(params, 'baseline') && ~isempty(params.baseline)
        %     baselineIdx = find(params.signal.window.timeRange >= params.baseline(1) & params.signal.window.timeRange <= params.baseline(2));
        %     if ~isempty(baselineIdx)
        %         baseline = mean(meanERP(:, baselineIdx), 2);
        %         meanERP = meanERP - repmat(baseline, 1, size(meanERP, 2));
        %     end
        % end
        
        % 各チャンネルのERPをプロット
        if isfield(params, 'plotChannels') && ~isempty(params.plotChannels)
            plotChannels = params.plotChannels;
        else
            % デフォルトで全チャンネルを使用
            plotChannels = 1:size(meanERP, 1);
        end
        
        % チャンネルグループごとにプロット
        channelGroups = min(10, length(plotChannels)); % 一度に表示するチャンネル数を制限
        for g = 1:ceil(length(plotChannels) / channelGroups)
            startCh = (g-1) * channelGroups + 1;
            endCh = min(g * channelGroups, length(plotChannels));
            currentChannels = plotChannels(startCh:endCh);
            
            figure('Name', ['ERP - 条件: ' num2str(conditions(i)) ' - チャンネルグループ ' num2str(g)]);
            
            for ch = 1:length(currentChannels)
                subplot(ceil(length(currentChannels)/2), 2, ch);
                timeVals = linspace(params.signal.window.timeRange{1}(1), ...
                                    params.signal.window.timeRange{1}(2), ...
                                    size(meanERP, 2));
                plot(timeVals, meanERP(currentChannels(ch), :));
                title(['チャンネル ' num2str(currentChannels(ch))]);
                xlabel('時間 (ms)');
                ylabel('振幅 (μV)');
                grid on;
                
                % ゼロ時点にラインを追加
                hold on;
                plot([0 0], ylim, 'k--');
                hold off;
            end
            
            % 図の保存
            saveas(gcf, fullfile(outDir, ['ERP_' num2str(conditions(i)) '_Group' num2str(g) '.png']));
            saveas(gcf, fullfile(outDir, ['ERP_' num2str(conditions(i)) '_Group' num2str(g) '.fig']));
            close(gcf);
        end
        
        % すべてのチャンネルのERPデータを保存
        save(fullfile(outDir, ['ERP_' num2str(conditions(i)) '.mat']), 'meanERP', 'params');
    end

  
    
    % 条件間比較（2つ以上の条件がある場合）
    if length(conditionData) >= 2
        figure('Name', 'ERP条件比較');
        
        % 代表的なチャンネルを選択（パラメータで指定されていればそれを使用）
        if isfield(params, 'compareChannels') && ~isempty(params.compareChannels)
            compareChannels = params.compareChannels;
        else
            % デフォルトで最初の数チャンネルを使用
            compareChannels = 1:min(4, size(conditionData{1}, 1));
        end
        
        % 生成する時間軸を統一
        timeVals = linspace(params.signal.window.timeRange{1}(1), ...
                            params.signal.window.timeRange{1}(2), ...
                            size(meanERP, 2));
        
        % 各チャンネルごとに条件比較
        for ch = 1:length(compareChannels)
            subplot(ceil(length(compareChannels)/2), 2, ch);
            
            hold on;
            for i = 1:length(conditionData)
                if ~isempty(conditionData{i})
                    meanERP = mean(conditionData{i}, 3);
                    
                    % % ベースライン補正
                    % if isfield(params, 'baseline') && ~isempty(params.baseline)
                    %     baselineIdx = find(params.signal.window.timeRange >= params.baseline(1) & params.signal.window.timeRange <= params.baseline(2));
                    %     if ~isempty(baselineIdx)
                    %         baseline = mean(meanERP(:, baselineIdx), 2);
                    %         meanERP = meanERP - repmat(baseline, 1, size(meanERP, 2));
                    %     end
                    % end
                    
                    plot(timeVals, meanERP(compareChannels(ch), :), 'DisplayName', num2str(conditions(i)));
                end
            end
            hold off;
            
            title(['チャンネル ' num2str(compareChannels(ch))]);
            xlabel('時間 (ms)');
            ylabel('振幅 (μV)');
            legend('show');
            grid on;
            
            % ゼロ時点にラインを追加
            hold on;
            plot([0 0], ylim, 'k--');
            hold off;
        end
        
        % 図の保存
        saveas(gcf, fullfile(outDir, 'ERP_Comparison.png'));
        saveas(gcf, fullfile(outDir, 'ERP_Comparison.fig'));
        close(gcf);
    end
    
    disp('ERP解析が完了しました');
end

function analyzeERSP(EEG, conditionData, params,conditions)
    % ERSP（事象関連スペクトルパワー）解析を実行
    % 
    % 入力:
    %   conditionData - 条件ごとのデータセル配列
    %   params - 解析パラメータ構造体
    
    disp('ERSP解析を実行中...');
    
    % 出力ディレクトリの設定
    outDir = fullfile(params.analysis.outputDir, 'ERSP');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    % ERSPのパラメータ設定
    if isfield(params.analysis, 'ersp') && ~isempty(params.analysis.ersp)
        erspParams = params.analysis.ersp;
        disp("ERSPパラメータが指定されました");
    else
        error('ERSPパラメータが指定されていません');
    end
    
    % ベースライン期間の設定
    if isfield(params, 'baseline') && ~isempty(params.baseline)
        erspParams.baseline = params.baseline;
    end
    


    % 各条件のERSPを計算
    for i = 1:length(conditionData)
        if isempty(conditionData{i})
            continue;
        end
        
        % 解析するチャンネルの設定
        if isfield(erspParams, 'erspChannels') && ~isempty(erspParams.erspChannels)
            erspChannels = erspParams.erspChannels;
        else
            % デフォルトで全チャンネルを使用
            erspChannels = 1:size(conditionData{i}, 1);
        end
        

        % チャンネルごとにERSPを計算
        for ch = 1:length(erspChannels)
            channel = erspChannels(ch);
            
            % 全試行のデータを取得
            trialData = squeeze(conditionData{i}(channel, :, :));
            disp(size(trialData));
            figure('Name', ['ERSP - 条件: ' num2str(conditions(i)) ' - チャンネル ' num2str(channel)]);
            % newtimefを使用してERSPを計算
            [ersp, itc, powbase, times, freqs] = newtimef(trialData, ...
                EEG.pnts, ...                           % データ長
                [EEG.xmin EEG.xmax], ...          % 時間範囲
                EEG.srate, ...                                 % サンプリングレート
                erspParams.cycles, ...                            % 周波数ビンに対する小波サイクル数
                'freqs', erspParams.freqs, ...                    % 周波数範囲
                'timesout', erspParams.timesout, ...              % 時間ビンの数
                'plotersp', 'on', ...                            % ERSPをプロットしない
                'plotitc', 'on', ...                             % ITCをプロットしない
                'verbose', 'on');                                % 冗長な出力を抑制
            
            
            disp(num2str(conditions(i)));
            % 図の保存
            saveas(gcf, fullfile(outDir, ['ERSP_' num2str(conditions(i)) '_Ch' num2str(channel) '.png']));
            saveas(gcf, fullfile(outDir, ['ERSP_' num2str(conditions(i)) '_Ch' num2str(channel) '.fig']));
            close(gcf);
            
            % ERSPデータを保存
            erspData = struct(...
                'ersp', ersp, ...
                'itc', itc, ...
                'powbase', powbase, ...
                'times', times, ...
                'freqs', freqs, ...
                'channel', channel, ...
                'condition', conditions(i), ...
                'params', params ...
            );
            save(fullfile(outDir, ['ERSP_' num2str(conditions(i)) '_Ch' num2str(channel) '.mat']), 'erspData');
        end
    end
    
    disp('ERSP解析が完了しました');
end

function analyzeTopoplot(EEG,conditionData, params,conditions)
    % トポプロット解析を実行
    % 
    % 入力:
    %   conditionData - 条件ごとのデータセル配列
    %   params - 解析パラメータ構造体
    
    disp('トポプロット解析を実行中...');
    
    % チャンネル位置情報がない場合は終了
    if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
        warning('チャンネル位置情報（EEG.chanlocs）が指定されていません。トポプロットは実行できません。');
        return;
    end
    
    % 出力ディレクトリの設定
    outDir = fullfile(params.analysis.outputDir, 'Topoplot');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    % トポプロットする時間点の設定
    if isfield(params.analysis.topography, 'topoTimes') && ~isempty(params.analysis.topography.topoTimes)
        topoTimes = params.analysis.topography.topoTimes;
    else
        % デフォルトで5つの時間点
        topoTimes = linspace(params.signal.window.timeRange{1}(1), params.signal.window.timeRange{1}(end), 5);
    end
    
    % 各条件のトポプロットを作成
    for i = 1:length(conditionData)
        if isempty(conditionData{i})
            continue;
        end
        
        % 平均ERP計算（チャンネル×時間）
        meanERP = mean(conditionData{i}, 3);
        
        % ベースライン補正（指定されていれば）
        % if isfield(params.analysis.topography, 'baseline') && ~isempty(params.analysis.topography.baseline)
        %     baselineIdx = find(params.signal.window.timeRange{1} >= params.analysis.topography.baseline(1) & params.signal.window.timeRange{1} <= params.analysis.topography.baseline(2));
        %     if ~isempty(baselineIdx)
        %         baseline = mean(meanERP(:, baselineIdx), 2);
        %         meanERP = meanERP - repmat(baseline, 1, size(meanERP, 2));
        %     end
        % end
        
        % 各時間点でのトポプロット
        figure('Name', ['トポプロット - 条件: ' num2str(conditions(i))]);
        
        for t = 1:length(topoTimes)
            % topoTimes は秒で与えられているため、ミリ秒に変換
            
            % 最も近い時間点を見つける
            timeIdx = round(abs(params.signal.window.timeRange{1}(1) - topoTimes(t))  * params.device.sampleRate);

            disp(timeIdx);
            % トポプロット
            subplot(1, length(topoTimes), t);
            topoplot(meanERP(:, timeIdx), EEG.chanlocs, 'electrodes', 'on');
            title([num2str(topoTimes(t)) ' s']);
            
            % 最後のサブプロットに色のバーを追加
            colorbar;
        end
        
        % 図の保存
        saveas(gcf, fullfile(outDir, ['Topoplot_' num2str(conditions(i)) '.png']));
        saveas(gcf, fullfile(outDir, ['Topoplot_' num2str(conditions(i)) '.fig']));
        close(gcf);
        
        % 全時間帯のトポマップを生成（より密な時間間隔で）
        if isfield(params, 'topoTimeDense') && params.topoTimeDense
            % 密な時間間隔設定
            denseTopoTimes = linspace(params.signal.window.timeRange{1}(1), params.signal.window.timeRange{1}(end), min(20, length(params.signal.window.timeRange)));
            
            figure('Name', ['トポプロット（密） - 条件: ' num2str(conditions(i))]);
            
            for t = 1:length(denseTopoTimes)
                [~, timeIdx] = min(abs(params.signal.window.timeRange{1} - denseTopoTimes(t)));
                
                subplot(4, 5, t);
                topoplot(meanERP(:, timeIdx), EEG.chanlocs, 'electrodes', 'off');
                title([num2str(params.signal.window.timeRange{1}(timeIdx)) ' ms']);
            end
            
            colorbar;
            
            % 図の保存
            saveas(gcf, fullfile(outDir, ['TopoplotDense_' num2str(conditions(i)) '.png']));
            saveas(gcf, fullfile(outDir, ['TopoplotDense_' num2str(conditions(i)) '.fig']));
            close(gcf);
        end
        
        % トポプロットデータを保存
        topoData = struct(...
            'meanERP', meanERP, ...
            'times', params.signal.window.timeRange{1}, ...
            'topoTimes', topoTimes, ...
            'condition', conditions(i), ...
            'params', params ...
        );
        save(fullfile(outDir, ['Topoplot_' num2str(conditions(i)) '.mat']), 'topoData');
    end
    
    disp('トポプロット解析が完了しました');
end