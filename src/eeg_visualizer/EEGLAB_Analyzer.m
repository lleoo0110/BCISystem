function EEG = EEGLAB_Analyzer(processedData, labels, params)
    % EEGLAB_Analyzer - GUIを使わずにEEGLABの解析を行う関数
    disp('--- EEGLAB_Analyzer 解析開始 ---');
    
    % 入力引数とパラメータのバリデーション
    if ~isfield(params, 'analysis') || isempty(params.analysis)
        error('解析タイプ（params.analysis）が指定されていません');
    end

    % EEGLABが初期化されているか確認し、初期化されていなければスタートする
    if ~exist('ALLEEG', 'var')
        eeglab("nogui");
        close(gcf); % GUIは閉じる
    end
    
    if ~isfield(params.device, 'sampleRate') || isempty(params.device.sampleRate)
        error('サンプリングレート（params.device.sampleRate）が指定されていません');
    end
    
    if ~isfield(params.signal.window, 'timeRange') || isempty(params.signal.window.timeRange)
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

    if iscell(allValues)
        if all(cellfun(@isnumeric, allValues))
            numValues = cell2mat(allValues);
            conditions = unique(numValues);
            isNumeric = true;
            disp("数値型の条件を抽出しました");
            disp(conditions);
        else
            allValues = cellfun(@num2str, allValues, 'UniformOutput', false);
            conditions = unique(allValues);
            isNumeric = false;
            disp("混合型の条件を抽出しました");
        end
    else
        conditions = unique(allValues);
        isNumeric = false;
        disp("文字列型の条件を抽出しました");
    end

    % 条件ごとのデータ分割
    conditionData = cell(length(conditions), 1);
    for i = 1:length(conditions)
        if isNumeric
            if iscell(allValues)
                numValues = cell2mat(allValues);
                trials = find(numValues == conditions(i));
            else
                trials = find([labels.value] == conditions(i));
            end
        else
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
    EEG.times = params.signal.window.timeRange{1}(1):1/EEG.srate:params.signal.window.timeRange{1}(2) - 1/EEG.srate;
    disp(size(EEG.times));
    EEG.nbchan = size(processedData, 1);
    EEG.pnts = size(processedData, 2);
    EEG.trials = size(processedData, 3);
    EEG.data = processedData;
    EEG.xmin   = params.signal.window.timeRange{1}(1);      % 開始時間 (秒)
    EEG.xmax   = params.signal.window.timeRange{1}(2) - 1/EEG.srate; % 終了時間 (秒)
    
    % 5. イベント情報の追加
    for i = 1:length(labels)
        EEG.event(i).type   = num2str(labels(i).value);
        EEG.event(i).latency = labels(i).sample;
        EEG.event(i).urevent = i;
        EEG.event(i).epoch = i;
    end

    for i = 1:EEG.trials
        EEG.epoch(i).event = { i }; 
        EEG.epoch(i).eventtype = EEG.event(i).type;
        EEG.epoch(i).eventlatency = { (-EEG.xmin)*1000 };
        EEG.epoch(i).eventurevent = EEG.event(i).urevent;
    end
    
    if isfield(params, 'chanlocs') && ~isempty(params.chanlocs)
        EEG.chanlocs = params.chanlocs;
    end

    % チャンネルロケーションのロード
    try
        if params.device.name == "EPOCX"
            ced_file = 'chanlocs/epocx_14ch.ced';
        elseif params.device.name == "EPOCFLEX"
            ced_file = 'chanlocs/emotiv_flex32ch.ced';
        end
        if ~exist(ced_file, 'file')
            error('ファイル "%s" が見つかりません。', ced_file);
        end
    
        EEG.chanlocs = readlocs(ced_file, 'filetype', 'autodetect');
        fprintf('チャンネルロケーションを %s からロードしました。\n', ced_file);
    
    catch ME
        fprintf('チャンネルロケーション %s のロードに失敗しました:\n', ced_file);
        disp(ME.message);
    
        for i = 1:EEG.nbchan
            EEG.chanlocs(i).labels = ['Ch' num2str(i)];
            EEG.chanlocs(i).X = 0;
            EEG.chanlocs(i).Y = 0;
            EEG.chanlocs(i).Z = 0;
            EEG.chanlocs(i).theta = NaN;
            EEG.chanlocs(i).phi = NaN;
        end
    
        EEG = pop_chanedit(EEG, 'rename',  {1:EEG.nbchan,{EEG.chanlocs.labels}} );
        fprintf('ダミーのチャンネルロケーションを設定しました。\n');
    end

    if ~isfield(EEG, 'urchan') || isempty(EEG.urchan)
        EEG.urchan = 1:EEG.nbchan;
    end

    % EEGデータをsetファイルとして保存
    parentDir = fullfile(params.analysis.outputDir, params.info.name);
    if ~exist(parentDir, 'dir')
        mkdir(parentDir);
        disp(['親ディレクトリを作成しました: ' parentDir]);
    end
    outputFileName = fullfile(parentDir, 'EEG_Analyzer_Output.set');
    pop_saveset(EEG, 'filename', outputFileName);
    disp(['EEGデータを ' outputFileName ' に保存しました']);
    
    % 各解析関数の呼び出し（進行状況ログ付き）
    if(params.analysis.ersp.enable)
        analyzeERSP(EEG,conditionData, params,conditions);
    end
    if(params.analysis.erp.enable)
        analyzeERP(EEG,conditionData, params,conditions);
    end
    if(params.analysis.topography.enable)
        analyzeTopoplot(EEG,conditionData, params,conditions);
    end
    disp('--- EEGLAB解析が完了しました ---');
end

function analyzeERP(EEG, conditionData, params, conditions)
    disp('--- ERP解析を実行開始 ---');
    
    parentDir = fullfile(params.analysis.outputDir, params.info.name);
    if ~exist(parentDir, 'dir')
        mkdir(parentDir);
        disp(['親ディレクトリを作成しました: ' parentDir]);
    end
    outDir = fullfile(parentDir, 'ERP');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    meanERPs = cell(length(conditionData), 1);
    for i = 1:length(conditionData)
        if isempty(conditionData{i})
            meanERPs{i} = [];
            continue;
        end

        meanERPs{i} = mean(conditionData{i}, 3);
    end

    timeVals = linspace(params.signal.window.timeRange{1}(1), ...
                        params.signal.window.timeRange{1}(2), ...
                        size(meanERPs{1}, 2));

    for i = 1:length(conditionData)
        if isempty(meanERPs{i})
            continue;
        end

        figure('Name', ['ERP - 条件: ' num2str(conditions(i)) ' - トポグラフィカル表示'], 'Position', [200, 200, 600, 600]);

        if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
            error('チャンネル位置情報（EEG.chanlocs）が見つかりません。');
        end

        y = [EEG.chanlocs.X];
        x = [EEG.chanlocs.Y];

        if isempty(x) || isempty(y)
            error('チャンネル位置情報が不完全です。');
        end

        maxVal = max(max(abs(x)), max(abs(y)));
        x = x / maxVal * 0.4;
        y = y / maxVal * 0.4;
        subplotSize = 0.1;

        for ch = 1:length(EEG.chanlocs)
            posX = x(ch);
            posY = y(ch);

            plotLeft = posX - subplotSize/2;
            plotBottom = posY - subplotSize/2;
            plotWidth = subplotSize;
            plotHeight = subplotSize;

            ax = axes('Position', [plotLeft+0.5, plotBottom+0.5, plotWidth, plotHeight]);
            hPlot = plot(timeVals, meanERPs{i}(ch, :), 'LineWidth', 1);
            set(hPlot, 'HitTest','off');
            set(gca, 'XTick', [], 'YTick', []);
            set(gca, 'XColor', 'none', 'YColor', 'none');
            hold on;
            plot([0 0], ylim, 'k');
            plot(xlim, [0 0], 'k');

            if isfield(EEG.chanlocs, 'labels') && ~isempty(EEG.chanlocs(ch).labels)
                channelName = EEG.chanlocs(ch).labels;
            else
                channelName = num2str(ch);
            end
            text(min(timeVals), max(ylim), channelName, 'FontSize', 8, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
            xlim([min(timeVals), max(timeVals)]);
            box off;
            set(gca, 'Color', [0.95, 0.95, 0.95]);

            set(ax, 'ButtonDownFcn', @(src, event) openLargeERPPlot(timeVals, meanERPs{i}(ch, :), channelName,conditions(i)));
        end

        axes('Position', [0, 0, 1, 1]);
        axis off;
        axis equal;
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        title(['ERP - 条件: ' num2str(conditions(i))], 'FontSize', 16);

        saveas(gcf, fullfile(outDir, ['ERP_Topographic_' num2str(conditions(i)) '.png']));
        saveas(gcf, fullfile(outDir, ['ERP_Topographic_' num2str(conditions(i)) '.fig']));
        print(gcf, fullfile(outDir, ['ERP_Topographic_' num2str(conditions(i)) '_HighRes.png']), '-dpng', '-r300');
        close(gcf);
    end

    disp('--- ERP解析が完了しました ---');

    function openLargeERPPlot(timeVals, erpData, channelName, condition)
        figure('Name', ['Large ERP Plot - 条件' num2str(condition) ' - channel - ' channelName ], 'Position', [300, 300, 800, 400]);
        plot(timeVals, erpData, 'LineWidth', 2);
        xlabel('Time (ms)');
        ylabel('Amplitude');
        title(['ERP - ' channelName]);
        grid on;
    end
end

function analyzeERSP(EEG, conditionData, params, conditions)
    disp('--- ERSP解析を実行開始 ---');
    
    parentDir = fullfile(params.analysis.outputDir, params.info.name);
    if ~exist(parentDir, 'dir')
        mkdir(parentDir);
        disp(['親ディレクトリを作成しました: ' parentDir]);
    end
    outDir = fullfile(parentDir, 'ERSP');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
        disp(['ERSP出力ディレクトリを作成しました: ' outDir]);
    end
    
    if isfield(params.analysis, 'ersp') && ~isempty(params.analysis.ersp)
        erspParams = params.analysis.ersp;
        disp("ERSPパラメータが指定されました。");
    else
        error('ERSPパラメータが指定されていません');
    end
    
    if isfield(params, 'baseline') && ~isempty(params.baseline)
        erspParams.baseline = params.baseline;
        disp("ベースライン期間が設定されました。");
    end
    
    for i = 1:length(conditionData)
        if isempty(conditionData{i})
            disp(['条件 ' num2str(conditions(i)) ' のデータが空です。スキップします。']);
            continue;
        end
        
        disp(['条件 ' num2str(conditions(i)) ' のERSP解析を開始します...']);
        
        if isfield(erspParams, 'erspChannels') && ~isempty(erspParams.erspChannels)
            erspChannels = erspParams.erspChannels;
            disp('指定されたチャンネルで解析を実施します。');
        else
            erspChannels = 1:size(conditionData{i}, 1);
            disp('全チャンネルで解析を実施します。');
        end
        
        erspResults = cell(length(erspChannels), 1);
        channelScales = cell(length(erspChannels), 1);
        
        trialData = squeeze(conditionData{i}(:, :, :));
        EEG.data = trialData;
        disp('全試行のデータを統合しました。');
        
        for ch = 1:length(erspChannels)
            channel = erspChannels(ch);
            disp(['条件 ' num2str(conditions(i)) ' - チャンネル ' num2str(channel) ' のERSP解析を開始します...']);
            
            [ersp, itc, ~, times, ~, ~, ~] = pop_newtimef(EEG, 1, channel, ...
                [EEG.xmin EEG.xmax]* 1000, ... 
                erspParams.cycles, ... 
                'freqs', erspParams.freqs, ... 
                'plotersp', 'off', ... 
                'plotitc', 'off', ... 
                'plotphase', 'off', ... 
                'padratio', 4, ...
                'verbose', 'off');
            
            maxVal = max(abs(ersp(:)));
            channelScales{ch} = [-maxVal maxVal];
            
            erspResults{ch} = struct('cycles', erspParams.cycles, 'timesout', erspParams.timesout, ...
                'srate', EEG.srate,'ersp', ersp, 'itc', itc, 'times', times, 'freqs', erspParams.freqs, ...
                'channel', channel, 'colorScale', channelScales{ch});
            disp(['条件 ' num2str(conditions(i)) ' - チャンネル ' num2str(channel) ' のERSP解析完了。']);
        end
        
        if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
            error('チャンネル位置情報（EEG.chanlocs）が見つかりません。');
        end
        yCoords = [EEG.chanlocs.X];
        xCoords = [EEG.chanlocs.Y];
        if isempty(xCoords) || isempty(yCoords)
            error('チャンネル位置情報が不完全です。');
        end
        maxValCoord = max(max(abs(xCoords)), max(abs(yCoords)));
        xNorm = xCoords / maxValCoord * 0.4;
        yNorm = yCoords / maxValCoord * 0.4;
        
        subplotSize = 0.1;
        
        disp(['条件 ' num2str(conditions(i)) ' のトポグラフィックERSP図を作成中...']);
        fig = figure('Name', ['ERSP - 条件: ' num2str(conditions(i))], 'Position', [200, 200, 800, 600]);
        fig.UserData.EEG = EEG;
        for idx = 1:length(erspChannels)
            channel = erspResults{idx}.channel;
            if channel <= length(EEG.chanlocs)
                channelName = EEG.chanlocs(channel).labels;
            else
                channelName = num2str(channel);
            end
            posX = xNorm(channel);
            posY = yNorm(channel);
            
            ax = axes('Position', [posX + 0.5 - subplotSize/2, posY + 0.5 - subplotSize/2, subplotSize, subplotSize]);
            imagesc(erspResults{idx}.times, erspResults{idx}.freqs, erspResults{idx}.ersp);
            axis xy;
            clim(channelScales{idx});
            set(gca, 'XTick', [], 'YTick', []);
            title(channelName, 'FontSize', 8);
            
            colormap(ax, "jet");
            flds = {'ersp','itc','powbase','times','trialData'};
            existingFlds = flds(isfield(erspResults{idx}, flds));
            if ~isempty(existingFlds)
                erspResults{idx} = rmfield(erspResults{idx}, existingFlds);
            else
                disp('ERSPフィールドが見つかりません');
            end
            set(ax, 'ButtonDownFcn', @(src, event) openLargeERSPPlot(erspResults{idx}, channelName));
        end
        
        axes('Position', [0, 0, 1, 1], 'Visible', 'off');
        title(['ERSP - 条件: ' num2str(conditions(i))], 'FontSize', 16);
        
        set(gcf, 'Renderer', 'painters');
        drawnow;
        
        saveas(gcf, fullfile(outDir, ['ERSP_Topographic_' num2str(conditions(i)) '.png']));
        savefig(gcf, fullfile(outDir, ['ERSP_Topographic_' num2str(conditions(i)) '.fig']));
        close(gcf);
        disp(['条件 ' num2str(conditions(i)) ' のERSP図の保存完了。']);
    end
    
    disp('--- ERSP解析が完了しました ---');
end

function openLargeERSPPlot(erspStruct, channelName)
    fig = gcf;
    figure('Name', ['Large ERSP Plot - ' channelName], 'Position', [300, 300, 800, 600]);
    title(['ERSP - ' channelName]);
    storedEEG = fig.UserData.EEG;
    [~, ~, ~, ~, ~, ~, ~] = pop_newtimef(storedEEG, 1, erspStruct.channel, ...
        [storedEEG.xmin storedEEG.xmax] * 1000, ...
        erspStruct.cycles, ...
        'freqs', erspStruct.freqs, ...
        'plotersp', 'on', ...
        'plotitc', 'on', ...
        'plotphase', 'off', ...
        'padratio', 4, ...
        'verbose', 'off');
end

function analyzeTopoplot(EEG, conditionData, params, conditions)
    disp('--- トポプロット解析を実行開始 ---');
    
    if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
        warning('チャンネル位置情報（EEG.chanlocs）が指定されていません。トポプロットは実行できません。');
        return;
    end
    
    parentDir = fullfile(params.analysis.outputDir, params.info.name);
    if ~exist(parentDir, 'dir')
        mkdir(parentDir);
        disp(['親ディレクトリを作成しました: ' parentDir]);
    end
    outDir = fullfile(parentDir, 'Topoplot');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    if isfield(params.analysis.topography, 'topoTimes') && ~isempty(params.analysis.topography.topoTimes)
        topoTimes = params.analysis.topography.topoTimes;
    else
        topoTimes = linspace(params.signal.window.timeRange{1}(1), params.signal.window.timeRange{1}(end), 5);
    end
    
    for i = 1:length(conditionData)
        if isempty(conditionData{i})
            continue;
        end
        
        meanERP = mean(conditionData{i}, 3);
        
        figure('Name', ['トポプロット - 条件: ' num2str(conditions(i))]);
        
        for t = 1:length(topoTimes)
            timeIdx = round(abs(params.signal.window.timeRange{1}(1) - topoTimes(t))  * params.device.sampleRate);
            disp(['条件 ' num2str(conditions(i)) ' - 時間インデックス: ' num2str(timeIdx)]);
            subplot(1, length(topoTimes), t);
            topoplot(meanERP(:, timeIdx), EEG.chanlocs, 'electrodes', 'on');
            title([num2str(topoTimes(t)) ' s']);
            colorbar;
        end
        
        saveas(gcf, fullfile(outDir, ['Topoplot_' num2str(conditions(i)) '.png']));
        saveas(gcf, fullfile(outDir, ['Topoplot_' num2str(conditions(i)) '.fig']));
        close(gcf);
        
        if isfield(params, 'topoTimeDense') && params.topoTimeDense
            denseTopoTimes = linspace(params.signal.window.timeRange{1}(1), params.signal.window.timeRange{1}(end), min(20, length(params.signal.window.timeRange)));
            
            figure('Name', ['トポプロット（密） - 条件: ' num2str(conditions(i))]);
            
            for t = 1:length(denseTopoTimes)
                [~, timeIdx] = min(abs(params.signal.window.timeRange{1} - denseTopoTimes(t)));
                subplot(4, 5, t);
                topoplot(meanERP(:, timeIdx), EEG.chanlocs, 'electrodes', 'off');
                title([num2str(params.signal.window.timeRange{1}(timeIdx)) ' ms']);
            end
            
            colorbar;
            
            saveas(gcf, fullfile(outDir, ['TopoplotDense_' num2str(conditions(i)) '.png']));
            saveas(gcf, fullfile(outDir, ['TopoplotDense_' num2str(conditions(i)) '.fig']));
            close(gcf);
        end
        
        topoData = struct(...
            'meanERP', meanERP, ...
            'times', params.signal.window.timeRange{1}, ...
            'topoTimes', topoTimes, ...
            'condition', conditions(i), ...
            'params', params ...
        );
        save(fullfile(outDir, ['Topoplot_' num2str(conditions(i)) '.mat']), 'topoData');
    end
    
    disp('--- トポプロット解析が完了しました ---');
end
