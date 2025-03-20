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
        analyzeERSP(EEG,conditionData, params,conditions);
    end
    if(params.analysis.erp.enable)
        % analyzeERP(EEG,conditionData, params,conditions);
    end
    if(params.analysis.topography.enable)
        % analyzeTopoplot(EEG,conditionData, params,conditions);
    end
    disp('EEGLAB解析が完了しました');
end

function analyzeERP(EEG, conditionData, params, conditions)
    % ERP（事象関連電位）解析を実行 - トポグラフィカル表示
    % 
    % 入力:
    %   EEG - EEGLABデータ構造体（チャンネル位置情報を含む）
    %   conditionData - 条件ごとのデータセル配列
    %   params - 解析パラメータ構造体
    %   conditions - 条件番号配列

    disp('ERP解析（トポグラフィカル表示）を実行中...');

    % 出力ディレクトリの設定
    outDir = fullfile(params.analysis.outputDir, 'ERP');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % 条件ごとの平均ERPを計算
    meanERPs = cell(length(conditionData), 1);
    for i = 1:length(conditionData)
        if isempty(conditionData{i})
            meanERPs{i} = [];
            continue;
        end

        % 平均ERP計算（チャンネル×時間）
        meanERPs{i} = mean(conditionData{i}, 3);
    end

    % 時間軸の設定
    timeVals = linspace(params.signal.window.timeRange{1}(1), ...
                        params.signal.window.timeRange{1}(2), ...
                        size(meanERPs{1}, 2));

    % 条件ごとにトポグラフィカル表示
    for i = 1:length(conditionData)
        if isempty(meanERPs{i})
            continue;
        end

        % トポグラフィカルプロット
        figure('Name', ['ERP - 条件: ' num2str(conditions(i)) ' - トポグラフィカル表示'], 'Position', [200, 200, 600, 600]);

        % チャンネル位置情報が存在するか確認
        if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
            error('チャンネル位置情報（EEG.chanlocs）が見つかりません。');
        end

        % チャンネル位置を取得
        y = [EEG.chanlocs.X];
        x = [EEG.chanlocs.Y];

        % 位置情報がない場合のデフォルト処理
        if isempty(x) || isempty(y)
            error('チャンネル位置情報が不完全です。');
        end

        % x, yの値をスケーリング（プロットの配置に合わせる）
        % 値を-1から1の範囲に正規化
        maxVal = max(max(abs(x)), max(abs(y)));
        x = x / maxVal * 0.4; % 0.8は余白のため
        y = y / maxVal * 0.4;

        % プロットの大きさを設定
        subplotSize = 0.1; % 各サブプロットの相対サイズ

        % 各チャンネルの位置に小さなERPプロットを配置
        for ch = 1:length(EEG.chanlocs)
            % 現在のチャンネルの位置を取得
            posX = x(ch);
            posY = y(ch);

            % ERPプロットのサイズと位置を計算
            plotLeft = posX - subplotSize/2;
            plotBottom = posY - subplotSize/2;
            plotWidth = subplotSize;
            plotHeight = subplotSize;

            % サブプロットを作成
            ax = axes('Position', [plotLeft+0.5, plotBottom+0.5, plotWidth, plotHeight]);
            hPlot = plot(timeVals, meanERPs{i}(ch, :), 'LineWidth', 1);
            % 防止：子オブジェクトがクリックを捕捉しないように設定
            set(hPlot, 'HitTest','off');
            % 軸の設定を最小限に
            set(gca, 'XTick', [], 'YTick', []);
            set(gca, 'XColor', 'none', 'YColor', 'none');
            hold on;
            plot([0 0], ylim, 'k');
            plot(xlim, [0 0], 'k');

            % チャンネル名を取得
            if isfield(EEG.chanlocs, 'labels') && ~isempty(EEG.chanlocs(ch).labels)
                channelName = EEG.chanlocs(ch).labels;
            else
                channelName = num2str(ch);
            end
            text(min(timeVals), max(ylim), channelName, 'FontSize', 8, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
            xlim([min(timeVals), max(timeVals)]);
            box off;
            set(gca, 'Color', [0.95, 0.95, 0.95]);

            % 設定：クリック時に大画面でERPプロットを表示するコールバックを追加
            set(ax, 'ButtonDownFcn', @(src, event) openLargeERPPlot(timeVals, meanERPs{i}(ch, :), channelName,conditions(i)));
        end

        % メインの図の設定
        axes('Position', [0, 0, 1, 1]);
        axis off;
        axis equal;
        xlim([-1.1, 1.1]);
        ylim([-1.1, 1.1]);
        title(['ERP - 条件: ' num2str(conditions(i))], 'FontSize', 16);

        % 図の保存
        saveas(gcf, fullfile(outDir, ['ERP_Topographic_' num2str(conditions(i)) '.png']));
        saveas(gcf, fullfile(outDir, ['ERP_Topographic_' num2str(conditions(i)) '.fig']));
        % 高解像度版も保存
        print(gcf, fullfile(outDir, ['ERP_Topographic_' num2str(conditions(i)) '_HighRes.png']), '-dpng', '-r300');
        % close(gcf);
    end

    disp('ERP解析（トポグラフィカル表示）が完了しました');

    % ローカルコールバック関数：クリックされたプロットを大画面で表示
    function openLargeERPPlot(timeVals, erpData, channelName,condition)
        figure('Name', ['Large ERP Plot - 条件' num2str(condition) ' - channel - ' channelName ], 'Position', [300, 300, 800, 400]);
        plot(timeVals, erpData, 'LineWidth', 2);
        xlabel('Time (ms)');
        ylabel('Amplitude');
        title(['ERP - ' channelName]);
        grid on;
    end
end

function analyzeERSP(EEG, conditionData, params, conditions)
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
    
    % グローバルカラースケールの設定（オプション）
    useGlobalScale = false;
    if isfield(params.analysis.ersp, 'useGlobalScale') && ~isempty(params.analysis.ersp.useGlobalScale)
        useGlobalScale = params.analysis.ersp.useGlobalScale;
    end
    
    if isfield(params.analysis.ersp, 'colorScale') && ~isempty(params.analysis.ersp.colorScale)
        erspParams.colorScale = params.analysis.ersp.colorScale;
        useGlobalScale = true; % 明示的に設定されている場合はグローバルスケールを使用
    end
    
    % カラーマップの設定
    if isfield(params.analysis.ersp, 'colorMap') && ~isempty(params.analysis.ersp.colorMap)
        erspParams.colorMap = params.analysis.ersp.colorMap;
    else
        erspParams.colorMap = 'jet'; % デフォルト
    end

    % 各条件のERSPを計算・プロット
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
        
        % 各チャンネルのERSP結果を格納するセル配列
        erspResults = cell(length(erspChannels), 1);
        channelScales = cell(length(erspChannels), 1); % 各チャンネルのカラースケール
        
        % 各チャンネルごとにERSPを計算しファイル保存
        for ch = 1:length(erspChannels)
            channel = erspChannels(ch);
            channelName = EEG.chanlocs(ch).labels;
            
            % 全試行のデータを取得
            trialData = squeeze(conditionData{i}(channel, :, :));
            [ersp, itc, powbase, times, freqs] = newtimef(trialData, ...
                EEG.pnts, ...                           % データ長
                [EEG.xmin EEG.xmax], ...                % 時間範囲
                EEG.srate, ...                          % サンプリングレート
                erspParams.cycles, ...                  % 周波数ビンに対する小波サイクル数
                'freqs', erspParams.freqs, ...          % 周波数範囲
                'timesout', erspParams.timesout, ...    % 時間ビンの数
                'plotersp', 'off', ...                  % プロットは行わず
                'plotitc', 'off', ...                    
                'verbose', 'on');                       
            
            % チャンネル個別のカラースケールを計算
            maxVal = max(abs(ersp(:)));
            channelScales{ch} = [-maxVal maxVal]; % 対称的なカラースケール
            
            erspResults{ch} = struct('cycles', erspParams.cycles, 'timesout', erspParams.timesout, ...
                'pnts', EEG.pnts, 'xmin', EEG.xmin, 'xmax', EEG.xmax, 'srate', EEG.srate, 'trialData', trialData, ...
                'ersp', ersp, 'itc', itc, 'powbase', powbase, 'times', times, 'freqs', freqs, ...
                'channel', channel, 'colorScale', channelScales{ch});
            
            % ERSPデータの保存
            erspData = struct(...
                'ersp', ersp, ...
                'itc', itc, ...
                'powbase', powbase, ...
                'times', times, ...
                'freqs', freqs, ...
                'channel', channel, ...
                'condition', conditions(i), ...
                'colorScale', channelScales{ch}, ...
                'params', params ...
            );
            save(fullfile(outDir, ['ERSP_' num2str(conditions(i)) '_' channelName '.mat']), 'erspData');
        end
        
        % グローバルスケールを使用する場合、全チャンネルのデータから計算
        if useGlobalScale && (~isfield(erspParams, 'colorScale') || isempty(erspParams.colorScale))
            allErspValues = [];
            for idx = 1:length(erspChannels)
                allErspValues = [allErspValues; erspResults{idx}.ersp(:)];
            end
            maxVal = max(abs(allErspValues));
            if isfield(erspParams, 'ERSP_CAXIS_LIMIT') && erspParams.ERSP_CAXIS_LIMIT ~= 0
                erspParams.colorScale = erspParams.ERSP_CAXIS_LIMIT * [-1 1];
            else
                erspParams.colorScale = [-1 1] * 1.1 * maxVal;
            end
            disp(['自動計算されたグローバルカラースケール: [' num2str(erspParams.colorScale(1)) ' ' num2str(erspParams.colorScale(2)) ']']);
        end
        
        % チャンネル位置情報を取得し正規化
        if ~isfield(EEG, 'chanlocs') || isempty(EEG.chanlocs)
            error('チャンネル位置情報（EEG.chanlocs）が見つかりません。');
        end
        yCoords = [EEG.chanlocs.X];
        xCoords = [EEG.chanlocs.Y];
        if isempty(xCoords) || isempty(yCoords)
            error('チャンネル位置情報が不完全です。');
        end
        maxVal = max(max(abs(xCoords)), max(abs(yCoords)));
        xNorm = xCoords / maxVal * 0.4;
        yNorm = yCoords / maxVal * 0.4;
        
        subplotSize = 0.15; % サブプロットの大きさ（必要に応じて調整）
        
        % 各条件のトポグラフィックなERSP図の作成
        figure('Name', ['ERSP - 条件: ' num2str(conditions(i))], 'Position', [200, 200, 800, 600]);
        
        for idx = 1:length(erspChannels)
            channel = erspResults{idx}.channel;
            channelName = EEG.chanlocs(idx).labels;
            % 対応するチャンネル位置（EEG.chanlocs のインデックスを利用）
            posX = xNorm(channel);
            posY = yNorm(channel);
            
            % 電極位置に小さなサブプロットを配置
            ax = axes('Position', [posX + 0.5 - subplotSize/2, posY + 0.5 - subplotSize/2, subplotSize, subplotSize]);
            
            imagesc(erspResults{idx}.times, erspResults{idx}.freqs, erspResults{idx}.ersp);
            axis xy;
            
            % カラースケールの適用（グローバルまたはチャンネル個別）
            if useGlobalScale
                clim(erspParams.colorScale);
            else
                clim(channelScales{idx});
            end
            
            set(gca, 'XTick', [], 'YTick', []);
            title(channelName, 'FontSize', 8);
            
            % カラーマップの適用
            colormap(ax, erspParams.colorMap);
            
            % サブプロットクリック時に拡大表示するコールバック
            set(ax, 'ButtonDownFcn', @(src, event) openLargeERSPPlot(erspResults{idx}, channelName));
        end
        
        % メインの図の設定（全体タイトルとカラーバー）
        axes('Position', [0, 0, 1, 1], 'Visible', 'off');
        title(['ERSP - 条件: ' num2str(conditions(i))], 'FontSize', 16);
        
        % カラーバー（スケールタイプを示す追加情報）
        % cbar = colorbar('Position', [0.92, 0.1, 0.02, 0.8]);
        % ylabel(cbar, 'dB', 'FontSize', 10);
        % if useGlobalScale
        %     annotation('textbox', [0.85, 0.02, 0.14, 0.03], 'String', 'グローバルスケール', 'EdgeColor', 'none');
        % else
        %     annotation('textbox', [0.85, 0.02, 0.14, 0.03], 'String', 'チャンネル個別スケール', 'EdgeColor', 'none');
        % end
        
        % 図の保存
        saveas(gcf, fullfile(outDir, ['ERSP_Topographic_' num2str(conditions(i)) '.png']));
        saveas(gcf, fullfile(outDir, ['ERSP_Topographic_' num2str(conditions(i)) '.fig']));
        print(gcf, fullfile(outDir, ['ERSP_Topographic_' num2str(conditions(i)) '_HighRes.png']), '-dpng', '-r300');
        close(gcf);
        
        % チャンネル個別のERSPプロットも保存（オプション）
        if isfield(erspParams, 'saveIndividualPlots') && erspParams.saveIndividualPlots
            for idx = 1:length(erspChannels)
                channelName = EEG.chanlocs(erspChannels(idx)).labels;
                figure('Position', [300, 300, 700, 500]);
                
                % ERSP
                subplot(2, 1, 1);
                imagesc(erspResults{idx}.times, erspResults{idx}.freqs, erspResults{idx}.ersp);
                axis xy;
                if useGlobalScale
                    clim(erspParams.colorScale);
                else
                    clim(channelScales{idx});
                end
                colormap(erspParams.colorMap);
                colorbar;
                title(['ERSP - 条件: ' num2str(conditions(i)) ' - チャンネル: ' channelName], 'FontSize', 12);
                xlabel('時間 (ms)', 'FontSize', 10);
                ylabel('周波数 (Hz)', 'FontSize', 10);
                
                % ITC
                subplot(2, 1, 2);
                imagesc(erspResults{idx}.times, erspResults{idx}.freqs, erspResults{idx}.itc);
                axis xy;
                colormap(erspParams.colorMap);
                colorbar;
                title(['ITC - 条件: ' num2str(conditions(i)) ' - チャンネル: ' channelName], 'FontSize', 12);
                xlabel('時間 (ms)', 'FontSize', 10);
                ylabel('周波数 (Hz)', 'FontSize', 10);
                
                % 保存
                saveas(gcf, fullfile(outDir, ['ERSP_' num2str(conditions(i)) '_' channelName '_Plot.png']));
                close(gcf);
            end
        end
    end
    
    disp('ERSP解析が完了しました');
end


% コールバック関数：クリックされたサブプロットを大画面で表示
function openLargeERSPPlot(erspStruct,channelName)
    figure('Name', ['Large ERSP Plot - ' channelName], 'Position', [300, 300, 800, 600]);
    % imagesc(erspStruct.times, erspStruct.freqs, erspStruct.ersp);
    % axis xy;
    % xlabel('Time (ms)');
    % ylabel('Frequency (Hz)');
    title(['ERSP - ' channelName]);
    % colormap jet;
    % colorbar;

    [ersp, itc, powbase, times, freqs] = newtimef(erspStruct.trialData, ...
        erspStruct.pnts, ...                           % データ長
        [erspStruct.xmin erspStruct.xmax], ...                 % 時間範囲
        erspStruct.srate, ...                           % サンプリングレート
        erspStruct.cycles, ...                   % 周波数ビンに対する小波サイクル数
        'freqs', erspStruct.freqs, ...           % 周波数範囲
        'timesout', erspStruct.timesout, ...     % 時間ビンの数
        'plotersp', 'on', ...                   % プロットは行わず
        'plotitc', 'on', ...                    
        'verbose', 'on');   
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
        % if isfield(params.analysis.topography, 'baseline') && !isempty(params.analysis.topography.baseline)
        %     baselineIdx = find(params.signal.window.timeRange{1} >= params.analysis.topography.baseline(1) & params.signal.window.timeRange{1} <= params.analysis.topography.baseline(2));
        %     if !isempty(baselineIdx)
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