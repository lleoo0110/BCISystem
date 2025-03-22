function EEG = create_eeglabset(rawData, labels, params,saveDir)

    % 前提: rawData (channels x data points), labels (struct array) がワークスペースにある
    
    % 1. EEGLAB環境の初期化
    eeglab("nogui");
    
    % 2. EEG構造体の作成
    EEG = eeg_emptyset; % 空のEEG構造体を作成
    
    % 3. データのロード
    EEG.data = rawData; % 生データをEEG構造体にロード
    EEG.nbchan = size(rawData, 1); % チャンネル数を設定
    EEG.pnts   = size(rawData, 2); % データポイント数を設定
    EEG.srate  = params.device.sampleRate;   % サンプリングレート (Hz) - これは仮の値です。実際のサンプリングレートに合わせてください。
    EEG.xmin   = 0;      % 開始時間 (秒) - これは仮の値です。必要に応じて調整してください。
    EEG.xmax   = (EEG.pnts-1) / EEG.srate; % 終了時間 (秒) を計算
    
    % 4. チャンネルロケーションのロード
    % epocx_14ch.ced から chanlocs を読み込む
    try
        % ファイルの存在確認
        % ファイルの存在確認
        if params.device.name == "EPOCX"
            ced_file = 'epocx_14ch.ced';
        elseif params.device.name == "EPOCFLEX"
            ced_file = 'emotiv_flex32ch.ced';
        end
        % ced_file = 'epocx_14ch.ced'; % またはフルパス
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
    
    % 5. イベント情報の追加
    %    labels構造体からイベント情報をEEG構造体に追加
    for i = 1:length(labels)
        EEG.event(i).type   = num2str(labels(i).value);   % イベントタイプ
        EEG.event(i).latency = labels(i).sample;  % レイテンシー (サンプル単位)
        EEG.event(i).urevent = i;                  % イベントのID
    end
    
    % 6. イベント情報を反映
    EEG = eeg_checkset( EEG, 'eventconsistency');
    
    % 7. データセットの保存
    % [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, 0); % 0 は CURRENTSET を変更しないことを意味
    % EEG = pop_saveset( EEG, 'filename','imported_data.set','filepath', pwd); % 現在のディレクトリに保存
    
    % 8. 前処理 (フィルタリング、リファレンスなど)
    % ハイパスフィルタ
    % EEG = pop_eegfiltnew( EEG, params.signal.frequency.min, params.signal.frequency.max, [], 0, [], 0);
    % EEG = pop_eegfilt( EEG, 1, 0, [], [0]); % 1 Hz ハイパスフィルタ
    
    % リファレンス
    % EEG = pop_reref( EEG, []); % 平均リファレンス
    
   
    % 9. エポックの作成
    EEG = pop_epoch( EEG, arrayfun(@(x)num2str(x.value), labels, 'UniformOutput', false), [params.signal.window.timeRange{1}(1) params.signal.window.timeRange{1}(2)], 'epochinfo', 'yes');  % 例: -1秒から2秒のエポック
    
    % 10. ベースライン補正
    % basemin = params.signal.window.stimulus(1) + 1.0/EEG.srate;
    % disp(basemin);
    % EEG = pop_rmbase( EEG, [basemin 0]); % -1000msから0msをベースラインとして除去
    

     % データセットの保存
    % [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
    EEG = pop_saveset( EEG, 'filename',[params.info.name,'.set'],'filepath', saveDir);

    % 最後に、処理したデータセットを EEGLAB GUI にロードする
    % ALLEEG(CURRENTSET) = EEG;
    % eeglab redraw; % EEGLAB GUI を更新
end