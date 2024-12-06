function testUDP()
   % クリーンアップ
   clear('receiver', 'sender');
   instrreset;
   
   % 共通のパラメータ取得
   params = getConfig('epocx');

   % モード選択ダイアログ
   choice = questdlg('モードを選択してください', ...
       'UDP Test', ...
       '受信モード', '送信モード', 'キャンセル', ...
       '受信モード');
   
   % 選択に基づいて処理を分岐
   switch choice
       case '受信モード'
           startReceiver(params);
       case '送信モード'
           startSender(params);
   end
end

function startReceiver(params)
   % figureハンドルの初期化
   fig = figure('Name', 'UDP Receiver', ...
                'NumberTitle', 'off', ...
                'MenuBar', 'none', ...
                'Position', [100 100 500 400]);
   
   try
       % 受信開始
       receiver = UDPManager(params);
       
       % receiverオブジェクトを保存（figureハンドルが有効であることを確認）
       if ishandle(fig)
           setappdata(fig, 'receiver', receiver);
       else
           error('Figure handle is invalid');
       end

       % パネルの作成
       panel = uipanel('Parent', fig, ...
                      'Position', [0.05 0.2 0.9 0.7]);

       % 受信ログ表示用のリストボックス
       listbox = uicontrol('Parent', panel, ...
                          'Style', 'listbox', ...
                          'Position', [10 10 460 250], ...
                          'String', {}, ...
                          'Max', 2, ...
                          'Min', 0);

       % 終了ボタン
       uicontrol('Parent', fig, ...
                'Style', 'pushbutton', ...
                'String', '終了', ...
                'Position', [200 20 100 40], ...
                'Callback', @(~,~)closeReceiver(fig));

       % トリガー情報の表示
       updateLog(listbox, '=== 受信トリガー一覧 ===');
       mappings = params.udp.receive.triggers.mappings;
       fields = fieldnames(mappings);
       for i = 1:length(fields)
           trigger = mappings.(fields{i});
           updateLog(listbox, sprintf('テキスト: "%-20s" 値: %d', ...
               trigger.text, trigger.value));
       end
       updateLog(listbox, '========================');

       updateLog(listbox, '受信待機を開始します...');
       
       receiver.start(@(value)handleTrigger(value, listbox));

       % ウィンドウが閉じられるまで待機
       uiwait(fig);

   catch ME
       % エラーハンドリングの改善
       errordlg(['エラーが発生しました: ', ME.message], 'Error');
       
       % figureハンドルが有効な場合のみクリーンアップを実行
       if ishandle(fig)
           if isappdata(fig, 'receiver')
               receiver = getappdata(fig, 'receiver');
               if ~isempty(receiver) && isvalid(receiver)
                   receiver.stop();
                   delete(receiver);
               end
           end
           delete(fig);
       end
   end
end

function startSender(params)
   % figureハンドルの初期化
   fig = figure('Name', 'UDP Sender', ...
                'NumberTitle', 'off', ...
                'MenuBar', 'none', ...
                'Position', [600 100 400 500]);
   
   try
       % パネルの作成
       panel = uipanel('Parent', fig, ...
                      'Position', [0.05 0.2 0.9 0.7]);

       % 送信ログ表示用のリストボックス
       listbox = uicontrol('Parent', panel, ...
                          'Style', 'listbox', ...
                          'Position', [10 10 360 250], ...
                          'String', {}, ...
                          'Max', 2, ...
                          'Min', 0);

       % トリガーボタンの作成
       mappings = params.udp.receive.triggers.mappings;
       fields = fieldnames(mappings);
       
       buttonHeight = 40;
       spacing = 10;
       startY = 400;
       
       % トリガーボタン
       for i = 1:length(fields)
           trigger = mappings.(fields{i});
           uicontrol('Parent', fig, ...
                    'Style', 'pushbutton', ...
                    'String', [trigger.text, 'トリガー送信'], ...
                    'Position', [75 startY-i*(buttonHeight+spacing) 250 buttonHeight], ...
                    'Callback', {@sendTrigger, trigger.text, listbox});
       end

       % カスタムメッセージ送信ボタン
       uicontrol('Parent', fig, ...
                'Style', 'pushbutton', ...
                'String', 'カスタムメッセージ送信', ...
                'Position', [75 startY-(length(fields)+1)*(buttonHeight+spacing) 250 buttonHeight], ...
                'Callback', {@sendCustom, listbox});

       % 終了ボタン
       uicontrol('Parent', fig, ...
                'Style', 'pushbutton', ...
                'String', '終了', ...
                'Position', [150 20 100 40], ...
                'Callback', @(~,~)closeSender(fig));

       % 送信用UDPManager作成
       sender = UDPManager(params);
       
       % senderオブジェクトを保存（figureハンドルが有効であることを確認）
       if ishandle(fig)
           setappdata(fig, 'sender', sender);
       else
           error('Figure handle is invalid');
       end
       
       % ウィンドウが閉じられるまで待機
       uiwait(fig);

   catch ME
       % エラーハンドリングの改善
       errordlg(['エラーが発生しました: ', ME.message], 'Error');
       
       % figureハンドルが有効な場合のみクリーンアップを実行
       if ishandle(fig)
           if isappdata(fig, 'sender')
               sender = getappdata(fig, 'sender');
               if ~isempty(sender) && isvalid(sender)
                   delete(sender);
               end
           end
           delete(fig);
       end
   end
end

function updateLog(listbox, message)
   % 現在のログを取得
   currentLog = get(listbox, 'String');
   if isempty(currentLog)
       currentLog = {};
   end
   
   % 新しいメッセージを追加
   currentLog{end+1} = message;
   
   % リストボックスを更新
   set(listbox, 'String', currentLog);
   
   % 最新の項目までスクロール
   set(listbox, 'Value', length(currentLog));
   
   % コマンドウィンドウにも表示
   disp(message);
end

function handleTrigger(value, listbox)
   timestamp = datestr(now, 'HH:MM:SS.FFF');
   switch value
       case 0
           msg = '未定義のトリガー';
       case 1
           msg = '安静トリガー';
       case 2
           msg = '炎トリガー';
       case 3
           msg = '氷トリガー';
       otherwise
           msg = ['未知のトリガー(', num2str(value), ')'];
   end
   updateLog(listbox, [timestamp, ' - 受信: ', msg]);
end

function sendTrigger(hObject, ~, text, listbox)
   try
       % figureハンドルの有効性を確認
       fig = ancestor(hObject, 'figure');
       if ~ishandle(fig)
           error('Figure handle is invalid');
       end
       
       % senderオブジェクトを取得
       if isappdata(fig, 'sender')
           sender = getappdata(fig, 'sender');
           if isempty(sender) || ~isvalid(sender)
               error('Sender object is invalid');
           end
           
           timestamp = datestr(now, 'HH:MM:SS.FFF');
           sender.send(text);
           updateLog(listbox, [timestamp, ' - 送信: ', text]);
       else
           error('Sender object not found');
       end
   catch ME
       updateLog(listbox, [datestr(now, 'HH:MM:SS.FFF'), ' - エラー: ', ME.message]);
   end
end

function sendCustom(hObject, ~, listbox)
   try
       % figureハンドルの有効性を確認
       fig = ancestor(hObject, 'figure');
       if ~ishandle(fig)
           error('Figure handle is invalid');
       end
       
       % senderオブジェクトを取得
       if isappdata(fig, 'sender')
           sender = getappdata(fig, 'sender');
           if isempty(sender) || ~isvalid(sender)
               error('Sender object is invalid');
           end
           
           answer = inputdlg('送信するメッセージを入力してください:', ...
                           'メッセージ入力', [1 50]);
           if ~isempty(answer)
               timestamp = datestr(now, 'HH:MM:SS.FFF');
               sender.send(answer{1});
               updateLog(listbox, [timestamp, ' - 送信: ', answer{1}]);
           end
       else
           error('Sender object not found');
       end
   catch ME
       updateLog(listbox, [datestr(now, 'HH:MM:SS.FFF'), ' - エラー: ', ME.message]);
   end
end

function closeReceiver(fig)
   try
       % figureハンドルの有効性を確認
       if ~ishandle(fig)
           return;
       end
       
       % receiverオブジェクトを取得
       if isappdata(fig, 'receiver')
           receiver = getappdata(fig, 'receiver');
           if ~isempty(receiver) && isvalid(receiver)
               receiver.stop();
               pause(0.1); % 受信ループが確実に停止するまで待機
               delete(receiver);
           end
       end
   catch ME
       warning(ME.identifier, '%s', ME.message);  % identifierを含めた正しい形式
   end
   
   % figureの削除を確実に実行
   if ishandle(fig)
       delete(fig);
   end
end

function closeSender(fig)
   try
       % figureハンドルの有効性を確認
       if ~ishandle(fig)
           return;
       end
       
       % senderオブジェクトを取得
       if isappdata(fig, 'sender')
           sender = getappdata(fig, 'sender');
           if ~isempty(sender) && isvalid(sender)  % receiverをsenderに修正
               delete(sender);
           end
       end
   catch ME
       warning(ME.identifier, '%s', ME.message);  % identifierを含めた正しい形式
   end
   
   % figureの削除を確実に実行
   if ishandle(fig)
       delete(fig);
   end
end