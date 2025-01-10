% データ結合の使用例

% 設定の読み込み
params = getConfig('epocx');

% DataConcatenatorの初期化
concatenator = DataConcatenator(params);

% データの結合実行
[concatenatedData, success] = concatenator.concatenateDataFiles();

if success
    fprintf('データの結合が正常に完了しました。\n');
    
    % 結合されたデータの情報表示
    fprintf('\n=== 結合データの情報 ===\n');
    fprintf('サンプル数: %d\n', size(concatenatedData.rawData, 2));
    fprintf('チャンネル数: %d\n', size(concatenatedData.rawData, 1));
    fprintf('ラベル数: %d\n', length(concatenatedData.labels));
    
    % ラベルの分布を表示
    labelValues = [concatenatedData.labels.value];
    uniqueLabels = unique(labelValues);
    fprintf('\nラベルの分布:\n');
    for i = 1:length(uniqueLabels)
        count = sum(labelValues == uniqueLabels(i));
        fprintf('ラベル %d: %d個\n', uniqueLabels(i), count);
    end
else
    fprintf('データの結合に失敗しました。\n');
end