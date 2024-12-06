function [loadedData, info] = testSaveLoad
    % テスト用のパラメータ設定
    params = getConfig('epocx');

    % テストディレクトリの作成
    if ~exist('./test_data', 'dir')
        mkdir('./test_data');
    end

    % テストデータの生成
    testData = struct();
    testData.rawEEG = randn(8, 1000);            % 8チャンネル x 1000サンプル
    testData.label = randi([1 3], 10, 1); % ラベル
    testData.processedData = randn(8, 1000);      % 処理済みデータ
    testData.processedLabel = randi([1 3], 10, 1); % 処理済みラベル
    testData.cspFeatures = randn(10, 20);         % CSP特徴量
    testData.svmClassifier = struct('model', 'dummy_model'); % ダミーの分類器
    testData.metadata = struct('sampleRate', 250); % メタデータ

    % DataManagerのインスタンス化
    dm = DataManager(params);

    try
        % === 保存のテスト ===
        fprintf('\n=== Testing save functionality ===\n');
        dm.saveDataset(testData);

        % === 読み込みのテスト ===
        fprintf('\n=== Testing load functionality ===\n');
        loadedData = dm.loadDataset();

        % === データ情報の取得テスト ===
        fprintf('\n=== Testing dataset info functionality ===\n');
        info = dm.getDatasetInfo(fullfile(params.acquisition.load.path, ...
                                        params.acquisition.load.filename));
        
        % テスト結果の表示
        fprintf('\nTest Results:\n');
        fprintf('Loaded fields: %s\n', strjoin(fieldnames(loadedData), ', '));
        fprintf('Dataset info fields: %s\n', strjoin(fieldnames(info), ', '));
        
        fprintf('\nTest completed successfully!\n');
        
    catch ME
        fprintf('\nTest failed: %s\n', ME.message);
        fprintf('Error details:\n%s\n', getReport(ME));
    end
end