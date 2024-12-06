function testCSP(procData, procLabels)
    % CSPアルゴリズムのテスト関数
    % 使用例：
    % testCSP(processedData, processedLabel);
    
    % テスト用のパラメータ設定
    params = getConfig('epocx');
    
    % モードに基づいてテストを実行
    if strcmp(params.feature.csp.mode, 'offline')
        testOfflineMode(params, procData, procLabels);
    else
        testOnlineMode(params, procData);
    end
end

function testOfflineMode(params, processedData, processedLabel)
    try
        fprintf('\n=== Testing offline mode ===\n');
        
        % 変数の初期化
        rawEEG = [];
        label = [];
        cspFeatures = [];
        cspFilters = [];
        svmClassifier = [];
        metadata = [];
        
        % CSPFeatureExtractorのインスタンス化
        csp = CSPFeatureExtractor(params);

        % CSPの学習
        fprintf('Training CSP...\n');
        csp.trainCSP(processedData, processedLabel);

        % フィルタの取得
        [cspFilters, numPatterns] = csp.getCSPFilters();
        fprintf('Number of CSP filters: %d\n', length(cspFilters));

        % 特徴抽出
        fprintf('Extracting features...\n');
        cspFeatures = csp.extractFeatures(processedData);

        % 結果の表示
        fprintf('\nFeature extraction results:\n');
        fprintf('Feature matrix size: %dx%d\n', size(cspFeatures));
        
        % メタデータの作成
        metadata = struct(...
            'featureSize', size(cspFeatures), ...
            'numFilters', length(cspFilters) ...
        );

        % データの保存部分の例
        dm = DataManager(params);
        saveData = dm.createSaveData(...
            'params', params, ...
            'rawEEG', rawEEG, ... 
            'label', label, ...
            'processedData', processedData, ...
            'processedLabel', processedLabel, ...
            'cspFilters', cspFilters, ...
            'cspFeatures', cspFeatures, ...
            'svmClassifier', svmClassifier, ...
            'metadata', metadata ...
        );
        dm.saveDataset(saveData);
        
        fprintf('\nOffline mode test completed successfully!\n');
        
    catch ME
        fprintf('\nTest failed: %s\n', ME.message);
        fprintf('Error details:\n%s\n', getReport(ME));
    end
end

function testOnlineMode(params, processedData)
    try
        fprintf('\n=== Testing online mode ===\n');
        
        % CSPFeatureExtractorのインスタンス化
        csp = CSPFeatureExtractor(params);

        % 単一エポックのテストデータ
        singleEpoch = processedData{1,1};

        % フィルタの有無を確認
        loadPath = fullfile(params.acquisition.load.path, ...
                          params.acquisition.load.filename);
                          
        if exist(loadPath, 'file')
            fprintf('Testing feature extraction with single epoch...\n');
            onlineFeatures = csp.extractFeatures(singleEpoch);
            fprintf('Online feature vector size: %dx%d\n', size(onlineFeatures));
            fprintf('\nOnline mode test completed successfully!\n');
        else
            fprintf('No pre-trained filters available for online testing\n');
            fprintf('Please run offline mode first to generate the filters\n');
        end
        
    catch ME
        fprintf('\nTest failed: %s\n', ME.message);
        fprintf('Error details:\n%s\n', getReport(ME));
    end
end