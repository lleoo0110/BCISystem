function results = testSVM(features, labels, mode)
    % testSVM - SVMの学習と評価を行う

    % 使用例:
    %   results = testSVM(features, labels, 'offline') % オフラインモード: 学習と評価
    %   results = testSVM(features, [], 'online')  % オンラインモード: 予測
    %   results = testSVM(cspFeatures, processedLabel, 'offline')
    %   results = testSVM(cspFeatures(1,1:14), [], 'online')
    
    if nargin < 3
        mode = 'offline';
    end
    
    try
        params = getConfig('epocx');
        classifier = SVMClassifier(params);
        dm = DataManager(params);
        
        switch lower(mode)
            case 'offline'
                classifier.setMode('offline');
                results = classifier.trainOffline(features, labels);
                
                saveData = dm.createSaveData(...
                    'params', params, ...
                    'svmClassifier', results.classifier, ...
                    'results', results.performance, ...
                    'timestamp', datetime('now'));
                
                dm.saveDataset(saveData);
                
            case 'online'
                %モデルを一度だけ読み込む
                loadData = dm.loadDataset(params.acquisition.load.filename);
                classifier.svmModel = loadData.svmClassifier;  % モデルをセット
                
                % classifier.setMode('online');
                results = classifier.predictOnline(features);
                
                saveData = dm.createSaveData(...
                    'params', params, ...
                    'svmClassifier', results.classifier, ...    % 使用したSVMClassifierを保存
                    'results', results.predict, ...
                    'timestamp', datetime('now'));
                
                dm.saveDataset(saveData);
                
            otherwise
                error('Invalid mode specified. Use ''offline'' or ''online''.');
        end
        
    catch ME
        error('SVM testing failed: %s', ME.message);
    end
end