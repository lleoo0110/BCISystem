% パラメータの設定
params = getConfig('epocx');

% SignalProcessorのインスタンス化
processor = SignalProcessor(params);

% データ処理
processor.process(eegData, stimulusStart);

% 結果の取得
processedData = processor.processedData;
processedLabels = processor.processedLabels;
dataInfo = processor.getProcessingInfo();

% 結果の可視化
processor.visualizeProcessing();

% 結果の保存
processor.save('processed_data.mat');