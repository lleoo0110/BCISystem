classdef DataConcatenator < handle
    properties (Access = private)
        params          % パラメータ設定
        dataManager    % データ管理用オブジェクト
    end
    
    methods (Access = public)
        function obj = DataConcatenator(params)
            obj.params = params;
            obj.dataManager = DataManager(params);
        end
        
        function [concatenatedData, success] = concatenateDataFiles(obj)
            try
                % データの読み込みと結合
                [rawData, labels] = obj.loadAndConcatenateData();
                if isempty(rawData) || isempty(labels)
                    error('データの結合に失敗しました');
                end
                
                % 結合データの構造体作成
                concatenatedData = struct();
                concatenatedData.params = obj.params;
                concatenatedData.rawData = rawData;
                concatenatedData.labels = labels;
                
                % 結合データの保存
                % savedFile = obj.dataManager.saveDataset(concatenatedData);
                % fprintf('結合データを保存しました: %s\n', savedFile);
                
                success = true;
                
            catch ME
                warning(ME.identifier, '%s', ME.message);
                concatenatedData = [];
                success = false;
            end
        end
    end
    
    methods (Access = private)
        function [mergedRawData, mergedLabels] = loadAndConcatenateData(obj)
            try
                % データファイルの選択（複数ファイル）
                [fileNames, filePath] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, ...
                    'Select data files', ...
                    'MultiSelect', 'on');
                
                if isequal(fileNames, 0)
                    error('ファイルが選択されていません');
                end
                
                % 単一ファイル選択時の処理
                if ~iscell(fileNames)
                    fileNames = {fileNames};
                end
                
                % 初期化
                totalSamples = 0;
                totalLabels = 0;
                fileData = cell(length(fileNames), 1);
                
                % 各ファイルのサンプル数とラベル数を計算
                fprintf('データファイルの解析中...\n');
                for i = 1:length(fileNames)
                    fullPath = fullfile(filePath, fileNames{i});
                    data = load(fullPath);
                    fileData{i} = data;
                    
                    if ~isfield(data, 'rawData') || ~isfield(data, 'labels')
                        error('ファイル %s に必要なフィールドがありません', fileNames{i});
                    end
                    
                    totalSamples = totalSamples + size(data.rawData, 2);
                    if ~isempty(data.labels)
                        totalLabels = totalLabels + length(data.labels);
                    end
                end
                
                % 結合データの初期化
                channelCount = size(fileData{1}.rawData, 1);
                mergedRawData = zeros(channelCount, totalSamples);
                mergedLabels = struct('value', {}, 'sample', {});
                
                % データの結合
                fprintf('データの結合を開始...\n');
                currentSample = 1;
                labelIndex = 1;
                
                for i = 1:length(fileData)
                    data = fileData{i};
                    
                    % rawDataの結合
                    numSamples = size(data.rawData, 2);
                    mergedRawData(:, currentSample:currentSample+numSamples-1) = data.rawData;
                    
                    % ラベルの結合
                    if ~isempty(data.labels)
                        for j = 1:length(data.labels)
                            mergedLabels(labelIndex).value = data.labels(j).value;
                            mergedLabels(labelIndex).sample = data.labels(j).sample + (currentSample - 1);
                            labelIndex = labelIndex + 1;
                        end
                    end
                    
                    currentSample = currentSample + numSamples;
                    fprintf('ファイル %d/%d を処理しました\n', i, length(fileData));
                end
                
                fprintf('データの結合が完了しました\n');
                fprintf('合計サンプル数: %d\n', totalSamples);
                fprintf('合計ラベル数: %d\n', length(mergedLabels));
                
            catch ME
                fprintf('エラーが発生しました: %s\n', ME.message);
                mergedRawData = [];
                mergedLabels = [];
            end
        end
    end
end