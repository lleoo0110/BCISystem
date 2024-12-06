classdef CSPFeatureExtractor < handle
   properties (Access = private)
       params             % 設定パラメータ
       cspFilters         % CSPフィルタ（セル配列または配列）
       numPatterns        % パターン数
       trainedStatus      % 学習状態
       isEnabled          % CSP有効/無効フラグ
       storageType        % データ保存形式（'array' or 'cell'）
   end
   
   methods (Access = public)
       function obj = CSPFeatureExtractor(params)
           obj.params = params;
           obj.validateConfig();
           
           if isfield(params.feature.csp, 'patterns')
               obj.numPatterns = params.feature.csp.patterns;
           else
               obj.numPatterns = 2;
           end
           
           obj.trainedStatus = false;
           obj.cspFilters = {};
           obj.isEnabled = params.feature.csp.enable;
           obj.storageType = params.feature.csp.storageType;    
           
       end
       
       function cspFilters = trainCSP(obj, data, labels)
            if ~obj.isEnabled
                error('CSPFeatureExtractor:Disabled', 'CSP feature extraction is disabled');
            end

            try
                % クラスごとにデータを分割
                uniqueLabels = unique(labels);
                numClasses = length(uniqueLabels);

                % データの形式に基づいて処理を分岐
                if iscell(data)
                    dataClass = obj.splitDataClassesCell(data, labels, uniqueLabels);
                else
                    dataClass = obj.splitDataClassesArray(data, labels, uniqueLabels);
                end

                % one-vs-one方式でCSPフィルタを学習
                numPairs = nchoosek(numClasses, 2);

                % フィルタの保存形式に応じて初期化
                if strcmp(obj.storageType, 'array')
                    % 配列形式で初期化
                    if iscell(data)
                        [nChannels, ~] = size(data{1});
                    else
                        [nChannels, ~, ~] = size(data);
                    end
                    obj.cspFilters = zeros(nChannels, obj.numPatterns * 2, numPairs);
                else
                    % セル配列形式で初期化
                    obj.cspFilters = cell(numPairs, 1);
                end

                cspIndex = 1;
                for i = 1:numClasses-1
                    for j = i+1:numClasses
                        avgCov1 = obj.calculateAverageCovariance(dataClass{i});
                        avgCov2 = obj.calculateAverageCovariance(dataClass{j});
                        currentFilter = obj.calculateCSP(avgCov1, avgCov2);

                        % フィルタの保存形式に応じて格納
                        if strcmp(obj.storageType, 'array')
                            obj.cspFilters(:, :, cspIndex) = currentFilter;
                        else
                            obj.cspFilters{cspIndex} = currentFilter;
                        end
                        cspIndex = cspIndex + 1;
                    end
                end
                cspFilters = obj.cspFilters;
                obj.trainedStatus = true;

            catch ME
                error('CSP training failed: %s', ME.message);
            end
       end
       
       function features = extractFeatures(obj, data, cspfilters)
            if ~obj.isEnabled
                error('CSPFeatureExtractor:Disabled', 'CSP feature extraction is disabled');
            end

            try
                % データの次元を確認
                [nChannels, nSamples, nTrials] = size(data);

                % CSPフィルタの検証と数の取得
                if strcmp(obj.storageType, 'array')
                    validateattributes(cspfilters, {'numeric'}, {'3d'}, 'extractFeatures', 'cspfilters');
                    numPairs = size(cspfilters, 3);
                else
                    validateattributes(cspfilters, {'cell'}, {'vector'}, 'extractFeatures', 'cspfilters');
                    numPairs = length(cspfilters);
                end

                % 各試行に対して特徴抽出を実行
                features = zeros(nTrials, numPairs * obj.numPatterns * 2);
                for trial = 1:nTrials
                    % 現在の試行のデータを取得（2次元配列として）
                    trialData = data(:, :, trial);  % [channels × samples]
                    features(trial, :) = extractSingleTrial(obj, trialData, cspfilters, numPairs);
                end

            catch ME
                error('Feature extraction failed: %s', ME.message);
            end
        end
        
       function features = extractSingleTrial(obj, data, cspfilters, numPairs)
            % 単一試行の特徴抽出
            features = zeros(1, numPairs * obj.numPatterns * 2);
            featureIndex = 1;

            for pair = 1:numPairs
                if strcmp(obj.storageType, 'array')
                    currentFilter = cspfilters(:, :, pair);
                else
                    currentFilter = cspfilters{pair};
                end

                % フィルタリングと特徴量計算
                filteredData = currentFilter' * data;
                pairFeatures = log(var(filteredData, 0, 2));

                % 特徴量の格納
                features(1, featureIndex:featureIndex+length(pairFeatures)-1) = pairFeatures;
                featureIndex = featureIndex + length(pairFeatures);
            end
        end
       
       function [filters, numPatterns] = getCSPFilters(obj)
            try
                if ~obj.trainedStatus
                    error('CSPFeatureExtractor:NotTrained', 'CSP filters are not trained');
                end
                filters = obj.cspFilters;
                numPatterns = obj.numPatterns;
            catch ME
                warning(ME.identifier, '%s', ME.message);
                rethrow(ME);
            end
       end
        
        function convertFilterStorage(obj, newStorageType)
            if ~obj.trainedStatus
                return;
            end

            if strcmp(newStorageType, obj.storageType)
                return;
            end

            try
                if strcmp(newStorageType, 'array')
                    % セル配列から配列への変換
                    numPairs = length(obj.cspFilters);
                    [nChannels, nPatterns] = size(obj.cspFilters{1});
                    tempFilters = zeros(nChannels, nPatterns, numPairs);
                    for i = 1:numPairs
                        tempFilters(:, :, i) = obj.cspFilters{i};
                    end
                    obj.cspFilters = tempFilters;
                else
                    % 配列からセル配列への変換
                    numPairs = size(obj.cspFilters, 3);
                    tempFilters = cell(numPairs, 1);
                    for i = 1:numPairs
                        tempFilters{i} = obj.cspFilters(:, :, i);
                    end
                    obj.cspFilters = tempFilters;
                end
                obj.storageType = newStorageType;

            catch ME
                error('Filter storage conversion failed: %s', ME.message);
            end
        end
   end
   
   methods (Access = private)
       function validateConfig(obj)
           % 設定の検証
           if ~isfield(obj.params.feature.csp, 'enable')
               obj.params.feature.csp.enable = true;
           end
           
           if ~isfield(obj.params.feature.csp, 'mode')
               obj.params.feature.csp.mode = 'offline';
           elseif ~ismember(obj.params.feature.csp.mode, {'offline', 'online'})
               error('Invalid processing mode. Must be either ''offline'' or ''online''');
           end
           
           if ~isfield(obj.params.feature.csp, 'storageType')
               obj.params.feature.csp.storageType = 'cell';
           elseif ~ismember(obj.params.feature.csp.storageType, {'array', 'cell'})
               error('Invalid storage type. Must be either ''array'' or ''cell''');
           end
       end
       
       function filters = convertFilterFormat(obj, inputFilters)
           % フィルタの形式を変換
           if strcmp(obj.storageType, 'array')
               if iscell(inputFilters)
                   % セル配列から配列に変換
                   [nChannels, nPatterns] = size(inputFilters{1});
                   numPairs = length(inputFilters);
                   filters = zeros(nChannels, nPatterns, numPairs);
                   for i = 1:numPairs
                       filters(:, :, i) = inputFilters{i};
                   end
               else
                   filters = inputFilters;
               end
           else
               if ~iscell(inputFilters)
                   % 配列からセル配列に変換
                   numPairs = size(inputFilters, 3);
                   filters = cell(numPairs, 1);
                   for i = 1:numPairs
                       filters{i} = inputFilters(:, :, i);
                   end
               else
                   filters = inputFilters;
               end
           end
       end
       
       function features = extractFeaturesCell(obj, data, numPairs)
           numTrials = length(data);
           features = zeros(numTrials, numPairs * obj.numPatterns * 2);
           
           for trial = 1:numTrials
               trialData = data{trial};
               featureIndex = 1;
               for pair = 1:numPairs
                   if strcmp(obj.storageType, 'array')
                       currentFilter = obj.cspFilters(:, :, pair);
                   else
                       currentFilter = obj.cspFilters{pair};
                   end
                   pairFeatures = obj.extractCSPFeatures(trialData, currentFilter);
                   features(trial, featureIndex:featureIndex+length(pairFeatures)-1) = pairFeatures;
                   featureIndex = featureIndex + length(pairFeatures);
               end
           end
       end
       
       function features = extractFeaturesArray(obj, data, numPairs)
           [~, ~, numTrials] = size(data);
           features = zeros(numTrials, numPairs * obj.numPatterns * 2);
           
           for trial = 1:numTrials
               trialData = data(:, :, trial);
               featureIndex = 1;
               for pair = 1:numPairs
                   if strcmp(obj.storageType, 'array')
                       currentFilter = obj.cspFilters(:, :, pair);
                   else
                       currentFilter = obj.cspFilters{pair};
                   end
                   pairFeatures = obj.extractCSPFeatures(trialData, currentFilter);
                   features(trial, featureIndex:featureIndex+length(pairFeatures)-1) = pairFeatures;
                   featureIndex = featureIndex + length(pairFeatures);
               end
           end
       end
       
       function dataClass = splitDataClassesCell(~, data, labels, uniqueLabels)
           numClasses = length(uniqueLabels);
           dataClass = cell(numClasses, 1);
           
           for i = 1:numClasses
               classIndices = labels == uniqueLabels(i);
               dataClass{i} = data(classIndices);
           end
       end
       
       function dataClass = splitDataClassesArray(~, data, labels, uniqueLabels)
           numClasses = length(uniqueLabels);
           dataClass = cell(numClasses, 1);
           
           for i = 1:numClasses
               classIndices = labels == uniqueLabels(i);
               dataClass{i} = data(:, :, classIndices);
           end
       end
       
       function avgCov = calculateAverageCovariance(~, dataClass)
            if iscell(dataClass)
                % セル配列形式の場合
                [nChannels, ~] = size(dataClass{1});
                avgCov = zeros(nChannels, nChannels);

                nTrials = length(dataClass);
                for i = 1:nTrials
                    singleTrialData = dataClass{i};
                    avgCov = avgCov + cov(singleTrialData') / nTrials;
                end
            else
                % 配列形式の場合
                [nChannels, ~, nTrials] = size(dataClass);
                avgCov = zeros(nChannels, nChannels);

                for i = 1:nTrials
                    singleTrialData = dataClass(:, :, i);
                    avgCov = avgCov + cov(singleTrialData') / nTrials;
                end
            end
       end
       
       function cspFilters = calculateCSP(obj, avgCov1, avgCov2)
           compositeCovariance = avgCov1 + avgCov2;
           [eigVectors, eigValues] = eig(avgCov1, compositeCovariance);
           
           [~, order] = sort(diag(eigValues), 'descend');
           eigVectors = eigVectors(:, order);
           
           cspFilters = eigVectors(:, [1:obj.numPatterns, end-obj.numPatterns+1:end]);
       end
       
       function features = extractCSPFeatures(~, data, filters)
           filteredData = filters' * data;
           features = log(var(filteredData, 0, 2));
       end
   end
end