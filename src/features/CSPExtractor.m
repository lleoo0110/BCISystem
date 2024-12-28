classdef CSPExtractor < handle
    properties (Access = private)
        params       % パラメータ設定
        patterns    % CSPパターン数
        regParam    % 正則化パラメータ
    end
    
    methods (Access = public)
        function obj = CSPExtractor(params)
            obj.params = params;
            obj.patterns = params.feature.csp.patterns;
            obj.regParam = params.feature.csp.regularization;
        end
        
        function [filters, parameters] = trainCSP(obj, data, labels)
           try
               % クラスごとのデータを分離
               classes = unique(labels);
               covMatrices = cell(length(classes), 1);

               % 各クラスの共分散行列を計算
               for i = 1:length(classes)
                   classData = obj.getClassData(data, labels, classes(i));
                   covMatrices{i} = obj.calculateCovarianceMatrix(classData);
               end

               % CSPフィルタの計算
               [filters, parameters] = obj.computeCSPFilters(covMatrices);

           catch ME
               error('CSP training failed: %s', ME.message);
           end
       end
        
        function features = extractFeatures(obj, data, filters)
            try
                if iscell(data)
                    % セル配列の場合
                    numEpochs = length(data);
                    features = zeros(numEpochs, 2 * obj.patterns);
                    for epoch = 1:numEpochs
                        features(epoch, :) = obj.computeFeatures(data{epoch}, filters);
                    end

                elseif ndims(data) == 3
                    % 3次元配列の場合
                    numEpochs = size(data, 3);
                    features = zeros(numEpochs, 2 * obj.patterns);
                    for epoch = 1:numEpochs
                        features(epoch, :) = obj.computeFeatures(data(:,:,epoch), filters);
                    end

                elseif ismatrix(data)
                    % 2次元配列の場合（オンライン処理）
                    features(1, :) = obj.computeFeatures(data, filters);

                else
                    error('Unsupported data format. Expected cell array, 3D array, or 2D array.');
                end

            catch ME
                error('Feature extraction failed: %s', ME.message);
            end
        end
    end
    
    methods (Access = private)
        function classData = getClassData(~, data, labels, targetClass)
               % 指定クラスのデータを抽出
               if iscell(data)
                   classIdx = labels == targetClass;
                   classData = data(classIdx);
               else
                   classIdx = labels == targetClass;
                   classData = data(:,:,classIdx);
               end
           end
        
        function covMatrix = calculateCovarianceMatrix(~, data)
            % 共分散行列の計算
            if iscell(data)
                numTrials = length(data);
                [numChannels, ~] = size(data{1});
                covMatrix = zeros(numChannels);
                
                for trial = 1:numTrials
                    normalizedData = data{trial} ./ trace(data{trial} * data{trial}');
                    covMatrix = covMatrix + normalizedData * normalizedData';
                end
                
                covMatrix = covMatrix / numTrials;
            else
                [numChannels, ~, numTrials] = size(data);
                covMatrix = zeros(numChannels);
                
                for trial = 1:numTrials
                    normalizedData = data(:,:,trial) ./ trace(data(:,:,trial) * data(:,:,trial)');
                    covMatrix = covMatrix + normalizedData * normalizedData';
                end
                
                covMatrix = covMatrix / numTrials;
            end
        end
        
        function [filters, parameters] = computeCSPFilters(obj, covMatrices)
            % 正則化項の追加
            covMatrix1 = covMatrices{1} + obj.regParam * eye(size(covMatrices{1}));
            covMatrix2 = covMatrices{2} + obj.regParam * eye(size(covMatrices{2}));
            
            % 一般化固有値問題の解法
            [eigVectors, eigValues] = eig(covMatrix1, covMatrix1 + covMatrix2);
            
            % 固有値の並べ替え
            [~, sortIdx] = sort(diag(eigValues), 'descend');
            eigVectors = eigVectors(:, sortIdx);
            
            % フィルタの選択
            numFilters = min(obj.patterns, floor(size(eigVectors, 2)/2));
            filters = [eigVectors(:, 1:numFilters), eigVectors(:, end-numFilters+1:end)];
            
            % パラメータの保存
            parameters = struct(...
                'eigenValues', eigValues, ...
                'numFilters', numFilters, ...
                'regularization', obj.regParam);
        end
        
        function features = computeFeatures(~, data, cspFilters)
            try
                % データにCSPフィルタを適用
                filteredData = cspFilters' * data;

                % 特徴量の計算（フィルターされた信号の対数バリアンス）
                features = log(var(filteredData, 0, 2))';  % 直接行ベクトルとして計算
                
            catch ME
                error('Feature computation failed: %s\nStack trace:\n%s', ...
                    ME.message, getReport(ME, 'extended'));
            end
        end
    end
end