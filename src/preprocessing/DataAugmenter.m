classdef DataAugmenter < handle
    properties (Access = private)
        params              % パラメータ設定
        channelMap         % チャンネル名とインデックスのマッピング
        rng                % 乱数生成器
    end
    
    methods (Access = public)
        function obj = DataAugmenter(params)
            obj.params = params;
            obj.rng = RandStream('mlfg6331_64');
            obj.initializeChannelMap();
        end
        
        function [augData, augLabels, info] = augmentData(obj, data, labels)
           try
               [channels, samples, epochs] = size(data);

               info = struct('originalSize', size(data), 'augmentationApplied', false);

               if ~obj.params.signal.preprocessing.augmentation.enable
                   augData = data;
                   augLabels = labels;
                   return;
               end
               augRatio = obj.params.signal.preprocessing.augmentation.augmentationRatio;
               targetTotalEpochs = round(epochs * (1 + augRatio));
               uniqueLabels = unique(labels);

               labelCounts = zeros(length(uniqueLabels), 1);
               for i = 1:length(uniqueLabels)
                   labelCounts(i) = sum(labels == uniqueLabels(i));
               end
               
               maxCount = max(labelCounts);

               balanceAugPerClass = maxCount - labelCounts;
               disp(balanceAugPerClass);

               remainingAugs = targetTotalEpochs - (epochs + sum(balanceAugPerClass));

               if remainingAugs > 0
                   classRatios = labelCounts / sum(labelCounts);
                   additionalAugs = round(remainingAugs * classRatios);
                   augCountsPerClass = balanceAugPerClass + additionalAugs;
               else
                   augCountsPerClass = balanceAugPerClass;
               end
               disp(augCountsPerClass);

               totalAugmentations = sum(augCountsPerClass);

               augData = zeros(channels, samples, epochs + totalAugmentations);
               augData(:,:,1:epochs) = data;
               augLabels = zeros(epochs + totalAugmentations, 1);
               augLabels(1:epochs) = labels;

               info.augmentationApplied = true;
               info.methodsUsed = cell(totalAugmentations, 1);

               currentIdx = epochs + 1;
               for i = 1:length(uniqueLabels)
                   classLabel = uniqueLabels(i);
                   classEpochs = find(labels == classLabel);
                   numAugNeeded = augCountsPerClass(i);

                   for j = 1:numAugNeeded
                       baseIdx = classEpochs(randi(obj.rng, length(classEpochs)));
                       methods = obj.selectAugmentationMethods();
                       info.methodsUsed{currentIdx-epochs} = methods;
                       disp(methods);

                       augmentedEpoch = data(:,:,baseIdx);
                       for method = methods
                           augmentedEpoch = obj.applyAugmentation(augmentedEpoch, method{1});
                       end

                       augData(:,:,currentIdx) = augmentedEpoch;
                       augLabels(currentIdx) = classLabel;
                       currentIdx = currentIdx + 1;
                   end
               end
               
               info.finalSize = size(augData);
               info.totalAugmentations = totalAugmentations;
               info.augmentationRatio = totalAugmentations / epochs;
               info.classBalance = struct(...
                   'original', labelCounts, ...
                   'added', augCountsPerClass, ...
                   'final', labelCounts + augCountsPerClass);
               info.augmentationDetails = obj.calculateMethodStats(info.methodsUsed);

           catch ME
               fprintf('Error occurred in augmentation: %s\n', ME.message);
               error('DataAugmenter:AugmentationFailed', ...
                   'Data augmentation failed: %s', ME.message);
           end
        end
    end
    
    methods (Access = private)
        function initializeChannelMap(obj)
            channels = obj.params.device.channels;
            obj.channelMap = containers.Map(channels, 1:length(channels));
        end
        
        function methods = selectAugmentationMethods(obj)
            augParams = obj.params.signal.preprocessing.augmentation;
            availableMethods = {'noise', 'scaling', 'timeshift', 'mirror', 'channelSwap'};
            methods = {};
            
            numMethods = randi(obj.rng, augParams.combinationLimit);
            
            enabledMethods = availableMethods(cellfun(@(m) ...
                augParams.methods.(m).enable, availableMethods));
            if ~isempty(enabledMethods)
                shuffledMethods = enabledMethods(randperm(obj.rng, length(enabledMethods)));
                for i = 1:min(numMethods, length(shuffledMethods))
                    method = shuffledMethods{i};
                    if obj.shouldApplyMethod(method)
                        methods{end+1} = method;
                    end
                end
            end
        end
        
        function shouldApply = shouldApplyMethod(obj, method)
            methodParams = obj.params.signal.preprocessing.augmentation.methods.(method);
            shouldApply = methodParams.enable && rand(obj.rng) < methodParams.probability;
        end
        
        function augmented = applyAugmentation(obj, data, method)
            methodParams = obj.params.signal.preprocessing.augmentation.methods.(method);
            
            switch method
                case 'noise'
                    augmented = obj.addNoise(data, methodParams);
                case 'scaling'
                    augmented = obj.applyScaling(data, methodParams);
                case 'timeshift'
                    augmented = obj.applyTimeshift(data, methodParams);
                case 'mirror'
                    augmented = obj.applyMirror(data);
                case 'channelSwap'
                    augmented = obj.swapChannels(data, methodParams);
                otherwise
                    augmented = data;
            end
        end
        
        function augmented = addNoise(obj, data, config)
            if ~config.enable
                augmented = data;
                return;
            end

            noiseTypes = config.types;
            selectedType = noiseTypes{randi(obj.rng, length(noiseTypes))};

            switch selectedType
                case 'gaussian'
                    % ガウシアンノイズ生成
                    noise = obj.rng.randn(size(data)) * sqrt(config.variance);
                    augmented = data + noise;
                case 'pink'
                    % ピンクノイズ生成
                    noise = obj.generatePinkNoise(data) * sqrt(config.variance);
                    augmented = data + noise;
                otherwise
                    augmented = data;
            end
        end
        
        function augmented = applyScaling(obj, data, config)
            if ~config.enable
                augmented = data;
                return;
            end
            
            scale = config.range(1) + (config.range(2) - config.range(1)) * rand(obj.rng);
            augmented = data * scale;
        end
        
        function augmented = applyTimeshift(obj, data, config)
            if ~config.enable
                augmented = data;
                return;
            end
            
            maxShiftSamples = round(config.maxShift * size(data, 2));
            shiftAmount = randi(obj.rng, [-maxShiftSamples, maxShiftSamples]);
            augmented = circshift(data, [0, shiftAmount]);
        end
        
        function augmented = applyMirror(~, data)
            augmented = -data;  % 振幅の反転
        end
        
        function augmented = swapChannels(obj, data, config)
            if ~config.enable || isempty(config.pairs)
                augmented = data;
                return;
            end
            
            augmented = data;
            pairs = config.pairs;
            selectedPair = pairs{randi(obj.rng, length(pairs))};
            
            ch1Idx = obj.channelMap(selectedPair{1});
            ch2Idx = obj.channelMap(selectedPair{2});
            
            temp = augmented(ch1Idx, :);
            augmented(ch1Idx, :) = augmented(ch2Idx, :);
            augmented(ch2Idx, :) = temp;
        end
        
        function noise = generatePinkNoise(obj, dataSize)
            [channels, samples] = size(dataSize);
            noise = zeros(channels, samples);
            
            for ch = 1:channels
                f = (1:samples/2)';
                amplitude = 1./sqrt(f);
                phase = 2*pi*rand(obj.rng, samples/2, 1);
                s = amplitude .* exp(1i*phase);
                s = [0; s; flipud(conj(s(1:end-1)))];
                noise(ch,:) = real(ifft(s));
            end
            noise = noise ./ std(noise(:));
        end
        
        function stats = calculateMethodStats(~, methodsUsed)
            stats = struct();
            allMethods = {'noise', 'scaling', 'timeshift', 'mirror', 'channelSwap'};
            
            for i = 1:length(allMethods)
                method = allMethods{i};
                count = sum(cellfun(@(x) any(strcmp(x, method)), methodsUsed));
                stats.(method) = struct(...
                    'count', count, ...
                    'percentage', count / length(methodsUsed) * 100);
            end
        end
    end
end