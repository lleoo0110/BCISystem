classdef EEGDataFormatter
    % データの整形と出力
    % 使用例：EEGDataFormatter.exportProcessedData('filename.mat');

    % 包括的レポートの生成
    % 使用例：EEGDataFormatter.generateComprehensiveReport('filename.mat');

    % 完全なデータパッケージの出力
    % 使用例：EEGDataFormatter.exportFullDataPackage('filename.mat');

    methods (Static)
        function exportProcessedData(filename)
            % DataLoaderを使用してデータ読み込み
            loadedData = DataLoader.loadDataBrowserWithPrompt('データ整形と出力');
            
            % 出力ディレクトリの作成
            [filepath, name, ~] = fileparts(filename);
            outputDir = fullfile(filepath, [name '_formatted_data']);
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
            
            % 処理済みデータの出力
            obj.exportRawData(loadedData, outputDir);
            obj.exportProcessedSignals(loadedData, outputDir);
            obj.exportFeatureData(loadedData, outputDir);
            obj.exportClassificationResults(loadedData, outputDir);
            
            fprintf('データの整形と出力が完了しました: %s\n', outputDir);
        end
        
        function exportRawData(loadedData, outputDir)
            % 生データの出力
            if isfield(loadedData, 'rawData')
                rawData = loadedData.rawData;
                
                % CSVファイルに保存
                rawDataTable = array2table(rawData');
                writetable(rawDataTable, fullfile(outputDir, 'raw_eeg_data.csv'));
                
                % MATファイルに保存（チャンネル名を含む）
                if isfield(loadedData, 'params') && isfield(loadedData.params, 'device')
                    channelNames = loadedData.params.device.channels;
                    save(fullfile(outputDir, 'raw_eeg_data.mat'), 'rawData', 'channelNames');
                end
            end
        end
        
        function exportProcessedSignals(loadedData, outputDir)
            % 前処理済みデータの出力
            if isfield(loadedData, 'processedData')
                processedData = loadedData.processedData;
                
                % データサイズに応じて処理
                if ndims(processedData) == 3
                    % エポックデータの場合
                    for epoch = 1:size(processedData, 3)
                        epochData = processedData(:, :, epoch);
                        epochTable = array2table(epochData');
                        writetable(epochTable, fullfile(outputDir, sprintf('processed_epoch_%d.csv', epoch)));
                    end
                else
                    % 2Dデータの場合
                    processedTable = array2table(processedData');
                    writetable(processedTable, fullfile(outputDir, 'processed_signals.csv'));
                end
            end
        end
        
        function exportFeatureData(loadedData, outputDir)
            % 特徴量データの出力
            if isfield(loadedData, 'results')
                % パワー特徴量
                if isfield(loadedData.results, 'power')
                    bandNames = fieldnames(loadedData.results.power.bands);
                    powerTable = array2table(...
                        cell2mat(cellfun(@(band) loadedData.results.power.bands.(band), ...
                        bandNames, 'UniformOutput', false)), ...
                        'VariableNames', bandNames);
                    writetable(powerTable, fullfile(outputDir, 'power_features.csv'));
                end
                
                % FAA特徴量
                if isfield(loadedData.results, 'faa')
                    faaTable = struct2table(loadedData.results.faa);
                    writetable(faaTable, fullfile(outputDir, 'faa_features.csv'));
                end
                
                % α/β比特徴量
                if isfield(loadedData.results, 'abRatio')
                    abRatioTable = struct2table(loadedData.results.abRatio);
                    writetable(abRatioTable, fullfile(outputDir, 'ab_ratio_features.csv'));
                end
                
                % 感情特徴量
                if isfield(loadedData.results, 'emotion')
                    emotionTable = struct2table(loadedData.results.emotion);
                    writetable(emotionTable, fullfile(outputDir, 'emotion_features.csv'));
                end
                
                % CSP特徴量
                if isfield(loadedData.results, 'csp') && isfield(loadedData.results.csp, 'features')
                    cspTable = array2table(loadedData.results.csp.features, ...
                        'VariableNames', arrayfun(@(x) sprintf('Feature_%d', x), ...
                        1:size(loadedData.results.csp.features, 2), 'UniformOutput', false));
                    writetable(cspTable, fullfile(outputDir, 'csp_features.csv'));
                end
            end
        end
        
        function exportClassificationResults(loadedData, outputDir)
            % 分類結果の出力
            if isfield(loadedData, 'classifier')
                % 各分類器の結果を保存
                classifiers = {'svm', 'ecoc', 'cnn'};
                
                for i = 1:length(classifiers)
                    classifier = classifiers{i};
                    if isfield(loadedData.classifier, classifier)
                        % パフォーマンス情報の保存
                        performanceStruct = loadedData.classifier.(classifier).performance;
                        
                        % テキストファイルに詳細情報を保存
                        fid = fopen(fullfile(outputDir, sprintf('%s_classification_results.txt', classifier)), 'w');
                        
                        % 分類器の性能情報を書き出し
                        fprintf(fid, '%s分類器の性能\n', upper(classifier));
                        fprintf(fid, '------------------------\n');
                        
                        % 一般的な性能指標
                        if isfield(performanceStruct, 'overallAccuracy')
                            fprintf(fid, '全体の正解率: %.2f%%\n', performanceStruct.overallAccuracy * 100);
                        end
                        
                        % 交差検証結果
                        if isfield(performanceStruct, 'crossValidation')
                            cv = performanceStruct.crossValidation;
                            fprintf(fid, '交差検証結果:\n');
                            fprintf(fid, '  平均精度: %.2f%%\n', cv.accuracy * 100);
                            fprintf(fid, '  標準偏差: %.2f%%\n', cv.std * 100);
                        end
                        
                        % クラス別の性能指標
                        if isfield(performanceStruct, 'precision')
                            fprintf(fid, '\nクラス別精度:\n');
                            for j = 1:length(performanceStruct.precision)
                                fprintf(fid, '  クラス %d:\n', j);
                                fprintf(fid, '    精度: %.2f%%\n', performanceStruct.precision(j) * 100);
                                fprintf(fid, '    再現率: %.2f%%\n', performanceStruct.recall(j) * 100);
                                fprintf(fid, '    F1スコア: %.2f%%\n', performanceStruct.f1score(j) * 100);
                            end
                        end
                        
                        % 混同行列
                        if isfield(performanceStruct, 'confusionMatrix')
                            fprintf(fid, '\n混同行列:\n');
                            confusionMat = performanceStruct.confusionMatrix;
                            for r = 1:size(confusionMat, 1)
                                fprintf(fid, '%s\n', num2str(confusionMat(r, :)));
                            end
                        end
                        
                        fclose(fid);
                        
                        % 混同行列のCSVエクスポート
                        if isfield(performanceStruct, 'confusionMatrix')
                            confusionTable = array2table(performanceStruct.confusionMatrix);
                            writetable(confusionTable, fullfile(outputDir, sprintf('%s_confusion_matrix.csv', classifier)));
                        end
                    end
                end
            end
        end
        
        function generateComprehensiveReport(filename)
            % DataLoaderを使用してデータ読み込み
            loadedData = DataLoader.loadDataBrowserWithPrompt('包括的レポート生成');
            
            % 出力ディレクトリの作成
            [filepath, name, ~] = fileparts(filename);
            outputDir = fullfile(filepath, [name '_comprehensive_report']);
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
            
            % レポートファイルの作成
            reportFilePath = fullfile(outputDir, 'comprehensive_eeg_analysis_report.md');
            fid = fopen(reportFilePath, 'w');
            
            % レポートヘッダー
            fprintf(fid, '# EEG解析包括レポート\n\n');
            fprintf(fid, '## 基本情報\n');
            fprintf(fid, '- 解析日時: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            
            % デバイス情報
            if isfield(loadedData, 'params') && isfield(loadedData.params, 'device')
                device = loadedData.params.device;
                fprintf(fid, '\n## デバイス情報\n');
                fprintf(fid, '- デバイス名: %s\n', device.name);
                fprintf(fid, '- チャンネル数: %d\n', device.channelCount);
                fprintf(fid, '- サンプリングレート: %d Hz\n', device.sampleRate);
                fprintf(fid, '- チャンネル: %s\n', strjoin(device.channels, ', '));
            end
            
            % データ前処理情報
            if isfield(loadedData, 'processingInfo')
                fprintf(fid, '\n## 前処理情報\n');
                processingInfo = loadedData.processingInfo;
                
                % ダウンサンプリング
                if isfield(processingInfo, 'downsample')
                    fprintf(fid, '### ダウンサンプリング\n');
                    fprintf(fid, '- 目標サンプリングレート: %d Hz\n', processingInfo.downsample.targetRate);
                end
                
                % アーティファクト除去
                if isfield(processingInfo, 'artifact')
                    fprintf(fid, '### アーティファクト除去\n');
                    fprintf(fid, '- 方法: %s\n', processingInfo.artifact.method);
                end
                
                % フィルタリング
                if isfield(processingInfo, 'firFilter')
                    fprintf(fid, '### FIRフィルタリング\n');
                    fprintf(fid, '- フィルタ次数: %d\n', processingInfo.firFilter.filterOrder);
                    fprintf(fid, '- 設計方法: %s\n', processingInfo.firFilter.designMethod);
                end
            end
            
            % 特徴量解析情報
            if isfield(loadedData, 'results')
                fprintf(fid, '\n## 特徴量解析\n');
                
                % パワー特徴量
                if isfield(loadedData.results, 'power')
                    fprintf(fid, '### パワースペクトル解析\n');
                    bandNames = fieldnames(loadedData.results.power.bands);
                    for i = 1:length(bandNames)
                        bandName = bandNames{i};
                        bandPower = loadedData.results.power.bands.(bandName);
                        fprintf(fid, '- %sバンド:\n', bandName);
                        fprintf(fid, '  - 平均パワー: %.4f\n', mean(bandPower));
                        fprintf(fid, '  - 最大パワー: %.4f\n', max(bandPower));
                        fprintf(fid, '  - 最小パワー: %.4f\n', min(bandPower));
                    end
                end
                
                % FAA特徴量
                if isfield(loadedData.results, 'faa')
                    fprintf(fid, '### 前頭葉非対称性 (FAA) 解析\n');
                    faaValues = arrayfun(@(x) x.faa, loadedData.results.faa);
                    fprintf(fid, '- 平均FAA値: %.4f\n', mean(faaValues));
                    fprintf(fid, '- 覚醒状態分布:\n');
                    arousalStates = {loadedData.results.faa.arousal};
                    uniqueStates = unique(arousalStates);
                    for j = 1:length(uniqueStates)
                        count = sum(strcmp(arousalStates, uniqueStates{j}));
                        fprintf(fid, '  - %s: %d\n', uniqueStates{j}, count);
                    end
                end
                
                % 感情解析
                if isfield(loadedData.results, 'emotion')
                    fprintf(fid, '### 感情解析\n');
                    emotionStates = {loadedData.results.emotion.state};
                    uniqueEmotions = unique(emotionStates);
                    fprintf(fid, '- 感情状態分布:\n');
                    for j = 1:length(uniqueEmotions)
                        count = sum(strcmp(emotionStates, uniqueEmotions{j}));
                        fprintf(fid, '  - %s: %d\n', uniqueEmotions{j}, count);
                    end
                end
            end
            
            % 分類器性能
            if isfield(loadedData, 'classifier')
                fprintf(fid, '\n## 分類器性能\n');
                classifiers = {'svm', 'ecoc', 'cnn'};
                
                for i = 1:length(classifiers)
                    classifier = classifiers{i};
                    if isfield(loadedData.classifier, classifier)
                        performance = loadedData.classifier.(classifier).performance;
                        
                        fprintf(fid, '### %s分類器\n', upper(classifier));
                        if isfield(performance, 'overallAccuracy')
                            fprintf(fid, '- 全体の正解率: %.2f%%\n', performance.overallAccuracy * 100);
                        end
                        
                        if isfield(performance, 'crossValidation')
                            cv = performance.crossValidation;
                            fprintf(fid, '- 交差検証結果:\n');
                            fprintf(fid, '  - 平均精度: %.2f%%\n', cv.accuracy * 100);
                            fprintf(fid, '  - 標準偏差: %.2f%%\n', cv.std * 100);
                        end
                    end
                end
            end
            
            fclose(fid);
            
            % レポートのプレビューと保存場所の表示
            fprintf('包括的レポートを生成しました: %s\n', reportFilePath);
            type(reportFilePath);
        end
        
        % CSV, MATファイル, Markdownレポートを統合して出力するメソッド
        function exportFullDataPackage(filename)
            % DataLoaderを使用してデータ読み込み
            loadedData = DataLoader.loadDataBrowserWithPrompt('包括的データパッケージ出力');
            
            % 出力ディレクトリの作成
            [filepath, name, ~] = fileparts(filename);
            outputDir = fullfile(filepath, [name '_full_data_package']);
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
            
            % すべてのデータエクスポートメソッドを呼び出し
            obj.exportRawData(loadedData, outputDir);
            obj.exportProcessedSignals(loadedData, outputDir);
            obj.exportFeatureData(loadedData, outputDir);
            obj.exportClassificationResults(loadedData, outputDir);
            
            % 包括的レポートの生成
            reportFilePath = fullfile(outputDir, 'comprehensive_eeg_analysis_report.md');
            
            % レポートファイルの作成
            fid = fopen(reportFilePath, 'w');
            
            % レポートヘッダー
            fprintf(fid, '# EEG解析包括レポート\n\n');
            fprintf(fid, '## 基本情報\n');
            fprintf(fid, '- 解析日時: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            
            % デバイス情報
            if isfield(loadedData, 'params') && isfield(loadedData.params, 'device')
                device = loadedData.params.device;
                fprintf(fid, '\n## デバイス情報\n');
                fprintf(fid, '- デバイス名: %s\n', device.name);
                fprintf(fid, '- チャンネル数: %d\n', device.channelCount);
                fprintf(fid, '- サンプリングレート: %d Hz\n', device.sampleRate);
                fprintf(fid, '- チャンネル: %s\n', strjoin(device.channels, ', '));
            end
            
            % 出力されたファイルの一覧
            fprintf(fid, '\n## エクスポートされたファイル\n');
            exportedFiles = dir(outputDir);
            exportedFiles = exportedFiles(~[exportedFiles.isdir]);
            
            fprintf(fid, '| ファイル名 | サイズ (KB) |\n');
            fprintf(fid, '|-----------|-------------|\n');
            
            for i = 1:length(exportedFiles)
                fprintf(fid, '| %s | %.2f |\n', ...
                    exportedFiles(i).name, ...
                    exportedFiles(i).bytes / 1024);
            end
            
            % データ保存の詳細を追加
            fprintf(fid, '\n## データパッケージ情報\n');
            fprintf(fid, '- 出力ディレクトリ: %s\n', outputDir);
            
            fclose(fid);
            
            % 圧縮ファイルの作成
            zipFilePath = fullfile(filepath, [name '_full_data_package.zip']);
            zip(zipFilePath, outputDir);
            
            fprintf('データパッケージを生成しました:\n');
            fprintf('- 出力ディレクトリ: %s\n', outputDir);
            fprintf('- ZIPファイル: %s\n', zipFilePath);
            
            % レポートのプレビュー
            type(reportFilePath);
        end
        
        % 解析結果の可視化と統計情報生成
        function visualizeAndAnalyzeResults(filename)
            % DataLoaderを使用してデータ読み込み
            loadedData = DataLoader.loadDataBrowserWithPrompt('解析結果の可視化');
            
            % 出力ディレクトリの作成
            [filepath, name, ~] = fileparts(filename);
            outputDir = fullfile(filepath, [name '_analysis_visualization']);
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
            
            % 特徴量可視化
            figHandles = [];
            
            % パワースペクトル解析
            if isfield(loadedData, 'results') && isfield(loadedData.results, 'power')
                figHandles(end+1) = figure('Name', 'パワースペクトル解析', 'Position', [100, 100, 1200, 800]);
                bandNames = fieldnames(loadedData.results.power.bands);
                powerValues = cell2mat(cellfun(@(band) loadedData.results.power.bands.(band), ...
                    bandNames, 'UniformOutput', false));
                
                subplot(2,2,1);
                boxplot(powerValues, 'Labels', bandNames);
                title('周波数帯域別パワー分布');
                ylabel('パワー値 (μV²/Hz)');
                xtickangle(45);
                
                subplot(2,2,2);
                bar(categorical(bandNames), mean(powerValues));
                title('平均パワー値');
                ylabel('平均パワー (μV²/Hz)');
                
                subplot(2,2,3);
                heatmap(bandNames, bandNames, corr(powerValues), ...
                    'ColorbarVisible', 'on', 'Title', '帯域間相関');
                
                subplot(2,2,4);
                histogram(powerValues(:));
                title('全帯域パワー値分布');
                xlabel('パワー値 (μV²/Hz)');
                ylabel('頻度');
            end
            
            % FAA解析
            if isfield(loadedData, 'results') && isfield(loadedData.results, 'faa')
                figHandles(end+1) = figure('Name', 'FAA解析', 'Position', [100, 100, 1200, 800]);
                faaValues = arrayfun(@(x) x.faa, loadedData.results.faa);
                arousalStates = {loadedData.results.faa.arousal};
                
                subplot(2,2,1);
                histogram(faaValues);
                title('FAA値の分布');
                xlabel('FAA値');
                ylabel('頻度');
                
                subplot(2,2,2);
                states = unique(arousalStates);
                counts = cellfun(@(x) sum(strcmp(arousalStates, x)), states);
                pie(counts, states);
                title('覚醒状態の分布');
                
                subplot(2,2,3);
                scatter(faaValues, strcmp(arousalStates, 'aroused'));
                title('FAA値と覚醒状態');
                xlabel('FAA値');
                ylabel('覚醒状態');
                
                subplot(2,2,4);
                boxplot(faaValues);
                title('FAA値のボックスプロット');
            end
            
            % 感情解析
            if isfield(loadedData, 'results') && isfield(loadedData.results, 'emotion')
                figHandles(end+1) = figure('Name', '感情解析', 'Position', [100, 100, 1200, 800]);
                valence = arrayfun(@(x) x.coordinates.valence, loadedData.results.emotion);
                arousal = arrayfun(@(x) x.coordinates.arousal, loadedData.results.emotion);
                emotionStates = {loadedData.results.emotion.state};
                
                subplot(2,2,1);
                scatter(valence, arousal);
                title('感情座標マップ');
                xlabel('価（Valence）');
                ylabel('覚醒度（Arousal）');
                
                subplot(2,2,2);
                uniqueStates = unique(emotionStates);
                counts = cellfun(@(x) sum(strcmp(emotionStates, x)), uniqueStates);
                bar(categorical(uniqueStates), counts);
                title('感情状態の分布');
                xtickangle(45);
                
                subplot(2,2,3);
                histogram(valence);
                title('価（Valence）の分布');
                xlabel('価の値');
                
                subplot(2,2,4);
                histogram(arousal);
                title('覚醒度（Arousal）の分布');
                xlabel('覚醒度の値');
            end
            
            % 図の保存
            for i = 1:length(figHandles)
                saveas(figHandles(i), fullfile(outputDir, sprintf('analysis_visualization_%d.png', i)));
            end
            
            fprintf('解析結果の可視化が完了しました。\n保存先: %s\n', outputDir);
        end
    end
end