classdef EmotionVisualizer
    % 感情解析結果の可視化
    % 使用例：EmotionVisualizer.visualize('filename.mat');
    
    methods (Static)
        function visualize(filename)
            % DataLoaderを使用してデータ読み込み
            loadedData = DataLoader.loadDataBrowserWithPrompt('感情解析');
            
            % 感情結果の検証
            if ~isfield(loadedData, 'results') || ~isfield(loadedData.results, 'emotion')
                error('感情結果が見つかりませんでした');
            end
            
            % 感情座標の抽出
            valence = arrayfun(@(x) x.coordinates.valence, loadedData.results.emotion);
            arousal = arrayfun(@(x) x.coordinates.arousal, loadedData.results.emotion);
            emotionStates = {loadedData.results.emotion.state};
            
            % プロットの初期化
            figure('Name', '感情解析結果', 'Position', [100 100 1600 1000]);
            
            % 2D感情マップ
            subplot(2, 3, 1);
            scatter(valence, arousal, 50, 'filled', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'black');
            title('感情座標マップ');
            xlabel('価（Valence）');
            ylabel('覚醒度（Arousal）');
            grid on;
            axis([-1 1 -1 1]);
            
            % 感情状態の分布
            subplot(2, 3, 2);
            uniqueStates = unique(emotionStates);
            stateCounts = cellfun(@(x) sum(strcmp(x, emotionStates)), uniqueStates);
            bar(categorical(uniqueStates), stateCounts, 'FaceColor', 'green', 'EdgeColor', 'black');
            title('感情状態の分布');
            xlabel('感情状態');
            ylabel('頻度');
            xtickangle(45);
            
            % 価（Valence）のヒストグラム
            subplot(2, 3, 3);
            histogram(valence, 20, 'FaceColor', 'red', 'EdgeColor', 'black');
            title('価（Valence）の分布');
            xlabel('価（Valence）');
            ylabel('頻度');
            
            % 覚醒度（Arousal）のヒストグラム
            subplot(2, 3, 4);
            histogram(arousal, 20, 'FaceColor', 'magenta', 'EdgeColor', 'black');
            title('覚醒度（Arousal）の分布');
            xlabel('覚醒度（Arousal）');
            ylabel('頻度');
            
            % 感情座標の密度マップ
            subplot(2, 3, 5);
            [X, Y] = meshgrid(linspace(-1, 1, 50), linspace(-1, 1, 50));
            Z = zeros(size(X));
            for i = 1:numel(X)
                distances = sqrt((valence - X(i)).^2 + (arousal - Y(i)).^2);
                Z(i) = sum(distances < 0.2);  % 近傍点のカウント
            end
            imagesc(X(1,:), Y(:,1), Z);
            colorbar;
            title('感情密度マップ');
            xlabel('価（Valence）');
            ylabel('覚醒度（Arousal）');
            
            % 感情状態の詳細分布
            subplot(2, 3, 6);
            [counts, centers] = hist3([valence', arousal'], {linspace(-1,1,20), linspace(-1,1,20)});
            imagesc(centers{1}, centers{2}, counts');
            colorbar;
            title('感情状態の2D分布');
            xlabel('価（Valence）');
            ylabel('覚醒度（Arousal）');
            
            % 結果の保存
            obj.saveResults(filename, valence, arousal, emotionStates);
        end
        
        function saveResults(filename, valence, arousal, emotionStates)
            % 結果の整形と保存
            [filepath, name, ~] = fileparts(filename);
            outputDir = fullfile(filepath, [name '_Emotion_results']);
            
            % 出力ディレクトリの作成
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
            
            % 図の保存
            saveas(gcf, fullfile(outputDir, 'Emotion_analysis.png'));
            
            % 数値データのCSV保存
            resultTable = table(...
                valence', ...
                arousal', ...
                emotionStates', ...
                'VariableNames', {'Valence', 'Arousal', 'Emotion_State'});
            writetable(resultTable, fullfile(outputDir, 'Emotion_data.csv'));
            
            % 統計情報のテキスト保存
            fid = fopen(fullfile(outputDir, 'Emotion_statistics.txt'), 'w');
            fprintf(fid, '感情解析結果\n');
            fprintf(fid, '-------------\n');
            fprintf(fid, '価（Valence）\n');
            fprintf(fid, '  平均値: %.3f\n', mean(valence));
            fprintf(fid, '  最大値: %.3f\n', max(valence));
            fprintf(fid, '  最小値: %.3f\n', min(valence));
            fprintf(fid, '  標準偏差: %.3f\n\n', std(valence));
            
            fprintf(fid, '覚醒度（Arousal）\n');
            fprintf(fid, '  平均値: %.3f\n', mean(arousal));
            fprintf(fid, '  最大値: %.3f\n', max(arousal));
            fprintf(fid, '  最小値: %.3f\n', min(arousal));
            fprintf(fid, '  標準偏差: %.3f\n\n', std(arousal));
            
            fprintf(fid, '感情状態の分布\n');
            uniqueStates = unique(emotionStates);
            for i = 1:length(uniqueStates)
                count = sum(strcmp(uniqueStates{i}, emotionStates));
                percentage = count / length(emotionStates) * 100;
                fprintf(fid, '  %s: %d (%.2f%%)\n', uniqueStates{i}, count, percentage);
            end
            fclose(fid);
            
            fprintf('感情解析結果を保存しました: %s\n', outputDir);
        end
    end
end