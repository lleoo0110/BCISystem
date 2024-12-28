classdef FAAVisualizer
    % FAA解析結果の可視化
    % 使用例：FAAVisualizer.visualize('filename.mat');
    
    methods (Static)
        function visualize(filename)
            % DataLoaderを使用してデータ読み込み
            loadedData = DataLoader.loadDataBrowserWithPrompt('FAA解析');
            
            % FAA結果の検証
            if ~isfield(loadedData, 'results') || ~isfield(loadedData.results, 'faa')
                error('FAA結果が見つかりませんでした');
            end
            
            % FAA値の抽出
            faaValues = arrayfun(@(x) x.faa, loadedData.results.faa);
            arousalStates = arrayfun(@(x) strcmp(x.arousal, 'aroused'), loadedData.results.faa);
            
            % プロットの初期化
            figure('Name', 'FAA解析結果', 'Position', [100 100 1200 800]);
            
            % FAA値のヒストグラムと統計情報
            subplot(2, 2, 1);
            histogram(faaValues, 20, 'FaceColor', 'blue', 'EdgeColor', 'black');
            title('FAA値の分布');
            xlabel('FAA値');
            ylabel('頻度');
            
            % 統計情報の追加
            stats_text = sprintf(...
                '平均値: %.3f\n最大値: %.3f\n最小値: %.3f\n標準偏差: %.3f', ...
                mean(faaValues), max(faaValues), min(faaValues), std(faaValues));
            text(0.05, 0.95, stats_text, 'Units', 'normalized', 'VerticalAlignment', 'top');
            
            % FAA値と覚醒状態の関係
            subplot(2, 2, 2);
            scatter(faaValues, arousalStates, 50, 'filled', 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'black');
            title('FAA値と覚醒状態の関係');
            xlabel('FAA値');
            ylabel('覚醒状態 (0: 非覚醒, 1: 覚醒)');
            ylim([-0.1 1.1]);
            
            % FAA値のボックスプロット
            subplot(2, 2, 3);
            boxplot(faaValues);
            title('FAA値のボックスプロット');
            ylabel('FAA値');
            
            % 覚醒状態の分布
            subplot(2, 2, 4);
            arousalCounts = [sum(~arousalStates), sum(arousalStates)];
            pie(arousalCounts, {'非覚醒', '覚醒'});
            title('覚醒状態の分布');
            
            % 結果の保存
            obj.saveResults(filename, faaValues, arousalStates);
        end
        
        function saveResults(filename, faaValues, arousalStates)
            % 結果の整形と保存
            [filepath, name, ~] = fileparts(filename);
            outputDir = fullfile(filepath, [name '_FAA_results']);
            
            % 出力ディレクトリの作成
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
            
            % 図の保存
            saveas(gcf, fullfile(outputDir, 'FAA_analysis.png'));
            
            % 数値データのCSV保存
            resultTable = table(...
                faaValues, ...
                arousalStates, ...
                'VariableNames', {'FAA_Value', 'Arousal_State'});
            writetable(resultTable, fullfile(outputDir, 'FAA_data.csv'));
            
            % 統計情報のテキスト保存
            fid = fopen(fullfile(outputDir, 'FAA_statistics.txt'), 'w');
            fprintf(fid, 'FAA解析結果\n');
            fprintf(fid, '-------------\n');
            fprintf(fid, '平均値: %.3f\n', mean(faaValues));
            fprintf(fid, '最大値: %.3f\n', max(faaValues));
            fprintf(fid, '最小値: %.3f\n', min(faaValues));
            fprintf(fid, '標準偏差: %.3f\n', std(faaValues));
            fprintf(fid, '\n覚醒状態\n');
            fprintf(fid, '非覚醒: %d\n', sum(~arousalStates));
            fprintf(fid, '覚醒: %d\n', sum(arousalStates));
            fclose(fid);
            
            fprintf('FAA解析結果を保存しました: %s\n', outputDir);
        end
    end
end