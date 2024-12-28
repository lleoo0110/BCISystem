classdef PowerVisualizer
    % パワースペクトル解析の可視化
    % 使用例：PowerVisualizer.visualize('filename.mat');
    
    methods (Static)
        function visualize(filename)
            % DataLoaderを使用してデータ読み込み
            loadedData = DataLoader.loadDataBrowserWithPrompt('パワー解析');
            
            % パワー結果の検証
            if ~isfield(loadedData, 'results') || ~isfield(loadedData.results, 'power')
                error('パワー結果が見つかりませんでした');
            end
            
            % パワーバンドの取得
            bandNames = fieldnames(loadedData.results.power.bands);
            
            % 各バンドのパワー値を抽出
            powerValues = zeros(size(loadedData.processedData, 3), length(bandNames));
            for b = 1:length(bandNames)
                powerValues(:, b) = loadedData.results.power.bands.(bandNames{b});
            end
            
            % プロットの初期化
            figure('Name', 'パワー解析結果', 'Position', [100 100 1600 1000]);
            
            % バンドごとのパワー値ボックスプロット
            subplot(2, 3, 1);
            boxplot(powerValues, bandNames);
            title('周波数帯域別パワー分布');
            ylabel('パワー値 (μV²/Hz)');
            xtickangle(45);
            
            % 統計情報の表示
            subplot(2, 3, 2);
            meanValues = mean(powerValues);
            stdValues = std(powerValues);
            
            bar(categorical(bandNames), meanValues, 'FaceColor', 'blue', 'EdgeColor', 'black');
            title('各周波数帯域の平均パワー');
            xlabel('周波数帯域');
            ylabel('平均パワー値 (μV²/Hz)');
            
            % エラーバーの追加
            hold on;
            errorbar(1:length(bandNames), meanValues, stdValues, 'k.', 'LineWidth', 1.5);
            
            % 相対パワーの計算と可視化
            subplot(2, 3, 3);
            totalPower = sum(powerValues, 2);
            relativePower = powerValues ./ totalPower * 100;
            
            bar(categorical(bandNames), mean(relativePower), 'FaceColor', 'green', 'EdgeColor', 'black');
            title('相対パワー分布');
            xlabel('周波数帯域');
            ylabel('相対パワー (%)');
            
            % パワー値の時系列変化
            subplot(2, 3, 4);
            plot(powerValues, 'LineWidth', 1.5);
            title('パワー値の時系列変化');
            xlabel('エポック番号');
            ylabel('パワー値 (μV²/Hz)');
            legend(bandNames, 'Location', 'best');
            
            % パワー値間の相関
            subplot(2, 3, 5);
            corrMatrix = corr(powerValues);
            imagesc(corrMatrix);
            colorbar;
            title('周波数帯域間の相関');
            xticks(1:length(bandNames));
            yticks(1:length(bandNames));
            xticklabels(bandNames);
            yticklabels(bandNames);
            
            % パワー値の分布ヒストグラム
            subplot(2, 3, 6);
            for b = 1:length(bandNames)
                histogram(powerValues(:, b), 20, 'DisplayName', bandNames{b}, 'FaceAlpha', 0.5);
                hold on;
            end
            title('パワー値の分布');
            xlabel('パワー値 (μV²/Hz)');
            ylabel('頻度');
            legend('show');
            
            % 結果の保存
            obj.saveResults(filename, powerValues, bandNames);
        end
        
        function saveResults(filename, powerValues, bandNames)
            % 結果の整形と保存
            [filepath, name, ~] = fileparts(filename);
            outputDir = fullfile(filepath, [name '_Power_results']);
            
            % 出力ディレクトリの作成
            if ~exist(outputDir, 'dir')
                mkdir(outputDir);
            end
            
            % 図の保存
            saveas(gcf, fullfile(outputDir, 'Power_analysis.png'));
            
            % 数値データのCSV保存
            resultTable = array2table(powerValues, 'VariableNames', bandNames);
            writetable(resultTable, fullfile(outputDir, 'Power_data.csv'));
            
            % 統計情報のテキスト保存
            fid = fopen(fullfile(outputDir, 'Power_statistics.txt'), 'w');
            fprintf(fid, 'パワー解析結果\n');
            fprintf(fid, '-------------\n');
            
            % 総パワー
            totalPower = sum(powerValues, 2);
            
            % 各周波数帯域の統計情報
            fprintf(fid, '周波数帯域別統計情報\n');
            for b = 1:length(bandNames)
                fprintf(fid, '%sバンド:\n', bandNames{b});
                fprintf(fid, '  平均パワー値: %.4f\n', mean(powerValues(:, b)));
                fprintf(fid, '  最大パワー値: %.4f\n', max(powerValues(:, b)));
                fprintf(fid, '  最小パワー値: %.4f\n', min(powerValues(:, b)));
                fprintf(fid, '  標準偏差: %.4f\n', std(powerValues(:, b)));
                
                % 相対パワーの計算
                relativePower = mean(powerValues(:, b) ./ totalPower * 100);
                fprintf(fid, '  相対パワー: %.2f%%\n\n', relativePower);
            }
            
            % 総合統計情報
            fprintf(fid, '総合統計情報\n');
            fprintf(fid, '平均総パワー: %.4f\n', mean(totalPower));
            fprintf(fid, '最大総パワー: %.4f\n', max(totalPower));
            fprintf(fid, '最小総パワー: %.4f\n', min(totalPower));
            
            fclose(fid);
            
            fprintf('パワー解析結果を保存しました: %s\n', outputDir);
        end
    end
end