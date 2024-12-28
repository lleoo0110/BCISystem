classdef DataLoader
    methods (Static)
        function loadedData = loadDataBrowserWithPrompt(purpose)
            try
                % ダイアログタイトルの設定
                switch purpose
                    case 'normalization'
                        dialogTitle = '正規化パラメータ計算用のデータを選択';
                        promptMessage = '正規化パラメータ計算';
                    case 'csp'
                        dialogTitle = 'CSPフィルタ用のデータを選択';
                        promptMessage = 'CSPフィルタ計算';
                    case 'baseline'
                        dialogTitle = 'ベースライン用のデータを選択';
                        promptMessage = 'ベースライン計算';
                    case 'classifier'
                        dialogTitle = '分類器用のデータを選択';
                        promptMessage = '分類器データ';
                    otherwise
                        dialogTitle = 'データファイルを選択';
                        promptMessage = 'データ読み込み';
                end

                choice = questdlg(sprintf('%s用のデータを読み込みますか？', promptMessage), ...
                    'データ読み込みの確認', 'はい', 'いいえ', 'はい');

                if strcmp(choice, 'はい')
                    [filename, pathname] = uigetfile({'*.mat', 'MAT-files (*.mat)'}, dialogTitle);
                    
                    if filename ~= 0
                        fullpath = fullfile(pathname, filename);
                        loadedData = load(fullpath);
                        fprintf('%s用のデータを読み込みました: %s\n', purpose, fullpath);
                        
                        % データの検証
                        DataLoader.validateLoadedData(loadedData, purpose);
                        return;
                    else
                        error('ファイルが選択されませんでした．');
                    end
                else
                    error('ユーザーによってキャンセルされました．');
                end
                
            catch ME
                errordlg(sprintf('データ読み込みエラー: %s', ME.message), 'エラー');
                rethrow(ME);
            end
        end
        
        function validateLoadedData(loadedData, purpose)
            % 読み込んだデータの検証
            switch purpose
                case 'normalization'
                    if ~isfield(loadedData, 'rawData') && ~isfield(loadedData, 'processedData')
                        error('正規化用の有効なデータが見つかりません．');
                    end
                case {'csp', 'classifier'}
                    if ~isfield(loadedData, 'rawData')
                        error('分類用の有効なデータが見つかりません．');
                    end
                case 'baseline'
                    if ~isfield(loadedData, 'processedData')
                        error('ERD計算用の処理済みデータが見つかりません．');
                    end
            end
        end
    end
end