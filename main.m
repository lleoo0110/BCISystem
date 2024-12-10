% パラメータ例：
% params = getConfig('epocx');
% params = getConfig('epocx', 'preset', 'magic_offline');
% params = getConfig('epocx', 'preset', 'magic_online');

% 全初期化：clc; clear all; close all; instrreset;
% コマンドクリア：clc
% ワークスペースクリア：clear all
% ウィンドウ初期化：close all;
% UDP系エラー初期化：instrreset;


%% パスの設定
currentDir = pwd;
setupPaths(currentDir);


%%  指定のパラメータで実行
params = getConfig('epocx');
manager = EEGAcquisitionManager(params);


%% パス設定補助関数
function setupPaths(currentDir)
    % パスの設定と確認を行う関数
    % currentDir: 現在のディレクトリパス（通常はpwd）

    try
        %% LSLとconfigのパス設定
        lslDir = fullfile(currentDir, '..', 'LabStreamingLayer');
        configDir = fullfile(currentDir, 'config');
        
        % パスの存在確認
        if ~exist(lslDir, 'dir')
            error('LSLディレクトリが見つかりません: %s', lslDir);
        end
        if ~exist(configDir, 'dir')
            error('Configディレクトリが見つかりません: %s', configDir);
        end
        
        % パスを追加
        addpath(genpath(configDir));  
        addpath(genpath(lslDir));     
        
        % 追加したパスの表示
        fprintf('=== 追加されたパス ===\n');
        fprintf('\nConfig パス:\n');
        configPaths = strsplit(genpath(configDir), pathsep);
        for i = 1:length(configPaths)
            if ~isempty(configPaths{i})
                fprintf('  %s\n', configPaths{i});
            end
        end
        
        fprintf('\nLSL パス:\n');
        lslPaths = strsplit(genpath(lslDir), pathsep);
        for i = 1:length(lslPaths)
            if ~isempty(lslPaths{i})
                fprintf('  %s\n', lslPaths{i});
            end
        end
        
        % LSLのライブラリパスを設定に反映
        params = getConfig('epocx');  % デフォルト設定を取得
        params.lsl.stream.libraryPath = lslDir;  % LSLパスを更新
        
        fprintf('\nパスの設定が完了しました\n');
        
    catch ME
        error('パス設定エラー: %s', ME.message);
    end
end