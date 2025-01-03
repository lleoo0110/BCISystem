% パラメータ例：
% params = getConfig('epocx');
% params = getConfig('epocx', 'preset', 'magic');
% params = getConfig('epocx', 'preset', 'ddaExperiment');

% params = getConfig('epocx', 'preset', 'magic_visualA');
% params = getConfig('epocx', 'preset', 'magic_visualB');
% params = getConfig('epocx', 'preset', 'magic_visualC');

% params = getConfig('epocx', 'preset', 'ahaloud');

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
manager = EEGAcquisitionManager(params);      % データ計測


function setupPaths(currentDir)
    try
        % BCISystemのルートディレクトリを取得
        rootDir = fullfile(currentDir, '..');
        
        % BCISystemの主要ディレクトリを定義
        srcDir = fullfile(rootDir, 'src');
        mainDirs = {
            'classification'
            'communication'
            'core'
            'data'
            'features'
            'main'
            'preprocessing'
            'utils'
            'visualization'
        };
        
        % LSLとconfigのパス設定
        lslDir = fullfile(rootDir, 'LabStreamingLayer');
        configDir = fullfile(rootDir, 'config');
        
        % パスの存在確認と追加
        % ソースディレクトリの各フォルダを追加
        fprintf('\nソースディレクトリのパス:\n');
        for i = 1:length(mainDirs)
            dirPath = fullfile(srcDir, mainDirs{i});
            if exist(dirPath, 'dir')
                addpath(genpath(dirPath));
            else
                warning('ディレクトリが見つかりません: %s', dirPath);
            end
        end
        
        % LSLパスの追加
        fprintf('\nLSLパス:\n');
        if exist(lslDir, 'dir')
            addpath(genpath(lslDir));
        else
            error('LSLディレクトリが見つかりません: %s', lslDir);
        end
        
        % Configパスの追加
        fprintf('\nConfigパス:\n');
        if exist(configDir, 'dir')
            addpath(genpath(configDir));
        else
            error('Configディレクトリが見つかりません: %s', configDir);
        end
        
        % LSLのライブラリパスを設定に反映
        params = getConfig('epocx');
        params.lsl.stream.libraryPath = lslDir;
        
        fprintf('\nパスの設定が完了しました\n');
        
    catch ME
        error('パス設定エラー: %s', ME.message);
    end
end