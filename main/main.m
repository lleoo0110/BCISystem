% パラメータ例：
% params = getConfig('epocx');
% params = getConfig('epocx', 'preset', 'character');
% params = getConfig('epocx', 'preset', 'magic');

% 全初期化：clc; clear all; close all;
% コマンドクリア：clc
% ワークスペースクリア：clear all
% ウィンドウ初期化：close all;
% UDPポートの初期化：delete(udpportfind)


%% パスの設定
currentDir = pwd;
setupPaths(currentDir);


%%  指定のパラメータで実行
params = getConfig('epocx', 'preset', 'character');
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
            'preprocessing'
            'utils'
            'visualization'
        };
        
        % LSLとconfigのパス設定
        lslDir = fullfile(rootDir, 'LabStreamingLayer');
        configDir = fullfile(rootDir, 'config');
        
        % パスの存在確認と追加
        % ソースディレクトリの各フォルダを追加
        for i = 1:length(mainDirs)
            dirPath = fullfile(srcDir, mainDirs{i});
            if exist(dirPath, 'dir')
                addpath(genpath(dirPath));
            else
                warning('ディレクトリが見つかりません: %s', dirPath);
            end
        end
        
        % LSLパスの追加
        if exist(lslDir, 'dir')
            addpath(genpath(lslDir));
        else
            error('LSLディレクトリが見つかりません: %s', lslDir);
        end
        
        % Configパスの追加
        if exist(configDir, 'dir')
            addpath(genpath(configDir));
        else
            error('Configディレクトリが見つかりません: %s', configDir);
        end
        
        fprintf('\nパスの設定が完了しました\n');
        
    catch ME
        error('パス設定エラー: %s', ME.message);
    end
end