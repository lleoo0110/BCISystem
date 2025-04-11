% filepath: BCISystem/main/analysis_all_folders.m
% 全フォルダ内のデータを解析するスクリプト例

%% パスの設定
currentDir = pwd;
fprintf('開始: 現在のディレクトリ = %s\n', currentDir);
setupPaths(currentDir);

%% 解析パラメータの設定
params = getConfig('epocflex', 'preset', 'eeglab_template');
analyzer = EEGAnalyzer(params);  % アナライザーの初期化
fprintf('パラメータ設定完了\n');

%% all_foldersディレクトリ内の各フォルダを取得
allFoldersPath = fullfile(currentDir, 'all_folders');  % all_foldersのパスを指定
fprintf('全フォルダパス: %s\n', allFoldersPath);
folderList = dir(allFoldersPath);
folderList = folderList([folderList.isdir] & ~ismember({folderList.name}, {'.','..'}));
fprintf('解析対象フォルダ数: %d\n', length(folderList));

%% 各フォルダごとにデータを解析
for idx = 1:length(folderList)
    currentFolder = fullfile(allFoldersPath, folderList(idx).name);
    fprintf('\n===== フォルダ %d/%d 解析開始: %s =====\n', idx, length(folderList), currentFolder);
    
    % 現在のフォルダ内のデータファイル（例: .matファイル）を取得
    files = dir(fullfile(currentFolder, '*.mat'));
    if isempty(files)
        fprintf('警告: フォルダ %s に解析対象のデータが見つかりませんでした。\n', currentFolder);
        continue;
    end
    fileNames = {files.name};
    
    % 各データファイル毎に解析を実施
    for i = 1:length(fileNames)
        fullpath = fullfile(currentFolder, fileNames{i});
        fprintf('\n--- ファイル %d/%d 解析開始 ---\n', i, length(fileNames));
        fprintf('  データファイル読み込み: %s\n', fullpath);
        
       analyzer.analyze(fullpath);
        
    end
    fprintf('===== フォルダ %s の解析終了 =====\n', currentFolder);
end
%% analyze済みデータのEEGLABでの解析
analyzedDataFolder = fullfile(currentDir, 'analyzedData');
matFiles = dir(fullfile(analyzedDataFolder, '*.mat'));
if isempty(matFiles)
    fprintf('警告: フォルダ %s の analyzedData に .mat ファイルが見つかりません。\n', currentFolder);
else
    for j = 1:length(matFiles)
        analyzedFile = fullfile(analyzedDataFolder, matFiles(j).name);
        fprintf('読み込み中: %s\n', analyzedFile);
        data = load(analyzedFile);
        if isfield(data, 'processedData') && isfield(data, 'labels')
            processedData = data.processedData;
            labels = data.labels;
            EEG = EEGLAB_Analyzer(processedData, labels, params,matFiles(j).name);
            fprintf('EEG 解析完了: %s\n', analyzedFile);
        else
            fprintf('Warning: %s に必要な変数(processedData, labels)が存在しません。\n', analyzedFile);
        end
    end
end
%% パス設定補助関数（analysis.mと同様）
function setupPaths(currentDir)
    try
        rootDir = fullfile(currentDir, '../..');
        srcDir = fullfile(rootDir, 'src');
        mainDirs = {...
            'classification'
            'communication'
            'core'
            'data'
            'features'
            'preprocessing'
            'utils'
            'visualization'
            'eeg_visualizer'
        };
        
        lslDir = fullfile(rootDir, 'LabStreamingLayer');
        configDir = fullfile(rootDir, 'config');
        
        for i = 1:length(mainDirs)
            dirPath = fullfile(srcDir, mainDirs{i});
            if exist(dirPath, 'dir')
                addpath(genpath(dirPath));
                fprintf('パス追加: %s\n', dirPath);
            else
                warning('ディレクトリが見つかりません: %s', dirPath);
            end
        end
        
        if exist(lslDir, 'dir')
            addpath(genpath(lslDir));
            fprintf('LSLパス追加: %s\n', lslDir);
        else
            error('LSLディレクトリが見つかりません: %s', lslDir);
        end
        
        if exist(configDir, 'dir')
            addpath(genpath(configDir));
            fprintf('Configパス追加: %s\n', configDir);
        else
            error('Configディレクトリが見つかりません: %s', configDir);
        end
        
    catch ME
        error('パス設定エラー: %s', ME.message);
    end
end