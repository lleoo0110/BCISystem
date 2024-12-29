# メインスクリプト説明書

このドキュメントでは、BCISystemの2つの主要なスクリプト（main.mとanalysis.m）の詳細な説明を提供します。

## main.m - リアルタイムEEG処理スクリプト

### 概要
`main.m`は、EEGデータのリアルタイム収集、処理、および分類を行うためのメインスクリプトです。このスクリプトは、オンラインモードでシステムを実行する際に使用します。

### 主な機能
- リアルタイムEEGデータ収集
- リアルタイム信号処理
- リアルタイム特徴抽出と分類
- GUIベースのリアルタイムデータ可視化
- UDPによるトリガー送受信

### 使用方法

1. 基本的な実行
```matlab
% パスの設定とシステムの初期化
cd path/to/BCISystem/main
params = getConfig('epocx');
manager = EEGAcquisitionManager(params);
```

2. プリセットを指定して実行
```matlab
% 特定のプリセットで初期化
params = getConfig('epocx', 'preset', 'magic');
manager = EEGAcquisitionManager(params);
```

### 注意事項
- メモリ使用量の監視が必要
- CPUリソースの確保が重要
- リアルタイム処理の遅延に注意

## analysis.m - オフライン解析スクリプト

### 概要
`analysis.m`は、記録済みのEEGデータを詳細に解析するためのスクリプトです。このスクリプトは、オフラインモードでデータ解析を行う際に使用します。

### 主な機能
- 記録済みEEGデータの読み込み
- 詳細な信号処理と特徴抽出
- 分類器の学習と評価
- 結果の可視化と保存
- パラメータの最適化

### 使用方法

1. 基本的な実行
```matlab
% パスの設定とシステムの初期化
cd path/to/BCISystem/main
params = getConfig('epocx');
analyzer = EEGAnalyzer(params);
analyzer.analyze();
```

2. プリセットを指定して実行
```matlab
% 特定のプリセットで初期化
params = getConfig('epocx', 'preset', 'magic');
analyzer = EEGAnalyzer(params);
analyzer.analyze();
```

### 解析オプション
analysis.mでは以下の解析が可能です：

1. 前処理
   - アーティファクト除去
   - ベースライン補正
   - フィルタリング
   - ダウンサンプリング
   - 正規化

2. 特徴抽出
   - パワースペクトル解析
   - FAA（前頭葉α波非対称性）
   - α/β比
   - CSP（Common Spatial Pattern）
   - 感情特徴量

3. 分類器学習
   - SVM
   - ECOC
   - CNN

### データ保存形式
解析結果は以下の形式で保存されます：

```matlab
saveData = struct();
saveData.params = params;           % 解析パラメータ
saveData.rawData = rawData;        % 生データ
saveData.processedData = procData; % 処理済みデータ
saveData.features = features;      % 抽出特徴量
saveData.results = results;        % 解析結果
```

## 共通の初期化処理

両スクリプトで使用される初期化処理：

```matlab
% パスの設定
currentDir = pwd;
setupPaths(currentDir);

% パス設定関数
function setupPaths(currentDir)
    % BCISystemのルートディレクトリを取得
    rootDir = fullfile(currentDir, '..', '..');
    
    % 主要ディレクトリの設定
    srcDir = fullfile(rootDir, 'src');
    mainDirs = {'classification', 'communication', 'core', 'data', 
                'features', 'main', 'preprocessing', 'utils', 'visualization'};
    
    % LSLとconfigのパス設定
    lslDir = fullfile(rootDir, '..', 'LabStreamingLayer');
    configDir = fullfile(rootDir, 'config');
    
    % パスの追加
    for i = 1:length(mainDirs)
        addpath(genpath(fullfile(srcDir, mainDirs{i})));
    end
    addpath(genpath(lslDir));
    addpath(genpath(configDir));
end
```

## エラー処理

両スクリプトで共通のエラー処理手順：

1. 全初期化
```matlab
clc; clear all; close all; instrreset;
```

2. 個別初期化
- コマンドクリア: `clc`
- ワークスペースクリア: `clear all`
- ウィンドウ初期化: `close all`
- UDP系エラー初期化: `instrreset`

## トラブルシューティング

1. パスエラー
   - setupPaths関数が正しく実行されているか確認
   - 必要なディレクトリが存在するか確認

2. メモリエラー
   - 大きなデータセットを扱う際はクリア処理を適宜実行
   - 不要な変数を削除

3. LSL/UDP通信エラー
   - instrreset実行で通信をリセット
   - ポート設定を確認

## 補足情報

- プログラムの実行中は、MATLABのパフォーマンスに影響を与える他のアプリケーションを閉じることを推奨
- 大規模なデータセットを扱う場合は、64bit版MATLABの使用を推奨
- GPUを使用する場合は、ドライバが最新であることを確認