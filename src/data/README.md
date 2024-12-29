# Data Module

## 概要

このディレクトリは、BCISystemで使用・生成される全てのデータファイルを管理するためのメインストレージです。システムで取得される脳波データ、学習済みモデル、前処理結果、および解析結果などが保存されます。

## データ構造

### データファイルの基本構造 (.mat)
```matlab
data = struct(
    'rawData',         % 生脳波データ
    'labels',          % イベントマーカー情報
    'params',          % パラメータ設定
    'processedData',   % 前処理済みデータ
    'processedLabel',  % 処理済みラベル
    'processingInfo',  % 処理情報
    'classifier',      % 分類器情報
    'results'          % 解析結果
);
```

### 主要データフィールドの詳細

#### rawData
- 生の脳波データ
- チャンネル × サンプル数

#### labels
```matlab
labels = struct(
    'value',   % トリガー値
    'time',    % タイムスタンプ
    'sample'   % サンプル位置
);
```

#### processedData
- 前処理済み・エポック化されたデータ
- チャンネル × サンプル × エポック

#### processedLabel
- エポックごとのクラスラベル

## 使用方法

### データの保存
```matlab
% DataManagerクラスを使用
dataManager = DataManager(params);
filename = dataManager.saveDataset(data);
```

### データの読み込み
```matlab
% DataLoaderクラスを使用
loadedData = DataLoader.loadDataBrowserWithPrompt('analysis');
```

## 注意事項

1. **データ整合性**
   - rawDataとlabelsの対応関係を維持すること
   - processedDataとprocessedLabelの順序を維持すること

2. **ファイル管理**
   - 実験ごとに独立したファイルを作成
   - 重要なデータは別途バックアップを推奨

3. **容量管理**
   - rawDataは特に大容量になるため、定期的な整理を推奨
   - 不要な中間データは適宜削除

4. **命名規則**
   - 日付とセッションIDを含めること
   - 実験条件を識別可能な命名を使用
   - アンダースコアで単語を区切る

## トラブルシューティング

1. **メモリ不足の場合**
   - 大きなrawDataの読み込み時は注意
   - 必要に応じてデータを分割して処理

2. **ファイル読み込みエラー**
   - MATLABバージョンの互換性を確認
   - ファイル破損の可能性を確認

3. **データ構造の不整合**
   - 各フィールドのサイズと形式を確認
   - processingInfoで処理履歴を確認
