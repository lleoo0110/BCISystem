# Data Module

## 概要
このモジュールは、データの管理、読み込み、保存を担当します。

## クラスの概要
1. `DataManager.m`
   - データセーブ管理
   - ファイル名生成
   - データフィールド選択的保存
   - メタデータ管理

2. `DataLoader.m`
   - データファイル読み込み
   - データ検証
   - データプロンプト

## 主な機能
- データセット保存
- データセット読み込み
- データ情報取得
- データ検証
- フィールド選択的保存/読み込み

## 使用例
```matlab
% データセーブ
data_manager = DataManager(params);
savedFile = data_manager.saveDataset(data_struct);

% データロード
loaded_data = DataLoader.loadDataBrowserWithPrompt('オフライン解析');
```

## サポートされるデータ形式
- MAT形式
- 選択的フィールド保存
- メタデータ付き

## データ保存オプション
- 生データ
- 前処理データ
- ラベル情報
- 分類器結果
- 解析情報

## 注意点
- ファイルサイズに注意
- データ整合性の確認
- プライバシーとセキュリティ
- 一意のファイル名生成
