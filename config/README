# Configuration Directory

## 概要
このディレクトリには、BCIシステムの設定ファイルが格納されています。設定は柔軟で、異なる実験やデバイス用にカスタマイズ可能です。

## ファイル構造
- `template_preset.m`: デフォルト設定テンプレート
- `getConfig.m`: 設定ローダー

## 設定パラメータ
- デバイス設定
- データ収集設定
- 信号処理パラメータ
- 特徴量抽出設定
- 分類器設定
- GUI表示設定

## 使用方法
```matlab
% プリセット設定を読み込む
params = getConfig('epocx');  % デフォルトEPOC X設定
params = getConfig('epocx', 'preset', 'magic_offline');  % オフラインモード
```

## カスタマイズ
新しいプリセットを追加するには、`template_preset.m`をコピーして編集してください。

## 注意
設定ファイルは慎重に編集してください。不適切な変更はシステムの動作に影響を与える可能性があります。
