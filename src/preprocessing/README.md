# Preprocessing Module

## 概要

このフォルダには、脳波データの前処理に関連するすべてのコンポーネントが含まれています。各コンポーネントは特定の前処理タスクを担当し、モジュール化された設計により柔軟な組み合わせが可能です。

## 機能一覧

このモジュールは以下の前処理機能を提供します：

1. アーティファクト除去 (ArtifactRemover.m)
2. ベースライン補正 (BaselineCorrector.m)
3. データ拡張 (DataAugmenter.m)
4. ダウンサンプリング (DownSampler.m)
5. 信号の正規化 (EEGNormalizer.m)
6. エポック化 (Epoching.m)
7. FIRフィルタリング (FIRFilterDesigner.m)
8. ノッチフィルタリング (NotchFilterDesigner.m)

## 各コンポーネントの詳細

### アーティファクト除去 (ArtifactRemover)

眼球運動（EOG）、筋電（EMG）、基線変動などのアーティファクトを検出・除去します。

主な機能：
- 複数のアーティファクト除去手法（EOG、EMG、閾値ベース）
- 適応的な閾値設定
- アーティファクト情報のログ機能

使用例：
```matlab
remover = ArtifactRemover(params);
[cleanedData, info] = remover.removeArtifacts(data, 'all');
```

### ベースライン補正 (BaselineCorrector)

信号の基線変動を補正し、安定したベースラインを確保します。

主な機能：
- 複数の補正手法（区間、トレンド、DC、移動平均）
- 適応的な窓サイズ設定
- オーバーラップ処理対応

使用例：
```matlab
corrector = BaselineCorrector(params);
[correctedData, info] = corrector.correctBaseline(data, 'interval');
```

### データ拡張 (DataAugmenter)

学習データセットを拡張し、分類器の性能向上を支援します。

主な機能：
- ノイズ付加
- スケーリング
- 時間シフト
- 信号反転
- チャンネル入れ替え

使用例：
```matlab
augmenter = DataAugmenter(params);
[augData, augLabels, info] = augmenter.augmentData(data, labels);
```

### ダウンサンプリング (DownSampler)

サンプリングレートを調整し、計算効率を向上させます。

主な機能：
- アンチエイリアシングフィルタ
- 可変デシメーション係数
- 自動パラメータ最適化

使用例：
```matlab
sampler = DownSampler(params);
[downsampledData, info] = sampler.downsample(data, targetRate);
```

### 信号の正規化 (EEGNormalizer)

信号振幅を標準化し、チャンネル間の比較を容易にします。

主な機能：
- 複数の正規化手法（Z-score、Min-Max、Robust）
- チャンネル単位の正規化
- オンライン処理対応

使用例：
```matlab
normalizer = EEGNormalizer(params);
[normalizedData, normParams] = normalizer.normalize(data);
```

### エポック化 (Epoching)

連続データをイベントベースのエポックに分割します。

主な機能：
- 時間ベースのエポック化
- 奇数-偶数ペアによるエポック化
- ベースライン期間の設定
- オーバーラップ処理

使用例：
```matlab
epoching = Epoching(params);
[epochs, epochLabels, epochTimes] = epoching.epoching(data, labels);
```

### フィルタリング (FIRFilterDesigner & NotchFilterDesigner)

特定の周波数帯域の信号を抽出または除去します。

主な機能：
- FIRフィルタ設計
- ノッチフィルタ設計
- 自動パラメータ最適化
- フィルタ特性の解析

使用例：
```matlab
firFilter = FIRFilterDesigner(params);
[filteredData, filterInfo] = firFilter.designAndApplyFilter(data);

notchFilter = NotchFilterDesigner(params);
[filteredData, filterInfo] = notchFilter.designAndApplyFilter(data);
```

## パラメータ設定

各コンポーネントは`params`構造体を通じて設定を行います。基本的な設定は`template_preset.m`に定義されていますが、必要に応じてカスタマイズ可能です。

パラメータの主な項目：
- サンプリングレート
- フィルタの仕様
- アーティファクトの閾値
- エポック化の条件
- 正規化の方法
- データ拡張の設定

## 使用上の注意

1. メモリ使用量
   - 大規模なデータセットを処理する場合は、メモリ使用量に注意してください
   - 必要に応じてバッチ処理を検討してください

2. 処理順序
   - 一般的な推奨順序：
     1. ダウンサンプリング
     2. アーティファクト除去
     3. フィルタリング
     4. ベースライン補正
     5. 正規化
     6. エポック化
     7. データ拡張

3. パラメータ調整
   - データの特性に応じて適切なパラメータを設定してください
   - 処理結果を可視化して、設定の妥当性を確認することを推奨します

4. エラー処理
   - 各コンポーネントは適切なエラー処理を実装していますが、入力データの検証も重要です
   - エラーメッセージを確認し、適切な対処を行ってください

## トラブルシューティング

よくある問題と解決策：

1. メモリ不足エラー
   - データをより小さなチャンクに分割して処理
   - 不要なデータをクリア
   - より効率的なアルゴリズムの使用を検討

2. 処理結果が期待と異なる
   - パラメータの設定を確認
   - 入力データの妥当性を確認
   - 処理順序が適切か確認

3. 処理が遅い
   - ダウンサンプリングの使用を検討
   - バッチサイズの調整
   - 並列処理の活用を検討

## 開発者向け情報

新しいコンポーネントを追加する場合の注意点：

1. 命名規則
   - クラス名は機能を明確に示す
   - メソッド名は動詞で開始

2. コードスタイル
   - エラー処理を適切に実装
   - パラメータのバリデーションを実装
   - 詳細なコメントを記述

3. テスト
   - 単体テストの作成
   - エッジケースの考慮
   - パフォーマンステストの実施
