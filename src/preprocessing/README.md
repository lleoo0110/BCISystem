# Preprocessing Module

## 概要
このモジュールは、脳波信号の前処理を担当し、データの品質と解析可能性を向上させます。

## 前処理クラス
1. `ArtifactRemover.m`
   - アーティファクト除去
   - EOG/EMGノイズ除去
   - 異常値補間

2. `BaselineCorrector.m`
   - ベースライン補正
   - トレンド除去
   - DC成分除去

3. `DataAugmenter.m`
   - データ拡張
   - ノイズ付加
   - 時間シフト
   - チャンネル操作

4. `DownSampler.m`
   - サンプリングレート調整
   - アンチエイリアシングフィルタ

5. `EEGNormalizer.m`
   - データ正規化
   - Z-score変換
   - Min-Max scaling

6. `Epoching.m`
   - データのエポック化
   - イベントトリガーベース
   - オーバーラップ設定

7. `FIRFilterDesigner.
