# Features Module

このディレクトリには脳波信号から特徴量を抽出するための各種クラスが含まれています。実装されている特徴抽出手法は、脳波解析で一般的に使用される手法をカバーしています。

## 目次

1. [概要](#概要)
2. [特徴抽出クラス](#特徴抽出クラス)
3. [使用方法](#使用方法)
4. [設定パラメータ](#設定パラメータ)
5. [実装例](#実装例)
6. [注意事項](#注意事項)

## 概要

featuresモジュールは以下の特徴量抽出を提供します：

- パワースペクトル解析（各周波数帯域のパワー値計算）
- 前頭非対称性指標（Frontal Alpha Asymmetry: FAA）
- アルファ/ベータ比（Arousal指標）
- 共通空間パターン（Common Spatial Pattern: CSP）
- 感情状態推定（2次元感情モデルに基づく）

## 特徴抽出クラス

### PowerExtractor
周波数帯域別のパワー値を算出するクラスです。
- デルタ波(0.5-4Hz)、シータ波(4-8Hz)、アルファ波(8-13Hz)、ベータ波(13-30Hz)、ガンマ波(30-45Hz)の各帯域のパワーを計算
- Welch法またはバンドパスフィルタによる計算方法を選択可能
- 相対パワー値やログ変換などの正規化オプションを提供

### FAAExtractor
前頭葉の左右差に基づく感情価推定を行うクラスです。
- 左右前頭葉のアルファ波パワーの非対称性を計算
- 接近-回避傾向の指標として利用可能
- 設定可能な閾値による2値分類機能

### ABRatioExtractor
アルファ波とベータ波の比率から覚醒度を推定するクラスです。
- α/β比の計算による覚醒状態の評価
- 高覚醒/低覚醒の2状態分類
- カスタマイズ可能な閾値設定

### CSPExtractor
2クラス分類のための空間フィルタを学習・適用するクラスです。
- 教師あり学習による最適な空間フィルタの計算
- 次元圧縮と特徴強調の同時実現
- 正則化パラメータによるオーバーフィッティング制御

### EmotionExtractor
2次元感情モデルに基づく感情状態推定を行うクラスです。
- Valence（快-不快）とArousal（覚醒-睡眠）の2軸で感情を評価
- FAAとABRatioの組み合わせによる感情象限の推定
- 9種類の基本感情状態への分類機能

## 使用方法

各特徴抽出クラスは以下の基本的な使用パターンに従います：

1. クラスのインスタンス化
```matlab
extractor = PowerExtractor(params);  % パラメータ構造体を渡してインスタンス化
```

2. 特徴量の抽出
```matlab
features = extractor.calculateXXX(data);  % データから特徴量を計算
```

3. 結果の取得
```matlab
results = extractor.getResults();  % 計算結果の取得（必要な場合）
```

## 設定パラメータ

特徴抽出の設定は`template_preset.m`の`feature`構造体で定義されます。主な設定項目：

```matlab
feature = struct(
    'power', struct(...),    % パワー解析の設定
    'faa', struct(...),      % FAA解析の設定
    'abRatio', struct(...),  % α/β比の設定
    'csp', struct(...),      % CSP解析の設定
    'emotion', struct(...)   % 感情推定の設定
);
```

各設定の詳細は`template_preset.m`のコメントを参照してください。

## 実装例

PowerExtractorを使用した周波数解析の例：

```matlab
% パラメータの取得
params = getConfig('epocx');

% PowerExtractorのインスタンス化
powerExtractor = PowerExtractor(params);

% アルファ帯域(8-13Hz)のパワー値を計算
alphaBand = [8 13];
alphaPower = powerExtractor.calculatePower(eegData, alphaBand);

% スペクトル計算
[pxx, f] = powerExtractor.calculateSpectrum(eegData);

% ERSP計算
[ersp, times, freqs] = powerExtractor.calculateERSP(eegData);
```

## 注意事項

1. データフォーマット
   - 入力データは[channels × samples]または[channels × samples × epochs]の形式
   - チャンネル数はデバイス設定と一致している必要があります

2. サンプリングレート
   - 特徴抽出時はパラメータで指定されたサンプリングレートと一致していることを確認
   - ダウンサンプリング後のデータを使用する場合は適切なレートを設定

3. メモリ使用
   - 大規模データセットを処理する際はメモリ使用量に注意
   - 必要に応じてバッチ処理を検討

4. エラー処理
   - 各メソッドは適切なエラーチェックとエラーメッセージを提供
   - try-catch構文での例外処理を推奨

5. 処理時間
   - CSPやERSPなど計算負荷の高い処理はオフライン解析での使用を推奨
   - オンライン処理では軽量な特徴量（パワー値、α/β比など）の使用を推奨