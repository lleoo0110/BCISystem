# Feature Extraction Module

## 概要
このモジュールは、脳波データから高度な特徴量を抽出します。

## 特徴量抽出クラス
1. `PowerExtractor.m`
   - 周波数帯域パワー計算
   - スペクトル解析
   - ERSP (Event-Related Spectral Perturbation)

2. `FAAExtractor.m`
   - 前頭非対称性（FAA）分析
   - 左右前頭葉活動比較

3. `ABRatioExtractor.m`
   - アルファ/ベータ比計算
   - 覚醒状態推定

4. `CSPExtractor.m`
   - 共通空間パターン（CSP）フィルタ
   - 特徴量生成
   - 次元削減

5. `EmotionExtractor.m`
   - 感情状態推定
   - 座標変換
   - 感情クラス分類

## 主な機能
- マルチ特徴量抽出
- 柔軟な計算方法
- 正規化オプション
- 高度な信号処理

## 使用例
```matlab
% パワー特徴量抽出
power_extractor = PowerExtractor(params);
alpha_power = power_extractor.calculatePower(eeg_data, [8 13]);

% 感情抽出
emotion_extractor = EmotionExtractor(params);
emotion_result = emotion_extractor.classifyEmotion(eeg_data);
```

## サポートされる特徴量
- パワースペクトル
- 周波数帯域パワー
- 前頭非対称性
- α/β比
- CSP特徴量
- 感情座標

## 注意点
- パラメータ設定の重要性
- 計算コストの考慮
- データ前処理の影響
- 特徴量の解釈
