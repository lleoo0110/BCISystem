# Classification Module

## 概要
このモジュールは、脳波データの高度な機械学習分類アルゴリズムを実装しています。

## クラスの概要
1. `SVMClassifier.m`: サポートベクターマシン分類器
   - 線形・非線形カーネル対応
   - ハイパーパラメータ最適化
   - 確率出力
   - クロスバリデーション

2. `ECOCClassifier.m`: エラー修正出力符号分類器
   - マルチクラス分類対応
   - 複数の基本学習器
   - ロバスト性の向上

3. `CNNClassifier.m`: 畳み込みニューラルネットワーク分類器
   - 深層学習による特徴抽出
   - 複数の畳み込み層
   - GPU対応

## 主な機能
- モデルのトレーニング
- オンライン予測
- パフォーマンス評価
- 特徴量抽出

## 使用例
```matlab
% SVMでのトレーニングと予測
svm_classifier = SVMClassifier(params);
results = svm_classifier.trainSVM(features, labels);
[predicted_label, score] = svm_classifier.predictOnline(test_features);
```

## 注意点
- パラメータ設定に注意
- 過学習に注意
- 適切な特徴量選択が重要
