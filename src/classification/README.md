# Classification Module

## 概要
このモジュールは、脳波（EEG）信号の分類処理を行うための分類器を提供します。SVMやECOC、CNNなど、複数の分類アルゴリズムを実装しており、リアルタイム処理とオフライン解析の両方に対応しています。

## 分類器の種類

### 1. SVMClassifier
Support Vector Machine（SVM）を用いた2クラス分類器です。

#### 主な特徴
- 線形/非線形カーネルに対応
- 確率的出力のサポート
- 最適閾値の自動探索
- クロスバリデーションによる性能評価

#### 使用例
```matlab
% SVMClassifierの初期化
classifier = SVMClassifier(params);

% モデルの学習
results = classifier.trainSVM(features, labels);

% オンライン予測
[label, score] = classifier.predictOnline(features, model, threshold);
```

### 2. ECOCClassifier
Error-Correcting Output Codes (ECOC)を用いたマルチクラス分類器です。

#### 主な特徴
- One-vs-All、All-vs-Allのコーディング方式
- SVMをベース分類器として使用
- 確率的出力のサポート
- クロスバリデーション評価

#### 使用例
```matlab
% ECOCClassifierの初期化
classifier = ECOCClassifier(params);

% モデルの学習
results = classifier.trainECOC(features, labels);

% オンライン予測
[label, score] = classifier.predictOnline(features, model);
```

### 3. CNNClassifier
畳み込みニューラルネットワーク（CNN）を用いた分類器です。

#### 主な特徴
- カスタマイズ可能なネットワークアーキテクチャ
- GPUサポート
- バッチ処理による高速化
- 学習過程の可視化

#### 使用例
```matlab
% CNNClassifierの初期化
classifier = CNNClassifier(params);

% モデルの学習
results = classifier.trainCNN(data, labels);

% オンライン予測
[label, score] = classifier.predictOnline(features, model);
```

## パラメータ設定

### 共通パラメータ
- `enable`: 分類器の有効/無効設定
- `evaluation`: 性能評価の設定
  - `kfold`: クロスバリデーションの分割数
  - `metrics`: 評価指標の選択

### SVMパラメータ
```matlab
params.classifier.svm = struct(
    'enable', true,          % SVMの有効化
    'optimize', false,       % ハイパーパラメータ最適化
    'probability', true,     % 確率的出力の有効化
    'kernel', 'linear',      % カーネル関数の選択
    'threshold', struct(     % 閾値設定
        'rest', 0.5,         % デフォルト閾値
        'useOptimal', true,  % 最適閾値の使用
        'range', [0.1:0.05:0.9]  % 閾値探索範囲
    )
);
```

### ECOCパラメータ
```matlab
params.classifier.ecoc = struct(
    'enable', true,          % ECOCの有効化
    'optimize', false,       % パラメータ最適化
    'probability', true,     % 確率的出力
    'kernel', 'linear',      % 基本分類器のカーネル
    'coding', 'onevsall'     % コーディング方式
);
```

### CNNパラメータ
```matlab
params.classifier.cnn = struct(
    'enable', false,         % CNNの有効化
    'architecture', struct(  % ネットワーク構造
        'inputSize', [28 28 1],  % 入力サイズ
        'numClasses', 2,         % クラス数
        'layers', struct(...),   % 層の設定
        'dropoutRate', 0.5       % ドロップアウト率
    ),
    'training', struct(      % 学習パラメータ
        'optimizer', 'adam', % 最適化アルゴリズム
        'maxEpochs', 30,    % 最大エポック数
        'miniBatchSize', 128 % バッチサイズ
    )
);
```

## 性能評価

各分類器は以下の評価指標を提供します：
- 正解率（Accuracy）
- 適合率（Precision）
- 再現率（Recall）
- F1スコア
- AUC（2クラス分類の場合）
- 混同行列

評価結果は`performance`構造体に保存され、以下の情報を含みます：
```matlab
performance = struct(
    'accuracy', accuracy,          % 全体の正解率
    'cvMeanAccuracy', cvAcc,      % 交差検証の平均正解率
    'cvStdAccuracy', cvStd,       % 交差検証の標準偏差
    'confusionMat', confMat,      % 混同行列
    'classwise', struct(...)      % クラスごとの性能
);
```

## 使用上の注意

1. メモリ管理
- 大規模なデータセットを扱う場合は、`batchSize`パラメータを適切に設定してください。
- CNNを使用する場合、GPUメモリの使用量に注意が必要です。

2. モデルの保存
- 学習済みモデルは自動的に保存されます。
- 保存されたモデルはオンライン処理で再利用できます。

3. エラー処理
- 各分類器は例外処理を実装しています。
- エラーメッセージを確認し、適切な対処を行ってください。

## トラブルシューティング

よくある問題と解決方法：

1. メモリ不足エラー
```matlab
% バッチサイズを小さくする
params.classifier.cnn.training.miniBatchSize = 64;
```

2. 過学習
```matlab
% ドロップアウトを追加
params.classifier.cnn.architecture.dropoutRate = 0.5;

% 正則化を強める
params.classifier.svm.hyperparameters.boxConstraint = 1;
```

3. 学習が不安定
```matlab
% 学習率を下げる
params.classifier.cnn.training.initialLearnRate = 0.0001;

% バッチ正規化を追加
params.classifier.cnn.architecture.batchNorm = true;
```