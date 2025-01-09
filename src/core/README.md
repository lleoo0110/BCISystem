# Core Module

## 概要
BCISystem Core Moduleは、脳波(EEG)データの収集、解析、および処理を行うシステムの中核を担うコンポーネントです。リアルタイムデータ処理とオフライン解析の両方に対応し、柔軟で拡張性の高い脳波解析基盤を提供します。

## 主要コンポーネント

### EEGAcquisitionManager
リアルタイムデータ収集と処理を担当する中心的なクラスです。
- LSLを介したEEGデータのストリーミング受信
- UDPを介したトリガー信号の送受信
- リアルタイムデータ処理とフィードバック
- データの保存と管理

### EEGAnalyzer
オフラインでのデータ解析を担当するクラスです。
- 保存されたEEGデータの読み込みと解析
- 特徴抽出と分類器の学習
- 解析結果の評価と可視化

## システム要件

### 必須要件
- MATLAB R2020b以降
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

### 推奨要件
- Deep Learning Toolbox
- Parallel Computing Toolbox

## セットアップ手順

1. **環境設定**
```matlab
% システムのルートディレクトリで実行
setupPaths(pwd);
```

2. **パラメータ設定**
```matlab
% デフォルト設定の読み込み
params = getConfig('epocx');
% または特定のプリセットを使用
params = getConfig('epocx', 'preset', 'presetA');
```

3. **システムの初期化**
```matlab
% リアルタイム処理用
manager = EEGAcquisitionManager(params);

% オフライン解析用
analyzer = EEGAnalyzer(params);
```

## 使用方法

### リアルタイム処理モード

1. **データ収集の開始**
```matlab
% GUIから開始するか、直接コマンドを実行
manager.start();
```

2. **処理の一時停止/再開**
```matlab
manager.pause();
manager.resume();
```

3. **データ収集の終了**
```matlab
manager.stop();
```

### オフライン解析モード

1. **データの読み込みと解析**
```matlab
analyzer.analyze();
```

## 高度な設定

### カスタムパラメータの設定
`template_preset.m`をベースに、以下の設定をカスタマイズできます：

- 信号処理パラメータ
  - フィルタリング設定
  - エポック化設定
  - アーティファクト除去設定

- 特徴抽出パラメータ
  - パワースペクトル解析設定
  - CSPフィルタ設定
  - 感情推定パラメータ

- 分類器設定
  - SVM/ECOC/CNNの選択
  - ハイパーパラメータ
  - クロスバリデーション設定

### データ保存設定
```matlab
% 保存設定の例
params.acquisition.save.enable = true;
params.acquisition.save.path = './data';
params.acquisition.save.name = 'subject01';
```

## トラブルシューティング

### よくある問題と解決方法

1. **LSL接続エラー**
   - LSLライブラリのパスが正しく設定されているか確認
   - デバイスのストリーミングが開始されているか確認

2. **メモリ不足エラー**
   - バッファサイズの調整
   - データの一時保存間隔の短縮

3. **処理の遅延**
   - 処理窓サイズの調整
   - 特徴抽出の設定見直し

### デバッグ情報の取得
```matlab
% デバッグモードの有効化
params.debug.enable = true;
params.debug.level = 'verbose';
```

## 拡張と開発

### 新機能の追加
1. 適切なフォルダに新規クラスファイルを作成
2. 必要なインターフェースを実装
3. `template_preset.m`に関連パラメータを追加
```

### 依存関係
- LSLManager（通信モジュール）
- UDPManager（通信モジュール）
- DataManager（データ管理モジュール）
- 各種特徴抽出器（特徴抽出モジュール）
- 各種分類器（分類モジュール）
