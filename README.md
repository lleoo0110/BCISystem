# Brain-Computer Interface (BCI) System

## プロジェクト概要

このプロジェクトは、脳波（EEG）データのリアルタイム収集、処理、分析のための包括的なBrainーComputer Interface (BCI)システムです。機械学習アルゴリズムを用いて、脳波信号から意味のある情報を抽出し、様々な応用に活用できます。

## 主な特徴

- リアルタイムデータ収集
- 多様な前処理オプション
- 高度な特徴抽出アルゴリズム
- 複数の機械学習分類器（SVM, ECOC, CNN）
- インタラクティブなGUIコントロール
- オンラインおよびオフライン解析モード

## システム要件

- MATLAB R2020b 以降
- Signal Processing Toolbox
- Machine Learning Toolbox
- Statistics and Machine Learning Toolbox
- Lab Streaming Layer (LSL)

## インストール

1. リポジトリをクローン
```bash
git clone https://github.com/your-username/bci-system.git
```

2. MATLAB内でプロジェクトのルートディレクトリを追加
3. 
```matlab
addpath(genpath('path/to/bci-system'));
```

3. 必要な依存関係をインストール
   - [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer)

## 使用方法

### オフラインモード
```matlab
% データの解析
params = getConfig('epocx');
analyzer = EEGAnalyzer(params);
analyzer.analyze();
```

### オンラインモード
```matlab
% オンラインデータ収集と処理
params = getConfig('epocx');
manager = EEGAcquisitionManager(params);
```

## サポートされているデバイス

- Emotiv EPOC X
- OpenBCI (8チャンネル、16チャンネル)
- MN8

## 主要モジュール

- **Classification**: 機械学習分類アルゴリズム
- **Communication**: デバイスとの通信管理
- **Core**: データ処理の中核
- **Feature**: 特徴量抽出
- **Preprocessing**: 信号前処理
- **Visualization**: GUI and データ可視化

## パラメータ設定とカスタマイズ
`config/presets/default/template_preset.m`を参照し、独自の設定を作成できます。
