# Visualization Module

## 概要
このモジュールは、脳波データの高度な可視化と対話的なユーザーインターフェースを提供します。

## 主要コンポーネント
1. `GUIControllerManager.m`
   - 統合GUIコントロール
   - リアルタイムデータ表示
   - インタラクティブな設定パネル
   - 状態管理

## 可視化機能
- リアルタイム生データプロット
- 処理済みデータ表示
- パワースペクトル解析
- イベント関連スペクトル摂動（ERSP）
- カラーマップカスタマイズ
- 周波数帯域ハイライト

## 主な特徴
- 柔軟な表示設定
- 自動/手動スケーリング
- リアルタイム更新
- インタラクティブなコントロール
- マルチチャンネル表示

## 使用例
```matlab
% GUIコントローラの初期化
gui_manager = GUIControllerManager(params);

% データ表示の更新
display_data = struct(...
    'rawData', raw_signal, ...
    'processedData', processed_signal, ...
    'spectrum', power_spectrum, ...
    'ersp', event_spectrum ...
);
gui_manager.updateDisplayData(display_data);
```

## GUIコントロール機能
- 開始/停止ボタン
- 一時停止/再開
- モード切り替え
- パラメータ調整スライダー
- トリガーラベル付け

## サポートされる可視化タイプ
- 時系列データ
- パワースペクトログラム
- イベント関連スペクトル
- 周波数帯域マッピング

## グラフィックカスタマイズ
- カラーマップ選択
- スケール調整
- 周波数帯域表示
- 背景色設定

## パフォーマンス最適化
- 更新レート制御
- 効率的なプロット更新
- リソース管理
- 描画負荷の軽減

## 注意点
- リアルタイム描画の計算コスト
- メモリ管理
- パフォーマンスチューニング
- デバイス互換性

## 拡張性
- カスタムプロットタイプの追加
- サードパーティライブラリとの統合
- プラグイン可能な可視化モジュール
