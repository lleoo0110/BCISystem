# Communication Module

## 概要
このモジュールは、外部デバイスとのデータ通信を管理します。

## クラスの概要
1. `LSLManager.m`: Lab Streaming Layer (LSL)マネージャ
   - EEGデバイスからのデータストリーム管理
   - リアルタイムデータ取得
   - シミュレーションモード
   - デバイス情報の表示

2. `UDPManager.m`: UDP通信マネージャ
   - トリガー送受信
   - 動的アドレス更新
   - メッセージエンコーディング
   - エラーハンドリング

## 主な機能
- データストリーミング
- トリガー処理
- デバイス接続管理
- リアルタイム通信

## 使用例
```matlab
% LSLマネージャの初期化と使用
lsl_manager = LSLManager(params);
[data, timestamp] = lsl_manager.getData();

% UDPマネージャの使用
udp_manager = UDPManager(params);
udp_manager.sendTrigger(trigger_value);
```

## サポートデバイス
- Emotiv EPOC X
- OpenBCI
- その他のLSL対応デバイス

## 注意点
- ネットワーク設定に注意
- タイムスタンプの同期
- 通信エラーのハンドリング
