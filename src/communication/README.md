# Communication Module

BCIシステムの通信コンポーネントについての説明書です。このフォルダには脳波計からのデータ取得と外部システムとの通信を担当するコンポーネントが含まれています。

## 概要

通信コンポーネントは以下の2つの主要なクラスで構成されています：

1. LSLManager - 脳波データのストリーミング受信を管理
2. UDPManager - トリガー信号やイベントマーカーの送受信を管理

## LSLManager

### 概要
Lab Streaming Layer (LSL)を使用して、脳波計からのリアルタイムデータストリームを受信します。

### 主な機能
- 脳波データストリームの自動検出と接続
- データのリアルタイム受信
- シミュレーションモードでのテストデータ生成
- チャンネル情報の管理

### 使用方法
```matlab
% LSLManagerのインスタンス化
params = getConfig('epocx');  % デバイス設定の読み込み
lslManager = LSLManager(params);

% データの取得
[data, timestamp] = lslManager.getData();
```

### 設定パラメータ
- `params.lsl.simulate.enable` - シミュレーションモードの有効/無効
- `params.device.streamName` - LSLストリーム名
- `params.device.channelCount` - チャンネル数
- `params.device.sampleRate` - サンプリングレート

## UDPManager

### 概要
UDP通信を使用して、トリガー信号やイベントマーカーの送受信を行います。

### 主な機能
- トリガー信号の受信と送信
- 通信アドレスとポートの動的更新
- エンコーディング設定の管理
- トリガーマッピングの管理

### 使用方法
```matlab
% UDPManagerのインスタンス化
params = getConfig('epocx');  % 通信設定の読み込み
udpManager = UDPManager(params);

% トリガーの送信
udpManager.sendTrigger(1);  % トリガー値1を送信

% トリガーの受信
trigger = udpManager.receiveTrigger();
```

### 設定パラメータ
- `params.udp.receive.port` - 受信ポート番号
- `params.udp.send.port` - 送信ポート番号
- `params.udp.receive.address` - 受信アドレス
- `params.udp.send.address` - 送信アドレス

## エラーハンドリング

両コンポーネントとも、通信エラーが発生した場合は適切なエラーメッセージを生成し、システムの安定性を維持します。

```matlab
try
    % 通信処理
catch ME
    warning(ME.identifier, '通信エラー: %s', ME.message);
end
```

## デバッグとトラブルシューティング

### 一般的な問題と解決方法

1. LSL接続エラー
   - LSLライブラリが正しくインストールされているか確認
   - ストリーム名が正しく設定されているか確認
   - デバイスが正しく接続されているか確認

2. UDPエラー
   - ポート番号が他のアプリケーションと競合していないか確認
   - ファイアウォール設定を確認
   - ネットワーク接続状態を確認

## 依存関係

- Lab Streaming Layer (LSL) MATLAB SDK
- MATLAB Instrument Control Toolbox (UDPソケット通信用)

## 注意事項

1. シミュレーションモード使用時は実データが混入しないよう注意してください
2. UDPポートは事前に空いていることを確認してください
3. マルチスレッド環境での使用時は適切な同期処理を実装してください
