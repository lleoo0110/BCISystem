classdef ReshapeLSTMLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties (Learnable)
        Weights
        Bias
    end
    
    methods
        function layer = ReshapeLSTMLayer(name)
            layer.Name = name;
            layer.Description = 'Reshape LSTM output to 1x1x128 format';
            layer.Type = 'Reshape';
            layer.Weights = [];
            layer.Bias = [];
        end
        
        function Z = predict(layer, X)
            % セルの場合は先頭要素を取り出し、dlarray なら数値配列へ変換
            if iscell(X), X = X{1}; end
            if isa(X, 'dlarray'), X = extractdata(X); end
            
            [d, b] = size(X);
            
            % 初回実行時に重み初期化
            if isempty(layer.Weights)
                layer.Weights = randn(128, d) * sqrt(2/d);
                layer.Bias    = zeros(128,1);
            end
            
            % 全結合 (Weights, Bias) を適用
            features = layer.Weights * X + layer.Bias;
            
            % [1,1,128,batch] へ reshape
            features = reshape(features, [1,1,128,b]);
            
            % 出力を dlarray として返す（次元ラベルは SSCB 等）
            Z = dlarray(features, 'SSCB');
        end
        
        function [Z, memory] = forward(layer, X)
            % memory には後方伝搬で必要な入力（数値配列）を保存
            if iscell(X)
                memory = X{1};
            else
                memory = X;
            end
            Z = layer.predict(X);
        end
        
        function [dLdX, dLdW, dLdB] = backward(layer, X, Z, dLdZ, memory)
            % dLdZ が dlarray の場合は数値配列へ
            if isa(dLdZ, 'dlarray')
                dLdZ = extractdata(dLdZ);
            end
            
            [d, b] = size(memory);
            % dLdZ を [128, b] に reshape
            dLdFeatures = reshape(dLdZ, [128, b]);
            
            %----------------------
            % 1) dLdX の計算
            %----------------------
            %   dLdX = W' * dLdFeatures
            dLdX = layer.Weights' * dLdFeatures;
            % dLdX は学習の微分追跡をしないなら数値配列のままでもよいが、
            % dlarray に変換しておく場合は以下
            dLdX = dlarray(dLdX);
            
            %----------------------
            % 2) dLdW の計算
            %----------------------
            %   dLdW = dLdFeatures * memory'
            %   memory が forward 時の入力 [d, b]
            dLdW = dLdFeatures * memory';
            
            %----------------------
            % 3) dLdB の計算
            %----------------------
            %   バッチ方向に勾配を合計
            dLdB = sum(dLdFeatures, 2);
        end
    end
end