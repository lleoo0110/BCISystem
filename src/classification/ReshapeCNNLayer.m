classdef ReshapeCNNLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties (Learnable)
        Weights
        Bias
    end
    
    methods
        function layer = ReshapeCNNLayer(name)
            layer.Name = name;
            layer.Description = 'Reshape CNN output to 1x1x128 format';
            layer.Type = 'Reshape';
            layer.Weights = [];
            layer.Bias = [];
        end
        
        function Z = predict(layer, X)
            [h, w, c, b] = size(X);
            features = reshape(X, [h*w*c, b]);
            
            % 初回実行時に重み初期化
            if isempty(layer.Weights)
                inputSize = h*w*c;
                layer.Weights = randn(128, inputSize) * sqrt(2/inputSize);
                layer.Bias    = zeros(128,1);
            end
            
            % 全結合 (Weights, Bias) を適用
            features = layer.Weights * features + layer.Bias;
            
            % [1,1,128,batch] へ reshape
            Z = reshape(features, [1,1,128,b]);
        end
        
        function [Z, memory] = forward(layer, X)
            Z = layer.predict(X);
            % memory には後方伝搬で必要な情報（ここでは X）を保存
            memory = X;
        end
        
        function [dLdX, dLdW, dLdB] = backward(layer, X, Z, dLdZ, memory)
            % 入力 X (memory) の形状
            [h, w, c, b] = size(memory);
            
            % dLdZ (勾配) を [128, b] へ reshape
            dLdFeatures = reshape(dLdZ, [128, b]);
            
            %----------------------
            % 1) dLdX の計算
            %----------------------
            %   dLdX = W' * dLdFeatures
            dLdX = layer.Weights' * dLdFeatures;
            dLdX = reshape(dLdX, [h, w, c, b]);
            
            %----------------------
            % 2) dLdW の計算
            %----------------------
            %   入力 X を [h*w*c, b] にフラット化しておき、
            %   dLdW = dLdFeatures * X'
            features = reshape(memory, [h*w*c, b]);
            dLdW = dLdFeatures * features';
            
            %----------------------
            % 3) dLdB の計算
            %----------------------
            %   dLdB はバッチ方向に勾配を合計
            dLdB = sum(dLdFeatures, 2);
        end
    end
end