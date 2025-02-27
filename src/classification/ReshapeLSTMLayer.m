classdef ReshapeLSTMLayer < nnet.layer.Layer & nnet.layer.Formattable
    % ReshapeLSTMLayer
    %   LSTMブランチの全結合層出力 [NumFeatures x N] を 4 次元テンソル
    %   [1 1 NumFeatures N] にリシェイプするカスタムレイヤーです。
    
    properties
        NumFeatures   % LSTM全結合層の出力ユニット数
    end
    
    properties (Learnable, SetAccess = private)
        % Learnableパラメータはありません
    end
    
    properties
        OutputFormat  % 出力のフォーマット（例："SSCB"）
    end
    
    methods
        function layer = ReshapeLSTMLayer(name, numFeatures)
            layer.Name = name;
            layer.Description = "Reshape LSTM branch output to [1 1 NumFeatures N]";
            layer.NumFeatures = numFeatures;
            layer.OutputFormat = "";
        end
        
        function Z = predict(layer, X)
            N = size(X,2);
            if size(X,1) ~= layer.NumFeatures
                error('Input first dimension (%d) does not match expected NumFeatures (%d).', ...
                      size(X,1), layer.NumFeatures);
            end
            Z = reshape(X, [1, 1, layer.NumFeatures, N]);
            if isa(X, 'dlarray')
                Z = dlarray(Z, 'SSCB');
            end
        end
        
        function [Z, memory] = forward(layer, X)
            Z = layer.predict(X);
            memory = [];
        end
        
        function dX = backward(layer, X, Z, dZ, memory)
            dX = reshape(dZ, size(X));
        end
        
        function layer = setOutputFormat(layer, format)
            layer.OutputFormat = format;
        end
    end
end
