classdef ReshapeCNNLayer < nnet.layer.Layer & nnet.layer.Formattable
    % ReshapeCNNLayer
    %   CNNブランチの出力を 4 次元テンソル [1 1 NumFeatures N] に
    %   リシェイプするカスタムレイヤーです。
    
    properties
        NumFeatures   % CNNブランチの最終出力特徴数
    end
    
    properties (Learnable, SetAccess = private)
        % Learnableパラメータはありません
    end
    
    properties
        OutputFormat  % 出力のフォーマット（例："SSCB"）
    end
    
    methods
        function layer = ReshapeCNNLayer(name, numFeatures)
            % コンストラクタ
            if nargin < 2
                numFeatures = [];
            end
            layer.Name = name;
            layer.Description = "Reshape CNN branch output to [1 1 NumFeatures N]";
            layer.NumFeatures = numFeatures;
            layer.OutputFormat = ""; % 初期は空文字
        end
        
        function Z = predict(layer, X)
            % 既存の処理で Z を reshape して得る
            if ismatrix(X)
                N = size(X,2);
                if size(X,1) ~= layer.NumFeatures
                    error('Input first dimension (%d) does not match expected NumFeatures (%d).', ...
                          size(X,1), layer.NumFeatures);
                end
                Z = reshape(X, [1, 1, layer.NumFeatures, N]);
            elseif ndims(X)==3
                if size(X,1)==1
                    N = size(X,3);
                    if size(X,2) ~= layer.NumFeatures
                        error('Input second dimension (%d) does not match expected NumFeatures (%d).', ...
                              size(X,2), layer.NumFeatures);
                    end
                    Z = reshape(X, [1, 1, layer.NumFeatures, N]);
                else
                    N = size(X,3);
                    flattened = reshape(X, [], N);
                    if size(flattened,1) ~= layer.NumFeatures
                        error('Flattened dimension (%d) does not match expected NumFeatures (%d).', ...
                              size(flattened,1), layer.NumFeatures);
                    end
                    Z = reshape(flattened, [1, 1, layer.NumFeatures, N]);
                end
            elseif ndims(X)==4
                Ztemp = squeeze(mean(mean(X,1),2));  % [C x N]
                if size(Ztemp,1) ~= layer.NumFeatures
                    error('Pooled channel dimension (%d) does not match expected NumFeatures (%d).', ...
                          size(Ztemp,1), layer.NumFeatures);
                end
                N = size(Ztemp,2);
                Z = reshape(Ztemp, [1, 1, layer.NumFeatures, N]);
            else
                error('Unsupported input dimensions.');
            end
        
            % もし入力が dlarray なら、同じフォーマット情報を出力にも付与する
            if isa(X, 'dlarray')
                % ここでは必ず 'SSCB' とするか、もしくは X.Format を利用する
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
            % 出力フォーマットを設定するメソッド
            layer.OutputFormat = format;
        end
    end
end
