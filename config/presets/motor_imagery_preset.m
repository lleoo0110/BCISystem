function preset = motor_imagery_preset()
    preset = struct(...
        'signal', struct(...
            'window', struct(...
                'analysis', 2.0, ...          % 解析窓2秒
                'stimulus', 5.0, ...          % 刺激時間5秒
                'bufferSize', 10, ...         % バッファサイズ10秒
                'updateBuffer', 0.5 ...       % 0.5秒ごとに更新
            ), ...
            'frequency', struct(...
                'min', 8, ...                 % μ波帯域の下限
                'max', 30, ...                % β波帯域の上限
                'bands', struct(...
                    'mu', [8 12], ...         % μ波帯域
                    'beta', [13 30] ...       % β波帯域
                ) ...
            ), ...
            'filter', struct(...
                'notch', struct(...
                    'enabled', true, ...
                    'frequency', [50 60], ...  % 電源ノイズ除去
                    'bandwidth', 2 ...
                ), ...
                'fir', struct(...
                    'enabled', true, ...
                    'scaledPassband', true, ...
                    'filterOrder', 100, ...
                    'designMethod', 'window', ...
                    'windowType', 'hamming' ...
                ) ...
            ) ...
        ), ...
        'feature', struct(...
            'power', struct(...
                'enable', true, ...
                'method', 'welch', ...         % Welch法でパワー計算
                'normalize', true, ...
                'welch', struct(...
                    'windowLength', 256, ...
                    'overlap', 0.5, ...
                    'nfft', 512 ...
                ), ...
                'bands', struct(...
                    'names', {{'alpha', 'beta'}} ...
                ) ...
            ), ...
            'csp', struct(...
                'enable', true, ...           % CSP特徴量を使用
                'patterns', 6, ...            % 上位3ペアのCSPパターン
                'regularization', 0.05 ...
            ) ...
        ), ...
        'classifier', struct(...
            'svm', struct(...
                'enable', true, ...
                'type', 'ecoc', ...
                'kernel', 'rbf', ...
                'optimize', true, ...
                'probability', true, ...
                'hyperparameters', struct(...
                    'optimizer', 'gridsearch', ...
                    'boxConstraint', [0.1, 1, 10], ...
                    'kernelScale', [0.1, 1, 10] ...
                ) ...
            ) ...
        ) ...
    );
end