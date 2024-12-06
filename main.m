% 使用例：params = getConfig('epocx');  manager = EEGAcquisitionManager(params);
% UDP系エラー初期化：clear all; instrreset;
% params = getConfig('epocx');   manager = EEGAcquisitionManager(params);
% params = getConfig('epocx', 'preset', 'motor_imagery');

params = getConfig('epocx');
manager = EEGAcquisitionManager(params);
