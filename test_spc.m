% test_spc.m
clc; clear; close all;


addpath(genpath(pwd));


data_dir     = './balance_uni/';
file_prefix  = '15_0.6balance_uni__';  
idx_list     = 1:9;                    


lambda   = 1e3;
alpha_spc = 1;
T        = 5;


err = nan(numel(idx_list),1);

for ii = 1:numel(idx_list)
    idx = idx_list(ii);
    fname = fullfile(data_dir, [file_prefix, num2str(idx), '.mat']);
    S = load(fname);  

    Y_obs = S.Y_obs;
    gt    = S.ground_truth(:);
    valid = (gt > 0);

    [~, ~, pred, ~] = SPC(Y_obs, lambda, alpha_spc, T);
    pred = pred(:);
    err(ii) = mean(pred(valid) ~= gt(valid));

    fprintf('idx=%d  error=%.4f\n', idx, err(ii));
end

