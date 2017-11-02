function [metrics] = pr_evaluate_voxel_logit(path, h5File, channel, pp)
% Takes inputs of the predicted volume (matrix or location) and the truth
% volume (matrix or location) and sweeps through thresholds of probability,
% evalating F1 (and F1 related metrics) at each bin. Then saves this data
% to saveloc.

evaluation_bins = 500;

test = false;
val = false;
train = false;
state = '';

detectFileOnes = h5read(strcat(path, h5File), strcat('/', channel, '/ones'));
detectFileZeros = h5read(strcat(path, h5File), strcat('/', channel, '/zeros'));
sDataTest = h5read(strcat(path, h5File), strcat('/', channel, '/truth'));

% Are we test? in which case load the preoptimized.
if contains(h5File, 'test')
    % Load optimized settings.
    load(strcat(path, '/', channel, '_voxel_metrics_optimized_val.mat'))
    metrics.thresholds = thresholds;
    test = true;
    state = 'test';
    evaluation_bins = 1;
    fprintf('Beginning voxel-level hyperparameter optimization (pr_evaluation_voxel_logit.m; test).\n');
else
    metrics.thresholds = linspace(-10,10,evaluation_bins);
    fprintf('Beginning voxel-level hyperparameter optimization (pr_evaluation_voxel_logit.m; val).\n');
    if contains(h5File, 'train')
        train = true;
        state = 'train';
    else
        val = true;
        state = 'val';
    end    
end

st = tic;

precision = zeros(evaluation_bins,1);
recall = zeros(evaluation_bins,1);
TP = zeros(evaluation_bins,1);
FP = zeros(evaluation_bins,1);
FN = zeros(evaluation_bins,1);
F1 = zeros(evaluation_bins,1);

parfor i = 1:evaluation_bins
    metricsTemp = pr_voxel(detectFileOnes>detectFileZeros+metrics.thresholds(i), sDataTest);
    
    precision(i) = metricsTemp.precision;
    recall(i) = metricsTemp.recall;
    TP(i) = metricsTemp.TP;
    FP(i) = metricsTemp.FP;
    FN(i) = metricsTemp.FN;
    F1(i) = metricsTemp.F1;
end

metrics.precision = precision;
metrics.recall = recall;
metrics.TP = TP;
metrics.FP = FP;
metrics.FN = FN;
metrics.F1 = F1;

t = toc(st);
fprintf('Voxel-level hyperparameter optimization complete, time elapsed: %0.2f.\n', t);

save(strcat(path, '/', channel, '_voxel_metrics_full_', state), 'metrics')

if val
    [~, optimalBin] = max(metrics.F1);
    thresholds = [metrics.thresholds(optimalBin)];
    precision = [metrics.precision(optimalBin)];
    recall = [metrics.recall(optimalBin)];
    TP = [metrics.TP(optimalBin)];
    FP = [metrics.FP(optimalBin)];
    FN = [metrics.FN(optimalBin)];
    F1 = [metrics.F1(optimalBin)];
    save(strcat(path, '/', channel, '_voxel_metrics_optimized_val'), 'thresholds', 'precision', 'recall', 'F1');
end


function [metrics] = pr_voxel(detectVol, truthVol)
% Adapted from W. Gray Roncal - 02.12.2015
% Feedback welcome and encouraged.  Let's make this better!
% Other options (size filters, morphological priors, etc. can be added.
%

TP = sum(sum(sum((detectVol ~= 0) & (truthVol ~= 0))));
FP = sum(sum(sum((detectVol ~= 0) & (truthVol == 0))));
FN = sum(sum(sum((detectVol == 0) & (truthVol ~= 0))));

metrics.precision = TP./(TP+FP);
metrics.recall = TP./(TP+FN);
metrics.TP = TP;
metrics.FP = FP;
metrics.FN = FN;
metrics.F1 = 2*metrics.precision*metrics.recall/(metrics.precision+metrics.recall);
