function [metrics] = pr_evaluate_synapse_logit(path, h5File, channel, pp)
% Function to compute a precision recall curve across a wide range of parameters
%
% **Inputs**
%
% h5File: (string)
%   - Location of a .h5 file that contains the predicted logits, as well as
%     the ground truth values.
%
% channel: (string)
%   - Absolute filepath of the analyte of interest in the target h5 file.
%     {'USyn', 'syn', 'ves', 'mem'}
%
% set: (string)
%   - The set from which we are comapring.
%     {'train', 'validation', 'test'}
%
% metricsFile: (string)
%   - Location of mat file containing metrics data from conducting the sweep.
%
% **Outputs**
%
%	No explicit outputs.  Output file is saved to disk rather than
%	output as a variable to allow for downstream integration with LONI.
%
% ** Notes**
%
% This function grid searches the parameter space with no optimizations for clarity.  pr_object provides an alternative, simpler sweep.

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
    load(strcat('./', path, '/', channel, '_synapse_metrics_optimized_val.mat'))
    test = true;
    state = 'test';
    
    disp('Beginning application of synapse-level hyperparameters (pr_evaluate_synapse_logit.m; test).')

else
    % 
    minSize2DVals = [0, 10, 25, 50, 75, 100, 150];
    maxSize2DVals = [500, 750, 1000, 1250, 1500, 2000]; 
    minSize3DVals = [250, 500, 750, 1000, 1250, 1500]; 
    thresholdVals = -5:0.1:5;
    minSliceVals = [1,2,3];
    if contains(h5File, 'train')
        train = true;
        state = 'train';
    else
        val = true;
        state = 'val';
    end    
    
    disp('Beginning synapse-level hyperparameter optimization (pr_evaluate_synapse_logit.m).')

end

t = tic;
clear precision recall dMtx thresh minSize
count = 1;
pad = [0,0,0]; % assume padding is taken care of externally
maxCount = length(minSize2DVals)*length(maxSize2DVals)*length(minSize3DVals)*length(thresholdVals)*length(minSliceVals);

%% Construct the parameter matrix for the parfor loop.
%{minSize2D, maxSize2D, minSize3D, threshold, minSlice}.
param_matrix = zeros(maxCount, 5);
counter = 1;

for minSize2D = minSize2DVals
    for maxSize2D = maxSize2DVals
        for minSize3D = minSize3DVals
            for threshold = thresholdVals
                for minSlice = minSliceVals
                    param_matrix(counter, 1) = minSize2D;
                    param_matrix(counter, 2) = maxSize2D;
                    param_matrix(counter, 3) = minSize3D;
                    param_matrix(counter, 4) = threshold;
                    param_matrix(counter, 5) = minSlice;
                    counter = counter + 1;
                end
            end
        end
    end
end

% Define storage vectors.
precision = zeros(maxCount, 1);
recall = zeros(maxCount, 1);
f1 = zeros(maxCount, 1);
thresh = zeros(maxCount, 1);
minSize2DOut = zeros(maxCount, 1);
minSize3DOut = zeros(maxCount, 1);
maxSize2DOut = zeros(maxCount, 1);
minSliceOut = zeros(maxCount, 1);

%% Hyp-opt loop.

st = tic;

for i = 1:maxCount
    
    minSize2D = param_matrix(i,1);
    maxSize2D = param_matrix(i,2);
    minSize3D = param_matrix(i,3);
    threshold = param_matrix(i,4);
    minSlice = param_matrix(i,5);
    
    temp_prob = detectFileOnes;
    %fprintf('NOW PROCESSING SEARCH %d of %d...\n', i, maxCount)
    % POST PROCESSING
    % threshold prob
    temp_prob(temp_prob + threshold > detectFileZeros) = 1;
    temp_prob(temp_prob + threshold <= detectFileZeros) = 0;

%                     se = strel('disk',2);
%                     temp_prob = imdilate(temp_prob,se);

    % Check 2D size limits first
    cc = bwconncomp(temp_prob,4);
    %Apply object size filter
    for jj = 1:cc.NumObjects
        if length(cc.PixelIdxList{jj}) < minSize2D || length(cc.PixelIdxList{jj}) > maxSize2D
            temp_prob(cc.PixelIdxList{jj}) = 0;
        end
    end

    % get size of each region in 3D
    cc = bwconncomp(temp_prob,6);

    % check 3D size limits and edge hits
    for ii = 1:cc.NumObjects
        %to be small
        if length(cc.PixelIdxList{ii}) < minSize3D
            temp_prob(cc.PixelIdxList{ii}) = 0;
        end
    end
    temp_prob = pr_minSliceEnforce(temp_prob,minSlice);

    % re-run connected components
    detectcc = bwconncomp(temp_prob,18);
    detectMtx = labelmatrix(detectcc);

    % POST PROCESSING
    stats2 = regionprops(detectcc,'PixelList','Area','Centroid','PixelIdxList');

    %fprintf('Number Synapses detected: %d\n',length(stats2));

    % 3D metrics

    %Scale 1:
    truthMtx = sDataTest(pad(1)+1:end-pad(1),pad(2)+1:end-pad(2), pad(3)+1:end-pad(3));
    truthObj = bwconncomp(truthMtx,18);

    TP = 0; FP = 0; FN = 0; TP2 = 0;

    for j = 1:truthObj.NumObjects
        temp =  detectMtx(truthObj.PixelIdxList{j});

        if sum(temp > 0) >= 10 %50 %at least 25 voxel overlap to be meaningful
            TP = TP + 1;

            % TODO any detected objects can only be used
            % once, so remove them here

            % This does not penalize (or reward) fragmented
            % detections
            detectIdxUsed = unique(temp);
            detectIdxUsed(detectIdxUsed == 0) = [];

            for jjj = 1:length(detectIdxUsed)
                detectMtx(detectcc.PixelIdxList{detectIdxUsed(jjj)}) = 0;

            end
        else
            FN = FN + 1;
        end
    end

    %length(detectObj)
    for j = 1:detectcc.NumObjects
        temp =  truthMtx(detectcc.PixelIdxList{j});
        %sum(temp>0)
        if sum(temp > 0) >= 10%50 %at least 25 voxel overlap to be meaningful
            %TP = TP + 1;  %don't do this again, because already
            % considered above
            TP2 = TP2 + 1;
        else
            FP = FP + 1;
        end
    end

    precision(i) = TP./(TP+FP);
    recall(i) = TP./(TP+FN);
    f1(i) = (2*precision(i)*recall(i)) / (precision(i)+recall(i));
    thresh(i) = threshold;
    minSize2DOut(i) = minSize2D;
    minSize3DOut(i) = minSize3D;
    maxSize2DOut(i) = maxSize2D;
    minSliceOut(i) = minSlice;
    %fprintf('precision: %f recall: %f threshold %f minSize2D %d minSize3D %d maxSize2D %d minSlice %d\n',metrics.precision(count),metrics.recall(count), threshold, minSize2D, minSize3D, maxSize2D, minSlice);
end

metrics.precision = precision;
metrics.recall = recall;
metrics.F1 = f1;
metrics.thresh = thresh;
metrics.minSize2DOut = minSize2DOut;
metrics.minSize3DOut = minSize3DOut;
metrics.maxSize2DOut = maxSize2DOut;
metrics.minSliceOut = minSliceOut;

t = toc(st);
fprintf('Synapse-level hyperparameter optimization complete, time elapsed: %0.2f.\n', t);

save(strcat('./', path, '/', channel, '_synapse_metrics_full_', state), 'metrics');

if val
    [~, optimalBin] = max(metrics.F1);
    minSliceVals = [metrics.minSliceOut(optimalBin)];
    maxSize2DVals = [metrics.maxSize2DOut(optimalBin)];
    minSize3DVals = [metrics.minSize3DOut(optimalBin)];
    thresholdVals = [metrics.thresh(optimalBin)];
    minSize2DVals = [metrics.minSize2DOut(optimalBin)];
    precision = [metrics.precision(optimalBin)];
    recall = [metrics.recall(optimalBin)];
    F1 = [metrics.F1(optimalBin)];

    save(strcat('./', path, '/', channel, '_synapse_metrics_optimized_val'), 'minSliceVals', 'maxSize2DVals', 'minSize3DVals', 'thresholdVals', 'minSize2DVals', 'F1', 'recall', 'precision');
end

