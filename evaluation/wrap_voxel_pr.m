function wrap_voxel_pr(path, channel, numWorkers)
% Wrap the calling of the evaluation functions such that the hyperparameter
% sweep on the training set is done, and then automatically applied to the
% test set.

if exist('numWorkers')
    pp = configure_parpool(numWorkers);
else
    pp = configure_parpool;
end

pr_evaluate_voxel_logit(path, '/validation_results.h5', channel, pp);
pr_evaluate_voxel_logit(path, '/test_results.h5', channel, pp)

end
