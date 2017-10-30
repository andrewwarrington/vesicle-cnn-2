function vesiclerf_example()
% Driver function to demonstrate vesicle-rf functionality for new users.
%
% **Inputs**
%
%	None.  Driver script is self-contained.
%
% **Outputs**
%
%	None.  Driver script is self-contained.
%
% **Notes**
%

padX = 50; padY = 50; padZ = 2;

%Get data

% When deploying vesicle-rf, the core of the code consists of
% vesiclerf_probs and vesiclerf_object

%% TRAIN CLASSIFIER
vesiclerf_train('example_classifier')
%% PIXEL CLASSIFICATION
vesiclerf_probs('example_classifier', padX, padY, padZ, 'classProbTrain.mat', 'train')
vesiclerf_probs('example_classifier', padX, padY, padZ, 'classProbValidation.mat', 'validation')
vesiclerf_probs('example_classifier', padX, padY, padZ, 'classProbTest.mat', 'test')
%% OBJECT PROCESSING
vesiclerf_object('classProbTest.mat', 0.90, 0, 5000, 2000, 1, 0, 0, 'testObjVol.mat', 0, 0, 0)

%% METRICS COMPUTATION
pr_objects('testObjVol','sDataPR','metrics_short')
pr_evaluate_full('classProbTest','sDataPR','metrics_full')
load metrics_full
metrics
