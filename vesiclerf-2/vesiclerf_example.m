addpath(genpath('../../../../evaluation/'));
addpath(genpath('./tools'));

numWorkers = 1;
%pp = configure_parpool(numWorkers);

%% Create file locationps.
date = datestr(now,'yyyy_mm_dd__HH_MM_ss');
fname = strcat('./vesiclerf_', date);
mkdir(fname);
classifier_name = strcat(fname, '/train_classifier.mat');

%% Constants required for VRF.
padX = 50; padY = 50; padZ = 2;
scale = 1;

fid = fopen(strcat(fname, '/report.txt'),'w+');
fprintf(fid, 'Training VESICLE-RF classifier.\n');
fprintf(fid, strcat('Conducted at ', datestr(now, 'yyyy-mm-dd--HH-MM-ss'), '.\n'));
fclose(fid);

%% TRAIN CLASSIFIER.
vesiclerf_train(classifier_name, 'train', scale, fname)

%% PIXEL CLASSIFICATION
train_output = vesiclerf_probs('train', classifier_name, padX, padY, padZ, fname);
val_output = vesiclerf_probs('validation', classifier_name, padX, padY, padZ, fname);
test_output = vesiclerf_probs('test', classifier_name, padX, padY, padZ, fname);

%% METRICS COMPUTATION
wrap_synapse_pr(fname, 'syn')
wrap_voxel_pr(fname, 'syn')
