function vesiclerf_train(outputFile)
% Function to train a RF classifier on a fixed region of interest (AC4)
%
% **Inputs**
%
% outputFile: (string)
%   - Location of output file containing classifier, saved as a matlab file in a variable named 'classifier'
%
% **Outputs**
%
%	No explicit outputs.  Output file is saved to disk rather than
%	output as a variable to allow for downstream integration with LONI.
%
% **Notes**
%
%	Currently training region is hardcoded, but can be adjusted as needed by getting new '\*DataTrain'.

set = 'train'

addpath(genpath(pwd))

eData = load(strcat('../../../../../../kasthuri_data/', set, '/eData_', set, '.mat'));
eData = eData.cube.data;

mData = load(strcat('../../../../../../kasthuri_data/', set, '/mData_', set, '.mat'));
mData = mData.cube.data;

sData = load(strcat('../../../../../../kasthuri_data/', set, '/sData_', set, '.mat'));
sData = sData.cube.data;

vData = load(strcat('../../../../../../kasthuri_data/', set, '/vData_', set, '.mat'));
vData = vData.cube.data;

% Find valid pixels
mThresh = 0.75;
mm = (mDataTrain.data>mThresh);
mm = imdilate(mm,strel('disk',5));
mm = bwareaopen(mm, 1000, 4);
pixValid = find(mm > 0);

% Extract Features
st = tic
Xtrain = vesiclerf_feats(eDataTrain.data, pixValid, vDataTrain);

% Classifier training
Ytrain = create_labels_pixel(sDataTrain.data, pixValid, [50,50,2]);

% Classifier training
trTarget = find(Ytrain>0);
trClutter = find(Ytrain==0);

idxT = randperm(length(trTarget));
idxC = randperm(length(trClutter));

disp('Ready to classify')
nPoints = min(200000, length(trTarget));

trainLabel = [Ytrain(trTarget(idxT(1:nPoints))); ...
    Ytrain(trClutter(idxC(1:nPoints)))];

trainFeat = [Xtrain(trTarget(idxT(1:nPoints)),:); ...
    Xtrain(trClutter(idxC(1:nPoints)),:)];

classifier = classRF_train(double(trainFeat),double(trainLabel),...
    200,floor(sqrt(size(trainFeat,2))));%,extra_options);

train_time = toc(st);
fprintf(strcat('Training time: ', str(train_time), ' seconds.'))

disp('training complete')
figure, bar(classifier.importance), drawnow

save(outputFile,'classifier', '-v7.3');
