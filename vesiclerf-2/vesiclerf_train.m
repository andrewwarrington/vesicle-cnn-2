function vesiclerf_train(outputFile, set, scale, fname)
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

addpath(genpath(pwd))

%eData = h5read(strcat('../../../../../../kasthuri_data/Data/', set, '/', set, '.h5'),'/image');
%sData = h5read(strcat('../../../../../../kasthuri_data/Data/', set, '/', set, '.h5'),'/synapse');
%vData = h5read(strcat('../../../../../../kasthuri_data/Data/', set, '/', set, '.h5'),'/vesicle');
%mData = h5read(strcat('../../../../../../kasthuri_data/Data/', set, '/', set, '.h5'),'/membrane');

eData = load(strcat('../../../../../../kasthuri_data/Data/', set, '/eData_', set, '.mat'));
eData = eData.cube.data;

mData = load(strcat('../../../../../../kasthuri_data/Data/', set, '/mData_', set, '.mat'));
mData = mData.cube.data;

sData = load(strcat('../../../../../../kasthuri_data/Data/', set, '/sData_', set, '.mat'));
sData = sData.cube.data;

vData = load(strcat('../../../../../../kasthuri_data/Data/', set, '/vData_', set, '.mat'));
vData = vData.cube.data;

if exist('scale','var')==1
    imSize = size(eData);
    images = imSize(3);
     
    % Image Data.
    temp = eData;
    eData = zeros(imSize(1)*scale,imSize(2)*scale,images);
    for i = 1:images
        eData(:,:,i) = imresize(temp(:,:,i),scale);
    end
    
    % Synapse Data.
    temp = sData;
    sData = zeros(imSize(1)*scale,imSize(2)*scale,images);
    for i = 1:images
        sData(:,:,i) = imresize(temp(:,:,i),scale);
    end
    
    % Vesicle Data.
    temp = vData;
    vData = zeros(imSize(1)*scale,imSize(2)*scale,images);
    for i = 1:images
        vData(:,:,i) = imresize(temp(:,:,i),scale);
    end
    
    % Membrane Data.
    temp = mData;
    mData = zeros(imSize(1)*scale,imSize(2)*scale,images);
    for i = 1:images
        mData(:,:,i) = imresize(temp(:,:,i),scale);
    end
end


% Find valid pixels
mThresh = 0.75;
mm = (mData>mThresh);
%mm = imdilate(mm,strel('disk',5));
%mm = bwareaopen(mm, 1000, 4);
pixValid = find(mm > 0);

disp('Beginning training (vesiclerf_train.m).')
st = tic;

% Extract Features
Xtrain = vesiclerf_feats(eData, pixValid, vData);

% Classifier training
Ytrain = create_labels_pixel(sData, pixValid, [50,50,2]);

% Classifier training
trTarget = find(Ytrain>0);
trClutter = find(Ytrain==0);

idxT = randperm(length(trTarget));
idxC = randperm(length(trClutter));

nPoints = min(200000, length(trTarget)); % todo - should be 200,000

trainLabel = [Ytrain(trTarget(idxT(1:nPoints))); ...
    Ytrain(trClutter(idxC(1:nPoints)))];

trainFeat = [Xtrain(trTarget(idxT(1:nPoints)),:); ...
    Xtrain(trClutter(idxC(1:nPoints)),:)];

classifier = classRF_train(double(trainFeat),double(trainLabel),...
    200,floor(sqrt(size(trainFeat,2))));%,extra_options);  % todo - should be 200.
classifier.scale = scale;

save(outputFile,'classifier', '-v7.3');

t = toc(st);

fprintf('Training complete, time elapsed: %0.2f.\n', t);

fid = fopen(strcat(fname, '/report.txt'),'a');
fprintf(fid,'Training complete, time elapsed: %0.2f.\n', t);
fclose(fid);

