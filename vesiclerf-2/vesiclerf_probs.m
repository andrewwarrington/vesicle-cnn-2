function [f_out] = vesiclerf_probs(set, classifier_file, padX, padY, padZ, outFile, numWorkers)
% Function to compute classifier probabilities on an input data cube.
%
% **Inputs**
%
% edata: (string)
%   - Location of mat file containing uint8 EM data RAMONVolume, saved as 'cube'.
%
% vesicles: (string)
%   - Location of mat file containing uint32 vesicle data RAMONVolume, saved as 'cube'.
%
% membrane: (string)
%   - Location of mat file containing float32 membrane data RAMONVolume, saved as 'cube'.
%
% classifier_file: (string)
%   - Location of classifier mat file, created during training.  Classifier is saved as the variable 'classifier'.
%
% padX (uint)
%   - Number representing value to crop the output volume in the x dimension.
%
% padY (uint)
%		- Number representing value to crop the output volume in the y dimension.
%
% padZ (uint)
%   - Number representing value to crop the output volume in the z dimension.
%
% outFile (string)
%   - Location of output file, saved as a matfile containing a RAMONVolume named 'cube'.  Contains result of applying classifier to input data.  Output cube is a probability map (float32).

disp('Beginning classification (vesiclerf_probs.m).')

if exist('numWorkers')
    pp = configure_parpool(numWorkers);
else
    pp = configure_parpool;
end

%load classifier
if ~isfield(classifier_file,'importance')
    load(classifier_file)
else
    classifier = classifier_file;
end

clear classifier_file

scale = classifier.scale;

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
        
    % Synapse Data.
    temp = sData;
    sData = zeros(imSize(1)*scale,imSize(2)*scale,images);
    for i = 1:images
        sData(:,:,i) = imresize(temp(:,:,i),scale);
    end
    
end

st = tic;

% Find valid pixels
if exist('membrane') %#ok<EXIST>
    mThresh = 0.75;
    mm = (mData>mThresh);
%     mm = imdilate(mm,strel('disk',5));
%     mm = bwareaopen(mm, 1000, 4);
    pixValid = find(mm > 0);
else % this exists as a starting point if membranes are unavailable
    pixValid = 1:numel(eData);
end

% Extract Feats and Run Classifier
Xtest = vesiclerf_feats(eData, pixValid, vData);


em = eData;

[xs, ys, zs] = size(em);

DV = zeros(xs*ys*zs,1);

disp('	Running RF classifier...')

votes_sTest = zeros(xs*ys*zs,2);

idxToTest = pixValid;
chunk = 1E6;
nChunk = min(ceil(length(idxToTest)/chunk),length(Xtest));
xTestChunked = cell(nChunk,1);
tIdxChunked = cell(nChunk,1);
yTestChunked = cell(nChunk,1);

for i = 1:nChunk
    if i < nChunk
        tIdx = (i-1)*chunk+1:i*chunk;
    else
        tIdx = (i-1)*chunk+1:length(idxToTest);
    end
    tIdxChunked{i} = tIdx;
    xTestChunked{i} = double(Xtest(tIdx,:));
end

for i = 1:nChunk % PARFOR
    [~,yTestChunked{i}] = classRF_predict(xTestChunked{i},classifier);
end

for i = 1:nChunk
    votes_sTest(idxToTest(tIdxChunked{i}),:) = yTestChunked{i};
end
    
if nChunk > 0
    DV = votes_sTest(:,2)./sum(votes_sTest,2);
end

DV(isnan(DV)) = 0;
% postprocessing
disp('	Post-processing data...')

DV = reshape(DV,[xs,ys,zs]);

% Need to crop padded region
DV = DV(padX+1:size(DV,1)-padX,padY+1:size(DV,2)-padY,padZ+1:size(DV,3)-padZ);

cropped_pred = DV;
cropped_pred_softmax = log(cropped_pred ./ (1 - cropped_pred));

cropped_im = eData(padX+1:size(eData,1)-padX,padY+1:size(eData,2)-padY,padZ+1:size(eData,3)-padZ);

sDataPR = sData(padX+1:size(eData,1)-padX,padY+1:size(eData,2)-padY,padZ+1:size(eData,3)-padZ);

t = toc(st);

fprintf('Classification complete, time elapsed (including feature extraction): %0.2f.\n',t);

fid = fopen(strcat(outFile, '/report.txt'),'a');
fprintf(fid,'Classification complete, time elapsed (including feature extraction): %0.2f.\n', t);
fclose(fid); 

%% Create the H5 datasets.

f_out = strcat(outFile, '/', set, '_results.h5');

h5create(f_out, '/image', size(cropped_im));
h5write(f_out, '/image', cropped_im);

h5create(f_out, '/syn/truth', size(sDataPR));
h5write(f_out, '/syn/truth', sDataPR);

h5create(f_out, '/syn/ones', size(cropped_pred_softmax));
h5write(f_out, '/syn/ones', cropped_pred_softmax);

h5create(f_out, '/syn/zeros', size(cropped_pred_softmax));
h5write(f_out, '/syn/zeros', zeros(size(cropped_pred_softmax)));


% (c) [2014] The Johns Hopkins University / Applied Physics Laboratory All Rights Reserved. Contact the JHU/APL Office of Technology Transfer for any additional rights.  www.jhuapl.edu/ott
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%    http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
