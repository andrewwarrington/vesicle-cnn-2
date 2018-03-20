# vesicle-cnn-2
TL;DR: An updated version of the synapse detector, VESICLE, presented by Roncal et al. (2014).
Please cite Warrington, Andrew and Wood, Frank "Updating the VESICLE-CNN Synapse Detector" arXiv preprint arXiv:1710.11397 (2017) ([link](https://arxiv.org/abs/1710.11397?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+StatArxiv+%28arXiv.org%29)) for the architecture.
Please cite Kasthuri, Narayanan, et al. "Saturated reconstruction of a volume of neocortex." Cell 162.3 (2015): 648-661 ([link](http://www.cell.com/abstract/S0092-8674(15)00824-7)) for the data used in this work.

## Repo struture
./vesicle-cnn-2 contains scripts and examples for the updated classifier we present.

./vesiclerf-2 contains scripts for a slightly reworked version of the RF classifier to make use of multithreading and out data formatting. Development code still.

./Prior\_art contains (currently just a single) previous implementation of synapse detection algorithms, predominantly unchanged. Minor reworks have been applied for compatibility and hence have been included here. 

./evaluation contains a number of scripts and tools for use in MATLAB for evaluating the voxelwise and whole-synapse performance of each algorithm using the .h5 file format.

## Running the code
Before running the code, the datafile must be downloaded and unpacked. This is done automatically by calling the `download_data.sh` script. 
From there, navigating to ./vesiclecnn-2 will place you in the directory containing all of the scripts for training and deployment. 

We recommend however running the code from a Docker container for reproducibility and portability. Docker images are maintained on the Docker Hub, both for ([CPU]https://hub.docker.com/r/ajwarrington/vcnn2-cpu/) and ([GPU]https://hub.docker.com/r/ajwarrington/vcnn2/) training. The CPU version is considered stable, although the GPU version is not (the CUDA installation seems system dependent...). Therefore, for GPU training, we recommend building the docker image locally using the command `docker build -t ajwarringtion/vcnn2 -f Dockerfile.gpu .`. The image is then deployed using the command `nvidia-docker run -it ajwarrington/vcnn2`. Of course, CUDA, docker and nvidia-docker must be installed and configured for this to work. We use a modified version of the ([Deepo docker image]https://hub.docker.com/r/ufoym/deepo/) as the baseimage for this. Deepo is a great, lightweight docker image that we found to be pretty usable for deep learning related tasks.

Most of the functionality is in the `vesicle-cnn-2.py` script, with a collection of utilities defined in `util.py`.
The default architecture is predefined in the main script. Modifications to the main architecture require changing the variables defined near the start of `vesicle-cnn-2.py`.
The architecture is then deployed using the `Makefile` provided. The command `make train-only` will load the data using the defaut partioning and weights, and will save the results of the training to its down directory, tagged with the date and time. `make train-and-deploy` will train a new network as before, but will also deploy the newly trained network to the train, validate and test sets. If the network is deployed to the validation and test set, a MATLAB script, derived from the original VESICLE implementation will also be called solving for the whole-synapse PR characteristics of the newly trained classifier. `make deploy-pretrained` deploys a pretrained network to all datasets when a filepath to a valid architecture is provided.

## Data availability
WE DO NOT OWN THIS DATA, WE SIMPLY REPACKAGE IT FOR CONVENIENCE. ALL REFERENCES FOR THIS DATA __MUST__ BE ATTRIBUTED TO Kasthuri, Narayanan, et al. "Saturated reconstruction of a volume of neocortex." Cell 162.3 (2015): 648-661 ([link](http://www.cell.com/abstract/S0092-8674(15)00824-7)). AND REFERENCES FOR PUBLIC ACCESS TO DATA TO [NeuroData](https://neurodata.io>). 

Data will be made available for download at <http://www.robots.ox.ac.uk/~andreww/synapse_data.html>. 
For this code to operate without adjustment, this data must be downloaded and unzipped into the root file of this repository so that the scripts can automatically grab the required data.
