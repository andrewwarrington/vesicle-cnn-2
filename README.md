# vesicle-cnn-2
TL;DR: An updated version of the synapse detector, VESICLE, presented by Roncal et al. (2014).

## Repo struture
./vesiclecnn-2 contains scripts and examples for the updated classifier we present.

./vesiclerf-2 contains scripts for a slightly reworked version of the RF classifier to make use of multithreading and out data formatting. Development code still.

./vesiclecnn contains a slight rework of the original script such that it runs in a dockerized environment (we had no end of difficulty getting Caffe installed and so went down the docker route). We cannot get this network to train properly and hence has been used solely for timing estimates through, and getting this script training is a work in progress.

./Prior\_art contains (currently just a single) previous implementation of synapse detection algorithms, predominantly unchanged. Minor reworks have been applied for compatibility and hence have been included here. 

./evaluation contains a number of scripts and tools for use in MATLAB for evaluating the voxelwise and whole-synapse performance of each algorithm using the .h5 file format.

## Data availability
Data soon to be made available at <http://www.robots.ox.ac.uk/~andreww/synapse_detection.html>
