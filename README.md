# vesicle-cnn-2
TL;DR: An updated version of the synapse detector, VESICLE, presented by Roncal et al. (2014).
Please cite Warrington, Andrew and Wood, Frank "Updating the VESICLE-CNN Synapse Detector" arXiv preprint arXiv:1710.11397 (2017) ([link](https://arxiv.org/abs/1710.11397?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+StatArxiv+%28arXiv.org%29)) for the architecture.
Please cite Kasthuri, Narayanan, et al. "Saturated reconstruction of a volume of neocortex." Cell 162.3 (2015): 648-661 ([link](http://www.cell.com/abstract/S0092-8674(15)00824-7)) for the data used in this work.

## Repo struture
./vesiclecnn-2 contains scripts and examples for the updated classifier we present.

./vesiclerf-2 contains scripts for a slightly reworked version of the RF classifier to make use of multithreading and out data formatting. Development code still.

./Prior\_art contains (currently just a single) previous implementation of synapse detection algorithms, predominantly unchanged. Minor reworks have been applied for compatibility and hence have been included here. 

./evaluation contains a number of scripts and tools for use in MATLAB for evaluating the voxelwise and whole-synapse performance of each algorithm using the .h5 file format.

## Data availability
WE DO NOT OWN THIS DATA, WE SIMPLY REPACKAGE IT FOR CONVENIENCE. ALL REFERENCES FOR THIS DATA __MUST__ BE ATTRIBUTED TO Kasthuri, Narayanan, et al. "Saturated reconstruction of a volume of neocortex." Cell 162.3 (2015): 648-661 ([link](http://www.cell.com/abstract/S0092-8674(15)00824-7)). AND REFERENCES FOR PUBLIC ACCESS TO DATA TO [NeuroData](https://neurodata.io>). 

Data will be made available for download at <http://www.robots.ox.ac.uk/~andreww/synapse_data.html>. 
For this code to operate without adjustment, this data must be downloaded and unzipped into the root file of this repository so that the scripts can automatically grab the required data.
