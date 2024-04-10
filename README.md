<h1 align="center">Cellpose-Quantification</h1>

A repository for single-cell quantification using mask formats generated by cellpose!
\
<h2 align="center">Getting Started</h2>
<h3 align="center">Installing Dependancies</h2>

* Clone this git repository locally
* Setup Python environment and Install packages

```
pip install -r requirements.txt
```
<h3 align="center">Executing program</h2>

This program can be run from the command line with the following example command:
```
python RunQuantification-CLI.py ./path/to/images ./path/to/masks markers.csv --normalization
```
<h3 align="center">Expected Parameters</h2>

* ./path/to/image
  * Can take Input from a directory containing tiff images
* ./path/to/masks
  * Can take a directory .npy and image formats as inputs generated by cellpose
* markers.csv
  * The path to metadata relating to markers corresponding to each marker, formatted on one comma-separated line  CD45, DAPI, Ki67, etc
* --normalization
  * An optional parameter that normalises mean intensities using Z-score normalisation 
 

## Authors
[Miles Bailey](https://github.com/milesbailey121)  
[@milesbailey121](https://twitter.com/milesbailey121)
