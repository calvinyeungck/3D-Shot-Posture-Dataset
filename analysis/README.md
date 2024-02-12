# 3D Shot Posture Analysis
This guide pertains to conducting clustering analysis using 3D posture data. The clustering results and cluster analysis are elaborated upon in the paper.

## Clustering with Unsupervised Deep Learning and Kmean
Step 1. Pre-process the data
```
cd path/to/this/repo/
cd 3dsp_utils
python pre_process.py
``` 
Step 2. Perform Clustering
```
cd path/to/this/repo/
cd analysis
python deep_learning+kmean/clustering.py
```
### Additional features
1. Analyze the cluster with the ankle joint traveled distance and the knee, hip, and shoulder angle.
```
cd path/to/this/repo/
cd analysis
python deep_learning+kmean/clustering.py
```
2. Train the GCN LSTM AE model
```
cd path/to/this/repo/
cd analysis
python deep_learning+kmean/train.py
```
## Clustering with PCA and Kmean
Step 1. Pre-process the data
```
cd path/to/this/repo/
cd 3dsp_utils
python pre_process.py
``` 
Step 2. Perform Clustering
```
cd path/to/this/repo/
cd analysis
python pca+kmean/pca+kmean.py
```

## Reference
 Please consider citing our work if you find it helpful to yours:

```
TBA
```