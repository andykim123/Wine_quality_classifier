Milestone 2
===========

WUSTL SP18 CSE517 Machine Learning  
Application Project 
The program takes two user inputs for dataset paths of red and white wine datasets. 


---
### Include files
* __*clustering.py*__: Clusters dataset into 8 clusters
* __*gmm.py*__: Gaussian Mixture Model, produces probability density functions for k-clusters
* __*GP.py*__ : Multiclass classification - used built-in function and newton-cg as solver
* __*modelEvaluation.py*__: Uses t-test to compare two models
* __*kernelSVM.py*__: Uses RBF kernel with gamma=0.1 to model support vector machine

__Refer to milstone 1 for..__  
* __*adaBoost.py*__
* __*decisionTree*__
* __*MultinomialLogRegression.py*__
* __*randomForest.py*__

### Usage      

```
>> python clustering.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>
>> python gmm.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>
>> python GP.py <DATASET_PATH_RED> OPTIONAL(<TEXT_DATA_PATH>) <RUN INFILE BOOLEAN>
>> python modelEvaluation.py <NAME OF MODEL 1 FILE> <NAME OF MODEL 2 FILE> <DATASET_PATH>
>> python kernelSVM.py <DATASET_PATH_RED> <RUN INFILE BOOLEAN>
```

### Resources used
* [Scikit Learn](http://scikit-learn.org/stable/)
* "Wine Quality" dataset imported from [UCI data repository](http://archive.ics.uci.edu/ml/datasets.html)
### Authors:
* Jae Sang Ha
* Ryun Han
* Andy Dohoon Kim
* Nigel Kim
* Annie Lee
