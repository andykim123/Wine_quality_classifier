Milestone 1
===========
WUSTL SP18 CSE517 Machine Learning
Application Project
The program takes two user inputs for dataset paths of red and white wine datasets. 
---
### Include files
* __*MultinomialLogRegression.py*__ : Multiclass classification - used built-in function and newton-cg as solver
* __*LogisticRegression.py*__: Binary classification - used built-in function on binary classified wine datasets
* __*decisionTree.py*__: Decision Tree
* ***decisionTreeBagging.py***: Decision Tree with Bagging
* __*randomForest.py*__: Random Forest
* __*adaboost.py*__: Adaboost
### Usage      

```
>> python LogisticRegression.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>
>> python LogisticRegression.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>
>> python decisionTree.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>
>> python decisionTree_bagging.py <DATASET_PATH_RED> <DATASET_PATH_WHITE>
>> python randomForest.py <DATASET_PATH_RED> <DATASET_PATH_WHITE> <n_estimator>
>> python adaBoost.py <DATASET_PATH_RED> <DATASET_PATH_WHITE> n_estimator
```

### Resources used
* [Scikit Learn](http://scikit-learn.org/stable/)
* Wine Data set imported from [UCI data repository](http://archive.ics.uci.edu/ml/datasets.html)
### Authors:
* Jae Sang Ha
* Ryun Han
* Andy Dohoon Kim
* Nigel Kim
* Annie Lee
