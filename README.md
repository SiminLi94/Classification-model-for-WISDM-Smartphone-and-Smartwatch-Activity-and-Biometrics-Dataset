# Classification-model-for-WISDM-Smartphone-and-Smartwatch-Activity-and-Biometrics-Dataset
This project introduces several methods and results of finding a good classification model for WISDM Smartphone and Smartwatch Activity and Biometrics Dataset.
## Introduction
The smartphone dataset consists of fitness 18 different activities recordings of 51 people captured through smartphone enabled with inertial sensors. The goal of this project is to build a classification model that can precisely identify human fitness activities.
## Chosen Models
### 1. Random Forest
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that combined bagging and decision tree.

The advantages of random forests include:
- The predictive performance can compete with the best supervised learning algorithms. 
- They provide a reliable feature importance estimate.
- They offer efficient estimates of the test error without incurring the cost of repeated model training associated with cross-validation. 

The process of building a random forest is as follows:
- Use the Bootstrapping method to select m samples randomly from the original training set, and select n samples for a total of n_tree samples to generate n_tree training sets.
- For n_tree training sets, we train n_tree decision tree models separately.
- For a single decision tree model, suppose the number of training sample features is n, then each time you split, select the best feature to split based on the information gain / information gain ratio / Gini index.
- Each tree is split like this until all training examples of the node belong to the same class. No need for pruning during splitting of decision tree.
- The generated multiple decision trees form a random forest. For classification problems, the final classification results are determined by voting on multiple tree classifiers; for regression problems, the average prediction value of multiple trees is used to determine the final prediction results.
### 2. Gradient Boosting
The advantages of GradienBoosting include:
- Often provides predictive accuracy that cannot be beat.
- Lots of flexibility - can optimize on different loss functions and provides several hyperparameter tuning options that make the function fit very flexible.
- No data pre-processing required - often works great with categorical and numerical values as is.
- Handles missing data - imputation not required.
- 
The disadvantages of GradienBoosting include:
- GBMs will continue improving to minimize all errors. This can overemphasize outliers and cause overfitting. Must use cross-validation to neutralize.
- Computationally expensive - GBMs often require many trees which can be time and memory exhaustive.
### 3. Decision Tree
We also test the single decision tree model. The advantages are as followed:
- Compared to other algorithms decision trees requires less effort for data preparation during pre-processing.
- A decision tree does not require scaling of data as well.
- Missing values in the data also does NOT affect the process of building decision tree to any considerable extent.
- A Decision trees model is very intuitive and easy to explain to technical teams as well as stakeholders.

According our test results, Decision Tree has the lowest accuracy.

## Code and Process
Our program contains four part: data processing, choosing hyperparameters, selecting features and training the model.
### 1. Data Processing
- Getting data from arff files. The data from arff files are the result after computing the raw data. They are four arff files which represent four machines (phone accel, phone gyno, watch accel and watch gyno) Each file contains 91 attributes.
- Changing label from alphabets A-S ( represent the different 18 activities) to numbers 1-18 for model training.
- Combining phone accel, gyno, watch accel, gyno features according to the time order.
- Shuffling dataset. 
### 2. Hyper-parameters
We use sklearn gridsearch function to decide the best hyper-parameters of each model.
1) Radom Forest
We use sklearn gridsearch function to decide the best hyper-parameters.
grid search
max_depth=30(10,20,30)(avoid over fit) n_estimators=1000 (100,500,1000,1500) criterion='entropy'('gini' , 'entropy')
The accuracy of the best hyper-parameters is 0.884. But when we used: RandomForestClassifier(n_estimators=100, max_depth=20,criterion="entropy")
The accuracy is 0.882.
Because the accuracy only decreased by about 0.002, we choose the later to reduce the cost ( time and memory exhaustive ).
2) Decision Tree
max_depth=20(10,20,30)(avoid over fit) 
criterion='entropy'('gini' , 'entropy')
3) GradienBoosting
It takes a long time to train a GradienBoosting model, as a result, we trained a GradienBoosting model with sklearn default hyperparameters.

### 3. Feature Selection
Each arff file contains 91 attributes(so we have 91*4=364 features in total). We need to check which attributes are more correlative to our model.

The image about correlation of variances of all 364 features and labels is as Fig. 1.

Using the sklearn SelectFromModel function with random forest can help us reduce the feature. It calculate each entropy and choose those important features. Fig.2 shows the results for some features. We use threshold=0.005 to reduce the features number to.

We finally decided to use PCA (Principal components analysis) to reduce features to 50. Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components. Fig.3. shows the results.
