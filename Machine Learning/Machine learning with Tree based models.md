#ml #datacamp 

# CART
## Classification with Decision trees
A classification tree learns a sequence of if else questions about features in a model to extract the labels. it is able to capture non-linear relationships in data
![[Pasted image 20240104221924.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104221924.png/?raw=true)
the above diagram is an example of a classification tree. it first splits on one point and that point depends on one feature. and this goes on until the height reaches the mentioned amount.
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2,
												   stratify=y, 
												   random_state=1)
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)
```

**Decision Region:** it is the region in a feature space where all instances there belong to one class(label)
**Decision Boundary:** surface separating regions.
for linear classifiers, the decision boundary is a straight line.
![[Pasted image 20240104222729.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104222729.png/?raw=true)

![[Pasted image 20240104222744.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104222744.png/?raw=true)
however for Decision Trees, the regions are rectangular. This is because, at each split only one feature is involved. so there are gonna be different splits.

## Classification Tree Learning
the nodes of a tree always have one class as a majority so that when a prediction is needed, the majority is predicted.
the tree learns about what feature and at what point of the feature to split on, by maximizing information gain.
The existence of a node depends on the state of its predecessors.
lets say a parent node splits into `left` node and `right` node
for a given feature `f` and a given split point `sp`, the information gain is given by
![[Pasted image 20240104225751.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104225751.png/?raw=true)

where, I() is the impurity of a given node. the impurity is the measure of the amount of different classes in a node or the measure of the non-homogeneousness of a node.
the lesser the better. so info gain is formulated in such a way that, if the impurity of the left and right nodes are less then info gain is greater. so we intend on maximizing info gain
there are some ways to calculate this impurity
- gini index
- entropy
so, after the finding the values of `f` and `sp` for which IG is max, the split happens at that point and this continues on recursively, until the tree hits the max depth or until the IG value is 0, then the parent node is deemed as a leaf node.
```python
dt = DecisionTreeClassifier(max_depth=3, criterion="gini")
```
the `gini` method is slightly faster to compute and is the default.

## Decision Tree For Regression
```python
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=4, 
						  min_samples_leaf=0.1,
						  random_state=21)
```
`min_samples_leaf` means the minimum proportion of samples to be in a node for a split to be made. so if there are less than 10% of samples in a node, it won't be split further. 
unlike classifiers, decision tree regressors use the MSE as a way of learning where to split.
![[Pasted image 20240104232516.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104232516.png/?raw=true)

the target is just the mean of all targets node. so here, the tree tries to minimize the MSE and not maximize it.
a prediction is made by averaging all the values in the node. 
decision tree regressors are really useful for capturing non-linearities unlike linear models
![[Pasted image 20240104232756.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104232756.png/?raw=true)


# Bias-Variance Tradeoff
## Generalization Error
Generally, supervised learning involves the model fitting a function for the given data, where the function takes in data from the training set as inputs.
this function we fit can't predict useless noise in the dataset. so the function is rather close to ideal. to find this function f, lots of algorithms are used like regression, random forests, etc. 
when the calculated function fits the noise also, it leads to overfitting. when the function is not flexible enough to approximate the original data's function, it is underfitting.
![[Pasted image 20240106093214.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106093214.png/?raw=true)

![[Pasted image 20240106093231.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106093231.png/?raw=true)

the generalization error is used to see how good our model is. the formula is
Generalization error = bias * bias + variance + irreducible error
where,
**Bias** is how much the approximated function differs from the actual function. the lesser it is the better
high bias models lead to underfitting
**Variance** tells how inconsistent the function is over other training datasets. High variance lead to overfitting.
**Complexity** of the model tells us the flexibility of the model to approximate the function. increasing the depth of a tree and the min nodes increase complexity. however if complexity increases, variance increases and bias decreases. this is the bias-variance tradeoff so we need to choose a point where both contribute to the lowest generalization error.
![[Pasted image 20240106094119.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106094119.png/?raw=true)

![[Pasted image 20240106094204.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106094204.png/?raw=true)

## Diagnose bias and Variance problems
to make sure we find problems of overfitting and underfitting, we use test sets which are unseen by the data and take the error made on the test set as the generalization error. we can calculate the error on test set using cross validation.
we use K-fold cross validation, where we split the dataset into k folds and keep one fold as the test fold and the model trains on the remaining folds. this goes on until all the folds become the test set once. then the mean of all the errors is taken as the error of the model.
when the general error is lesser than the training error, but both errors are pretty high, it means underfitting due to high bias. to reduce bias we can increase the complexity of the model
when the general error is greater than the training error it means our model is overfitting due to high variance. to reduce this we reduce the model complexity
![[Pasted image 20240106100510.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106100510.png/?raw=true)
`n_jobs` = -1 means all the CPUs are used.
the MSE CV is the general error and the MSE due to test sets will be close to this value.

## Ensemble learning
ensemble learning is the process of training different models and aggregating all of their predictions to give a single prediction.
![[Pasted image 20240106103824.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106103824.png/?raw=true)

this aggregation can be done in many ways, one being hard voting for classifiers, where the final prediction is selected by which category has more votes. to implement this 
```python
from sklearn.ensemble import VotingClassifier
knn = KNN()
lr = LogisticRegression()
dt = DecisionTreeClassifier()
classifiers = [("Logistic", lr),
			   ("knn", knn),
			   ("decision Tree", dt)]
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)

print(accuracy_score(y_test, y_pred))
```


# Bagging and Random Forests
## Bagging 
bagging is a type of ensemble. normally in an ensemble we use different algorithms and average them all. in bagging, we use the same algorithm but on different subsets of the training data and then average the predictions.
Bagging's full form is Bootstrap Aggregation. this is because the subsets of the training data used are bootstrap samples of the training dataset.
![[Pasted image 20240106122014.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106122014.png/?raw=true)

![[Pasted image 20240106122031.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106122031.png/?raw=true)

![[Pasted image 20240106122057.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106122057.png/?raw=true)

in a bagging classifier the prediction is found by majority votes and in a regressor the prediction is found by averaging the predictions.
```python
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16)
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
bc.fit(X_train, y_train)
bc.predict(X_test)
```
`n_estimators` is the number of models to create.

## OOB evaluation
Out of Bag Evaluation is a technique used to find how good our model is without using cross validation. here, we use the "test" set of each bootstrap sample to find how good each model is. on average, each model only trains on 63% of the whole data. so we can use the remaining 37% unseen data to evaluate that particular model. we evaluate all the models of the ensemble and finally average them to get the OOB score used to evaluate the ensemble
![[Pasted image 20240106123452.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106123452.png/?raw=true)

```python
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, oob_score=True,
					  n_jobs=-1)
print(bc.oob_score_)
```
note that the score evaluated for each model is the accuracy for classification and R squared values for regression.

## Random Forests
random forest is a bagging technique where we use Decision trees as the estimator of the ensemble and we randomize the process even further by using not only subsets of rows but columns(features) too. usually the number of features to subset on is square root of the total number of rows.
![[Pasted image 20240106125411.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106125411.png/?raw=true)

and then final predictions are made by majority vote for Classifiers and averaging for regressors
```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=0.12)
```
there is something called feature importance for tree based models, which measures the predictive power of features in a dataset. to access it,
```python
imp = rf.feature_importances_
```
this gives the array of feature importance.
![[Pasted image 20240106130133.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106130133.png/?raw=true)

# Boosting
## ADABOOST
Boosting is an ensemble method where several weak learners come together to create a strong learner. in boosting an ensemble of models predict sequentially. and each model tries to be better than it's predecessor.
2 types of boosting are:
1. Ada Boost or Adaptive Boosting
2. Gradient Boosting
each model pays attention to wrong predictions and changes the weights accordingly.
Each model is also given a coefficient called alpha which tells the weight of that model in the final prediction.
alpha depends on the model's error
![[Pasted image 20240106174920.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106174920.png/?raw=true)

the green dots show corrected instances ignored by previous models.
a hyper parameter, learning rate, known as eta ranging from 0 to 1 is used to shrink the value of alpha. if the learning rate is small, then many estimators should be used.
then for predictions in classifiers, majority weighted voting is used 
for predictions in regressions, weighted averages are used.
```python
from sklearn.ensemble import AdaBoostClassifier
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
```

## Gradient Boosting
here, the same sequentially predicting models exist. however, it doesn't tweak weights of the training set. instead, the dependent variable changes.
the labels for the first model would be the original labels and then for the next model the labels are the residuals of all the independent variables. so the model tries to predict the residuals. then the next model tries to predict the previous model's residues and so on. the learning rate is used here to modify the labels for the next model and just like in ADA, it has a tradeoff with number of estimators.
the learning rate is also known as shrinkage here
![[Pasted image 20240106182012.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106182012.png/?raw=true)

then the predictions of a regression model is given by
![[Pasted image 20240106182046.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106182046.png/?raw=true)
which makes sense, because the target = prediction + residual and we have many residuals. a similar algorithm is used for classification too.
```python
from sklearn.ensemble import GradientBoostingRegressor
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1)
```

## Stochastic Gradient Boosting
there are some disadvantages to gradient boosting like,
- its very exhaustive
- the variance is low coz each CART probably does the same split
to overcome these we use stochastic Gradient boosting, where the training dataset of each estimator or model is going to be a random subset of the data like in Random Forests. both the rows and features are gonna be sampled(40% to 80%)
![[Pasted image 20240106183234.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106183234.png/?raw=true)

```python
from sklearn.ensemble import GradientBoostingRegressor
sgbt = GradientBoostingRegressor(max_depth=1,
								subsample=0.8,
								max_features=0.2,
								n_estimators=300)
```
predictions are made just like in gradient boosting

# Hyperparameter Tuning
## Grid Search
the hyperparameters of CART like `max_depth`, `min_samples_leaf` can be learned/tuned by some hyperparameter tuning methods like,
- Grid Search
- Random Grid Search
- Bayesian Optimization
- Genetic Algorithms

in grid search, we set different hyperparameter values and find how each combination of hyperparameter scores and choose the best one out of them all. the scoring is done by cross validation.
![[Pasted image 20240106184332.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240106184332.png/?raw=true)

```python
dt.get_params() -> #prints the parameters
from sklearn.model_selection import GridSearchCV
params_dt = {"min_samples_leaf":[0.04, 0.08, 0.09],
			 "max_depth":[5, 6, 8],
			  "max_features":[0.2, 0.4, 0.6]}
grid_dt = GridSearchCV(estimator=dt,
					  param_grid=params_dt,
					  scoring="accuracy",
					  cv=10,
					  n_jobs=-1)
grid_dt.fit(X_train, y_train)
best_params = grid_dt.best_params_
best_score = grid_dt.best_score_
best_model = grid_dt.best_estimator_
```
`best_estimator_` gives the best model. we can use this to predict labels. 
**the same can be done for a Random Forest also. it has all the parameters of a CART and some more.**
