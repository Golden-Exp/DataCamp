#ml #datacamp 

# Classification
## Machine learning
machine learning is the process of making computers learn to make decisions from data without being explicitly programmed
there are two types:
1. unsupervised learning is discovering hidden patterns and structures from unlabeled data. like predicting what categories, features fall into when we don't know the categories ourselves
2. supervised learning is where the values to be predicted are already known, and a model is built with the aim of accurately predicting values of previously unseen data.
there are two types of supervised learning
1. classification: when we want to predict the category the features fall into
2. regression: when the value to predict is a continuous variable

there are some requirements for supervised learning
1. there should be no missing values
2. data should be in numeric format
3. data should be in Data Frames or arrays

to convert our data to the required form we use EDA(Explanatory Data Analysis)
there is a specific `scikit` learn syntax that can be used for any model
```python
from sklearn.module import Model
model = Model()
model.fit(X, y)
predictions = model.predict(X_new)
print(predictions)
```
the Model is the various models we will learn about.

## K-nearest Neighbors
lets try classifying using KNN. for classifying data
1. build a model
2. model learns from labeled data we pass
3. pass unlabeled data as input
4. model predicts unseen data

the idea of K-nearest neighbors is to predict the label by looking at the k nearest neighbors and deciding the category using majority. for example if k=3, then we see which 3 points from the training data are nearest to the given data. then we decide using majority, that is if 2 are of the same category, then the unseen data is predicted to be of that category.
![[Pasted image 20231226190113.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231226190113.png/?raw=true)
let the black dot be the unseen data to be predicted. then we see the 3 nearest dots. 2 of them are red so we label the dot as the category of red's.

to see more clearly, lets visualize a scatter plot between two features with the dependent variable being 0 or 1(binary classification). when k is decided to be 15, 
![[Pasted image 20231226191046.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231226191046.png/?raw=true)

anyone falling under the red boundary, churns while the grey don't.
to use KNN from `scikit` learn.
```python
from sklearn.neighbors import KNeighborsClassifier
X = df[["col1", "col2"]].values -> converts into an array
y = df["dependent col"].values -> converts into an array
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)
predictions = knn.predict(X_new)
```

## Measuring Model Performance
for classification, we use accuracy as a metric to see how our model is performing. 
accuracy is the number of correct predictions divided by the total number of predictions. but we can't measure accuracy on training data, because this data is already seen. so we use a separate data known as test data(also known as validation data), to test accuracy.

for this, we split the labeled data that we have into two, training set and test set. we set the proportion of data that needs to be split, so if 0.3 is given, then we need 30% of our data to be in the test set. also, if 10% of our data has independent variable=1 and the rest are 0, we need 10% of the train and test to have independent variable=1 and rest 0. this can be achieved by passing the dependent variable to stratify parameter.
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
												   random_state=21, 
												   stratify=y)
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
```
we can find the accuracy by passing in the test datasets to the score function of the model. 
to select the right value of k for our model, we can plot a model complexity curve.
![[Pasted image 20231226193115.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231226193115.png/?raw=true)
we can see that as k increases, the model starts underfitting, that is, it becomes a simple model. however as k decreases, the model starts overfitting, that is it becomes so complex and performs well on the training set but horribly on the test set.
to see which value for k is good, lets iterate of various accuracies of different models for different values of k
```python
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
	knn = KNeighborsClassifier(n_neighbors=neighbor)
	knn.fit(X_train, y_train)
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies = knn.score(X_test, y_test)

#plot the graph
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.show()

```
![[Pasted image 20231226193730.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231226193730.png/?raw=true)

as we can see, as k increases the model goes from overfitting to underfitting. the peak test accuracy was attained at k=13.

# Regression
## regression with 1 explanatory
`scikit learn` requires the dependent variables to be 2-D arrays. so before we create the model, we change the shape of our single feature.
```python
y = df.iloc[:, -1]
x = df.iloc[:, 0]
x = x.reshape(-1, 1) -> changes shape to 2-D
```

now we create the model
```python
from sklearn.linear_model import LinearRegression
linear = LinearRegresion()
linear.fit(x,y)
predictions = linear.predict(x)
plt.scatter(x,y)
plt.plot(x, predictions)
plt.show()
```
![[Pasted image 20231231200006.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231231200006.png/?raw=true)
linear regression gives a straight line of predictions

## How does linear regression work?
to calculate predictions with simple linear regressions we need to calculate the slope and the intercept of the regression line. we do this with **optimization**.
we choose a loss function and optimize the model coefficients(slope and intercept) such that the loss function is at its minimum.
the loss function used by `sklearn` is the RSS. Residual Sum of Squares. this function is calculated by first calculating the residuals of all the predictions, that is the distance between the prediction and the actual value. since some values are negative here, we square all values and then add them. this is our loss function. and this method of calculating the slopes with RSS is known as Ordinary Least Squares method and is used by `sklearn` under the hood.
![[Pasted image 20231231202249.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231231202249.png/?raw=true)

for simple linear regression, the slope and intercept is calculated by optimizing. then the prediction y = ax + b, where a is the slope and b is the intercept. for higher dimensions, we just extend the formula and calculate the coefficients with the same optimization
![[Pasted image 20231231202501.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231231202501.png/?raw=true)

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
												   stratify=y)
reg = LinearRegression()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

to measure the fit of our model, we use metrics. there are some metrics we can use for linear regression. one of them is R squared
this measures the amount of variance in the predictions, that can be explained by the explanatory variable. it ranges from 0 to 1, 1 being very good coz every variance can be explained and 0 being very bad, coz 0 variance explained
![[Pasted image 20231231203334.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231231203334.png/?raw=true)

to calculate the R squared we print the score 
```python
reg.score(X_test, y_test)
```

another metric is the mean squared error. we calculate this by squaring the residuals and taking the mean of them. this is MSE and the root is RMSE.
to calculate the MSE
```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False)
```
squared = False gives RMSE

## Cross Validation
R-squared can sometimes not be representative of the model's ability to generalize to unseen data mainly due to the random split we do. so we use a technique called cross validation to check the R-squared on many splits.
![[Pasted image 20231231205756.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231231205756.png/?raw=true)
we split our data into 5 folds and each time, we keep one fold as the test data and the remaining as training. so now we have 5 r-squared. we can now calculate the statistic of our choice on these and check them out.
```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)


print(np.mean(cv_results))
print(np.quantile(cv_results, [0.025, 0.975]))
```
the function returns an array of R-squared and now you can calculate the mean or median and std, with confidence intervals. we can also use the score parameter in it to determine the score we want to use.

## Regularized Regressions
a common symptom of overfitting is that the coefficients are larger. this is because the model tends to modify the coefficients more extremely to capture the noise in the training set too well. to avoid overfitting like this we use regularization. it involves modifying the loss function to penalize large coefficients.
one type of regularized regression is the ridge regression. this adds the OLS loss function to the sum of all the coefficients squared times a constant named alpha.
![[Pasted image 20231231232849.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231231232849.png/?raw=true)
if alpha is 0 then its just regular OLS and that might lead to overfitting. if very high then the coefficients will be too low and lead to underfitting. 
we can select the value for alpha by trial and error like we did for the value of k
to use ridge with `sklearn`
```python
from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1, 10, 100, 1000]:
	ridge = Ridge(alpha=alpha)
	ridge.fit(X_train, y_train)
	y_pred = ridge.predict(X_test)
	scores.append(ridge.score(X_test, y_test))
print(scores)
```
another technique is the Lasso regression. this is just like ridge except instead of the sum of squares of coefficients we take the sum of absolute values of the coefficients. the implementation is also the same as ridge.

```python
from sklearn.linear_model import Lasso
scores = []
for alpha in [0.1, 1, 10, 100, 1000]:
	lasso = Lasso(alpha=alpha)
	lasso.fit(X_train, y_train)
	y_pred = lasso.predict(X_test)
	scores.append(lasso.score(X_test, y_test))
print(scores)
```
lasso regression can be used for plotting feature importance plots
![[Pasted image 20231231235106.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231231235106.png/?raw=true)
to get the feature importance
```python
lasso_coeff = lasso.fit(X, y).coef_
```
this gives an array of values that we can plot
![[Pasted image 20231231235325.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020231231235325.png/?raw=true)


# Finetuning the model
## Assessing Fit
in classification we used accuracy as a metric for our model. although its not a bad metric, it can perform badly with some datasets. like when we have a dataset that has all of its y values as 0 except 10% of the data. now if we check the accuracy of a model that just gives 0 as the result for any data, we see that the accuracy is 99%. doesn't mean that our model is really good. so we go for other metrics
there is something called confusion matrix for binary classification, which is a 2x2 matrix, with 4 values.
1. True Negative - predicted 0 and actually 0
2. False Positive - predicted 1 but actually 0
3. False Negative - predicted 0 but actually 1
4. True Positive - predicted 1 and actually 1
![[Pasted image 20240101111814.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101111814.png/?raw=true)

some other metrics are :
**Precision:** number of true positives divided by the sum of true positives and false positives
![[Pasted image 20240101112051.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101112051.png/?raw=true)

**Recall:** number of true positives divided by the sum of true positives and false negatives. this is also known as sensitivity(the proportion of true positives)
![[Pasted image 20240101112148.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101112148.png/?raw=true)

**F1 score**: this is the harmonic mean of precision and recall.
2 * ((recall * precision) / (recall + precision))
to get the confusion matrix in python:
```python
from sklearn.metrics import classification_report, confusion matrix
print(confusion_matrix(y_test, y_preds))
print(classification_report(y_test, y_preds))
```

## Logistic Regression and ROC curve
although its called Regression, this type of model is used for classification. it predicts the probabilities of the y value being 1(regression). if its above 0.5, taken as 1 and below taken as 0.
```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
												   random_state=82)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

y_pred_probs = logreg.predict_proba(X_test)[:, 1] -> returns a 2-D array of probabilities of 0 and 1
```

the probability threshold by default is 0.5. we can change that for our model if we want.
Receiver Operating Characteristic (ROC) curves are used for checking how the model performs for different probability thresholds.
![[Pasted image 20240101125317.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101125317.png/?raw=true)

for a model that predicts randomly, if p(threshold) is 0 then all are predicted to be positive and the true positive and false positive rate increases.
if p is 1 then all are predicted to be 0 and TP and FP rates are both 0.
to plot the ROC curve for our model,
```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
```
we then plot the variables.
![[Pasted image 20240101125726.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101125726.png/?raw=true)

to quantify the ROC curve we use AUC that is the Area Under Curve a metric for ROC. for a model that randomly guesses, the area is 0.5. for our model,  its 0.67. the area ranges from 0 to 1 and for a really bad model, the area is 0(because the curve just goes from 0,0 to 1,0 and then to 1,1) and for a really good model the area would be 1(because the curve just goes from 0,0 to 1,0 to 1,1) 
![[Pasted image 20240101130016.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101130016.png/?raw=true)

to find the ROC AUC in `sklearn`
```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))
```

## Hyperparameter Tuning
hyper parameters are parameters we set for our model before fitting them. hyperparameter tuning is a way of finding the right hyperparameter for our model, by trying it with multiple values and see which gives us the best score.
![[Pasted image 20240101140251.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101140251.png/?raw=true)

one way of hyperparameter tuning is the Grid search with cross validation method. we can choose from a grid of hyperparameter values. for example we can select between 2 values for `n_neighbors` in a KNN model and the metric
![[Pasted image 20240101140448.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101140448.png/?raw=true)
here the score of 0.8716 for 11 neighbors in the Euclidean metric is calculated by cross validation with k-folds and the mean value is shown here. here we see that `n_neighbors=5` and metric=Euclidean gives the best score.
to use this with `sklearn`
```python
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha":np.arange(0.0001,1,10),
			 "solver":["sag", "lsqr"]}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```
however there's a catch for this method. when the number of hyperparameter and k folds increase, the number of fits also increase. this is computationally expensive. so an alternate is Randomized Grid Search
```python
from sklearn.model_selection import RandomizedSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha":np.arange(0.0001,1,10),
			 "solver":["sag", "lsqr"]}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```
`n_iter` is used to set the number of hyperparameters tested.

# Preprocessing and Pipelines
## Preprocessing data
real world data has data that are not optimal for `sklearn` models. so we might have to process our data before passing them to our models. 1 of them is to always have numeric features. if a column has a categorical values, to convert it to numerical we use one-hot encoding. this creates new columns for each category with 1 denoting that the row's category is that column. we can have n-1 new columns coz if all the new columns had 0 this meant that the category is the remaining column. this is needed for some models.
```python
import pandas as pd
coded_df_cols = pd.get_dummies(df["col"], drop_first=True)
```
`drop_first` drops 1 column. we can then join this with our original data frame using `pd.concat` and then drop the categorical column.

## Missing values
to handle missing data, we can drop columns with only 5% missing values. if more than that we can choose to impute the data with a statistic of our choice. to impute with `sklearn` we use imputer class
remember that we must always impute after we split the data, because imputing before might lead to the model knowing about the test set before learning known as data leakage.
```python
from sklearn.impute import SimpleImputer
X_cat = music_df["genre"].values.reshape(-1,1)
X_num = music_df.drop(["genre", "popularity"], axis=1).values
X_train_cat, X_test_cat, y_train_cat, y_test_cat = 
train_test_split(X_cat, y, test_size=0.2)
X_train_num, X_test_num, y_train_num, y_test_num = 
train_test_split(X_cat, y, test_size=0.2)

imp_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

imp_num = SimpleImputer()  #by default the strategy is mean
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
X_train = np.append(X_train_cat, X_train_num, axis=1)
X_test = np.append(X_test_cat, X_test_num, axis=1)
```
imputers are known as transformers as they transform the data.
we can also use a pipeline to impute, which is an object used to run a series of transformations and build a model in a single workflow.
```python
from sklearn.pipeline import Pipeline
steps = [("imputation", SimpleImputer()),
		 ("logistic_regression", LogisticRegression())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
												   random_state=82)
logreg = pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```
steps should be a list of tuples with the step going to be done and the object to do that. the step to be done can have any name but that name should be the one that is used when we use `GridSearchCV` for the parameters
remember that all the steps except for the last should be a transform.

## Centering and Scaling
centering and scaling our data is also known as standardizing or normalizing. this is done to scale our data to similar values, that is we decrease the distance between values of different features. for example, if the values of one feature is fully in 1000s and other is negative, this is too much scaling and affects models like KNN which relies on distance.
there are many ways to scale our data. one of them is
- one is **Standardization** where we subtract all the values by the mean of its feature and divide by variance. so all features are centered around 0 and have a variance of 1.
- we can also subtract from the minimum and divide by the range. so the minimum will be 0 and maximum will be 1
- can also **normalize** the data so that it ranges from -1 to 1

to standardize in `sklearn`
```python
from sklearn.preprocessing import StandardScaler
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3,
												   random_state=84)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
we can also scale in pipelines because this is also a type of transform
```python
steps = [('scaler', StandardScaler()),
		('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
												   random_state=84)
knn = pipeline.fit(X_train, y_train)
knn.predict(X_test)
```

we can also perform Cross validation with pipelines
```python
from sklearn.model_selection import GridSearchCV
steps = [('scaler', StandardScaler()),
		('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
												   random_state=84)
parameters = {"knn__n_neighbors":np.arange(1, 50)}
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
cv.predict(X_test)
cv.best_score_
cv.best_params_
```

## Evaluating multiple models
different models perform well based on many things
choosing the right model depends heavily on what we want to know.
![[Pasted image 20240101183347.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101183347.png/?raw=true)

different models follow similar metrics so it is easy to compare between them.
![[Pasted image 20240101183508.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101183508.png/?raw=true)

some models require scaling. some are:
- KNN
- Linear Regression
- Logistic Regression
- ANNs
to select the best of many models,
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
												   random_state=84)
scaler = StandardScalar()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {"Logistic Regression":LogisticRegression(), "knn":KNeighborsClassifier(), "Decision Tree":DecisionTreeClassifier()}
results = []
for model in models.values():
	kf = KFold(n_splits=6, random_state=6, shuffle=True)
	cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
	results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()
```
![[Pasted image 20240101211515.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101211515.png/?raw=true)

```python
for name, model in models.items():
	model.fit(X_train_scaled, y_train)
	test_score = model.score(X_test_scaled, y_test)
	print(name+": "+test_score)
```
