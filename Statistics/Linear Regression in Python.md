#statistics #datacamp 
# Simple Linear Regression modelling
## Regression
Regression models are a class of statistical models that let you explore the relationship between a response variable and some explanatory variables.
so with that relationship, given the explanatory variables we can predict the value of the response variable
response variable - dependent variable - Y
explanatory variable - independent variable - Xs

#### Jargon
Linear Regression means the type of regression where the dependent variable is numeric/continuous
Logistic regression means the type of regression where the dependent variable is logical/Categorical.
Simple Linear/Logical Regression : means there is only 1 explanatory variable.

It is always a good idea to visualize your data before making regression models for your data. a useful tool for this is seaborn's `regplot` function. this adds a trendline to the plot to see whether it is a good idea to use a regression model. if the line's closer to the points on the plot, it means it might be a good fit.

#### Packages
`statsmodel` - we are using this for now. this package is good for giving insights rather than predicting
`scikit-learn` - we'll see this in later lessons. useful for predictions rather than insight.

## Fitting a linear regression
the straight line in the `regplot` has 2 very important features.
the ==intercept== and the ==slope==
==Intercept==: is the value of y when x is zero.
==slope==: is the steepness of the line and defined as the amount y increases by when x is increased by 1
![[Pasted image 20231217153222.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217153222.png/?raw=true)
here the red line is the intercept and the green indicates the slope.
to see the actual slope and intercepts we need to fit a regression model and look at its parameters.
```python
from statsmodels.formula.api import ols
model = ols("total_payment_sek ~ n_claims", data=df)
model = model.fit()
print(model.params)
```

OLS stands for Ordinary Least Squares and is a type of regression
the syntax for `ols` function for a regression model is 
`model_name = ols("y column ~ x column", data=df)` and to fit it we use `model_name.fit()`
once fit we can see the parameters of the model
**The Parameters of simple linear regression model are the slope and the intercept**
![[Pasted image 20231217153816.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217153816.png/?raw=true)

so the equation for this model is 
y = 3.413824 * x + 19.994486

## Categorical Explanatory Variables
when the explanatory variables are categories, we do the same thing as last time to fit the model with a small change. but first its a good idea to visualize the plot using seaborn's `displot` function
then as usual
```python
from statsmodels.formula.api import ols
model = ols("mass_g ~ species", data=df).fit()
print(model.params)
```

![[Pasted image 20231217155955.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217155955.png/?raw=true)

this gives us 4 parameters. these are all the categories in the categorical variable.
the first one is the intercept and the rest of the parameters are relative to the first one.
all of them are the mean mass of each species relative to the first mean. for example 617-235 gives 382 the original mean that we need as parameter.
this type of relative representation is useful with multiple explanatory variables but single variables don't require this. to change the representation we add a +0 to the parameters of OLS to note that all parameters should be relative to 0
```python
from statsmodels.formula.api import ols
model = ols("mass_g ~ species + 0", data=df).fit()
print(model.params)
```

![[Pasted image 20231217160025.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217160025.png/?raw=true)

now we got all the parameters relative to 0 and note that all of them are the mean mass of each category. this is because, this is how we predict in simple linear regression models with categorical explanatory variables. we just give out the mean of the asked category.
**so the parameters of a regression model with categorical variables as explanatory variables will be the mean dependent variable of each category.**

# Prediction and Model objects
## Making Predictions
to make predictions with a regression model you made, use the `.predict()` function.
```python
from statsmodels.formula.api import ols
model = ols("y column ~ x column", data=df).fit()
model.predict(df1) -> gives out df with the predicted answers
```
a Data Frame of explanatory variables with column name being the same name as the explanatory variable name provided should be passed to the predict function to get a data frame of predictions. to make it easier
```python
df1 = dataframe with explanatory variables
df1.assign(new_col = model.predict(df1))
df1 -> contains both the explanatory and predicted variables
```
when we plot them in the scatter plot of our original data.
![[Pasted image 20231217162207.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217162207.png/?raw=true)

we can see that all the prediction always fall on the regression line.
#### Extrapolating
when we predict data for explanatory variables outside of the range of data we have originally, that is known as extrapolation. we can extrapolate with regression models however, sometimes it gives us wrong values.
![[Pasted image 20231217162455.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217162455.png/?raw=true)

like here the predicted mass is negative which is definitely wrong. we need to understand our data properly before deciding if it is ok to extrapolate.

## Working with model objects
the models created have many attributes that we can use.
`model.fittedvalues` gives us the predicted values of the original data. that is predicting values for the explanatory data in the original dataset
`model.resid` gives us by how much does the predicted value(fitted values) differ from the actual values. they are the original values - the predicted values
the actual value = fitted value + residual

`model.summary()` is a function that gives out all the statistical points for the data. this includes the dependent variable, independent variable, and some statistical attributes like R-squared and stuff which we'll see about later
![[Pasted image 20231217174913.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217174913.png/?raw=true)

the first part is the metrics and dependent variables part.
next we see the coefficients of the model.
last part contains diagnostic statistics

## Regression to the Mean
regression to the mean is a property that data we sample have. it says that extreme values that occur when we sample, become closer to the mean when we sample another time.
that is, extreme values eventually becomes not extreme and they do not persist over time.
response values are the sum of the predicted values and the residues.
the residuals occur due to randomness in our world. and they are more when the cases are extreme. that makes our model fail to predict for extreme cases. especially for simple linear regression models.
however that doesn't make our model bad or does it make it necessary to take the randomness into account. we don't need to do any of that because of the property that any sampled data has: extreme values don't persist over time.
![[Pasted image 20231217180255.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217180255.png/?raw=true)

this is a `regplot` plotted on a famous dataset collected by Pearson in the 19th century. it contains the heights of fathers and sons. from the plot above with the regression line in the black, and the green line signifying equal heights, we can see that one part of the regression line falls to the left of the green line meaning short fathers had short sons but not as short as them, likewise on the right means very tall fathers had tall sons but not as tall as them.
(values to the left of the green line means the y values are greater than x
values to the right of the green line suggest that x values are greater than y
all for the collected data)

## Transforming variables
sometimes, the `regplot` might have a curved graph and the points may not seem to fall on the regression line. to fix this we transform the variables to make it into a linear regression.
for example, the lengths vs mass data of a fish species named perch has the following `regplot`
![[Pasted image 20231217184223.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217184223.png/?raw=true)

we can see that the points don't follow the line. creating a regression model for this would lead to bad results. so we need to transform. the transformation should be based on the data.
![[Pasted image 20231217184329.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217184329.png/?raw=true)

this is a perch fish. looking at it, we can see that it has a round body. that might mean that even if the length is short, it might be wider and have more mass. that means the mass and length don't have a linear relationship. However, the volume and mass might have a linear relationship as volume increases mass increases. so we cube the length as length cube is proportional to the volume.
now if we see the `regplot`
![[Pasted image 20231217184537.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217184537.png/?raw=true)

here the points follow the line more. this is good. so now we create a model for this. remember, when predicting, feed the model the cube of the given length to get the correct predicted mass.
also now if we plot the predicted points against the normal lengths not cubed, we see that the predicted points follow the original curve which is so cool to me.
![[Pasted image 20231217184740.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217184740.png/?raw=true)

another transformation is to take the square root of both the explanatory variable and the dependent variable. this can be done when the `regplot` shows that all the points are clustered to the right and we can't seem to see if they fit the line.
it is generally good for right skewed data.
![[Pasted image 20231217184946.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217184946.png/?raw=true)

after square roots:
![[Pasted image 20231217185004.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217185004.png/?raw=true)

**Remember to take the square of the predictions from this model because it is predicting the square roots also remember to feed in square roots of the original.**
undoing the transformation of your response variable is known as back transformation.

# Assessing Model Fit
lets see how to say if our model is good at predicting what we want.

## Quantifying model fit
there are a few metrics to see if our model is any good. some of them are:
### Coefficient of determination
this is called r-squared when simple linear regression and R-squared when many explanatory variables.
it denotes the proportion of variance in the response variable that is predictable from the explanatory variable.
ranging from 0 to 1 -  0 being worst and 1 being a perfect fit.
for example, an R-squared value of 0.89 means the explanatory variable can explain 89% of the variability in the dependent variable.
this just the correlation of the explanatory variable and the dependent variable squared **for simple linear regression**
to get the coefficient use the attribute
```python
print(model.rsquared)
```

### Residual Standard Error
it is the measure of the typical size of a residual. it has the same unit as the response variable. MSE is the Squared Residual Standard Error.
to get RSE we need to get MSE and square root
```python
print(model.mse_resid)
print(np.sqrt(model.mse_resid))
```
to calculate the RSE, first we square all the residuals and then add them up. then we calculate the degrees of freedom(the number of rows of explanatory variable - number of parameters) we divide by the degrees of freedom and perform square root to get the RSE.

### Root Mean Squared Error
this is similar to RSE and calculated just like it, except you don't divide by the degrees of freedom, you just divide by the number of rows of explanatory variable. MSE is the mean squared error and is just the RMSE squared.
the course recommend that you use RSE all the time instead of RMSE.
![[Pasted image 20231217212518.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217212518.png/?raw=true)

## Visualizing model fit
properties of residuals of a good model is that 
- they are normally distributed
- they have a mean of zero
there are 3 plots for visualizing residuals
### Residual Plot
this is a plot between the residuals and the fitted values. ideally the residuals should be zero for any fitted value and we can see if our data follows that 0 using a LOWESS line which is a smooth curve that follows our data.
for creating one, 
```python
sns.residplot(x="independent variable", y="dependent variable", 
			  data=df, lowess=True)
plt.plot()
```
![[Pasted image 20231217214915.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217214915.png/?raw=true)

the model on the right isn't a very good fit.

### Q-Q plots
these plots show whether our residuals follow a normal distribution. the x axis contains the quantiles from the normal distribution. the y values are the sample quantiles derived from our dataset. if the points follow the straight line they are normally distributed. if no, they are not.
```python
from statsmodels.api import qqplot
qqplot(data=model.resid, fit=True, line=45)
```
![[Pasted image 20231217215241.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217215241.png/?raw=true)

the model on the right doesn't track the line. meaning the residuals are not normally distributing meaning its a bad fit. meanwhile the plot on the left shows that its a good fit coz the residuals are normally distributed.
the extremes are the two rows of the dataset with the largest residuals

### Scale Location Plot
this plot shows the **square root of the standardized residuals** versus the fitted values.
to get this plot we have a few steps to do.
```python
norm_resid = model.get_influence().resid_studentized_internal
sqrt_norm_resid = np.sqrt(np.abs(norm_resid))
sns.regplot(y=sqrt_norm_resid, y=model.fittedvalues, ci=None, lowess=True)
```
first we calculate the standardized residuals. then we square root it and finally we plot it against the fitted values.
![[Pasted image 20231217220020.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217220020.png/?raw=true)

here we see that in the model on the left although the residuals tend to increase a lil bit when the fitted values increase, it isn't a big change. however the model on the right is very bad having the size of residuals going up and down.

## Outliers, Leverage and influence
outliers are extreme explanatory variables. in a `regplot` an outlier can be the points along the extreme side of the line or can be very far away from the line.
![[Pasted image 20231217224949.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217224949.png/?raw=true)

here the orange points and the cross are all outliers
**Leverage:** this is a measure of how extreme the explanatory values are.
**Influence:** this measures how much the model would change if you left the observation out of the dataset while modelling.
the influence of each observation is based on the leverage and the size of the residual of the observation.
we can get them with the following methods
```python
model = ols("y variable ~ x variable", data=df).fit()
summary = model.get_influence().summary_frame()
leverage = summary["hat_diag"] -> gives the leverage of each point
influence = summary["cooks_d"] -> gives the influence of each point
```
the `summary_frame()` returns a data frame with many metrics for all rows of the data.
`hat_diag` column contains the leverage of each column which means Hatrix diagonal
`cooks_d` column contains the influence calculated by the Cook's distance method
we can find the most influential rows by sorting on the `cooks_d` column.
now we'll try removing the influential rows
![[Pasted image 20231217225851.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231217225851.png/?raw=true)

just removing one extreme data has totally changed the slope of our regression line.

# Simple Logistic Regression
## Logistic Regression
when the response variable is binary like true or false and 1s and 0s, we use logistic regression.
if we use the same linear regression for these types of data, we might predict values outside of the only 2 values(0 and 1). the predictions for a binary response variable can be seen as probability for 1. so it can't be negative or greater than 1.
![[Pasted image 20231218124827.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218124827.png/?raw=true)

so using the same OLS method for these data won't do any good. instead we use logistic
```python
from statsmodels.formula.api import logit
model = logit("y variable ~ x variable", data=df).fit()
print(model.params)
```
the parameters are the same as intercepts and slopes except they have different interpretations now.
also now the predictions follow a logistic curve(S-shape), which I think is most probably a sigmoid.
![[Pasted image 20231218125131.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218125131.png/?raw=true)

here we can see that the values never go beyond 0 and 1.
**Remember that logistic regression models don't actually predict the binary variable but the probabilities of the binary variable.**
to plot logistic regressions we use the `logistic=True` parameter in `regplots`

## Predictions and Odds ratio
to make predictions we just use the same predict method on a Data Frame with the column name being the same name as the name of the explanatory variable given.
```python
print(model.predict(df2))
```
the predictions follow the line in the logistic `regplot`.
![[Pasted image 20231218130930.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218130930.png/?raw=true)

note that these values are probabilities and not the actual predictions. to make it easier we can get the most likely response. that is if the probability is less than 0.5 its a 0 and if its greater than 0.5 its 1.
so, 
```python
predictions["probs"] = model.predict(df1)
predictions["most_likely"] =np.round(predictions)
```
now the predictions are either 0 or 1
![[Pasted image 20231218131222.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218131222.png/?raw=true)

another way of representing is the odds ratio.
the odds ratio can be calculated by the probability of something happens divided by the probability it doesn't.
![[Pasted image 20231218131356.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218131356.png/?raw=true)

```python
predictions["odds"] = predictions["probs"]/(1-predictions["probs"])
```

![[Pasted image 20231218131505.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218131505.png/?raw=true)

an odds ratio of 1 means both outcomes are equally likely to happen. if the odds ratio is less than 1 means the possibility of 0 is more and a ratio of more than 1 means 1 is more likely. here we see that the odds of 1 are 5 times more when the   "`time_since_last_purchase`" is more.
to make the odds ratio linear we apply log and call it log odds ratio

![[Pasted image 20231218131838.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218131838.png/?raw=true)

```python
predictions["log odds"] = np.log(predictions["odds"])
```
![[Pasted image 20231218131949.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218131949.png/?raw=true)


## Quantifying logistic regression fit
there are 4 categories that our predictions fall into
they are 
1. True Positive(when we predicted 1 and its actually 1)
2. False Positive(When we predicted 1 but its actually 0)
3. True Negative(When we predicted 0 and its actually 0)
4. False Negative(when we predicted 0 and its actually 1)
![[Pasted image 20231218133910.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218133910.png/?raw=true)


this above table is a confusion matrix. to get that, 
```python
actual_response = df["y variable"]
predicts = np.round(model.predict())
outcomes = pd.DataFrame({"predicted":predicts, "actual":actual_response})
outcomes.value_counts()
```
`model.predict()` gives the predicted values for the original dataset.

an easier way is to
```python
conf_matrix = model.pred_table()
print(conf_matrix)
```
![[Pasted image 20231218134504.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218134504.png/?raw=true)

to plot confusion matrices us the mosaic package
```python
from statsmodels.graphics.mosaicplot import mosaic
mosaic(conf_matrix)
```
![[Pasted image 20231218134613.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218134613.png/?raw=true)

some metrics for logistic regression plots are:
#### Accuracy
accuracy is the number of correct predictions divided by total predictions.
![[Pasted image 20231218135028.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218135028.png/?raw=true)

```python
tn = conf_matrix[0,0]
tp = conf_matrix[1,1]
fn = conf_matrix[1,0]
fp = conf_matrix[0,1]

acc = (tn+tp) / (tn+tp+fn+fp)
```

#### Sensitivity
sensitivity is the ratio of values predicted to be 1 and are actually 1 to the total number of 1s.
![[Pasted image 20231218135239.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218135239.png/?raw=true)

#### Specificity
specificity is the ratio of values predicted to be 0 and are actually 0 to the total number of 0s
![[Pasted image 20231218135456.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218135456.png/?raw=true)
all of these metrics need to be high for a good model fit.


![[Pasted image 20231218140552.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218140552.png/?raw=true)
