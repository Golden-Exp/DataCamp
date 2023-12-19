#datacamp  #statistics 
last time we just saw how to do linear and logistic regression for 1 explanatory variable known as simple regression. now we are gonna see multiple regression. i.e. regression with multiple explanatory variables

# Parallel slopes

## Parallel slope linear regression
parallel slopes is a special case of multiple regression where we have 2 explanatory variables, one being categorical and another being continuous. to do this

```python
from statsmodels.formula.api import ols
model = ols("dependent variable ~ cont col + cat col + 0", data=df).fit()
model.params
```
just like last time we add a +0 for categorical variables to tell the model not to calculate the slope and making the intercepts relative to 0.
now if we print the parameters for this model, we'll have a slope made from the continuous variable and intercepts for each categories in the categorical column.
**these parameters are different than when we fit one continuous or categorical independent variable.**
to plot this we use `matplotlib`. lets say we have 3 categories in our categorical variable. meaning we'll have 3 intercepts and 1 slope.
```python
sns.scatterplot(x="cont variable", y="dep variable",
			hue="cat variable", data=df)
in1, in2, in3, sl = model.params
plt.axline(xy1=(0, in1), slope=sl, color="blue")
plt.axline(xy1=(0, in2), slope=sl, color="orange")
plt.axline(xy1=(0, in3), slope=sl, color="red")

plt.show()
```
remember that an intercept is when the x value is 0 and slope is how much the y value changes by when x changes by 1.
![[Pasted image 20231218210248.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218210248.png)

since all the intercepts have the same slope this type of regression is known as parallel slopes as lines with same slope are parallel.

## Predictions with parallel slopes
predictions with parallel slopes regression is the same as simple linear regression. that is y = m * x + c. however the intercept c changes depending on the categorical variable because there is a separate intercept for each categories.
```python
from itertools import product
data = product(array of continuous variable, array of all categories)
# data will now contain all combinations of the cont var and categories

predictions = data.assign(dep_var = model.predict(data))
```
when the predictions are plotted
![[Pasted image 20231218224017.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218224017.png)

they always follow one of the slopes.

to manually implement the predictions
```python
conditions = [
			  data["cat"] == cat1,
			  data["cat"] == cat2,
			  data["cat"] == cat3
]
inter = [in1, in2, in3]
predictions["intercept"] = np.select(conditions, inter)
predictions["predictions"] = slope * predictions["cont"] + 
								predictions["intercept"] 
```
`np.select` selects the data after checking which condition is true. for example for the first data, if it passes the first condition, its given the value in the first index of the `inter` list. like this we give each of the rows the intercept they have due to their categories.

## Assessing model fit
recall from the last lesson that R-squared and RSE are two metrics of a model. R-Squared is proportion of variance in the response that is predictable by the model.
and RSE is typically the size of an average residual of the model.

when comparing R-squared of different models, we can see that a model with many explanatory variables is always better.
![[Pasted image 20231218231737.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218231737.png)

in fact as the number of explanatory variables increase, R-squared also increases. so sometimes even if r-squared looks too good, it might be a bad fit coz it might be an overfit.
overfit is when our model is really good at predicting values for the current dataset and fails for others. this tends to happen in multiple linear regression when we have many explanatory variables.
we may fail to notice an overfit with r-squared as it increases with the number of explanatory variables.
so a metric named r-squared adjoint is introduced which penalizes more number of rows. its also known as adjusted coefficient of determination
the formula is:
![[Pasted image 20231218232315.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218232315.png)
`nobs` is the number of observations and `nvar` is the number of explanatory variables. if `nvar` is higher then the value of the ratio is high. then the value of (1 - R-squared) times the ratio will be high. which means 1 - that will be low. so it penalizes more explanatory variables for noticing overfitting. it is also penalizing when R-squared is small, coz then 1 - R-squared is large and the product is large and 1 - that product is small.
```python
print(model.rsquared_adj)
```
prints the adjusted coefficient of determination.
we can see that both metric are high for two explanatory variables tho

![[Pasted image 20231218232858.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218232858.png)

same with RSE. two variables are better
![[Pasted image 20231218232928.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231218232928.png)

# Interactions
## model for each category
in parallel slopes we restrict each category to have the slope. this might inhibit our predicting capabilities. instead, lets try dividing the dataset for each category and make a separate model for each of them. if we do this, there would be different slopes and intercepts for each category. 
```python
cat1 = df[df[cat == "cat1"]]
cat2 = df[df[cat == "cat2"]]
cat3 = df[df[cat == "cat3"]]

m1 = ols("dep var ~ cont", data=cat1)
m2 = ols("dep var ~ cont", data=cat2)
m3 = ols("dep var ~ cont", data=cat3)
print(m1.params, m2.params, m3.params) -> gives different params

cat_1 = exp1.assign(predictions = m1.predict(exp1), cat="cat1")
cat_2 = exp2.assign(predictions = m2.predict(exp2), cat="cat2")
cat_3 = exp3.assign(predictions = m3.predict(exp3), cat="cat3")

prediction = pd.concat(cat1, cat2, cat3)

sns.lmplot(x="cont", y="dep", hue="cat", data=df) -> plots the regression
sns.scatterplot(x="cont", y="predictions", data=prediction, hue="cat")
plt.show()
```
![[Pasted image 20231219112438.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219112438.png)
here we see that there are separate regression lines for each category with different slopes. to plot this we use `lmplot` instead of `regplot`. we can see that each prediction follows its own regression line.
from the metrics, we see that some models are better than parallel slopes model while some were worse. this is to be expected because, the advantages are split between the models

## One model with interactions
instead of a model for each category we can have one model that gives different slopes and intercepts for each category. this is achieved with interacts.
interacts are when different explanatory variable influence other variables.
the effect of one explanatory variable changes depending on the value of another explanatory variable. for different values of a continuous variable, the response could be different depending on the categorical variable it has.
this is known as interactions. this means both the continuous and categorical variable interact.

there are two ways to specify interactions for models. one is implicit and the other is explicit
```python
model = ols("response ~ exp_1 * exp_2")

model = ols("response ~ exp1 + exp2 + exp1:exp2")
```
if we give a continuous and categorical variable like last time, without any +0, it will be confusing for us as all the values are relative to the intercept. but to give +0 and say that params has to be relative to 0 we have to interact explicitly.
```python
model = ols("response ~ cat + cont:cat + 0") ->gives everything separately

model = ols("response ~ cont + cat + cont:cat + 0") 
#the intecepts are given relative to 0 but the slopes are relative to the first slope
```
this type is used to have the params relative to 0 and to specify interactions
when we use this, the coefficients of the model will be the same as when we had separate models for each categories, i.e. it will have a slope and intercept for each category.
![[Pasted image 20231219121701.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219121701.png)

## Predictions
when predicting, you don't have to worry how you declare the interactions(implicitly or explicitly) because all of them are relative and will give the same predictions.
when plotting the predictions we get the same graph as the one we got for a model per category.
![[Pasted image 20231219123436.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219123436.png)
to manually calculate we have to manually select the right slope and intercept according to the category using `np.select()` and use the formula y = m * x + c

## Simpsons paradox
Simpsons paradox occurs when the slope of the model for the whole dataset is totally different from when the model is done with interactions that is when the model is used on subsets of data.
for example the `regplot` of an example dataset with only the continuous variable shows that

![[Pasted image 20231219124906.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219124906.png)
this shows a positive slope that as x increases y increases.
however when we build a model where the continuous variable interacts with a categorical variable and plot it

![[Pasted image 20231219125019.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219125019.png)
we see that for each category, the slope is negative.
which is the best model then? the answer depends on context of the data and the question you are trying to answer.
for example if you are checking a dataset to see the relationship between the number of hours a kid spends on video games and the test scores he gets, 

![[Pasted image 20231219125209.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219125209.png)

we can see that the interactions model is better because we gather insight that as age increases the test scores gradually increases however for each age as the number of hours on video games increases the scores decrease.

another example is ![[Pasted image 20231219125331.png]]
here we see the relationship between population density and the infection rate of a disease, the first plot shows that as the density increases the rate also increases. the second shows us that for each city, higher population leads to lower rate of infection. but this might be due to unknown factors like the wealth and demographic of each city. arguably, "infection rate increases with population density" is better and so the model with the whole dataset is better.

# Multiple Linear Regression

## Two numeric explanatory variables
to visualize three numeric variables(1 response and 2 independent) we might need a 3d scatter plot. however it is difficult to interpret so we use a normal scatter plot with the response variable set as hue.
![[Pasted image 20231219130822.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219130822.png)
from the plot we can see that as height and weight increases, the color gets darker meaning, it gets more heavier.

for the model, its the same as before
```python
model = ols("response ~ cont1 + cont2", data=df).fit()
```
the params we get for this model is a global intercept and a slope for each continuous variable. making predictions is the same as before. when the predictions are plotted
![[Pasted image 20231219131020.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219131020.png)

we can see that the top right colors are more darker indicating more mass.

when we use interactions for the model:
```python
model = ols("response ~ cont1 * cont2")
model.params
```
the params for this model are a global intercept and a slope for each continuous variable and a slope for the interaction
![[Pasted image 20231219131302.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219131302.png)

when the predictions are plotted
![[Pasted image 20231219131342.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219131342.png)
we can see that the predicted values match the colors of their surrounding data much better indicating a better fit.

## More than two explanatory variables
when there are more explanatory variables, it becomes difficult to plot the data. so we use a `FacetGrid` to facet on the values we need.
![[Pasted image 20231219192009.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219192009.png)
 this way we can see the relationship between as many variables. however as the number increases the plot becomes harder to interpret.
 but note that faceting means we facet on categorical variables.
 for models there isn't much of a change.
```python
#without interactions
model = ols("response ~ cont1 + cont2 + cat1 + 0", data=df).fit() 


#with interactions pairwise that is all two way interactions
model = ols("response ~ cont1 + cont2 + cat1 + cont1:cat1 + cont2:cat1 + cont1:cont2 + 0", data=df).fit()
or
model = ols("response ~ (cont1 + cont2 + cat1) ** 2  + 0", data=df).fit()


#with all possible interactions
model = ols("response ~ cont1 + cont2 + cat1 + cont1:cat1 + cont2:cat1 + cont1:cont2 + cont1:cont2:cat1 + 0", data=df).fit()
or
model = ols("response ~ cont1 * cont2 * cat1 + 0", data=df).fit()
```
the parameters for models like these are an intercept for each category and a slope for each continuous variable and one for each interactions.
predictions are the same as usual. however plotting them might be a pain so we won't do that.

## How does Linear regression work
OLS means Ordinary Least Squares. what does this mean? lets see.
now lets take simple linear regression as an example. we'll get a function with the coefficients we get from the model namely 
y = m * x + c
but how do we find these coefficients?
lets take random coefficients first. with these we get a function y = m1 * x + c1
the model's main goal is to reduce the size of residuals. so lets make that a parameter for **optimizing** the function. 
so we need the minimum value for the **sum of the squares of residuals** hence the name.
so for the random function, we'll calculate the predictions and see the sum of squares. and we update the slope and intercept again and check the sum of squares. we do this again and again to get the least value for sum of squares. this is known as **optimization**.
optimization of a function can be done with calculus. to get the least value/the greatest value(Maxima, Minima) we just have to apply derivative and see the value for x for which y is 0. that is our minimum point.
but for complex functions we get with more variables, there is a much more complex optimization involved. we use the minimize function from `scipy`
also the below implementation is for 1 explanatory variable. for more, the method gets complicated
```python
from scipy.optimize import minimize
def calc_sum_of_squares(coeffs):
	intercept, slope = coeffs
	y_pred = intercept + slope * df["exp1"]
	resid = y_pred - y_actual
	sum_s = np.sum(resid**2)
	return sum_s

print(minimize(fun=calc_sum_of_squares, x0=[0, 0]))
```
`minimize(fun=func to minimize, xo=[starting coefficients])`
minimize takes the return value of the function passed as the y and the whole function as the function to optimize. and the parameters to upgrade are taken as the parameters of the function. `x0` is the starting coefficients from which it upgrades to the optimized ones. 
here since its simple linear regression, we just calculate the predicted values as     m * x + c. if it had more explanatory variables, it would have been different, but the method is that it somehow calculates predictions for the starting coefficients and use them to calculate the sum of squares of residuals. then it optimizes the parameters to give a low sum of square of residuals. this how linear regression works. (mic drop)

# Multiple Logistic Regression
Logistic regression with multiple variable is the same as linear regression except we use the `logit` function instead.
interactions and others all are the same for this.

## The Logistic function
before logistic let us look at the gaussian or normal distribution. this distribution is well known by its bell curve. to check the gaussian distribution of given data, we call `norm.pdf()` pdf being Probability Density function.
![[Pasted image 20231219213647.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219213647.png)

the area under a normal distribution is known as the CDF of the curve. that is integrating the PDF gives us another function named CDF - Cumulative Distribution Function. just like PDF we call CDF using `norm.cdf()`. 
![[Pasted image 20231219213828.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219213828.png)
this is the CDF of a normal distribution. as we can see the y values are restricted from 0 to 1. this is a feature of all CDF of all distributions. we can think of CDF of x as transformation of x from the values to the probabilities. so for x=1 if we get y=0.84 that means the probability of a normally distributed variable x being less than 1 is 84%.
CDF transforms from x to probabilities. likewise, a way to transform from probabilities to values is the PPF - Perfect Point Function. also known as the inverse CDF or quantile function.
![[Pasted image 20231219214248.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219214248.png)
here we can see that the graph is just a flipped version of CDF. 
now lets do all of this for a logistic distribution.
the PDF of a logistic distribution looks just like a normal distribution but the tails are a bit fatter.
![[Pasted image 20231219214441.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219214441.png)

the CDF of a logistic distribution is known as the logistic function.
this CDF is also known as the sigmoid function and the formula is 
![[Pasted image 20231219214603.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219214603.png)
the inverse CDF or PPF is known as logit function. this is the same name as the log odds ratio we saw about in the previous lesson.
the formula for that is log(odds) or
![[Pasted image 20231219214710.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231219214710.png)

## How does logistic regression work
just like linear regression, logistic regression involves optimizing a metric. however the metric here is the likelihood 
the formula is 
sum of all        prediction * actual + (1-prediction) * (1-actual)
from this formula we can see too cases, 

case 1, actual =1:
then likelihood = prediction, so likelihood more nearer to 1 means that the prediction is more nearer to 1 which is good.
case 2, actual=0:
then likelihood = 1 - prediction, so likelihood more nearer to 1 that means the prediction is more nearer to 0 which is good.

in both cases if likelihood nearer to 1 the better. so the sum of all likelihood must be a greater value for the model to be good. however computing likelihood brings in many small numbers so it may bring up errors. so we apply log to the predictions before calculating likelihood and this metric is known as log likelihood.
if log likelihood increases, the model fit is better.

now just like linear regression, logistic regression is calculated using a slope and an intercept, but the resulting y value is passed to a logistic function to generate probabilities. from this we calculate predictions and from that we calculate the log likelihood.
```python
from scipy.optimize import minimize
from scipy.stats import logistic
def calc_log_likelihood(coeffs):
	intercept, slope = coeffs
	y_preds = logistic.cdf(slope * x_actual + intercept)
	log_like = np.log(y_preds) * y_actual + np.log(1-y_preds)*(1-y_actual)
	sum_l = -np.sum(log_like)
	return sum_l

print(minimize(fun=calc_log_likelihood, x0=[0,0]))
```
just like last time minimize is used to minimize the return value of the given function by optimizing the parameters of the function after starting at x0.

since we can only minimize, we apply negative sign to the sum of log likelihoods.

now first we calculate the predictions after passing the y values from y = mx + c  to the logistic function(sigmoid). then we calculate the log likelihood using the formula and adding them all up. then we add the - symbol to "MINIMIZE" the negative log likelihood, in turn maximizing maximizing log likelihood.

that's how it works for simple logistic regression. even for complex ones with more explanatory variables, only the calculation of predictions changes. then optimization of log likelihood takes place. how does the function optimize? idfk.
