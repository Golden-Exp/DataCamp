#datacamp #statistics

field of statistics -> practice of collecting and analyzing data
summary statistic -> summary or fact about some data

descriptive statistics -> describe the data
inferential statistics -> get an inference about a large population using sample data
![[Pasted image 20231030154849.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231030154849.png/?raw=true)

Two types of data
1. numeric
2. categorical
numeric data has
1. continuous    ->   values that can be measured (speed, time)
2. discrete          ->   counted data (no. of pets)
categorical data has
1. nominal    ->  not ordered(marriage status)
2. ordinal      ->  ordered(agreement)
![[Pasted image 20231030155314.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231030155314.png/?raw=true)
categorical data can also be represented with numbers(0 or 1 and 1 to 5 stuff like that)

which summary statistics and visualization to use depends on the type of data

numeric data makes use of scatter and line plots and statistics like mean
categorical data makes use of bar plots and counts 

### measures of center
3 measures of centers
1. mean
2. median
3. mode
mean -> average of data
median -> the middle of the sorted data
mode -> most frequent data

mode used more frequently used for categorical data 
median is affected to extreme values. so for example if u add an outlier to the data the mean changes more than the mode.
so mean better to use for symmetrical data
median better for skewed data

![[Pasted image 20231030160613.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231030160613.png/?raw=true)

symmetric data -> the mean and median are almost same
![[Pasted image 20231030160652.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231030160652.png/?raw=true)

skewed data
right skewed -> right tail
left skewed -> left tail
the mean tends to go towards the tail. so, 
in left skewed, mean < median
and in right skewed mean > median
```python
import statistics
import numpy as np
np.mean(df['column'])
np.median(df['column'])
statistics.mode(df['column'])
```
### measures of spread
![[Pasted image 20231030162811.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231030162811.png/?raw=true)

spread is the measure of how much spread apart the data is
variance
it is the average distance from each data point to the data's mean
to calculate we first calculate the distance of each point from the mean.
then to eliminate negative distances we square each value. then we add everything and divide by the no. of data -1. that is the variance -> the unit is squared
standard deviation is the root of variance -> the unit is normal and not squared
more variance = more spread

![[Pasted image 20231030163118.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231030163118.png/?raw=true)
```python
dists = msleep['sleep_total'] - np.mean(msleep['sleep_total']) 
sq_dists = dists ** 2
sum_sq_dists = np.sum(sq_dists)
variance = sum_sq_dists / (83 - 1)
std = variance ** 1/2
np.var(df['column'], ddof=1)
np.std(df['column'], ddof=1)
```
mean absolute deviation is the same as variance but instead of squaring to eliminate negatives we take the absolute values and mean them
```python
dists = msleep['sleep_total'] - mean(msleep['sleep_total']) 
np.mean(np.abs(dists))
```
std is more for longer and short for shorter due to squaring
![[Pasted image 20231030163321.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231030163321.png/?raw=true)

#### quantile
it is the data at that particular fraction of the sorted data
0.5 quartile is the median
```python
np.quantile(msleep['sleep_total'], [0, 0.25, 0.5, 0.75, 1]) ->  returns all the quantile values
```
quartiles -> split data into 5 -> 0, 0.25, 0.5, 0.75, 1
quintiles -> split data into 6 pieces -> 0, 0.2, 0.4, 0.6, 0.8, 1
for 11 its deciles
boxplots show quartiles
![[Pasted image 20231030163636.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231030163636.png/?raw=true)

```python
plt.boxplot(df['column'])
plt.show()
```
linspace function in numpy
```python 
np.linspace(0,1,5) -> returns values from 0 to 1 split into 5 parts i.e. 0, 0.2, 0.4, 0.6, 0.8, 1
```

interquartile range
height of box in boxplot that is 75th quantile - 25th quantile
```python
from scipy.stats import iqr 
iqr(msleep['sleep_total'])
```
outliers
data that is substantially different than others
conditions for an outlier 
data < q1 - iqr * 1.5 
data > iqr * 1.5 + q3
q1 - 0.25
q3 - 0.75
```python
from scipy.stats import iqr 
iqr = iqr(msleep['bodywt']) 
lower_threshold = np.quantile(msleep['bodywt'], 0.25) - 1.5 * iqr 
upper_threshold = np.quantile(msleep['bodywt'], 0.75) + 1.5 * iqr
msleep[(msleep['bodywt'] < lower_threshold) | (msleep['bodywt'] > upper_threshold)]
```
df.describe gives many summary statistics all at once

## Random numbers and probability
```python
df.sample(n, replace=True or False) -> returns n random rows
if replace true rows might be repeated
np.random.seed(n) -> sets seed
```
Independent events
when one event's outcome doesn't affect the probability of the second event (with replacement)
Dependent events
when one event's outcome affects the probability of the second event(without replacement)
#### probability distributions
describe the probability of each possible outcome
**expected value**: mean of the probability distribution
![[Pasted image 20231031094024.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031094024.png/?raw=true)

the probability distribution of rolling a die 

discrete probability distributions
the outcomes of the distribution are discrete(they can be counted)
if all the outcomes have same probability they are uniform discrete distributions

when sampling and then calculating the probability, if u sample a very large number the resulting distribution will be close to the theoretical distribution and the mean will be close to the expected value

![[Pasted image 20231031094349.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031094349.png/?raw=true)

*here in the graph the sample and probability distribution are compared since they should almost be the same. because if an outcome has high probability then its size in the sample histogram shud also be high*

this is law of large numbers -> as the sample size increases, the mean approaches the expected value
![[Pasted image 20231031094437.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031094437.png/?raw=true)


### continuous distributions
they are distributions of outcomes that cant be counted.
for example: the time of the day at which a bus arrives cant have discrete number of outcomes because even if it is determined to to arrive every 12 minutes, it can arrive at any time in those 12 minutes -> no. of outcomes is infinite
so, a line is drawn above 0 to 12 to show that every time is equally probable of being the time the bus shows up

![[Pasted image 20231031100910.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031100910.png/?raw=true)

this is continuous uniform distribution
some continuous distributions are non-uniform - that is the probabilities are not equal for all outcomes. however, the area under the graph is always 1

to see the probability of any outcome less than or equal to a particular outcome we use
```python
from scipy.statistics import uniform
uniform.cdf(the number, lower limit, upper limit) -> the  number is the particular outcome -> gives probabilty of all between 0 to the number
if between m and n needed
calculate 0 to m and 0 to n and subtract
uniform.cdf(m, ul, ll) - uniform.cdf(n, ul, ll)
```
to see greater than use 1 - lower than
to generate random numbers for a continuous distribution
```python
from scipy.stats import uniform
uniform.rvs(0, 5, size=10) -> lower limit=0, upperlimit=5, size of sample 10
```
![[Pasted image 20231031100943.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031100943.png/?raw=true)

### Binomial distributions
Probability distribution of the number of successes in a sequence of **independent** trials
example: to flip n coins m times how many heads are possible each time?
```python
from scipy.stats import binom
binom.rvs(m, n, size=10) -> m is the number of events done simultaneously(3 coins flipped at the same time) and n is probability of success for an event, size is the no. of times m events should be done simultaneously(3 coins should be flipped simultaneously 10 times. n-> here its 0.5)
returns a sample of 10 events with the no. of success in each
```
![[Pasted image 20231031184234.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031184234.png/?raw=true)

described by n and p
n -> no. of events
p -> probability of success

for probability of m successes in n events
```python
from scipy.stats import binom
binom.pmf(m, n, p) -> gives probability of m successes in n events
binom.cdf(m, n, p) -> gives probability of m or less than m successes in n events
```
the expected value of a binomial distribution is given by (p * n)
expected value of 10 tosses = 10 * 0.5 = 5 heads
**the event should be independent for calculating binomial distribution**
this is because if its dependent then after 1 event the probability changes and the formula won't work

### normal distribution
![[Pasted image 20231031190648.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031190648.png/?raw=true)

 a normal distribution is often called a bell curve. its mostly defined by its mean and standard deviation
 when mean = 0
 and std = 1
 it is a standard normal deviation
 its useful because many known data falls under this distribution.
 ![[Pasted image 20231031191324.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031191324.png/?raw=true)
 
 if it does we can use norm package to calculate the probabilities and quantiles and even create samples
 ```python
from scipy.stats import norm
norm.cdf(m, mean, std) -> gives percentage of sample that is below m -> this is the probability too coz percent
subtract with 1 to get greater than
and subtract two cdfs to get in between values
```
![[Pasted image 20231031191836.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031191836.png/?raw=true)

this also means that when u randomly pick a woman from the sample, there is a 16 percent chance that she is shorter than 154 cm.
```python
norm.ppf(percent, mean, std) -> gives what value is the given percent of sample less than
norm.ppf(0.9, mean, std) -> gives what value is 90% of the population less than
norm.rvs(mean, std, size=n) -> gives n samples of the normal distribution
```


## Central limit theorem
lets say u do an event n times. then u take the mean of the outcomes.
repeat this process m times to get m means
plot the means.
this is a sampling distribution, that is the distribution of a statistic of the sample
![[Pasted image 20231031193058.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031193058.png/?raw=true)

this can be done to any statistic not only mean. when doing it with a particular statistic the mean of the sampling distribution will be equal to the statistic of the sample.
the CLT states that when the no. of samples increases, a sampling distribution approaches a 
normal distribution
![[Pasted image 20231031193131.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031193131.png/?raw=true)

with CLT we can determine unknown characteristics of the underlying distribution using mean of sampling distribution
it is also more easy when there is a large population because we can take samples and determine the statistics

## Poisson distribution
events happening at a certain rate in an interval, but random
probability of some events happening after some fixed time is calculated with poisson distribution
the time interval should be consistent
lambda -> the average no. of events happening at a time interval
in the distribution, the peak will be at the lambda value

![[Pasted image 20231031194925.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031194925.png/?raw=true)

```python
from scipy.stats import poisson
poisson.pmf(n, lambda) -> gives the probability of n events occuring with that lambda
poisson.cdf(n, lambda) -> gives the probability of n and less than n events occuring with that lambda
poisson.rvs(lambda, size=n) -> gives n samples from the poisson distribution
```

### Exponential distribution
it calculates the time between 2 poisson events
uses lambda too.
since it measures the time, it is a continuous distribution
lets say on average 1 ticket generated every 2 mins. -> poisson distribution with lambda 1 and the time interval being 2 mins
then 0.5 tickets each minute on average -> this is the rate/scale 

![[Pasted image 20231031200253.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031200253.png/?raw=true)
![[Pasted image 20231031200710.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031200710.png/?raw=true)
as lambda increases the curve increase

![[Pasted image 20231031200338.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031200338.png/?raw=true)

how long until new ticket?
probability of 1 minute?
```python
from scipy.stats import expon
expon.cdf(1, scale=2) -> probability of waiting less than a minute
1 - expon.cdf(1, scale=2) -> probability of waiting greater than a minute
```
the scale is the average time interval for 1 event to take place -> here 2
## T distribution
also known as student's T distribution

![[Pasted image 20231031200840.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031200840.png/?raw=true)
similar to normal but the tails are thicker.
there's a parameter called degrees of freedom for t distribution -> affects the thickness of tails
when DOF is increases it becomes more and more like a normal distribution

![[Pasted image 20231031200954.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031200954.png/?raw=true)
## Log-normal distribution

a variable whose log is normally distributed is said to be log normal distribution

![[Pasted image 20231031201112.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031201112.png/?raw=true)
when log is not applied it looks like another distribution

## correlation
the relation b/w numeric variables can be visualized using a scatter plot.

![[Pasted image 20231031232920.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031232920.png/?raw=true)

likewise the numeric value for measuring the relationship between two variables is known as correlation coefficient.
it quantifies the linear relationship between two variables
always between -1 and 1
if closer to 1 very strongly related -> if x increases y increases
if closer to -1 inversely related -> if x increases y decreases
if closer to 0 not related at all -> knowing x doesn't tell us anything about y
scatterplots using seaborn
```python
import seaborn as sns
sns.scatterplot(x='colname', y='colname', data=df)
plt.show()
```
![[Pasted image 20231031233326.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031233326.png/?raw=true)
```python
sns.lmplot(x='colname', y='colname', data=df, ci=None)
plt.show()
```

![[Pasted image 20231031233348.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031233348.png/?raw=true)
lmplot draws a line to show us the relationship

correlation coefficient calculated by
```python
df['col'].corr([df['col']]) -> gives the correlation b/w the columns
```
many ways to calculate the correlation
most commonly used is Pearson product-moment coefficient ( r )
![[Pasted image 20231031233641.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031233641.png/?raw=true)

## correlation caveats
1. non - linear relationships
![[Pasted image 20231031234932.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031234932.png/?raw=true)

just because the correlation is low doesn't mean the data doesn't have a relationship like the above data.
to transform the relationship to linear we use transforms on the variable.
like
1. log
2. reciprocal
3. square root
![[Pasted image 20231031235809.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031235809.png/?raw=true)

![[Pasted image 20231031235828.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031235828.png/?raw=true)

since the data is too skewed, log transformation can be used

![[Pasted image 20231031235909.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231031235909.png/?raw=true)

combinations
log x and log y
1/x and sqrt(y)
why to transform?
-  certain statistical methods depend on having a linear relationship
	- correlation
	- linear regression
"Correlation doesn't imply causation"
just because two variables are correlated doesn't mean one of them causes the other. There might be a secret cofounder that is the reason for this association

![[Pasted image 20231101001014.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231101001014.png/?raw=true)

although drinking coffee and lung cancer is correlated, drinking coffee doesn't result in cancer. however it is associated to cancer because, drinking coffee is associated to smoking and smoking causes cancer. thus smoking is the cofounder here.

## experiments
they aim to answer: *what is the effect of the treatment on the response?*
treatment: independent variable
response: dependent variable
#### controlled experiments
participants are assigned to treatment group or control group
- treatment group goes through the treatment
- control group doesn't
the groups should be comparable. like there shouldn't be differences like one group have a higher average age than the other.
this leads to cofounding(bias)
gold standard experiments use: 
  - randomized controlled experiments -> the participants are chosen at random to reduce bias
  - placebo -> resembles treatment but it isn't. used to fool the participants into thinking they are under treatment so that in the end we can check if the result is due to the treatment or the idea of the treatment. participants dunno which group they are in
  - double blind trials -> person administering treatment also doesn't know which group they are assigning the participant to. prevents bias during analysis
  in the end: **fewer opportunities for bias = more reliable conclusion**
#### observational studies
people aren't assigned to groups. instead, they should already be part of a group, usually pre-existing
example: sales history, medical history, height.
observational studies can only establish association and can't establish cause, there can be certain cofounders u can't control.
#### longitudinal vs cross-sectional studies
![[Pasted image 20231101003034.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231101003034.png/?raw=true)

longitudinal -> same participant used again and again after some time
cross-sectional -> different participants used.
