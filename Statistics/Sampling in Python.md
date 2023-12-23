#datacamp #statistics 

# Intro to Sampling
working with a subset of the whole population is known as sampling.,
*population* is the complete set of data
*sample* is the subset of data you calculate on.
```python
df_sample = df.sample(n=10)
```
returns 10 unique rows from the population.
***Population Parameter:*** is a calculation made on the population dataset.
***Point Estimate or sample statistic:*** is a calculation made on the sample dataset.
note that the point estimates and population parameter done are identical but not the same. because they are estimates of the real population.

sampling is useful when we have a very large dataset and we cant waste resources working on it.

## Convenience Sampling
convenience sampling is when data collection is done in a way that is convenient for us, but leads to selection bias. 
for example, a sample survey was taken over telephones for which president was going to win. however the results from the survey widely varied from the actual results. this is because of convenience sampling. since the survey was done through telephones, only rich people contributed to the survey. that is not representative of the whole population.
we can easily spot convenience sampling with histograms. plot the population set and the sample set side by side. if the plots are kind of identical, that's good. else, the sample is bad. and the point estimates will also be way off from the actual value.
![[Pasted image 20231222233218.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231222233218.png)

![[Pasted image 20231222233235.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231222233235.png)

## "Random" Numbers
to get random rows for a sample we need random numbers. what is random? random is something that isn't done without method or consciousness. in real life some applications use, atmospheric noises, radioactive decay, and flipping coins to determine random numbers. but those are slow and costly. true randomness isn't cheap.
instead we use pseudo random numbers. it appears random, but the next number is actually calculated from the previous number. then the first number must be calculated from another number. that number is the seed. so that means, if we set the same seed, the string of random numbers will also be the same.

![[Pasted image 20231223000129.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223000129.png)

```python
np.random.distribution(parameters, size) ->returns random numbers from that distribution
```
the parameters for a normal distribution is loc and scale which is the mean and std respectively.

# Sampling Methods

## Simple random and systematic sampling
simple random sampling is the sampling we have been doing. just randomly selecting data. this might lead to selecting similar data or not selecting a large portion of a data.
![[Pasted image 20231223002647.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223002647.png)
each element has the same probability to be picked.
```python
df_sample = df.sample(n=5, random_state=3834321)
```
`random_state` is the same as seed.

systematic sampling is sampling done at regular intervals. 
![[Pasted image 20231223002854.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223002854.png)

to calculate the interval we divide the total length of the population by the number of rows we need in our sample. then we select the element after each interval.
```python
interval = len(df)//n
df_sample = df.iloc[::interval]
```
there is a problem with systematic sampling. it is only safe when we don't have patterns in our scatter plots. if we do its good to shuffle the data before sampling.
after sampling, its no use if you sample systematically or simply, coz its the same.
![[Pasted image 20231223003219.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223003219.png)
to shuffle,
```python
df = df.sample(frac=1)
df = df.reset_index(drop=True).reset_index()
```
frac is the proportion of data to be sampled. by choosing 1, we randomly select the whole data.
now,
![[Pasted image 20231223003347.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223003347.png)
no patterns

## Stratified and Weighted Random Sampling

when we want to sample based on categories, these types of samplings are used.
stratified sampling makes sure that the counts of each category in the sample is relatively similar to the counts of each category in the population.
to do this
```python
df_sample = df.groupby("category").sample(frac=0.1, random_state=10)
```
we just group by the categories and select 10% of the number of rows in each. so the counts will be similar. 
Similar to this there is equal counts stratified sampling. where instead of selecting based on the proportions, we select an exact number of rows from each category.
so there are equal number of rows from all categories.
```python
df_sample = df.groupby("category").sample(n=15, random_state=10)
```

weighted sampling, is when we need to increase the proportions of a certain category, so we add more weights to it. lets say we want to double the proportion of `cat1` in categories, when we sample. so we use weights.
```python
df["weights"] = torch.where(df["category"] == "cat1", 2 ,1)
df_sample = df.sample(frac=0.1, weights="weights")
```
the weight of `cat1` alone would be 2, meaning there's double the chance that it would be picked, so its proportion is doubled. this is used in political polls to correct the population for under/over representation.

## Cluster Sampling
cluster sampling is used when it is difficult to perform stratified sampling, because there are so many categories. in this type of sampling, we first take a sample of the categories and then we sample rows of that category only. 
this is a cheaper alternative to sampling all of the categories.
```python
categories = list(df["categories"].unique())
import random
cat_sample = random.sample(categories, k=5)#sample 5 categories

df = df[df["catgeory"].isin(cat_sample)]
df["category"] = df["category"].cat.remove_unused_categories()

df.groupby("category").sample(n=5, random_state=30)
```

![[Pasted image 20231223011132.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223011132.png)

this is a multistage sampling, where we first sampled the categories then the rows.
we can extend the number of stages and it is commonly used in national surveys.

## Comparison between methods
all of the methods give a pretty close point estimate to the population statistic. cluster sampling might be a little off, but it is designed to be that way ,due to limited resources.
![[Pasted image 20231223013136.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223013136.png)

same is seen when the population statistic of each category is compared to its point estimate. cluster sampling is a bad idea when the point estimate of each category is important, coz we sample on the categories too.

# Sampling Distributions

## Relative error of point estimates
when we compare point estimates to the population parameter, we see that as the sample size increases the point estimate becomes more accurate. that is it gets closer to the population parameter as we increase the rows in our sample.
Relative error is the absolute difference between the population parameter and the point estimate divided by the population parameter times 100 for percentage. the error widely changes just by adding a few rows to the sample when the sample size is small. however when the size is somewhat high, adding some rows won't change anything.
![[Pasted image 20231223121105.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223121105.png)

## Creating a Sampling Distribution
when we sample the same data many times, we get many different samples and each of them would have different point estimates. when we plot those different point estimates we get a sampling distribution.
each point estimate we calculate is known as a replicate. the distribution of replicates is known as the sampling distribution.
```python
replicates = []
for i in range(1000):
	replicate = df.sample(n=30)["col"].mean()
	replicates.append(replicate)

plt.hist(x=replicates, bins=30)
plt.show()
```
![[Pasted image 20231223122124.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223122124.png)

when the sample size decreases, the range of replicates broaden. when we increase it, the range decreases.
![[Pasted image 20231223122210.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223122210.png)
more sample size gives more accuracy.
By generating the sample statistic many times with different samples, you can quantify the amount of variation in those statistics.

## Approximate Sampling Distributions
exact sampling distribution: this is the distribution containing all the calculations done on all of the population dataset. lets say we have 4 dice. the dataset with all possible combinations is the population dataset.

```python
dice = expand_grid({
					"die1": [1,2,3,4,5,6],
					"die2": [1,2,3,4,5,6],
					"die3": [1,2,3,4,5,6],
					"die4": [1,2,3,4,5,6]
})
dice["mean_roll"] = dice.mean(axis=1).astype("category")
dice["mean_roll"].value_counts(sort=False).plot(kind="bar")
```
![[Pasted image 20231223191850.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223191850.png)
this is the exact sampling distribution. because this contains all the possible combinations. now when the number of dice increases the number of possible combinations also increase. to visualize this:
![[Pasted image 20231223192645.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223192645.png)

so now comes the use of approximate sampling distributions. to simulate the mean of 1000 die rolls, we select 4 from a range of 1 to 6 with replaceability and do this a 1000 times.
```python
import numpy as np
sample_means = []
for i in range(1000):
	sample_means.append(
	np.random.choice(list(range(1, 7)), size=4,replace=True).mean()
	)
plt.hist(x=sample_means, bins=20)
```
![[Pasted image 20231223193151.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223193151.png)
so using an approximation will give us a good estimate of how the population will behave.
The exact sampling distribution can only be calculated if you know what the population is and if the problems are small and simple enough to compute. Otherwise, the approximate sampling distribution must be used.

## Standard Errors and The Central Limit Theorem
![[Pasted image 20231223195016.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223195016.png)

when we increase the sample size when calculating a sampling distribution, the distribution appears closer to a gaussian/normal distribution. this is what the central theorem dictates.
![[Pasted image 20231223195601.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223195601.png)

now, if we take the mean of the means in the sampling distribution, the mean is closer to the mean of the population. this is a property of sampling distribution. and as the sample size increases, the mean is more accurate. 
![[Pasted image 20231223195844.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223195844.png)
however that is not true for standard deviations.
![[Pasted image 20231223200053.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223200053.png)
this is because as the sample size increases, the distribution gets narrower and so the spread decreases. however, the std of the population divided by the root of the sample size will give a close estimate of the std of the sampling distribution. this is useful to determine the right sample size.
this standard deviation of the sampling distribution is known as the standard error.
![[Pasted image 20231223200426.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231223200426.png)
so the sqrt of size times the sampling distribution's std gives us an approximate of the std of the population.

# Bootstrap distributions
## Intro to Bootstrapping
there are two types of sampling. resampling is when we sample with replacement, that is a row that is picked still has the same probability to be picked again.
without replacement means the sampling we have been doing, where once picked the row can't be picked again.
resampling with the `frac` parameter as 1 and the `replace` parameter as true will give us a dataset with the number of rows as that of the population. however some rows might be duplicates and some might be excluded. this concept is used in bootstrapping.
![[Pasted image 20231224000540.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231224000540.png)

bootstrapping is the opposite of sampling. here we try to derive the population from the sample. we can use this to understand the variability of a sample.
for bootstrapping,
1. create a sample with the parameters mention above and calculate a summary statistic of choice.
2. store the summary statistic in a list
3. repeat 1 and 2
```python
import numpy as np
means = []
for i in range(1000):
	means.append(
	np.mean(df_sample.sample(frac=1, replace=True)["col"])
	)

plt.hist(means)
```
![[Pasted image 20231224000508.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231224000508.png)
the plot of the distribution is known as bootstrap distribution
when u have the sample and not the population, you use bootstrapping. to calculate the summary statistic of the population when u have a sample, we use this.

## Comparing sampling and bootstrap distribution
when we take the mean of all the means we found from bootstrapping, we get the closest estimate of the mean of the sample.
not that it is the mean of the sample and not the population. the mean can only be closer to the population, if we got a good sample.
however the standard deviation is better. we can interpret std as the std of the summary statistic we calculated and as we did for standard errors last time. the standard error times the sample size's square root gives us a close estimate of the std of the population.
![[Pasted image 20231224001845.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231224001845.png)
the standard deviation of the population is best estimated by the bootstrap distribution. When you don't have all the values from the population or the ability to sample multiple times, you can use bootstrapping to get a good estimate of the population standard deviation.
![[Pasted image 20231224001919.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231224001919.png)

## Confidence Intervals
when calculating our point estimate, if we want to give a range of values under which the actual value might be, we calculate the confidence interval. confidence intervals are the ranges under which the point estimate might fall under.
***"values within one standard deviation of the mean"
 Confidence intervals account for uncertainty in our estimate of a population parameter by providing a range of possible values. We are confident that the true value lies somewhere in the interval specified by that range.***
to calculate this, we have three methods. 
```python
point_estimate = np.mean(bootstrap_distribution)
low = point_estimate - np.std(bootstrap_distribution)
high = point_estimate + np.std(bootstrap_distribution)
```
![[Pasted image 20231224004841.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231224004841.png)


another method, is to calculate the middle 95 percentile of the distribution to be the confidence interval.
```python
low = np.quantile(bootstrap_distribution, 0.025)
high = np.quantile(bootstrap_distribution, 0.975)
```
![[Pasted image 20231224004906.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231224004906.png)

the last method involves the inverse CDF function. it is the inverse of CDF which is the area of PDF of a distribution. this is the standard error method
![[Pasted image 20231224004935.png]](https://github.com/Golden-Exp/DataCamp/blob/main/statistics/Attachments/Pasted%20image%2020231224004935.png)

```python
point_estimate = np.mean(bootstrap_distribution)
std_error = np.std(bootstrap_distribution, ddof=1)
from scipy.stats import norm
low = norm.ppf(0.025, loc=point_estimate, scale=std_error)
high = norm.ppf(0.975, loc=point_estimate, scale=std_error)
```
"**There are two results from the last chapter that are really important for hypothesis testing, and it's important to make sure you understand them. Firstly, the standard deviation of a bootstrap distribution statistic is a good approximation for the standard error of the sampling distribution. Secondly, you calculated confidence intervals for statistics using both the quantile method and the standard error method, and they gave very similar answers. That means that the normal distribution tends to be a good approximation for bootstrap distributions.**"
