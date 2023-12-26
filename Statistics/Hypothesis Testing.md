#datacamp #statistics 
# Intro to Hypothesis Testing

## Hypothesis tests and z-scores
hypothesis testing is as the name suggests, testing out a hypothesis and seeing how right the final result is. lets say, we hypothesize about the mean of a certain feature in our data frame to be m1. so we calculate the sample mean
```python
sample_mean = df["col"].mean()
```
if we see that the mean is different, how so? and is it meaningful? to find out we plot the bootstrap distribution.
```python
means=[]
for i in range(5000):
	means.append(
	df.sample(frac=1, replace=True)["col"].mean()
	)
plt.hist(means)
plt.show()
```
![[Pasted image 20231224112254.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224112254.png/?raw=true)
we see that it is kind of like a normal distribution. 
now before testing, since values have arbitrary units and values, we need to standardize them.
a common way is to subtract the value from the mean and divide by the standard deviation.
for hypothesis testing, we use something similar , that is z-score. here we subtract the sample statistic from the hypothesized value and divide by the standard deviation. if we were right on the mark the z-score should be 0.
so we do that by subtracting `df["col"].mean()` from m1 and divide by std.
which is the standard error we get from the bootstrap distribution.
we get something like 1.707. is it good or bad? we'll see that later. that is one of the goals of hypothesis testing, to determine whether sample statistics are close to or far away from expected.
there is also something called the z distribution ,which is also known as standardized normal distribution, with loc(mean)=0 and scale(std)=1
![[Pasted image 20231224112952.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224112952.png/?raw=true)

## p-values
a **Hypothesis** is a statement about an unknown population parameter.
when hypothesis testing, there are 2 hypothesis competing to see which is right.
1. the Null Hypothesis
2. the Alternative Hypothesis.
the null hypothesis is what is assumed at first. this could have been derived from a research or other sources, but at first it is assumed that the null hypothesis is right.
when we propose evidence from the sample that is "significant", the null hypothesis is rejected or we choose the null hypothesis
we always refer the results of a test in terms of the null hypothesis. we either, reject the null hypothesis or choose it.
![[Pasted image 20231224122444.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224122444.png/?raw=true)

for a distribution, the tails are its left and right edges
***Hypothesis tests determine whether the sample statistics lie in the tails of the null distribution, which is the distribution of the statistic if the null hypothesis was true.***
there are 3 types of tailed tests:
1. Two-tailed: this is used when the alternative is different from the null. that is if we want to see the difference between them we check in the two extreme values in the tails
2. Right-tailed: this is used when the alternative uses phrases like "is greater than" when compared to null.
3. Left-tailed: this is used when the alternative uses phrases like "is lesser than" when compared to null.

now p-values are probabilities of getting a result, assuming null is true. so larger p-values mean larger support for null
Large p-values mean our statistic is producing a result that is likely not in a tail of our null distribution, and chance could be a good explanation for the result. Small p-values mean our statistic is producing a result likely in the tail of our null distribution. Because p-values are probabilities, they are always between zero and one.
![[Pasted image 20231224122507.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224122507.png/?raw=true)
 to calculate the p-value, we calculate the z-score and find the `norm.cdf` value to get probabilities. remember that `norm.cdf` gives probabilities of the value being less than given value. if we need to do a right tailed test, that is greater than ,we find 1 - `norm.cdf` to find the probability of greater than
```python
z_score = (sample_mean - hypo_mean)/std_error
p_value = 1 - norm.cdf(z_score, loc=0, scale=1)
```
## Significance Levels
the significance level of a hypothesis test is the threshold point for which we compare p-values. if the p-value is greater than alpha(significance level), then we choose the null hypothesis. else we reject the null hypothesis.
usually, 0.05, 0.1, 0.01, and 0.2 are common choices for alpha.
so, if p-value <= alpha -> reject null
and, if p-value > alpha -> choose null
alpha should be set prior to the test, because choosing one after calculating the p-values, might introduce bias.
we also use confidence intervals to denote where the point estimate would be, if we reject null. this is selected as the middle 1-alpha quantile.
so if we have alpha = 0.05 and p-value <= alpha, we reject null and say that the population parameter lies in the range of 1 - alpha so 1-0.05 = 0.95 
so the range is the 95% confidence interval
```python
lower = np.quantile(boot_distn, 0.025)
higher = np.quantile(boot_distn, 0.975)
```
these are the range of values under which the real population parameter lies.

there are 2 types of errors in hypothesis testing
1. **Type 1**: if p<=alpha and we reject null, there is a chance we made a false positive. that is the null was actually correct and not the alternative
2. **Type 2:** if p> alpha and we choose null, there is a chance we made a false negative. that is the null is actually wrong and alternative was right.

# Two Sample and ANOVA Tests

## Performing t-tests
Two sample problems involve comparing statistics across a group of variable.
for example, a hypothesis could be
the value of a column is higher for one category compared to another. 
to do hypothesis tests on these we need t-tests.
the null is that the value is same for both categories. that is the mean of values in category 1 is the same as the mean in category 2
the alternative is that the value is higher for category 1 when compared to category 2 that is, the mean of values in category 1 is greater than in category 2
we can also say
null: mean1 - mean2 = 0
alt: mean1 - mean2 > 0
we can directly check this by grouping over categories and checking the results. but, is the result accurate or can it be explained by sampling variability?
we use test statistics to confirm this:
test statistic is the difference between the mean1 and mean2, that is used for the hypothesis test.
z-scores are a type of standardized test statistic and are used when we do one sample tests.
for two sample tests we use test statistic, known as t. this can be calculated in a similar way as z-scores.
z = (sample stat - hypothesized stat)/standard error
t = (difference in sample stats - difference in population parameter)/standard error

the standard error can be calculated from the bootstrap distribution ,but a better way is:  s is calculated on the sample and not from the bootstrap

![[Pasted image 20231224131102.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224131102.png/?raw=true)

and since we assume null to be true, the difference in population parameter would be 0.
and so the final equation becomes:
![[Pasted image 20231224131153.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224131153.png/?raw=true)

with the resulting t-statistic, we can't make any conclusions. just like how p-values are calculated from z-scores and then a result is told, t-stats also have something like that.
![[Pasted image 20231224131526.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224131526.png/?raw=true)

## Calculating p-values from t-statistics
like how we calculated p-values for z-scores by passing the z-score to the CDF of a z-distribution(standardized normal distribution), we get p-values of t-statistics by passing the t-statistics to the CDF of the t-distribution. they are similar to normal distributions but have fatter tails
the t-distribution has a parameter called degrees of freedom. as the degrees of freedom increase, the t-distribution gets closer to a normal distribution.
![[Pasted image 20231224132820.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224132820.png/?raw=true)

the degrees of freedom is the number of independent values in our data. for example, lets say we have 5 values all of which are independent. if we don't know any other thing other than the 5 values, then the degrees of freedom is 5. however, if we know 4 values and the mean of the whole data, then the degrees of freedom is 4 because, only 4 are truly independent. the 5th value can be found using the mean.
so the degrees of freedom for our t-statistics is n1(number of observations in category 1) plus n2(number of observations in category 2) - 2
n1 + n2 - 2
the minus 2 is because we know 2 means the mean of category 1 and mean of category 2.
so, now we pass the t-stat to the CDF of t. since we are hypothesizing that mean1 - mean2 > 0, we use right tailed test and take 1 - CDF as p-value
```python
from scipy.stats import t
p_value = 1 - t.cdf(t_stat, df=deg_of_freedom)
```
now compare this to the significance level alpha that should have been set before calculating p-value. and reject or choose null.

## Paired t-tests
just like last time, lets say a hypothesis that mean1 < mean2. so now,
null: mean1 - mean2 = 0
alt: mean1 - mean2 < 0
but this time, a twist is that the values column are paired.
If you have repeated observations of something, then those observations form pairs. that is they have some common factor. this can be anything. for example, if we are comparing the Republican votes in 2008 and in 2012, both are paired because they are the same county. so this is known as a paired t-test.
for paired t-tests, the calculations become simpler.
the null and alt become
null: `mean_diff` = 0
alt: `mean_diff` < 0 
 and so, the t-stat is
 t-stat = (`mean_diff_samp` - `mean_diff_pop`)/std error from sample
 so,
![[Pasted image 20231224134744.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224134744.png/?raw=true)
  since we assume that null is true, `mean_diff_pop` = 0
  and then just like last time we pass the t-stat to the t-CDF. however the degrees of freedom is `n_diff` - 1
  there is a package for this:
```python
import pingouin
pingouin.ttest(x=sample_data["diff"], y=0, alternative="less")

pingouin.ttest(x=sample_data["value 1"], y=sample_data["value 2"],
			  paired=True, alternative="less")
```
this will give us the same answer as we calculated.
`x` is the value 1 or the difference in the values
`y` is 0 as it specifies the value of difference for null. else, it specifies value 2
`alternative` is what type of test: "two-sided", "less", 'greater' corresponding to the tail tests
`paired`=True is also used when you don't want to convert your data and just get the t-tests.

always notice if the columns are paired or not, coz the chances of false negative is high.

## ANOVA tests
what we saw up until now was considering we only had 2 categories. what if we had more than 2 categories? we use ANOVA tests that is Analysis of Variance.
a good way of first seeing the differences is by plotting a box plot
ANOVA tests can calculated using the `pengouin` package mentioned above. remember we are comparing values of a continuous column across a categorical column
```python
alpha = 0.2
pingouin.anova(data=df,
			 dv="cont",
			 between="cat")
```
this gives us a `p-value` that is less than alpha. so that means at least 2 categories of cat column has different `cont` values. but to find which categories, we have to do a t-test on all possible pairs of categories. lets say there are 5 categories. then we need to perform 10 pairwise tests. we can do this in `pengouin`
```python
pingouin.pairwise_tests(data=df, 
					   dv="cont",
					   between="cat",
					   padjust="none")
```
this gives a data frame of the results of each pair wise tests.
![[Pasted image 20231224141420.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224141420.png/?raw=true)
from the `p_unc` column we can see that 3 values have lesser p values.
when the number of categories increases, the number of pairs also increases, quadratically. and so, the number of tests also increase.
if alpha is 0.2, the probability of a false positive(p-value<0.2 but null is correct) is 0.2 for 1 test.
for 10 tests the probability of at least 1 false positive is 1 - probability of no false positive = 1 - (0.8 * 0.8 ...) = 1 - ((0.8) ^ 10) which is around 0.89
. so as the number of categories increase, the probability of a false positive also increase. so we need to adjust the p-values by increasing them to reduce the number of false positives as the number of categories increase. so we use `p_adjust`
there are many techniques of adjustment, one common one is Bonferroni adjustment.
```python
pigouin.pairwise_tests(data=df,
					  dv="cont",
					  between="cat",
					  padjust="bonf")
```
![[Pasted image 20231224142527.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224142527.png/?raw=true)

now we have only 2 with p-values greater than alpha.
![[Pasted image 20231224142556.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224142556.png/?raw=true)

there are many ways for `padjust`. giving it 'none' is usually not an option.

# Proportion tests
## One sample proportion tests
when we calculated the p-values with z-scores, we used the standard error for the formula of z-scores. this is calculated from the bootstrap distribution. now lets calculate the z-score without the bootstrap distribution.
previously,
z = (sample stat - hypothesized stat)/std error
![[Pasted image 20231224231528.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224231528.png/?raw=true)

now, 
the std error is equal to
std = square root(((hypo stat) * (1 - hypo stat)) / n)
![[Pasted image 20231224231547.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224231547.png/?raw=true)

so we calculate the z score with just the hypothesized statistic, the sample statistic and the number of observations
the reason why we use z-distributions here and not t is that, t-distributions are used to account for the uncertainty in the formula used.
since in the formula, a lot of the variables are just estimates, we use t-distribution which has fatter tails to account for it. with the formula above, z-distribution is fine.

and if we want to apply a two tailed test, we need to calculate the tails in both sides and add them. since the distribution is symmetric, we just haver to multiply the right tail by 2
```python
p_value = 2 * (1 - norm.cdf(z_score))
```

## Two sample proportion tests
for two sample proportion tests, i.e. across categories, we can use the same formula like last time.
lets say
null : mean1 - mean2 = 0
alt: mean1 - mean2 != 0
to calculate the z-score,
z = ((mean1 - mean2) - (hypo (mean1 - mean2)))/std error
that is equal to
z = ((mean1 - mean2) - 0)/std error
where, 
std error = square root(((`p_hat` * (1 - `p_hat`)) / n1) + (`p_hat` * (1 - `p_hat`)) / n2))
here `p_hat` = (n1 * p1 + n2 * p2)/(n1 + n2)
where p1 = mean1 and p2 = mean2
![[Pasted image 20231224234639.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231224234639.png/?raw=true)

to avoid doing all these we can use the `proportions_ztest()` from `statsmodels` 
```python
from statsmodels.stats.proportion import proportions_ztest
z_score, p_value = proportions_ztest(count=np.array([n1, n2]),
									nobs=np.array[(len(cat1), len(cat2))],
									alternative = "two-sided")
```
the alternatives can be "less", "larger" or "two-sided"

## Chi Square tests of independence
lets say cat1 and cat2 are two categorical variables. for cat1 testing independence of cat1 from cat2, we keep cat1 as response and cat2 as explanatory variable.
**Statistical independence:** proportion of response variable is the same across all categories of the explanatory variable.
to test this we use chi square, which is basically z-score squared.
for example, if we wanted to test the independence of cat1 from cat2, we calculate z-score as usual, with,
null: cat1 is independent of cat2
alt: cat1 is not independent of cat2
the test statistic used is chi square. it quantifies how far away the observed results are from the expected values if independence is true.
to test this, we first visualize the proportions of cat1 across cat2
![[Pasted image 20231225081106.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231225081106.png/?raw=true)
then we use the `chi2_independence` function from `pingouin`
```python
import pingouin
expected, observed, stats = pingouin.chi2_independence(data=df, x="cat2", 
													  y="cat2", 
													  correction=False)
```
The correction argument specifies whether or not to apply Yates' continuity correction, which is a fudge factor for when the sample size is very small and the degrees of freedom is one
it gives us three tables, out of which `stats` is the one we need.
![[Pasted image 20231225081844.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231225081844.png/?raw=true)

we then compare the p-values to our significance level to see which hypothesis to reject. 
now if we try swapping the variables, we can see if the variables are statistically independent of each other.
that is instead of is cat1 independent of cat2
we say are cat1 and cat2 independent?
also what tail test are we using? the answer is right tailed, because we are squaring the z-score so its always positive.

## Chi square goodness of fit tests
to compare the distribution of one categorical variable, we use chi square goodness fit.
lets say we hypothesize the distribution of a certain categorical variable's proportions
```python
total = len(df)
hypo = pd.DataFrame({"cat":["cat1", "cat2", "cat3"], 
					 "prop":[1/3, 1/3, 1/3]})
hypo["n"] = hypo["prop"] * total
```
null: the sample matches the hypothesized distribution
alt: the sample doesn't match the hypothesized distribution.
lets first visualize both distributions
![[Pasted image 20231225085828.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Statistics/Attachments/Pasted%20image%2020231225085828.png/?raw=true)

```python
from scipy.stats import chisquare
chisquare(f_obs=cat_df["n"], f_exp=hypo["n"])
```
it returns the p-value which we can compare with the significance level.

# Non - Parametric Testing
## Assumptions in hypothesis testing
we assume many things before starting hypothesis testing
### Randomness:
we assume that the sample is a random sample of the population. this is to be assumed because for all our tests, we need the sample to be representative of the population.
to verify this, enquire the data collection process
### Independence
all observations are assumed to be independent. when there are dependencies, we should know when. because as we saw with t-tests, the chances of a false positive increase. to verify this we could try t-tests, but it is ideal if we know any dependencies beforehand
### Large sample size
we assume that our sample is large enough for the central limit theorem to apply. if we don't have a large sample size, the confidence intervals size would be large and there is an increased chance of false positive errors
how large is large?

#### T-tests
1. one sample: sample size >= 30
2. Two samples: n1 >= 30 and n2 >= 30
3. ANOVA : ni >= 30, where i ranges from 1 to m(number of groups)
4. paired samples: at least 30 pairs of observations

#### Proportion tests
1. one sample: number of successes(n * `p_hat`) >= 10, number of failures(n * (1 - `p_hat`)) >= 10
2. two samples: n1 * `p_hat1` >= 10, n2 * `p_hat2` >= 10, same for failures

#### Chi Squared tests
1. number of successes for each group is greater than or equal to 5. same for failures.

we can use bootstrap distributions to verify if the sample is valid.

## Non - parametric tests
all of the tests we did so far, are parametric tests. these can only be done when the assumptions above are met.
if the assumptions aren't met, we use non-parametric tests. that is if the sample size is small, or the sampling distribution doesn't follow the normal distribution.
one of these tests is the Wilcoxon-signed rank test.
a rank is the position in which an element falls when its sorted.
the steps for finding the parameter w for the test of pairs is
1. Calculate the difference in mean of both pairs.
2. find the absolute difference.
3. find the ranks of the absolute difference
4. set `T_PLUS` as the sum of all the ranks of positive differences
5. set `T_MINUS` as the sum of all the ranks of negative differences
6. w is the minimum of the two(`T_PLUS`, `T_MINUS`)
to find p-values, use the `pingouin` package.
```python
pingouin.wilcoxon(x=df["cat1"], y=df["cat2"], alternative="less")
```

## non-parametric ANOVA tests and unpaired t-tests
the Wilcoxon-Mann-Whitney test is used when we have to do a t-test to see if hypothesis across categories are valid. to use this, we first have to convert the data into wide format.
```python
df = df[["cont", "cat"]]
wide_df = df.pivot(columns="cat", values="cont")
pingouin.mwu(x=wide_df["cat1"], y=wide_df["cat2"], alternative="greater")
```
just like how ANOVA is for t-tests with more categories, we use Kruskal-Wallis test for more categories.
```python
pingouin.kruskal(data=df,
				dv="cont",
				between="cat")
```
