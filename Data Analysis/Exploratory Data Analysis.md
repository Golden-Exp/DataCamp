#datacamp #data_analysis 
# The Dataset
## Initial Exploration
first after getting the dataset, review it using the `head` method to get a feel for the dataset and its columns
then try using `info` method to see the missing values and the data types of the column
use the `describe` method to see how the numerical columns are distributed.
for categorical columns use `value_counts` to check how many elements are there in each categories.
you can also plot a histogram to check the distribution of a particular column.

## Data Validation
use `df.dtypes` to check the data type of each column and to change the data type of a particular column we can use
`df[col] = df[col].astype(datatype)`

then to validate categorical variables we can use `.isin` to check which columns come under the category we want. we also use the ~ operator to check the reverse/complement
`df[col].isin([list of categories]` 

`df.select_dtypes("number" or "object")`
to validate numerical values use the boxplot graphs to see how they are distributed and their ranges. use the hue parameter to check the distribution for each category.

## Data Summarization
we can summarize the values in our dataset using `groupby` followed by the aggregating functions like mean and median
`df.groupby(col).mean()`

we can also use the `agg` function to aggregate multiple functions
`df[col].agg(["mean", "std"])`
or pass in a dictionary to get the aggregate functions for particular columns
`df[col].agg({"col_name":["mean", "median"], "col_name":"mean"})`

with `groupby` this becomes 
`df.groupby(col).agg(col_name_g = ("col", "agg_func"), col_name_g = ("col", "agg_func"))`
`col_name_g` is the new column's name in the grouped series object
`col` is the column's name from the data frame.

we can also use a `barplot` to visualize the mean of a particular column with confidence intervals
recall that bar plots are used to check the summary statistic of a numerical column varied across a category.

# Cleaning and Imputation

## Missing Data
missing data should be handled correctly or the sample will represent the wrong population and the whole process after that might be wrong.
missing data can be found by 
`df.isna().sum()`

by a rule of thumb, if the total number of missing values in a column is **_less than or equal to 5%_** of the total length of the column, the missing values of that column can be dropped using
`df.dropna(subset=[col names], in_place=True)`

the process of filling the missing values with a summary statistic of the column is known as _**imputing**_ the missing values.

imputing can either be done by filling the missing values with a summary statistic of the whole column, that is, mean, median or mode

or the summary statistic of a particular category the missing value belongs to. (imputing by sub groups) this again typically is either mean or median.
to choose between mean or median, try plotting a boxplot to see if there are many extreme values. if there are, then median's the better choice than mean. else mean is better.

`to_dict()` used to convert to dictionary
then, 
`df[missing_col] = df[missing_col].fillna(df['category col'].map(dict))`
means the values of the column are mapped to the dictionary which means when the missing value belongs to a certain category, the value that belongs to that category in the dictionary will be filled in place


## Converting categorical data
sometimes in an object column, the number of categories can be much more than necessary. so we need to filter convert them to smaller categories for easy access and analysis. to convert we first need to check if the category has key phrases so that we can group them in new categories.
`df[col].str.contains("phrase")`
if we need to check multiple phrases at once, 
`df[col].str.contains(phrase1|phrase2)`  -> don't leave space before and after the pipe.
if only the starting word needs to be checked
`df[col].contains("^phrase")

all these give us a series of True/False values which can be used as conditions. 
according to these conditions, the categories should be changed.
this can be done with `np.select()`
`df[new col] = np.select(conditions, new categories, default=default)`
both conditions and new categories are lists of the same length. with respect to the order, if the condition in conditions is true, then that value will be replaced by the value in new categories that has the corresponding index.

a count plot can be used to visualize the frequency of a categorical column.


## Numeric columns

Sometimes there might be some columns which are of object datatype and you might need to covert them to integer datatype.
here 
`df[col].str.replace(old, new)` might be useful. then apply `.astype()` method.

sometimes looking up the summary statistic every time can be inconvenient. so sometimes we can add a new column with a summary statistic of a category that row belongs to.
this can be done by
`df[col mean] = df.groupby(categorical col)[numeric col].transform(lambda x: x.mean())`
this will apply mean across each the numeric column for each category and add them.
can also use mean and median for this.

## Outliers
outliers are extreme values. they are either extremely high or low compared to the values in the column.
we can calculate outliers mathematically by the IQR .
if a value is greater than `75th quantile + (IQR*1.5)`, or
if a value is lesser than `25th quantile - (IQR*1.5)`
IQR is given by the difference between the 75th quantile and 25th quantile

histograms and the `df[col].describe()` method is very useful for detecting outliers.

after detecting outliers, what do we need to do with them?
if the outlier's existence is reasonable, like a player did extremely well in season compared to others, then it might be best to leave it in.
if the outlier itself is unreasonable, like having negative values for a column based on salary, then it would be best to remove them.

outliers skew the data, and change the mean too much. most statistical tests and machine learning models require a normally distributed dataset for calculations


# Relationships in data

## Patterns over time
columns can be converted into datetime data types by using the `parse_dates` parameter while importing or by using `pd.datetime()` method.
`pd.to_datetime(df[["day", "month", "year"]])` this parses all three columns here into a single date time column.
`df[col].dt.year` allows us to extract just the month. we can also extract just the year or day.

line plots are a really good way to visualize date time columns. when there are many x values for a single y value, it aggregates them and plots the mean line with the confidence interval.

## Correlations
correlations measure the linear relationship between two variables using the Pearson correlation method.
`df.corr()` is used to see all the correlation coefficients across all variables.
if its closer to 0 its a weak linear relationship, while if its closer to -1 or 1 its a strong linear relationship

remember that correlations only measure the **linear relationship** between variables. some variables might have a strong relationship in any other order and still have a weak linear relationship. so its always useful to verify the correlation using a scatter plot to see if what the correlation says is true.
one quick way to do this is by using the `sns.pairplot(df, vars=[cols])`. this gives a scatter plot across all variables in the form of a grid with the diagonals being histograms.

passing the correlations to a heatmap is even better because they are color coded and easy to understand.
`df.heatmap(df.corr(), annot=True)`


## Relationships between categorical data
to see relationships between categorical data, we visualize the data.
mostly we use distribution plots to check relationships between categorical data by adding a third variable with hue.
however with hue, distribution plots like histograms aren't good to read.
a much better plot to read with the hue parameter is the KDE plot.
KDE plots use a smoothing algorithm to represent the distribution as a curve. this would sometimes result in values that do not exist in the dataset to appear in the plot.
to change this we set the `cut=0` argument so that the starting and ending values of the distribution to be the minimum and maximum values, respectively.
![[Pasted image 20231124103426.png]]

if we also set the `cumulative=True` argument then it would plot the probability of the values in the x axis to occur/happen. the y axis is the probabilities
![[Pasted image 20231124103450.png]]
this plot can be interpreted as 
**the probability of the marriage duration to be less than or equal to 10 for a professional level education is 30%** 
Scatter plots can also be used to visualize categorical values, by adding the categorical column as the third variable with the hue parameter.

# After Exploration
there are many reasons to perform EDA. like,
1. Detecting patterns
2. generating hypotheses
3. pre-processing for machine learning
but to do any of this, we need the data to fulfil one important condition. the dataset should represent the population.
we can verify this
## Class frequencies
to represent the population, the categories in the data should be proportionate to its actual proportion in the population. if one class's frequency is more than the other and this is not reflected in the population, then the dataset is not accurate and will lead to biased results due to class imbalance.
we can check the count and plot them to see if class imbalance occurs.
we can also verify the proportions using `df[col].value_counts(normalize=True)`
### Cross tabulation
cross tabulation is a way to check class frequencies. it enables us to check frequencies of combination of classes.
we use `pd.crosstab()` for this
`pd.crosstab(index col, values col)`
![[Pasted image 20231124105803.png]]
![[Pasted image 20231124105845.png]]
the numbers here are the counts of flights from destination to source

for example if you know the summary statistic of a particular combination of values in the overall population, we can use crosstab to verify this in our dataset.
`pd.crosstab(index col, column col, values=col for values, aggfunc="function to aggregate over for the values")`
![[Pasted image 20231124110117.png]]
here the median of price when going from one destination to source is shown.

## Generating new features
new columns can help us detect more patterns with correlations and also increase machine learning models' performance.
sometimes we can convert some categorical variables to integer variables(not talking about encoding). like the column wasn't supposed to be a categorical column but a discrete numerical variable. then we can convert the column using `str.replace` and other string methods

we can also use the date time attributes to extract only the date or time and add them as separate columns which also help in finding more correlations because they become numerical now.

we can also create categorical columns from numerical columns by separating them in groups using the quantiles.
first define the labels -> list of the categories
then we define the bins -> the part of the data where each category belong to.
![[Pasted image 20231124111731.png]]
this can be interpreted as "Economy" class is between 0 and the 25th quantile and so on.

then we use 
`df[cat col] = pd.cut(values col to group, labels=labels, bins=bins)`

## Generating Hypotheses
an important result after EDA is to generate hypotheses. 
to verify if the results we got from EDA we need to use hypotheses testing.
hypothesis testing requires to come up with a hypothesis before collecting the data.
and also need a statistical test to perform
when you don't collect the data without the prior hypothesis in mind and start doing EDA to check verify a hypothesis, this might lead to bias because we know about the dataset and just generate a hypothesis that we know is right.
the act of excessive EDA, generating multiple hypotheses and  executing multiple statistical tests is known as data snooping or p-hacking

the act of generating hypotheses can be done with EDA. if we have a hunch about something we can just check it with EDA and if its true we proceed further with statistical tests.

