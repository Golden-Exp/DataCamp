#datacamp #seaborn
seaborn is a library built on matplotlib and pandas
it has the superior flexibility of matplotlib and lowers the complexity of matplotlib
since its built on pandas very easy to use with pandas. refer [[Basics of  Seaborn]] before this.

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x=x, y=y)
sns.countplot(y or x=list of categories)
```

Scatterplot takes in two variables and show the relationship between them
Count plot takes in a list of categories and show the number of values in each category

### Plots with Data Frames
```python
sns.countplot(x="name of column", data=df)
plt.show()
```

this only works with tidy data.
that means all rows have the same type of value and not be different. if untidy, the data should first be transformed to tidy then used.

### Third variable with colors
just like color in matplotlib we can use the hue parameter to differentiate with colors

```python
sns.scatterplot(x=x, y=y, data=df, hue="another col name", hue_order=[list of legend], palette={dict of legend to its color})
```
![[Pasted image 20231114100358.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231114100358.png/?raw=true)

## Relplots
relationship plots are plots that show the relationship between two variables.
they can be either line plots or scatter plots
seaborn's `relplot()` is better that scatter plot because it has the option of introducing a third variable as subplots
```python
sns.relplot(x="col name", y="col name", data=df, kind="scatter or line", 
			row="col name", col="col name", row_order=[list of elements],
			col_wrap = no.of columns)
```

### Customizations in Scatter plot
#### Size
```python
sns.relplot(x="col name", y="col name", data=df, kind="scatter", size="sequence or col with categories", hue="col name")
```
mentioning size allows us to differentiate different groups with the size of the points
![[Pasted image 20231115103813.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231115103813.png/?raw=true)

#### Style
```python
sns.relplot(x="col name", y="col name", data=df, kind="scatter", hue="col name", style="sequence or col with categories")
```
each category will have a different style of point
![[Pasted image 20231115103847.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231115103847.png/?raw=true)

#### Alpha
```python
sns.relplot(x="col name", y="col name", data=df, kind="scatter", hue="col name", style="sequence or col with categories", alpha=a number from 0 to 1)
```
alpha is the opacity of the points
0 - transparent
1 - opaque
![[Pasted image 20231115103910.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231115103910.png/?raw=true)


### Customizations in Line plots
line plots are used when we need to track something over time. like stock over time or the stats of a player over time.
```python
sns.relplot(x="name of col", y="name of col", data=df, kind="line", 
				style="to separate categories", hue="to differentiate with colors",
				 markers=True or False, dashes=True or False)
```
style creates different lines for different categories.
hue gives different colors for different lines.
markers mark the points
dashes false means the other lines don't have different dash styles

![[Pasted image 20231115231321.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231115231321.png/?raw=true)

```python
sns.relplot(x="col", y="col", data=df, kind="line", ci="sd or None")
```
when there are multiple observations per 1 x value seaborn calculates the summary statistic of those points and display them. by default its the mean. seaborn also 
***calculates the confidence interval for that mean***. 
this is represented by the shaded region. 
confidence interval indicates the uncertainty of our estimate.
confidence interval of the mean signify that about 95% of times the mean of y at x comes under this shaded region.
to get the spread of the dataset we use `cd="sd"`. this gives the std and so we can see the whole spread

![[Pasted image 20231115231518.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231115231518.png/?raw=true)

## Categorical plots
categorical plots, i.e. plots that represent categories.
bar plots , count plots , box plots and point plots
just like `relplots`, `catplots` can also arrange graphs in rows and columns.
### Count plots
count plots are plots that show the `value_counts` of each category.
```python
sns.catplot(x="col", data=df, kind="count", order=[order of categories])
```
![[Pasted image 20231116095600.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231116095600.png/?raw=true)

### Bar plots
bar plots are used to show the average amount of a numeric variable associated to each category.
for example, the average **bill** for each **day**. it also, by default shows the confidence interval, which shows that about 95% of the time, for the population, the mean of the bill is going to be around those points.
when the numerical/y part is True/False, bar charts show the percentage.
```python
sns.catplot(x="col", y="col", data=df, kind="bar", ci="None if u don't want")
```
![[Pasted image 20231116095852.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231116095852.png/?raw=true)

### Box plots
box plots show the distribution of the given numeric variable across the different categories. it is represented by its whiskers and the rectangle. the rectangle is the 25th and 75th percentile, while the median is the red line in between. the diamonds are outliers.
by default, the whiskers are 1.5 times the IQR range. if u want to change that set 
`whis=2.0` -> to make the whiskers 2 times the IQR range.
`whis=[5, 95]`  -> to set the whiskers as the 5th percentile and 95th percentile.
the `sym` parameter allows you to omit outliers if you give it an empty string
```python
sns.catplot(x="col", y="col", data=df, kind="box", sym="", whis=2.0)
```
![[Pasted image 20231116102702.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231116102702.png/?raw=true)

### Point plots
these plots show the mean of each category connected to each other by a line. it also shows the confidence interval just like a bar plot and a line plot. however, 
line plots compare between two numeric quantities while this is between a category and a numeric
point plots are better when we use a third variable with hue, because, its easier to comprehend them in a point plot.
```python
sns.catplot(x="col", y="col", data=df, kind="point", 
			estimator=mean or median, capsize=0.3, ci="None if u want", 
			dodge=True means the lines dont overlap, 
			join=False means the lines arent available)
```

estimator is used so we can calculate some other summary statistic instead of mean.
we use medians when we have more outliers. means get affected more by outliers than the median.
capsize is to add caps on confidence intervals.

### Customizations
seaborn has 5 preset styles to choose from
"white", "dark", "whitegrid", "darkgrid", "ticks"
white is the default
```python
sns.set_style("whitegrid")
```
can also change basic color palette of plot with
```python
sns.set_palette()
sns.set(font_scale=1.5) -> sets fontsize
```
it can be a palette from seaborn itself or a custom palette
palettes from seaborn consist of 2 colors merging or a single color from light to dark
`"RdBu", "PrGn"` are examples
if it ends with an `_r` -> reverse.
"Greys" means just grey from light to dark
or have a list of colors of hex codes and give it to `set_palette()`

`sns.set_context()` helps us change the scale of the plot
"paper", "talk", "notebook", "poster" -> few examples

### Titles for plots
seaborn plots create 2 types of objects
`FacetGrid` and `AxesSubplot` 
`FacetGrid` is created when subplot optimal plots are used, like `relplot` and `catplot`
this type of object supports rows and columns and consists of multiple `AxesSubplots`.
`AxesSubplot` is created when other plots are used like `scatterplot` 
```python
g = sns.catplot(x="", y="", data=df, kind="box")
g.fig.suptitle("title of plot", y="the height; by default it is 1")
plt.subplots_adjust(top=0.90) -> sets starting height of plot at 90% of full height. this is used to separate the title from the plot
```

note that it is `suptitle` and not subtitle

for `AxesSubplots` 
```python
g = sns.scatterplot(x="", y="", data=df)
g.set_title("title", y=height)
```
when subplots are used, we use
`g.set_titles("this is {col_name}")` to set the title for each subplot, `col_name` gives the name of the column.

to set x and y labels
```python
g.set(xlabel="label", y="label")
or 
g.set_axis_labels(xlabel, ylabel)
```

to customize ticks use
```python
plt.xticks(rotation=90)
plt.yticks(rotation-45)
```

[seaborn cheat sheet](https://images.datacamp.com/image/upload/v1676302629/Marketing/Blog/Seaborn_Cheat_Sheet.pdf)
