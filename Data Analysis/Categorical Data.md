#datacamp #data_analysis
# Pandas Categories
data that contain finite number of categories.
also known as qualitative data
has 2 types
1. ordinal
2. nominal
ordinal datatypes are datatypes that have a specific order. examples are ratings
nominal categories don't have an order. like marital status
can use `value_counts()` to see the frequencies.
can also use `normalize=True` to see the proportions of categories in the data
when a datatype of a column is object then that is most probably a categorical column.

there is a separate datatype in pandas known as category. By default, categorical columns are stored as object datatypes. to change this we use `astype` keyword to change it to category
`df[col].astype("category")`
the need for converting object datatypes to categorical is that categorical columns consume less space.
`df.nbytes` -> gives the space consumed

we can also convert using 
`df[obj col] = pd.Categorical(df[obj col], categories=[categories], ordered=True)`
this also registers the order of the categories if needed. here the index is used to see the order. 
so a higher index means a higher rank
when importing 
```python
dictionary = {"col name":"category"}
df = pd.read_csv("csvfile.csv", dtype=dictionary)
```
parses that column to category

grouping by categorical datatypes is a good way to check a summary statistic across categories.
`df.groupby([categorical columns]).size()` -> will give how much fall under each category.
if multiple categories given, then it will return the number of all elements that fall under all combinations of categories

# Methods on categorical columns

## New categories
most of the methods have keywords `new_categories` and `ordered`
```python 
df[cat col] = df[cat col].cat.set_categories(new_categories=[new categories])
df[cat col] = df[cat col].cat.add_categories(new_categories=[newly added categories])
df[cat col] = df[cat col].cat.remove_categories(removals=[categories to remove])
```
`set_categories` sets categories which are in the `new_categories` list. other categories are replaced with `NaN` values
`add_categories` adds new categories but doesn't update them to data.
`remove_categories` removes the categories in removals and updates them to `NaN`

## Updating categories
```python
df[cat col] = df[cat col].cat.rename_categories(dictionary with old and new names)
df[cat col] = df[cat col].cat.rename_categories(lambda x: x.title()) can also give lambda functions
```

the `rename_categories` method can't collapse multiple categories into the same category.
instead we use,
```python
df[cat col] = df[cat col].replace(dictionary with old cat to new cat)
df[cat col] = df[cat col].astype("category")
```

the replace method replaces the key to the value in the dictionary, so it can collapse multiple values into a single value.
replace is an object method and so once used, the column becomes object datatype.

## Reordering categories
```python
df[cat col] = df[cat col].cat.reorder_categories(new_categories=[ordered categories],
												 ordered = True)
```
this order will be used everywhere we use this particular column.

## Cleaning and accessing categories
sometimes the categories in the column needs to be cleaned.
this commonly occurs with spelling mistakes, whitespaces and other errors.
by using string methods and the replace method, we can resolve these issues and clean the column
do not forget to change the column back to category with `astype`
we can also use the `str` object to access elements with methods like `contains`, `isin`,  etc.
![[Pasted image 20231126142453.png]]
![[Pasted image 20231126142419.png]]

# Visualizing Categorical data
we mainly use the Seaborn's `catplot` to plot categorical plots. it has many options for what kind of plot you want to plot.

## Box plots
box plots show the distribution of a numeric variable. with the x parameter being a categorical variable and the y parameter being a numeric quantity, box plots can be used to easily see how a numeric quantity varies with categories. 
it is also useful to use the `whitegrid` style to denote outliers
![[Pasted image 20231126181704.png]]

## Bar Plots
bar plots are used to quickly check a summary statistic across different categories. the hue parameter is also used to check with a third variable.
![[Pasted image 20231126184516.png]]

## Point plots and count plots
point plots are the same as bar plots except they have points pointing to the mean with the confidence intervals. all the points are connected. point plots are in general better than bar plots.
count plots are bars the measure the count of number of elements in categories.
![[Pasted image 20231126194438.png]]![[Pasted image 20231126194504.png]]

## Additional plots
the `catplot` method is mainly preferred for its way of creating subplots with rows and columns.
this can be used to emphasis a particular category and plotting them across rows and columns instead of adding them as a hue parameter.

![[Pasted image 20231126195929.png]]
![[Pasted image 20231126200006.png]]

# Pitfalls and encoding
## Pitfalls
some categorical pitfalls are that
1. the savings in storage depends on the number of categories(if less categories more space saved)
2. the `.str` object converts the column to object. so we have to convert it to category again
3. the `.apply` also converts it to object
4. the different methods in `.cat` handle missing values differently
5. `numpy` functions don't work with categorical columns

so some precautionary things to do are to always check the `dtype` of the column.
also if you want to use a `numpy` function, convert to `int` use the function and convert it back to category

## Label Encoding
label encoding is the process of encoding categories as integers from 0 to n-1
missing values are generally encode with a -1
```python
df["cat col codes"] = df[cat col].cat.codes
```
this gives all the codes
the codes are assigned in alphabetical order
now for a code book we use the zip function
```python
codes = df[cat col].cat.codes
cats = df[cat col]
code_book = dict(zip(codes, cat))
returns a dictionary with unique key value pairs of codes and categories.

map used to check values
df[cat col codes].map(code_book)->returns all the categories corresponding to the code
```
 Boolean encoding used for some conditions to have separate Boolean codes to check if the condition is true or not
 ```python
 import numpy as np
 df["bool codes"] = np.where(df[cat col].str.contains("hello", regex=False), 1, 0)
```
`np.where(condition, if true, if false)`

## One-hot encoding
one-hot encoding is the process of Boolean encoding for all categories.
that is if there is a column for marital status, one-hot encoding means to create two columns `is_married` and `is_single` both the columns are Boolean encoding of the two categories.
this is useful because, when we use label encoding, the machine learning model might give a higher weight to one that has a higher code, that is alphabetically. we don't want this.
one-hot encoding ensures that this doesn't happen.

to do one-hot encoding in pandas, we use `pd.get_dummies()`. the resulting columns are known as dummy columns
```python
df_new = pd.get_dummies(df, columns=[colums to get dummy variables for], 
						prefix="prefix before each col like 'is'")
```

some things to note about `pd.get_dummies()` is that any column with object datatype gets a dummy column. this means that there would be many columns in the end. many columns can end up overfitting the model, so make sure you use the columns parameter before using
another thing is that, missing values don't get their own column. this is however fine, since we can just see that if a row is 0 on every dummy column, its a missing value of that column
