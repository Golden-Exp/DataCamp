#datacamp #data_analysis 
# Common Data Problems
## Datatype constraints
cleaning data should be the first thing to do with the data you received before starting anything else.
![[Pasted image 20231127095948.png]]

most commonly mistook datatypes are categories and numeric. sometimes, an integer or a float is stored as an object datatype in pandas, due to some extra string character. to perform the required summary statistics, you have to remove the extra string and convert the column to numerical.
also, sometimes, categorical columns are label-encoded and has numbers in them. they are, by default, stored as integer. we don't want this, so we convert it to category

```python
assert 1==1
returns nothing
```
`assert condition` will give an error if the condition is false, else it does nothing

## Out-of-range data
sometimes, a column that is supposed to have data that should be within a particular range, has data that doesn't come within the range. like ratings(0-5) and dates greater than today's date but the column can't have future dates.
to deal with out-of-range data we can 
1. we can drop the data(rule of thumb, only drop data when they constitute a small portion)
2. set custom minimums and maximums
3. treat the data as missing and impute them
4. assign a custom value if you know about the data well enough

```python
df = df.drop(df[df[col] > range].index) -> #drops those indices' columns
df.loc[df[col] > range, 'col'] = max value -> #assigning custom maximums or minimums

dt.date.today() -> returns the current date
```

## Duplicated values
duplicated rows are rows that have the same data except for one or two columns, which could have been man made errors. the main reason for duplicated data to exist is probably because of joining and merging data from various sources.

to check and remove duplicates,
```python
df.duplicated() -> returns a boolean series of whether it is a duplicated row
has parameters
dupes = df.duplicated(subset=[list of columns to check for duplicates], keep="last")
df[dupes].sort_values([subset]) -> see all duplicated rows
```
subset makes it so that only if the values of the given column are the same across 2 rows, it is deemed as duplicates.
keep has 3 possible values
1. first -> keep only the first occurrence
2. last -> keep only the last occurrence
3. False -> keep all occurrences
usually false is used to see how the row varies across other columns

to remove duplicates
```python
df.drop_duplicates(subset=[], keep)
```
usually `drop_duplicates` is used to delete only rows that are complete copies. so the subset column is usually not used.

for rows with duplicates across a subset of columns, we usually calculate a summary statistic of the differing row/rows and add that as the true value.
we do this with `groupby`
```python
df.groupby(by=the subset we used for duplicates). \ 
						agg(dictionary to do what summary statistic for what column).\ 
						reset_index()
```
![[Pasted image 20231127104007.png]]

# Text and Categorical problems
categorical columns might sometimes have inconsistent values, like categories that are not even valid.
this can be resolved by either dropping the row or by remapping the value to its correct category
## Membership constraints
membership constraints are when a category exists that is not part of all possible categories.
for example a Z+ blood type.

first we need to filter the inconsistent value. to do this we use an anti join
the left anti join takes in two tables and gives out values **that are only in left table and not in right table.**
![[Pasted image 20231127105902.png]]

we can use our incorrect column as one table and a column of all possible correct categories as another table
by using a left anti join here we get values that are only present in our column and are not present in the "all possible categories" column. meaning, we have successfully filtered out the required data

```python
unique_values = set(df["our_col"])  ->used to get unique values of all categories
inconsistent = unique_values.difference(df["all_possible_categories"]) #left anti join
rows = df["our_col"].isin(inconsistent)
df[rows] -> inconsistent rows
df[~rows] -> filtered out data
```
## Other Problems
### Variable Inconsistency
this occurs when categories that are supposed to be some category but are different due to spelling mistakes, trailing or leading whitespaces and cases(lower and upper)
this can be resolved using string methods
`df[col].str.upper() or .strip() or .replace()`

### Creating new categories
we can create new categories based on range of a particular numeric column using 2 ways
`qcut` and `cut`
```python
names=["ranges' names"]
df["new cat col"] = pd.qcut(df[numeric column], q=no. of ranges, labels=names)
```
this cuts the data based on the value of q and sets the ranges the names based on order.

to set the names based on the ranges we use cut
```python
ranges = [0, 100, 200, np.inf] -> so range from 0-100, 100-200, 200-inf
df["new cat col"] = pd.cut(df[numeric col], bins=ranges, labels=names)
```
### Collapsing data into new categories
this is to be done when you have many categories and want to collapse some categories and make new ones.
this is done by passing a dictionary to the replace method.
```python
mapping = {dictionary with keys as old categories and values as the new category you want}
df[col] = df[col].replace(mapping)
```
this maps the old values with the new values.

[!NOTE]
> Don't forget to convert the columns back to categorical datatype

## Text Data problems
text data or regular string data is stored as object datatype in pandas. 
these type of data can have some errors just like ones in categories like spelling mistakes and such.
we can use `.str.method()` to use the string methods and resolve these issues.
like `.str.len()` and `.replace`
if the data is too messy, then you can't type in individual conditions like these for all rows. instead we use regular expressions
regular expressions are used to find patterns in data
for example
```python
df[col] = df[col].replace(r'\D+', '')
```
here we are saying that anything other than a digit should be replaced with an empty string.


# Advanced Data Problems

## Unit Uniformity
it is necessary for all the data in a particular column to be of the same unit. for example, we can't have temperature measured in both Celsius and Fahrenheit. it should only be of one unit. same with dates. they should be of only one format.
in case of the units being different, we first spot which data has different units using visualizations and such.
then we replace those values with the converted values.

in case of the dates' format being inconsistent, we use `pd.to_datetime` to coerce all the dates into the same format. and formats that are wrong or cannot be recognized by pandas are deemed as `NaN` which is good for us too
```python
df[col] = pd.to_datetime(df[col], 
						infer_datetime_format=True, #convert all to one format
						errors="coerce")   #unrecognizable formats deemed as missing
```
to change the format we can use
`df[col].dt.strftime("%d-%m-%y")`

#### Ambiguous data
however sometimes, the format might be correct but we don't know what format it is.
for example, "2019-03-08" is this march 8th 2019 or august 3rd 2019?
to infer this we should know about our data and try to see what formats other data previously had.
if not sure, we can always convert this to a missing value.

## Cross Field validation
this is to ensure data integrity(that is to know that our data is valid), by cross verifying the values of columns using multiple columns.
for example, if the date of birth and age is given, we can check if the age is correct by subtracting the year of the birthday column from today's year. Then we can filter out values that don't match as inconsistencies.
```python
dates = dt.date.today().year - df[col].dt.year
cond = df["age"] == dates
consistent = df[cond]
inconsistent = df[~cond]
#or checking if the sum of some columns equal the total column
df[[list of columns]].sum(axis=1)
```
with inconsistencies
we can either, 
1. drop them
2. deem them as missing and impute them (idk why I use deem so much)
3. apply something that you know from that domain of the dataset to find the correct values
these can apply for different data. point is you need to know about your data for doing this.

## Completeness
missing data can caused due to either technical errors or human errors.
there are 3 types of missing data
1. MCAR - Missing Completely At Random - no relationship between missing data and other data
2. MAR - Missing At Random - there is a relationship between missing data and other ___observed___ data
3. MNAR - Missing Not At Random - there is a relationship between missing data and ___unobserved___ data
for example, if the missing values of a sensor all correspond to low temperatures that day, this is known as MAR.
if we didn't measure the temperatures that day and didn't know about this, that is known as MNAR.

we can use the `missingno` package to visualize the missing values.
```python
import missingno as msno
import matplotlib.pyplot as plt
msno.matrix(df)
plt.show()
```
![[Pasted image 20231127134519.png]]

this is basically the table visualized like a grid which is pretty cool
to find these relationships separate into missing rows and non-missing rows and apply the `describe` method on both and spot the difference
to check the relationship we can try plotting differently
![[Pasted image 20231127134731.png]]
with this type of visualization we can see that all the missing values come for lower temperatures.

to deal with missing values we either, 
1. impute with summary statistic
2. delete
![[Pasted image 20231127134858.png]]
```python
df.dropna(subset=[list of columns to check and drop])
df.fillna({col:value to be filled with})
```

# Record Linkage
## String Similarity
this is the concept of checking if two strings are similar. there is a parameter known as minimum edit distance. this calculates the minimum number of turns required to convert one word to another. the lesser the MED, the more similar the words are to each other.
![[Pasted image 20231127144729.png]]
there are 4 operations
1. insertion
2. deletion
3. substitution
4. transposing
there are many ways and algorithm to find the MED.
MED can be used to find whether a typo belongs to a particular category or not. so it is useful when the data is hand filled and is expected to have many errors. we can't manually remap all of them.

we'll be using the `Levenshtein` method through the `thefuzz` package
```python
from thefuzz import fuzz
fuzz.WRatio("Reading", "Reeding")
```
`WRatio` returns a score from 0 to 100 where 0 means totally different and 100 means a perfect match. this is different from MED.
we can also select the most similar strings from a sequence
```python
from thefuzz import process
string = "hell to you"
choices = ["hello", "hell yeah broo", "hey"]
process.extract(string, choices, limit=1)
```
limit keyword is to set the number of similar words you want
extract returns a list of tuples, with the number of tuples being the limit.
each tuple has the string match, `WRatio`, and the index.
this can be implemented for collapsing many typos, instead of replacing them
```python
for cat in categories:  #looping through the correct categories
	matches = process.extract(cat, df[typo col], limit=df.shape[0]) 
	for i in matches: #looping through all the matches
		if i[1] >= 80:    #if the WRatio is greater than 80
			df.loc[df[col] == i[0], 'typo col'] = cat  #changing the value
```


## Generating Pairs
some tables can't be joined with a join because they don't have a unique identifier common to both tables. in that case we use record linkage.
Record linkage is the act of linking data from two different sources regarding the same entity.
![[Pasted image 20231127155808.png]]

```python
import recordlinkage
indexer = recordlinkage.Index()
indexer.block('common col')
pairs = indexer.index(df1, df2)
comparator = recordlinkage.Compare()
comparator.exact('col in df1', 'col in df2', label="col name")
comparator.string('col in df1', 'col in df2', label="col name",
				  threshold=from 0 to 1)
matches = comparator.compute(pairs, df1, df2)
```
so, what the fk does the above code mean?
`indexer` is index object of `RecordLinkage` it is used to generate pairs based on the `block` column given.
to block a column means to generate pairs of rows where the blocked column is the same across both rows.
so all the pairs generated will have two rows, where both have the same value for the blocked column.

the pairs are generated with the `index` method.
if you print the pairs, it will be a `MultiIndex` object that is a 2-D array with the indices of the rows.
now to see matches, we use the `Compare` object.
it has 2 methods. 
1. `exact` -> to see which pairs' given column match exactly
2. `string` -> to see which pairs' given column match using string similarity threshold is the minimum `WRatio`
then the table of all these matches is done using the `compute`
![[Pasted image 20231127165326.png]]
![[Pasted image 20231127165559.png]]

first is the index of `df1` and then index of `df2` that has matching `block` columns then all the matches.
for the above table, its a complete match if, the date of birth and the surname are already matches(i.e. 1) or if the state and the address columns are already a match
so, to filter out the same columns, both `dob` and `surname` should be 1 or both `state` and `address` should be 1
this can be done like
![[Pasted image 20231127170202.png]]

## Linking Data Frames
now that we have got the matches and scored them, we are gonna link the two data frames after removing the duplicates.
duplicates here mean that rows appearing in both `df1` and `df2`. that is what we filtered for at the end above.
now after filtering, all we have to do is to subset `df2` so that it only has rows that are not **duplicates** and then we **append** it to `df1`.
```python
duplicate_rows = matches.index.get_level_values(1)  
#gives the 1st level index, i.e the index of df2
df2_new = df2[df2.index.isin(duplicate_rows)]
df2_dupes = df2[~df2.index.isin(duplicate_rows)]

full_df = df1.append(df2_new)
```
thus we joined the two Data Frames without a unique primary key. 
