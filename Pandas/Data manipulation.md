#datacamp #pandas
to create a Data frame we use a list of dictionaries where the key of each dictionary is the column name and the value is the value of that particular row. each element of the list corresponds to a row. 
we can also use a dictionary with the keys being the column name and the values being a list of the whole column
intro
```python
df.head(n) -> returns the first n rows of data
df.describe() -> returns summary statistics 
df.info() -> returns info abt the data
df.shape -> returns (rows, columns)
df.values -> returns the df in a 2-d array
df.columns -> returns the column names
df.index -> returns the row names or labels
```
indexing and slicing
```python
df.sort_values(column or list of column names, ascending=list of [True or False]) -> sorts the df acc to that column
df[column name or list of column name] -> subset
df[bool values] -> conditional filtering
df[(condition) & (condition)] -> logical operators
df['col'].unique() -> gives a list of all unique values
```
summary statistics
```python
df['column'].mean()
df['column'].max() or .min()
df['column'].quantile(0.75)
df['column'].agg([list of functions to have statistics]) or df.agg({column:func})
df['column'].cumsum() or .cummax() or .cummin or .cumprod()
```
categorical values
```python
df.drop_duplicates(subset=[column names])
df['column'].value_counts(sort=True, normalize=True, 
						  dropna=False) -> normalize returns proportions
									    -> dropna=False includes NaN values also
```                                     
grouped statistics
```python
df.groupby([list of column], as_index=True or False)["specific column u want to calculate statistics on"].func

df.pivot_table(values="the column to calc statistics on", index="the column to group by", columns="the second column to group by if u want to", fill_value=fills nan to whatever u want, margins=True sets the final row and column to mean values, aggfunc=list of functions to aggregate)
```
indexing and slicing
```python
df.set_index([list of column names])
df.reset_index(drop)->resets index if drop is true the index column is dropped
df.sort_index(level=list of indices, ascending=list of ascending values)
df.loc[rowname:rowname]
if two index levels
df.loc[(index1, index2):(index1, index2)]
df.loc[row:row, column:column]
df.loc[cond for filtering the rows, columns] 
#example 
user.loc[user["name"] == "sarah","name"] = "hannah"
df.iloc[row_index:row_index, column_index:column_index]
loc includes the last element of slice
iloc doesnt include the last element
```
visualizing dataframes
```python
df["column"].hist(alpha=0 to 1) -> distribution of a numeric variable x - range, y-no.of in the range
df.plot(kind="bar", title="") -> used to see relationship b/w cat amd numerical variable
df.plot(kind="line", x="", y="", rot=angle) -> visualizing changes in numeric variables over time
df.plot(kind="scatter") ->great for seeing relationship b/w 2 numeric variables
#we can plot two graphs and show to see layered graphs
plt.legend([list of graphs to label])
```
missing values
```python
df.isna().any() -> returns if any value is missing 
df.isna().sum() -> returns total no. of missing values in each column -> plot bar plots with this for visualization
df.dropna() -> drops na rows
df.fillna(0) -> fills with 0
```
csv files
```python
pd.read_csv(path, parse_dates=[list of cols for date], index_col="index col")
df.to_csv()
```
selecting data using query method
```python
df.query('col > value and col < value') -> smth like this use single quotes
df.query('col == "value" and condition') -> for strings double quotes are used
```
melt method for transforming long df to wide df
![[Pasted image 20231030003002.png]]
```python
df.melt(id_vars=[columns that u dont want to change], value_vars=[columns that u want under the column variable], var_name=variable name, value_name=value_name)
```
![[Pasted image 20231030003209.png]]

```python
df[new col] = df[col].map(dictionary)
will map the keys in col with its respective value in the dictionary
```
