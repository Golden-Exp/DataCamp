#datacamp #pandas
to join two tables on a common column we use the merge method
```python
df.merge(df2, on=col_name) -> this gives the columns of df first then df2. if two cols have same name they are given sufixxes x and y
if the common column has diff names in both tables, right_on and left_on keywords used
df.merge(df2, on=col name, suffixes=("_a, _b")) -> can give own suffixes too
```
## Mutating joins
### inner join
.merge gives the result of inner join
![[Pasted image 20231029185251.png]]

if the common column is unique and has only one row for one value of that column - > one to one relationship
![[Pasted image 20231029190448.png]]

if there are multiple rows for 1 value of the common column in the right table it is a one to many relationship
![[Pasted image 20231029190313.png]]

### left join
```python
left.merge(right, on=col, how="left")
```
![[Pasted image 20231029193238.png]]

left join one to one will have the same no. of rows as that of the left table
left join one to many will have either the same or greater no. of  rows as that of the left table
### right join
```python
left.merge(right, how="right")
```
![[Pasted image 20231029193903.png]]
### outer join
```python
left.merge(right, how="outer")
```
![[Pasted image 20231029211308.png]]

### self join
will be useful when dealing with
	-Hierarchical relationships 
	-Sequential relationships 
	-Graph data
basically when u deal with data that has links to other data in the same table kinda like linked lists
you can also use this to create new links with existing ones.
```python
df.merge(df, how=inner, suffixes=[]) -> giving inner here gives only the column that has links
```

merging on index
```python
df.merge(df2, on="index")
if index name diff on both df
df.merge(df2, left_on=ind1, right_on=ind2, left_index=True, right_index=True)

```

## Filtering joins
Filter observations from table based on whether or not they match an observation in another table

### semi join
inner join but only left table included 
and also no duplicates
![[Pasted image 20231029224534.png]]
we first do an inner join then we check for the values of the common column that are present in the merged df. subset and that is semi join
```python
genres_tracks = genres.merge(top_tracks, on='gid') 
top_genres = genres[genres['gid'].isin(genres_tracks['gid'])] 
print(top_genres.head())
```
### anti join
the values of the left table that were not included in inner join
![[Pasted image 20231029224913.png]]
we first perform a left join. then we locate the rows that are only available in the left df. then subset the common column of that and subset the left table using that common col list.
```python
genres_tracks = genres.merge(top_tracks, on='gid' , how='left' , indicator=True) 
#the indicator parameter adds an extra column _merge that says whether the row is part of both tables or only the left
gid_list = genres_tracks.loc[genres_tracks['_merge'] == 'left_only' , 'gid'] #getting the list of common column only available in the left table.
non_top_genres = genres[genres['gid'].isin(gid_list)] 
print(non_top_genres.head()
```

## concatenating tables 
to concatenate two tables use pd.concat()
```python
pd.concat([list of dfs to concat], join=type of join, ignore_index=True or False, keys=[list of keys for each df], sort=True or False, axis=0(vertical) or 1(horizontal)) -> if extra column it depends on the typa join whether the column will be included or excluded.
if inner no. if outer(default) yes
```

## verifying integrity
checking for duplicates while merging/concatenating

```python
df.merge(df2, validate='one_to_one') #checks if the merge is one to one if not error
likewise
'one_to_one'
'one_to_many'
'many_to_one'
'many_to_many'
pd.concat([], verify_integrity=True) -> checks if the index and only if the the index of two rows are same. if same error
```

## Ordered Merges
### merge_ordered()
same as merge except this time its ordered and also has some other parameters
![[Pasted image 20231029233612.png]]
*note:* default is outer for merge_ordered()
```python
pd.merge_ordered(df1, df2, on="", suffixes=(), fill_method='ffil') -> fills missing values with previous values
```

### merge_asof()
same as merge_ordered's left join except the matches aren't exact. it matches on the column closest to the value of the left table
```python
pd.merge_ordered(df1, df2, on="", suffixes=(), dirction="")
direction:
nearest -> nearest value
forward -> nearest greater than
backward(default) -> nearest lesser than
```

