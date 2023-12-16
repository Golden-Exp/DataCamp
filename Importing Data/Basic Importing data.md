#datacamp #importing
# Flat Files
## Text files
to import text files we use the open function with the mode being 'r'. should always practice while closing the files. to not close manually we use the with function
with is known as context manager
```python
filename = "abc.txt"
file = open(filename, mode="r")
text = file.read()
file.close()
print(text)

or 

filename="abc.txt"
with open(filename, mode="r") as file:
	text=file.read()
	print(text)
```
what we are doing above is that we are binding a variable in the context manager construct. while in this construct, the file is always open.
`file.readline() -> reads one line`

## Flat files
flat files are text files with records in them, without structured relationship. flat files consist of rows and columns where each record is separated by a delimiter.
for example `.csv` where the delimiter is a comma
and `.tsv` where the delimiter is a tab space.
we use `numpy` or `pandas` to import such files.
flat files are different from files that are actually tables like relational databases. sometimes flat files also contain headers, where each record is about what is the column name. we have to check whether our data has a header or not each time we import.

## importing flat files with Numpy
`loadtxt` and `genfromtxt` are used to import numeric flat files.
```python
import numpy as np
filename = "mnist.txt"
data = np.loadtxt(filename, delimiter=",", skiprows=1, 
				  usecols=[0,2], dtype=str)
print(data) -> prints the data as a numpy array
```
`skiprows` skips that number of rows -> used to skip headers
`usecols` are the columns to load([0,2] means only to use the first and third columns)
`dtype` is to mention what datatype the data is, else by default it will be loaded as string. this means numpy clashes when we have data of multiple datatypes.

to deal with different datatypes we use the `genfromtext` function.
```python
data = np.genfromtext("filename", delimiter="\t", name=True, dtype=None)
```
names parameter is true. that means there is a header. when `dtype` is None that means a structured array of 1-Dimension is created. that array has columns and rows. each row values are stored as a tuple.
you can access a column by calling the column's name.
`data[col name]` -> returns all values of the column as an array
another function just like `genfromtext` is `recfromcsv` -> same as that except the default `dtype` value is None

## Pandas
pandas is the most popular and efficient tool to have structured relationship. Dataframes are very useful with a lot of methods to manipulate the data and perform many analysis on them.
***A matrix contains rows and columns. A Data Frame contains variables and observations***

```python
data = pd.read_csv(filename, nrows=no.of rows, 
				   header=None means no header, sep=delimiter,
				   comment="what do lines to ignore start with", 
				   na_values="what values tofill missing values with")
data.head() -> first 5 rows
data.values -> returns the array version of the dataframe
```

# Other file types
## Pickle and excel
when you want to convert lists and dictionaries to files we use the pickle files to convert them into a sequence of bytes. this is known as pickling. this is a file type native to python
```python
import pickle
with open("file.pkl", mode="rb") as file:
	data = pickle.load(file)
print(data)
```

when you want excel sheets to be imported as Data Frames, we use pandas `ExcelFile`
```python
import pandas as pd
data = pd.ExcelFile("file.xlsx")
print(data.sheet_names)
df1 = data.parse(sheet_name, skiprows, names=[list of name of columns],
				 usecols=[list of indices of columns to use])
df1 = data.parse(index, skiprows, names=[list of name of columns],
				 usecols=[list of indices of columns to use])
```
when using `ExcelFile` the sheets are the ones that get imported. to get the sheet names we use the attribute `sheetnames` then we import a sheet as a data frame using, `data.parse(sheet name as string)` or by passing the index of the sheet name

## SAS and Stata
SAS - Statistical Analysis System -> used in business and biostatistics
Stata = Statistics + Data -> used in academic social science and research like economics
`.sas7bdat`(dataset files) and `.sas7bcat`(catalog files)
```python
import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT("filename.sas7bdat") as file:
	df_sas = file.to_data_frame()
print(df_sas.head())

df_stata = pd.read_stata("filename.dta")
```

## HDF5 files
Hierarchical Data Format version 5 can store a lot of numerical data scaling up to exabytes.
```python
import h5py
filename = "filename.hdf5"
data = h5py.File(filename, 'r')
for key in data.keys():
	print(key)   lets say there's a key named file
for key in data["file"].keys()
	print(key)  lets say there's a key named files in file
print(np.array(data["file"]["files"])) -> convert to array
```
you can imagine that the HDF5 files are like directories and each key is like a folder going inside gives you other folders. and printing those folders will probably give you a summary of the folder.
you can extract the data inside keys by converting it into a `numpy` array.

## MATLAB files
Matrix Laboratory or MATLAB is a popular software used as a numerical computing environment among Engineering and Science.
```python
import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat(filename)
print(mat[key name])
```
MATLAB files contain workspaces with the variables and their values. so when importing a MATLAB file, we get a dictionary with the keys being the variable names and the values being the values of the variables.

# Relational Databases in Python
## The relational database system
relational databases are types of databases to store tables that are linked to each other through the primary keys of the tables. the tables are entities and each row of a table is an instance of that entity.
each table should have a primary key in them, that is a column that has only unique values and can be used to identify rows. the primary key of one table can be a regular column in another table. with this we can link both tables by looking up the values of the regular column of one table in the table which the same column as its primary key.
some languages that have relational databases are PostgreSQL and MySQL
```python
from sqlalchemy import create_engine
engine = create_engine("sqlite:///Northwind.sqlite")
print(engine.table_names()) 
#prints all the names of the tables in the database
```
`create_engine("type of database:///name of database")`

## Querying in Relational Databases

![[Pasted image 20231129114036.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Importing%20Data/Pasted%20image%2020231129114036.png/?raw=true)

generally to query is to get data from the database.
```python
from sqlalchemy import create_engine
engine = create_engine("sqlite:///Northwinds.sqlite")
con = engine.connect()   #establishing connection

#exceuting SQL query and storing results
rs = con.execute("SELECT * FROM Orders")  

#fetching all rows and converting to dataframe
df = pd.DataFrame(rs.fetchall())

#setting the column names as the keys of the result of exexcution
df.columns = rs.keys()
con.close()

or 

with engine.connect() as con:
	rs = con.execute("SELECT 'column names' FROM table WHERE col > 6 ORDER BY col")
	df = pd.DataFrame(rs.fetchmany(5)) #fetching 5 rows
	df.columns = rs.keys()
```

we can also do all this in a single line with pandas
```python
df = pd.read_sql_query("SELECT 'col names' FROM TABLE WHERE COL > 6", engine)
```
`df = pd.read_sql_query("SQL QUERY", engine_name)`
to join tables the SQL query is 
`SELECT COL NAMES FROM TABLE1 INNER JOIN TABLE2 ON TABLE1.COL = TABLE2.COL`
here we join table 1 and table 2 using an inner join where the values of col in table 1 is equal to values of col in table 2
