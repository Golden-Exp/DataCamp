#datacamp #toolkit 
# Functions in python
```python
def square(value):    #here value is the parameter
	'''returns the square of the value''' #docstrings define what the function is doing
	return value*value
print(square(4))  #here 4 is the argument
```
to return multiple values return tuples
3 scopes
1. global
2. local
3. enclosing
4. built in
first python searches in the local scope then the global scope and finally the built in scope
`global a` makes it so that the variable a has global scope and if changed anywhere, the changes would be permanent

## Nested Functions
```python
def outer(x):
	def inner(n):
		return x**n
	return inner(2)
outer(3)
```
returns 3 ** 2
```python
def outer(x):
	def inner(n):
		return n**x
	return inner
square = outer(2)
print(square(3))
```
outer returns a function that can give any value(3) raised to the power of the original argument(2)
note that even after the inner function ends, it still remembers the value for x.
this is known as closure.
like global changes scope from local to global
`nonlocal a` changes the scope from enclosing to local
## Default arguments
```python
def func(a, b, c=1, d=2):
	print(a+b+c+d)
func(1, 2, 3, 4)
func(1,3,4)
func(5,6)
```

## Flexible arguments
sometimes we may not know the number of arguments. here, we use two keywords
```python
#defining a function with many arguments
def func(*args)
	for i in args:
		print(i)
a function that prints all the values given
func(1,25,8,"asdads")
```
note that the `*args` keyword takes all the arguments given into a tuple. and then we can iterate over the tuple and do what we want to do

when we want a keyword and an argument, i.e. a dictionary instead of a tuple we use `**kwargs`
```python
def func(**kwargs)
	for k,v in kwargs.items():
		print(k+":"+v)
func(post="mail", series="breaking bad")
```
here `kwargs` is a dictionary

## Lambda functions
lambda variable: return of function
```python
square = lambda x: x*x -> function to return x power 
square(8)
```
some uses of lambda function are:
### Map function
map(func, seq) -> applies the function to all the objects in the seq and returns the map object.
to see the results of the function over the sequence we use list(map)
now with lambda
```python
nums = [1,2,3,4]
y = map(lambda x: x*x, nums)
print(list(y)) returns [1,4,9,16]
```

### Filter function
filter takes a function and sequence as input just like map, but the function here should check for certain conditions and return true or false. only if its true, the element of the sequence stays. else its removed from the list.
```python
nums = [5, 10, 15, 20]
y = filter(lambda x: x>10, nums)
print(list(y)) returns [15, 20]
```

### Reduce function
```python
from functools import reduce
nums=[1,2,3,5]
z = reduce(lambda x, y: x+y, nums)
print(z)  ->prints 11
```
here first 1 and 2 are given to the lambda function, then the result and the next element, that is 3 and 3 are given, then 6 and 5 so finally 11.
cumulatively applies function over sequence

## Exception handling
```python
def sqrt(a):
	if(a<0):
		raise ValueError("x should be greater than 0")
	try:
		return x**0.5
	except: 
		print("x must be an integer or a float")
```
to only catch ValueErrors can use `except ValueError: `
raise raises an exception
try is like using trial and error
if its without error, the execution goes as usual
if an exception is caught, execution goes to except block
except can also catch specific exceptions

# Iterators
some special types can be iterated over. these types are known as **Iterables**
examples are lists, strings, dictionaries and file connections.
they are objects with the `iter()` method
when used, this method produces an iterator object.
this object gives the next value when `next()` is used on it
```python
word = 'da0'
i = iter(word)
print(next(i))  -> prints d
print(next(i))  -> prints a
print(next(i))  -> prints 0
print(next(i))  -> stop iteration error
i = iter(word)
print(next(*i)) -> unpacks and prints d a 0
```
giving next to a file connection gives the next line.

## Enumerate
gets an iterable object and returns an enumerate object with tuples of 2 elements
the index and the value
```python
a = [10,20,30,40, start= the number u want the indices to start with. its 0 by default]
o = enumerate(a)
print(list(o)) -> returns [(0,10), (1,20), (2,30), (3,40)]
for index, value in enumerate(a):
	print(index, value)
```

## Zip 
gets n number of iterable objects that are of the same length and returns a zip object with tuples of n elements
the two elements corresponding to the same index in both iterable objects
```python
a = "hello"
b = [1,2,3,4,5]
c = zip(a, b)
print(list(c))  -> prints [("h", 1), ("e", 2), ("l", 3), ("l", 4), ("o", 5)]
print(*z) also returns like the above output
for a,b in c:
	print(a+":"+b)
a, b = zip(*z) gives the unzipped version, that is the original itearbles
a dict of a zip returns the unique key value pairs

```
can also be used like `zip(a, b, c)` gives a list of tuples of 3 elements

## Chunks
sometimes when loading a .csv file, it might be too big to load and crashes the program.
in situations like these we load the dataframe in chunks, do the required calculations over that df and bring in the next chunk and repeat.
```python
sum = 0
for chunk in pd.read_csv("as.csv", chunksize=100):
	sum+=chunk['x']
```
thus we got the sum of the column x without loading the whole Dataframe
note that `pd.read_csv("file", shunk_size=10)` is already an iterator

# List Comprehensions
list comprehensions are used to collapse for loops into a single line of code
the syntax is `[(output expression) for iterator in iterable]`
```python
a = [1,2,4,5]
b = [i+1 for i in a]
print(b) -> gives [2,3,5,6]
```
can also be used to collapse nested loops
```python
a = []
for i in range(1,5):
	for j in range(8,10):
		a.append((i,j))

a = [(i,j) for i in range(1,5) for j in range(8,10)]
```

## Conditional comprehensions
```python
a = [num for num in range(11) if num%2==0] -> returns [0,2,4,6,8,10]
a = [num if num%2==0 else 0 for num in range(11)] ->[0,0,2,0,4,0,6,0,8,0,10]
```
we can implement a condition on the iterable or on the output expression

## Dictionary comprehensions
```python
a = {a:-a for a in range(1,5)} returns {1:-1, 2:-2, 3:-3, 4:-4} 
```
can also use comprehensions on dictionaries


# Generators
generators are the same as list comprehensions except we replace the square brackets with `()`
generator objects are iterable objects and can be iterated over.
when list comprehensions are used, a list of that memory is created an stored. however that is not the case for generators. only when u iterate over the generator object is the next element placed in memory.
this approach of having the next value stored only when needed is known as **_lazy evaluation_**
```python
g = (num for num in range(10))
print(next(g)) ->prints 0
print(next(g)) ->prints 1 and so on
for i in g:
	print(i) -> can also be used
```
any function/capability of list comprehensions can also be used with generators
we can use generators when we don't want to have a huge space of memory occupied by a list comprehension and can apply lazy evaluation in this situation
## Generator functions
these type of function returns a generator object. instead of `return` we use `yield`
```python
def func(n):
	for i in range(1, n+1):
		yield i
a = func(5)
for i in a:
	print(i) -> prints 1 to 5
```
basically, whenever we 'yield' an element it kinda gets 'appended' to the generator object that is being returned, if that makes sense
