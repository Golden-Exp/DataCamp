#datacamp #toolkit
# Best Practices
## Docstrings
docstrings describe what the function is. We use triple quotes to write docstrings in python.
there are 2 common formats for docstrings google and Numpy docs
```python
def func(arg1, arg2=43):
"""  Description of what the function does

	Args:
		arg1(str): description of what arg1 should be
		arg2(type, optional): description of what arg2 should be

	Returns:
		type of return: description of return
	
	Raises: if any exceptions are raised intentionally 
"""	
	return True

print(func.__doc__)  -> returns the docstring
import inspect
print(inspect.getdoc(func)) ->returns the docstring
```

## DOT & DRY
"Do not Repeat Yourself" and "Do One Thing"
when you copy and paste the same block of code with a few tweaks for your program its a good sign to create a function for that block.
this is "Don't Repeat Yourself".
also when writing a function, try to mostly make the function just do one thing. this is to keep it simple and to make the most of the created functions. this is 
"Do One Thing"
"Any fool can make code that a computer can understand. but only a good programmer can create a code that a human "

## Mutable and Immutable
when a mutable variable, passed to a function, is changed, the value of the variable changes outside the function too. this is not reflected with immutable datatypes like `int`, `string`.  the reason being when mutable datatypes are copied, they aren't true copies but just the same object being pointed by two variables. so when the object is changed, it is changed on both.
to avoid things like this to happen, when we want to use a mutable datatype as a default argument, we use the `None` datatype instead. this makes it so that the value is the same throughout multiple function calls.

# Context Managers
Context managers are special types of functions in python that set up a context for the code, runs the code in that context and closes the context.
an example of a context manager is the `open()` function we use to open files to read and write in them. any context manager starts with the keyword 'with'.
`with func_name(args) as return variable:`
some functions return things sometimes so we use the as keyword to have it be stored in a variable
```python
with open("file name") as file:
	a = file.read()
print(a)
```

writing context managers are useful when we need a function that has a context to set up and tear down, like with files and MySQL.
there are 5 parts to a context manager
1. Define the function
2. Add the set up code for the context
3. use the yield keyword
4. add any teardown code, if it needs it
5. add @contextlib.contextmanager decorator
```python
@contextlib.contextmanager
def context(arg):
	print("hello")
	yield "damn" 
	print("bye")
with context("s") as h:
	print(h)
```
the output will be
```
hello
damn
bye
```
any code before yield is executed then the return value of yield is returned and the code inside the context is run and finally the teardown code.
if you don't want to return anything just write yield and leave it as it is

nested context managers are also perfectly legal in python
```python
with open(file) as file:
	with open(file) as file:
		#code
```

also to make sure the teardown code runs even when an exception is raised, we can use the `try`, `catch` and `finally` blocks
```python
@context.contextmanager
def func():
	file = open(file)
	try:
		yield file
	finally:
		file.close()
```
finally runs no matter if we catch errors or don't

# Decorators
Functions are objects just like any other datatype. they are just like lists or strings. but remember when mentioning the function with parentheses it means we are calling the function. when we don't mention them we are referencing the function itself as an object
```python
def myfunc(arg):
	print("hello")
a = myfunc("f")  -> calling the function
b = myfunc    ->storing the function object in b
```

there are also nested functions that are defined inside functions.
```python
def func():
	def printo(s):
		print(s)
	return s
a = func()   -> printo function returned
a("hello")   -> printo function called
```

Scopes in python are classified into 4 types and when referring to a variable python checks for the variable in the following scopes in the given order
1. local
2. nonlocal
3. global
4. built-in
to change the scope of a variable we use the `global` and `nonlocal` keyword.

closures in python are nonlocal variables added to a returned function. they are essentially a tuple of variables.
```python
def foo():
	a = 5
	def bar():
		print(a)
	return bar
func = foo()
type(func.__closure__) -> tuple
len(func.__closure__) -> 1
print(func.__closure__[0].cell_contents) -> 5
```
here a is the nonlocal variable that is being accessed from the closure even though it is not in the scope of the nested function. that is why closure is used
```python
x = 25
def foo(v):
	def bar():
		print(v)
	return bar
x = foo(x)
print(x.__closure__[0].cell_contents) -> 25
```
even though we changed the value of x, it remains the same in the closure of the returned function.
so closure contains variables that are defined in the nonlocal scope and used by the returned function.

decorators are wrappers that can be wrapped around a function to change that function's behavior. Decorators are functions that get a function as an argument and return a modified version of the function
```python
@double_args
def multiply(a, b):
	return a*b
multiply(1, 5) -> returns 20 because double args doubles the arguments
```
here `double_args` is a decorator that returns a function call after doubling the arguments.
to construct a decorator we use a nested function called wrapper
```python
def double_args(func):
	def wrapper(a, b):
		return func(a*2, b*2)
	return wrapper
multiply = double_args(multiply)
multiply(1, 5) -> returns 20
```
here even though we change the value of multiply by overriding it, the old multiply is called with doubled arguments inside the wrapper function due to closure, which contains `func` in it, i.e. the old multiply function.
using `@double_args` is just a way of saying `multiply = double_args(multiply)`

since we won't know the number of arguments of the function that is being decorated we can use the `*args` keyword in the wrapper function.
```python
def double_args(func):
	def wrapper(*args):
		return func(*args)
	return wrapper
multiply = double_args(multiply)
```

# More On Decorators
decorators are useful when you need some functions to do some common thing. like a timer decorator that calculates the time took to run a function. this can be used as a decorator instead of having to define it each time for each function.

when using a wrapper on a function, the metadata of the function being decorated cannot be accessed because the original function was replaced by the wrapped function. 
```python
@decorator
def func(arg):
	print("hello")
print(func("h").__name__) -> returns wrapper name
```

to change this we use wraps from `functools`
```python
from functools import wraps
def timer(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		print(time(func))
		return func(*args, **kwargs)
	return wrapper
```
we use the `wraps(func)` as a decorator for the wrapper function in the decorator function. this decorator sets the metadata of the wrapper as the metadata of the function being decorated.
we use the `function.__wrapped__` -> to access the wrapper function instead of the decorated function

decorators can also take arguments. for this we use another nested function
lets say we need a decorator that runs a function n times, n passed in the argument
```python
@run_n_times(5)
def func(s):
	print(s) 
func("hello") -> should print hello 5 times
```
these types of decorators can be defined like this
```python
def run_n_times(n):
	def decorator(func):
		def wrapper(*args, **kwargs):
			for i in range(n):
				func(*args, **kwargs)
		return wrapper
	return decorator
```
we create a decorator function and return it with the n value used in it.
using @run_n_times(5) is the same as initializing a decorator
`run_five_times = run_n_times(5)`  and setting that decorator as the decorator of the given function
```python
run_five_times = run_n_times(5)
@run_five_times
def printo(h):
	print(h)
printo(5) -> prints 5 5 times

print = run_n_times(5)(print)
print("hello") -> prints hello 5 times
```

![[Pasted image 20231211113340.png]]

here tags takes in any number of arguments named tags. we then set a tags attribute for the wrapper that is equal to the tags given to tags. so that we can check what are the tags whenever we need to 