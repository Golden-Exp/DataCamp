#datacamp #matplotlib
```python
import matplotlib.pyplot as plt
plt.plot(x, y)                      #plots the line plot
plt.show()                          #to show the plot
```
plots a line plot of the two given axes

![[Pasted image 20231024230414.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Matplotlib/Attachments/Pasted%20image%2020231024230414.png/?raw=true)

```python
plt.scatter()                    #plots the line plot without the lines joining the dots
plt.show()
```
plots a scatter plot

![[Pasted image 20231024230252.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Matplotlib/Attachments/Pasted%20image%2020231024230252.png/?raw=true)


```python
plt.xscale('log')          --> #represents the x axis as powers of 10
plt.xlabel('year')         --> #changes the name of the x axis as year
plt.ylabel('returns')      --> #changes the name of the y axis as returns
plt.title('title')         --> #sets the title of the graph
plt.xticks(value, label)   --> #represent the given value as its corresponding label
plt.yticks(value, label)   --> #same as the above but for y axis
plt.scatter(x, y, s=size)  --> #the size parameter sets the sizes of the bubble(size should be the same length as x and y)
plt.scatter(x, y, c=col, alpha=0.6) --> #the col parameter specifies the color of the bubble and the alpha parameter tells the opacity
```

![[Pasted image 20231024231840.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Matplotlib/Attachments/Pasted%20image%2020231024231840.png/?raw=true)



```python
plt.hist(values, bins=10)
```
gives a histogram which categorizes the data into the ranges it comes under
bins is the number of ranges we need
![[Pasted image 20231024231952.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Matplotlib/Attachments/Pasted%20image%2020231024231952.png/?raw=true)

when plotting for a 2-d array/list, the plot shows different lines each following the data in each "column". so its usually better to transpose the array before plotting using
```python
np.transpose(array)
```
