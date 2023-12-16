#datacamp #matplotlib 
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(df['col'], df['col'])       #plots x and y axis
```
the convention is always y vs x or y against x

### Customizations
```python
ax.plot(x, y, marker="o") -> returns all the coinciding points represented as a circle
ax.plot(x, y, linestyle="--") -> changes lines to --
ax.plot(x, y, color='r') -> changes colr to red
```
more markers -> [markers](https://matplotlib.org/stable/api/markers_api.html)
more linestyles -> [linestyles](https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html)

![[Pasted image 20231113084945.png]]

if linestyle=None -> no line gives a scatter plot
```python
ax.set_xlabel("label") -> x label
ax.set_ylabel("label") -> y label
ax.set_title("title") -> title
```

### Multiple Plots
when we want to represent a lot of data, we cant do that in the same graph because it will look messy.
that's where subplots come in. we use the subplots to create a grid of plots. when we use the function to create that grid, the `ax` object is no longer a single element but rather an array.
**When having only one row or one column the ax object becomes a 1-D array. else it becomes a 2-D array**
```python
fig, ax = plt.subplots(3, 2, sharey=True) -> creates a grid of 3 rows and 2 columns
#sharey = True means the y axis will have the same dimensions for all plots in the grid
ax[0][0].plot(...)
```

![[Pasted image 20231113090323.png]]
![[Pasted image 20231113090345.png]]

#### Twin plots
```python
ax2 = ax.twinx() -> the x axis is common for both
ax2 = ax.twiny() -> the y axis is common for both
ax.tick_params("either x or y", colors='r') -> changes the ticks of either x or y to red
```
note that the color parameter of `tick_params` is **colors** and not **color**

![[Pasted image 20231113135336.png]]
![[Pasted image 20231113135402.png]]

two axes in the same plot. can use colors to differentiate
this is useful when u want to represent 2 different measurements in the same plot.

can also create twin plots at the beginning
```python
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols2, sharey=True, figsize=(7,4))
```
this creates 2 axes to plot

### Annotations
annotations are methods used to highlight a part of a graph.
```python
ax.annotate("label of annotation", xy = (x of the point to annotate, y of the point to annotate), xytext = (x coordinate of label, y cordinate of label), arrowprops={"arrowstyle":"->", "color":"grey"})
```
`xy` - the coordinates of the point to annotate
`xytext` - the coordinates of the label of the annotation
when `arrowprops` is left empty default properties are used
refer : [annotations](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html)
![[Pasted image 20231113141301.png]]
can also denote a line parallel to the x or y axis
```python
ax.axvline(x="the number where u want to draw the line", label="", linestyle="--")
```
![[Pasted image 20231119221726.png]]
### Bar charts
used to represent numbers for a certain function/group
```python
ax.bar(x, y) -> y is the numeric column and x is the ticks list 
ax.set_xticklabels(labels, rotation=angle we want to rotate)

#stacks the graphs and the starting point of this graph is the end of y
ax.bar(x2, y2, bottom = y, label="label for legend") 
ax.bar(x3, y3, bottom = y + y1) -> starts from the sum of y and y1
ax.legend() -> displays the legend
```
![[Pasted image 20231113222240.png]]

### Histograms
used to show the distribution of the numeric column
```python
ax.hist(data, bins=no.of bins or list of bins, histtype="step") -> returns a line histogram instead of a solid one
ax.hist(data2, bins=no.of bins or list of bins, histtype="step") -> stacks both histograms
```

![[Pasted image 20231113222558.png]]

### Statistical plotting
comparing visualizations for statistics

#### Error Bars
marks the summary statistic of that particular bar or point
the longer the error bar, the more the summary statistic.
so giving the std as error bar means we can see the spread of the data.
##### error bars in bar graphs
```python
ax.bar(x, y, yerr=x.mean() or x.std())
```
![[Pasted image 20231113223625.png]]

##### Error bars in line plots
```python
ax.errorbar(x, y, data of the summary statistic of all values of x)
```
![[Pasted image 20231113223752.png]]
![[Pasted image 20231113223805.png]]

#### Boxplots
boxplots are boxes with the IQR and lines denoting 1.5 * IQR + 75th percentile 
and 25th percentile - 1.5 * IQR anything greater or lesser are outliers

```python
ax.boxplot(list of data to plot)
```
![[Pasted image 20231113224118.png]]

the orange line is the median or 50th percentile

### Scatter plots
they are used when we need to compare two variables or two measured values instead of a measured value against a function or a group(bar charts)

```python
ax.scatter(x, y, label="label", c="values with the same length as x and y to encode the color with a third variable")
```
the c keyword is different from color
color uses the same color for all points
c uses a different color for high measurements and another for low measurements and the color transitions between
![[Pasted image 20231113225702.png]]
![[Pasted image 20231113225718.png]]

### Styles for plots
```python
plt.style.use("name of the style")
plt.style.use("default") -> changes back to default
```
![[Pasted image 20231113230636.png]]

refer [styles](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html) for more styles

### saving figures
```python
fig.savefig("name_of_file.png", quality=number between 0 to 100, dpi = any number)
```
saves file as the filename.png 
PNG files are of high quality but takes up space
JPG files useful for websites
SVG files useful for editing later.

quality keyword is between 0 and 100 and lower the quality more compressed it becomes. any quality number after 95 is meaningless

dpi is dots per inch. more the dpi more is the quality and more is the space taken.

```python
fig.set_size_inches([width, breadth])
```
sets the size of the figure



