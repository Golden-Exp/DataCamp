#datacamp  #seaborn 
### Distribution plots
just like `relplots` and `catplots` there are `displots` which plot distribution plots
by default it plots histograms
```python
sns.displot(x=df['col'], kind="hist")
```
note that we don't use the usual `sns.plot(x="", y="", data=df)`. using it won't show any error tho, so yea we can use it both ways
the types of `displots` are 
1. KDE plots (Kernel Density Element)
2. Histograms
3. Rug plot
4. ECDF plots (Estimated Cumulative Density Function)
KDE plots
![[Pasted image 20231119115225.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119115225.png/?raw=true)


Histograms
![[Pasted image 20231119115246.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119115246.png/?raw=true)

there is also a parameter called stat which changes the y axis from the count of x axis to something else in the column.

rug plots
![[Pasted image 20231119115356.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119115356.png/?raw=true)


ECDF plot
![[Pasted image 20231119115430.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119115430.png/?raw=true)


### Regression plots
`regplots` are used to show a connection between two variables.
you could say its a scatter plot with a regression line
plots that only care about 1 variable are known as univariate analysis like the previous plots
regression plots are for bivariate analysis
there are 2 ways to plot regression plots
1. `regplot`
2. `lmplot`
`lmplot` is superior because it has a good aspect ratio and can implement faceting
faceting is the concept of plotting multiple graphs while changing a single variable
can be implemented with hue and subplots (row and col argument)
```python
sns.regplot(x="", y="", data=df)
```
![[Pasted image 20231119120858.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119120858.png/?raw=true)


```python
sns.lmplot(x="", y="", data=df, hue="col", row="col")
```
third variable with hue
![[Pasted image 20231119121131.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119121131.png/?raw=true)


third variable with row/col
![[Pasted image 20231119121253.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119121253.png/?raw=true)



# Visualization aesthetics
```python
sns.set()  -> sets the style to default style which is darkgrid

for i in ['darkgrid', 'whitegrid', 'white', 'dark', 'ticks']
	sns.set_style(i)
	sns.displot(x=df['col'], kind="hist", bins=10)
```
experiment with each style to see which is the best for your visualization

sometimes removing the axes lines can help
```python
sns.set_style("white")
sns.displot(df['col'])
sns.despine(left=True)  -> removes the y axis line
```
## Colors
```python
sns.set(color_codes=True) -> enables us to use color codes like 'g' and 'r'
```
color palettes
seaborn has 6 color palettes
deep, muted, pastel, bright, dark, colorblind
`sns.palplot(color_palette_name)` displays the colors used by that palette
`sns.color_palette()` -> returns the current color palette
![[Pasted image 20231119194723.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119194723.png/?raw=true)


Custom color palettes
1. Circular  -> the color is not ordered
2. Diverging   -> two colors' low and high intersect
3. Sequential  -> the same color from low to high
![[Pasted image 20231119195021.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119195021.png/?raw=true)

[color palettes](https://seaborn.pydata.org/tutorial/color_palettes.html)

## Customizing with matplotlib
we can also assign axes to seaborn functions
so we can first create a figure and an axis using subplots, assign it to the seaborn function and use the axis object to do further customizations
![[Pasted image 20231119222252.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119222252.png/?raw=true)


can also use the subplots advantage of multiple plots in seaborn this way
![[Pasted image 20231119222508.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119222508.png/?raw=true)

![[Pasted image 20231119222553.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119222553.png/?raw=true)


note that the ax parameter and its functions only work for basic plots like `sns.histplot()` and `sns.scatterplot()` 
it won't work for `displots`, `relplots` and others

# Different types of plots
## Categorical plots
for categorical data there are a lot of plots
most of them work well when combined with a numerical variable
3 types
1. shows all observations
2. abstract representation
3. shows summary statistic
### All observations
#### Strip plots
![[Pasted image 20231119224517.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119224517.png/?raw=true)

here we can see how each DRG type(categorical) has varied Average Covered charges(numerical)
jitter parameter used to see the points more clearly

#### Swarm plots
plots all the points and, using an algorithm, makes sure that points stay in the same category and don't overlap. however it doesn't scale well with large data
![[Pasted image 20231119224826.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119224826.png/?raw=true)


### Abstract representations
#### Boxplots
used to see distribution with measures like median and outliers
![[Pasted image 20231119224907.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119224907.png/?raw=true)


#### Violin plots
combination of box plots and KDE plots. show the overall distribution.
uses KDE formula to plot. so may not plot all the data points
![[Pasted image 20231119225028.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119225028.png/?raw=true)


#### Boxen plots
hybrid between box plots and violin plots. can scale well with large datasets
![[Pasted image 20231119225218.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119225218.png/?raw=true)


### Statistical estimates
#### Bar plots
used to show the average of the category. used hue here to introduce a third variable.
![[Pasted image 20231119225323.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119225323.png/?raw=true)


#### Point plots
just like bar plots they show the average and confidence interval. however, its better to interpret when a third variable is introduced
however me personally, I think both plots are good with a third variable, and it varies with the dataset. so check for both and see which is better
![[Pasted image 20231119225521.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119225521.png/?raw=true)


#### Count plots
used to show the number of elements in each category
![[Pasted image 20231119225740.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119225740.png/?raw=true)


# Regression Plots
used to see a relation between x and y
```python
sns.regplot(x="", y="", data=df, order=by default 1, 
			x_bins=number, 
			x_jitter=width of each category for the points to stay between, 
			x_estimater = np.mean)
```
when order is greater than one the graph is tried to fit in a polynomial of that order.
`x_jitter` and `x_estimator` used with categorical variables
`x_estimator` plots the estimated summary statistic instead of all the points of the category with confidence intervals
![[Pasted image 20231119231418.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119231418.png/?raw=true)


`residplot` is used to check the appropriateness of the `regplot`
same parameters as `regplot`
![[Pasted image 20231119231445.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119231445.png/?raw=true)

ideally, if the `regplot` is correctly fit, the data would be randomly scattered over the horizontal line of a `residplot`

# Matrix Plots
plots that show the output as a grid.
they get their input as a grid so we use `pd.crosstabs` and `df.corr()` for this
`crosstab()` used to convert into a grid
![[Pasted image 20231119232734.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119232734.png/?raw=true)

the values are the elements in the values column corresponding to the points in the x and y column upon which the `aggfunc` is applied.
```python
df_crosstab = pd.crosstab(df['col'], df['col'], values=df['col'], aggfunc="mean")
sns.heatmap(df_crosstab, annot = True or False, 
			fmt="d to represent the numerical values in the cells", 
			cmap="color palette", cbar=True or False to represent the color bar,
			linewidth=to represent space between cells,
			center = choose the cells to represent the highes)
```
![[Pasted image 20231119233310.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%20231119233310.png/?raw=true) ![[Pasted image 20231119233328.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119233328.png/?raw=true)

with correlations,
![[Pasted image 20231119233515.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231119233515.png/?raw=true)


# Data Aware Grids
## Facet Grids
faceting is the concept of splitting the data into rows and columns for better analysis
`FacetGrid()` is used to create a row and column of any plot.
that is, any plot can be divided into rows based on a third variable using this.
`catplot` and `lmplot` are used to facet categorical plots and regression plots respectively using the row and col parameter.
they are much simpler to use than `FacetGrid`
```python
g = sns.FacetGrid(df, col="col") -> splits the grid into the categories of col
g.map(sns.boxplot, 'Tuition', other parameters for boxplots)
```

![[Pasted image 20231120171456.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231120171456.png/?raw=true)


Note that for seaborn to do all this, the data that we feed to seaborn should be tidy.
that is all the rows should be an observation

## Pair Grids

pair grids and pair plots are used to plot a grid of plot across different variables to check if they have a relation. all the plots in the grid need not be the same type of plot.
```python
PairGrid(df, vars=["col name", "col name"])
g.map(sns.scatterplot) ->this creates a 4x4 grid with all of them being scatterplots

g.map_diag(sns.hist)
g.map_offdiag(sns.regplot) -> creates a 4x4 grid with diagonals being histograms and others being regplots
```
![[Pasted image 20231120173954.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231120173954.png/?raw=true)


![[Pasted image 20231120174024.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231120174024.png/?raw=true)


`pairplot` is shortcut for pair grids
```python
sns.pairplot(df, vars=[colname, colname], kind="reg", diag_kind="hist")
```
note that by default `diag_kind` is hist **if hue is not used**
if hue is used the default `diag_kind` is KDE plots.
also note that the default for `kind` is scatter
refer [pairplots](https://seaborn.pydata.org/generated/seaborn.pairplot.html)
**also if you want particular columns on x and y axis and note the same, you can use `x_vars` and `y_vars`**

## Joint grids
used to compare two variables with many plots combined

```python
g = JointGrid(x="", y="", data=df)
g.map(sns.regplot, sns.histplot)
```

the first argument for map is the plot that should be in the center and the second one is the one that should be on the marigins
![[Pasted image 20231120180039.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231120180039.png/?raw=true)


`plot_joint` used to plot in the center
`plot_marginals` used to plot in the margins
![[Pasted image 20231120180104.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231120180104.png/?raw=true)


Joint plot is easier
![[Pasted image 20231120180152.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231120180152.png/?raw=true)

in joint plot, by default the margins kind is histogram

here's a sort of complex plot
![[Pasted image 20231120180234.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231120180234.png/?raw=true)


here we first use a `pairplot` to plot between the two variables and set the center plot to be scatter and set the data with some filtering. then we map the `pairplot` object with a `plot_joint` function to plot a KDE plot in the center
![[Pasted image 20231120180254.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Seaborn/Attachments/Pasted%20image%2020231120180254.png/?raw=true)


