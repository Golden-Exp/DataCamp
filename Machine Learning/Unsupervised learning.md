#ml #datacamp 

# Clustering for Dataset Exploration
## Intro
unsupervised learning is when we don't know what we are trying to predict. the main goal of supervised learning is to predict something. that is not the case of unsupervised learning. here the main goal is to find patterns in data. it is mainly pattern discovery and nothing else.
some examples of unsupervised learning are:
- **Clustering:** where we separate the data in groups or clusters with similar patterns
- **Dimensionality Reduction:** where we compress the data using the correlations and patterns found

to usually find patterns among features we visualize them. however as the number of features increase the graphs are difficult to interpret.
k-means clustering is a type of clustering to find a specified number of clusters which is k
```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)
labels = model.predict(samples)
print(labels)

labels = model.fit_predict(samples) -> fits and gives label
```
samples here are 2-d arrays like the ones we give to pandas for creating Data frames.
to find predict what clusters new unseen data belong to, the model calculates something called the centroids which is the mean of the samples in each cluster.
![[Pasted image 20240101223815.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101223815.png/?raw=true)
![[Pasted image 20240101223902.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240101223902.png/?raw=true)

`model.cluster_centers_` gives n values which the centroids of n clusters

## Evaluating a Cluster
to see how the clusters have been assigned compared to their initial groups we can create a correspondence table, using Pandas' cross tabulation.
```python
import pandas as pd
ct = pd.crosstab(df["col1"], df["col2"])
```
![[Pasted image 20240102121733.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240102121733.png/?raw=true)
another way to determine the quality is by the inertia of the clusters. this is the measure of spread of the samples in each clusters from its centroid. the more lower the inertia is the better. K Means aims to minimize this value of inertia.
```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)
```
but what if the data doesn't come with pre determined groups? we wouldn't know the number of groups. so how do we choose the right amount of clusters?
we can do that like how we do hyperparameters. by choosing different number of clusters and see how it goes.
when we plot the graph for this, a noticeable trend is that as the number of clusters increase the inertia always decrease. imagine we have n rows in our sample. if we decide to have n clusters, then each row would have its own cluster and the inertia will be 0. so its not a good idea to increase the number of clusters just because inertia decreases.
a good choice of number is the elbow point in the graph, where the inertia starts decreasing slowly for each increase
![[Pasted image 20240102123220.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240102123220.png/?raw=true)

## Transforming Features
sometimes K Means clustering will give worse results. this might be because the data doesn't have the same variance across all features. since K means is like KNN and depends on distance, standardizing the features beforehand might improve the results very much
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)
#now scaler can be used to transform any similar samples
samples_std = scaler.transform(samples)
```
we can use a pipeline for this.
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
```
pipeline is now our model object and can be used to get the labels.
another transform is normalizer.
standardizer -> rescales the variance of features/columns
normalizer -> rescales each sample/row

# Visualizing Clusters
## Visualizing hierarchies
there are two types of visualizations for clusters.
1. t-SNE
2. hierarchical clustering

hierarchical clustering is when we have as many clusters as the number of rows and then slowly merge them under smaller clusters and this goes on until there's only one cluster.
this can be very useful to show similarities between different known clusters
![[Pasted image 20240102225637.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240102225637.png/?raw=true)

this above graph was done from the scores given by each country for many songs. just from a dataset like that the model was able to find familiar countries. each row in the dataset was a country.
this type of hierarchical clustering is known as agglomerative clustering
what it does is it merges the two most closest clusters together one by one
this entire graph is known as a dendrogram. 
to plot a dendrogram,
```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method="complete")
dendrogram(mergings, 
		  labels=country_names,
		  leaf_rotation=90,
		  leaf_font_size=6)
plt.show()
```
linkage performs the clustering and dendrogram plots it

## Cluster Labels in Hierarchical Clustering
the y axis of the dendrogram is the distance between two clusters when they are merged. if for example, two clusters are merged at y = 15, then the max distance between samples in those two clusters is 15. and if we specify, height = 15, any clusters that have a distance between them greater than this should not be merged.
the distance between two clusters is found by the linkage method. when its given as "complete", the distance is taken to be the maximum distance between the samples of each cluster.
![[Pasted image 20240102231912.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240102231912.png/?raw=true)
for example, from the above diagram, cluster 1 is closest to cluster 2 according to complete linkage, because the maximum distance between cluster 1 and cluster 2 is lesser than the max distance between cluster 2 and cluster 3.
but if we take single linkage where the distance is the distance between the closest points, cluster 3 is closest to cluster 1.
there are other methods also.
we can also extract the clusters available at a specific height of the dendrogram
```python
from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method="complete")
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion="distance")
print(labels)
```
labels is the cluster number. 3-4 rows may come under the same cluster/label number.

**note that the labels start from 1 if we use `scipy` unlike in `sklearn`**

## t-SNE for 2-Dimensional Plots
t-SNE or t-distributed stochastic neighbor embedding is plot mainly for measuring the distance between clusters and the inertia of clusters
it approximates the nearness of samples.
![[Pasted image 20240102233143.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240102233143.png/?raw=true)
here we see that there are 3 clusters and 2 of them are close together, and can be concluded that the two clusters are closely related. these are the type of things we can interpret from t-SNE plots.
to plot them with `sklearn`
```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(samples)
xs = transformed[:, 0]
ys = transformed[:, 1]
plt.scatter(xs, ys, c=species)
plt.show()
```
note that we can only use `fit_transform` for TSNE. that means we have to use it each time we get new samples. also there is a hyperparameter called Learning Rate here, which is found by trial and error.
also note that the x and y values of a t-SNE plot has no meaning. in fact if you run the same code some times, each time we get different plot orientations, with the same relative distance between clusters
![[Pasted image 20240102234219.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240102234219.png/?raw=true)

this is a plot of stock movements of multiple companies. this shows how different companies' differ from others.

# Dimension Reduction
## Visualizing PCA Transformations
dimension Reduction finds patterns in data and uses these patterns to express the same data in a compressed and  memory efficient way.
the main thing that dimension reduction does is that it reduces the data to its bare bones, by removing any useless noise, which causes problems for regression and classification
sometimes, the data set undergoes dimensionality reduction and then goes to supervised learning for prediction.
PCA - Principal Component Analysis - a basic for dimension Reduction.
it performs dimension reduction with 2 steps
- decorrelation
- dimension reduction
decorrelation is done by rotating the samples to align with the coordinate axes.
it also shifts the data samples so that the mean of the samples is 0.
![[Pasted image 20240103113337.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240103113337.png/?raw=true)

note that no information is lost when doing this transform
for doing this with `sklearn`
```python
from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)
transformed = model.transform(samples)
```
since its not `fit_transform`, the same transform can be used for unseen data.
this returns an array similar in shape to the original array, with the same number of rows. the columns are the PCA features.
many features of a dataset are correlated. PCA transforms undo this correlation by shifting it to the axes.
![[Pasted image 20240103113906.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240103113906.png/?raw=true)

PCA learns the Principal Components of the data. that is the directions in which the sample varies the most.
it is the principal components that PCA aligns with the coordinate axes
![[Pasted image 20240103114411.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240103114411.png/?raw=true)

we can access the principal components by,
```python
print(model.components_)
```
gives an array. one row for each Principal component. and one row contains the coordinates

## Intrinsic Dimension
intrinsic dimension is the minimum number of features required to approximate the dataset.
for example, if we have a dataset with features latitude and longitude of a plane's movement, that dataset is intrinsically 1 because we could just have the displacement of the flight path
![[Pasted image 20240103121104.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240103121104.png/?raw=true)

this is the essential idea behind PCA and to find the compact representation of the given dataset.
lets take a sample with 3 features and plot it in a 3-d scatter plot
![[Pasted image 20240103121308.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240103121308.png/?raw=true)
we can see that it is scattered like a 2-d flat sheet. this means the intrinsic dimension of the sample is 2.
but this type of plots can only be made with 2 or 3 features. to get more, we can use PCA to find the features with the highest variance. and it makes sense. if the values are constant along one direction, the coordinates in that direction, will all have similar values. so the variance will be small.
![[Pasted image 20240103121626.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240103121626.png/?raw=true)
here we see that the 3rd feature has very low variance and can be eliminated. this is what we saw in the scatter plot
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA()
pca.fit(samples)
features = range(pca.n_components_)
plt.bar(x=features, pca.explained_variance_)
plt.show()
```
`pca.n_components` -> returns the number of features.
please note that the intrinsic dimension from PCA is ambiguous.
![[Pasted image 20240103121944.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240103121944.png/?raw=true)
here we can argue that the intrinsic dimension is 2 or 3. 

## Dimension Reduction with PCA
we can do this by giving the `n_components` parameters in the PCA model
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(samples)
transformed = pca.transform(samples)
```
the transformed samples will have 2 features only. the value of `n_components` can be the intrinsic dimension. 
when the reduced sample is plotted
![[Pasted image 20240103124828.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240103124828.png/?raw=true)

although this method of dimension reduction is good, there are alternatives for PCA. one example are word-frequency arrays.
they are kind of like cross tabulations, with each row corresponding to a document and each column to a word. the values inside are the frequencies of words in each document. 
sometimes many words will have a frequency of 0 across many documents except a few. these types are known as sparse arrays. a representation of sparse arrays can be done by `csr_matrices`. it remembers only the non zero entries in the array.
`sklearn` doesn't have CSR matrix so we use `TruncatedSVD` instead. this performs the same things as PCA but, accepts CSR matrices as inputs
```python
from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents) -> #this is the csr_matrix
transformed = model.transform(documents) #same as pca
```
another way of converting to `csr_matrix` is using `TfidfVectorizer`. we transform the docs with this. if we print it we get the `csr_matrix`. we can also get the feature names that is the words.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

#get words
words = tfidf.get_feature_names()
print(words)
```

so a common workflow of PCA and clustering is, 
1. make the `PCA`/`TruncatedSVD` instance depending on the question
2. create the `KMeans` instance
3. make a pipeline
4. fit the data - so the PCA transforms and reduces the components to the given `n_components` and `KMeans` performs Clustering on the reduced data
5. and we get the result

# Discovering interpretable features
## NMF
NMF is like PCA and is used for dimensionality reduction. however they are much more easier to interpret. it can't be applied to every dataset though. only arrays with non-negative numbers can be applied.
NMF decomposes samples as the sum of the parts of the sample.
![[Pasted image 20240103225825.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240103225825.png/?raw=true)

NMF works with `numpy` arrays and `csr_matrices`
lets see an example with a word frequency array. the frequency is found using something called the `tf-idf`. `tf` is the frequency of the word in the document.
`idf` is a weighting scheme that reduces the influence of frequent words like "the".
```python
from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples)
transformed = model.transform(samples)
```

NMF also has basic components and the number is the number of components we give at the beginning. the dimensions depend on the number of features. if there are 4 features and 2 components were given, then the components would be a 2-d array with 2 elements each having 4 elements corresponding to the 4 features.
the transformed features are also non negative by the way. to reconstruct the whole data from the transformed data, we can matrix multiply the transformed features with the basic components
![[Pasted image 20240104081315.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104081315.png/?raw=true)

that's why the name Matrix Factorization.
`nmf_features` = (`n_rows_of_sample` x `n_comp`) matrix
`components` = (`n_comp` x `no.of features`) matrix
so when both are matrix multiplied we get 
(`n_rows_of_sample` x `no.of features`) matrix that is, our data
So what this means is that NMF doesn't discard any features but actually decomposes them. unlike PCA

## The Interpretable Parts
when we see the basic components of an NMF model, we see that for each component(rows) there are an equal number of features as the number of features of the original sample. note that all of these components are topics.
each component of the NMF is a topic corresponding to the CSR matrix.
if its an image, the components are patterns in the image.
for example, if we create an NMF model for Wikipedia and create 10 components, each component correspond to specific topics like science, math which are separated according to the word frequencies. we won't know that a row's topic is science, but that component might have feature values with many scientific words.

![[Pasted image 20240104084524.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104084524.png/?raw=true)

for images, it corresponds to patterns. to convert images into having positive numbers pixels, we use grayscale images. where every shade is just a shade of gray. the pixel number represents the brightness, with black being 0 and white being 1.
![[Pasted image 20240104084731.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104084731.png/?raw=true)

since images are mostly squares or rectangles, we get 2-d arrays of images. we can flatten these arrays to get a single array of numbers, by reading the elements row by row from left to right and from top to bottom
![[Pasted image 20240104084908.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104084908.png/?raw=true)

so an image is now 1 array of numbers. and if we have a collection of images, that would be a 2-D array of positive numbers, where each row is an image and each column is a pixel - NMF can be applied.
To recover the image from the flat array, we can reshape
```python
sample = sample.reshape(original size)
plt.imshow(sample, cmap="gray", interpolation="nearest")
plt.show()
```

visualizing the components is a good way for looking at patterns and parts of data for NMF. for PCA, the components are not interpretable

## Collaborative Filtering
NMF features are similar for similar rows. so to group similar rows/articles, do we compare the feature values?
```python
from sklearn.decomposition import NMF
model = NMF(n_components=6)
features = model.fit_transform(samples)
```
although if two rows have similar feature values means they are similar, the vice-versa isn't true.
just because two articles are similar, doesn't mean they have the same feature values. 
![[Pasted image 20240104095638.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104095638.png/?raw=true)

however, all of the features that are similar will lie on the same line from the origin, even if the values are different. we can see that in the scatter plot between topics in the above image.
so it is a good idea to compare these lines to get the similarity of 2 documents. if they are ideally the same, they will be on the same line. if they are really similar, they will be closer to each other.
we'll compare the angle between the lines. but since using the angles directly can be hard, we'll use the cosine of the angle known as **cosine similarity**. this ranges from 0 to 1 and the higher the value, the better.
![[Pasted image 20240104100035.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104100035.png/?raw=true)
to calculate the cosine similarity,
```python
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)

#to find smth similar to one doc that has index 23
current_doc = norm_features[23, :]
similarities = norm_features.dot(current_doc) -> #gives similarity of all docs to the current_doc
```
we can also use pandas for better clarity
```python
import pandas as pd
from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
current_doc = df.loc["doc_name"]
similarities = df.dot(current_doc)
```
![[Pasted image 20240104100730.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Machine%20Learning/Attachments/Pasted%20image%2020240104100730.png/?raw=true)




