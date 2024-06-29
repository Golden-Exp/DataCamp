 #deeplearning #datacamp 
# Text with `Pytorch`
## Preprocessing
![[Pasted image 20240131200531.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240131200531.png/?raw=true)

this is the pipeline for text data in `PyTorch`. first we preprocess the data. there are many types of preprocessing. some are:
1. Tokenization
2. Stop word Removal
3. Stemming
4. Rare Word Removal
Tokenization is the process of converting text into tokens. these might be actual words, half words or even punctuations
```python
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer("basic_english") #to tokenize basic english
tokens = tokenizer("sentence")
print(tokens)
```

Stop word removal removes frequently occurring words that have less meaning to the text, like "and", "or", etc.
```python
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
filter_tokens = [token for token in tokens if token.lower() not in stop_words]
```

Stemming reduces the words to their basic forms. runs, running, ran become run
```python
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filter_tokens]
```

Rare word removal is just like the name suggests. it removes rarely occurring words.
```python
from nltk.probability import FreqDist
freq_dist = FreqDist(stemmed_tokens)
threshold = 2
common_tokens = [token for token in stemmed_tokens if freq_dist[token] > threshold]
```

## Encoding text
Encoding is used to convert text into machine readable language, i.e. meaningful numbers.
some encoding techniques are:
1. One hot encoding 
2. Bag of Words
3. TF-IDF

One hot encoding is used to represent a word using a tensor of zeroes except for 1 index, which is the index of that word in the vocab.
the vocab is the list of all words in the document
```
["cat", "dog", "rabbit"]
then after one hot encoding:
- cat - [1, 0, 0]
- dog - [0, 1, 0]
- rabbit - [0, 0, 1]
```

```python
import torch
vocab = ["cat", "dog", "rabbit"]
vocab_size = len(vocab)
one_hot_vectors = torch.eye(vocab_size) #returns a 2d tensor containing all encoded vectors
#to map words to their encodings
one_hot_dict = {word:one_hot_vectors[i] for i, word in enumerate(vocab)}
```

Bag of Words treat the document as a literal bag of words. here the order of the words don't matter and only the frequency is used.
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = ["doc1", "doc2"]
X = vectorizer.fit_transform(corpus)
print(X.to_array())
print(vectorizer.get_feature_names_out())
```
the transformed array is a 2-d matrix, with the rows being each document and each column representing a word. which column represents which row can be seen with `get_feature_names_out`

TF-IDF or Term Frequency - Inverse Document Frequency scores words based on their importance.
rare words are treated to be more important than frequent words
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
corpus = ["doc1", "doc2"]
X = vectorizer.fit_transform(corpus)
print(X.to_array())
print(vectorizer.get_feature_names_out())
```
the transformed array is a 2-d matrix, with the rows being each document and each column representing a word. which column represents which row can be seen with `get_feature_names_out`

## Datasets and Data Loaders
the final step is converting the encoded text into a data loader to get batches of data.
we use python's class for this.
```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
	def __init__(self, text):
		self.text = text
	def __len__(self):
		return len(self.text)
	def __getitem__(self, idx):
		return self.text[idx]
dataset = TextDataset(encoded_text)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

# Text Classification using `PyTorch`
## Text classification
text classification can classify texts in 3 ways - binary class, multi class and multi label.
another type of encoding is known as word embeddings. here, words are mapped to numerical values to preserve semantic meanings between words. 
for example, 'king' and 'queen' have some semantic meaning so when represented in 3 dimensions, they appear closer. in reality, the dimensions would be much more larger.
to create embeddings we assign a unique index to each word. this is the numerical representation of the words.
embeddings can follow any preprocessing techniques. mostly, it is used after tokenization.
`torch.nn.Embedding` takes in word indices and give out word vectors(embeddings with many dimensions). initially these embeddings are random. through training, these embeddings learn and change accordingly.
so, when we create an embedding of `num_embeddings`=10, and `embedding_dim`=3 we have numbers 0 to 9 mapped to a tensor of random coefficients and when we pass in for example [0, 3, 4], the corresponding 3 tensors are returned. so the embedding matrix is just for looking up the embeddings for each index.
```python
import torch
from torch import nn
words = ["the", "cat", "sat", "on", "the", "mat"]
word_to_idx = {word:i for i, word in enumerate(words)}
inputs = torch.LongTensor([word_to_idx[word] for word in words])
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=10)
output = embedding(inputs)
```
this returns 10 random numbers for each word in the list, even if the same word repeats itself. so, the value for `num_embeddings` is usually, the size of the vocab.

## CNNs for text classification
CNNs for images involve a grid filter passing through another grid. for Text we can imagine the filter and the text to be a grid with 1 row. so, it utilizes 1-d Convolutional layers. A stride is how the next position of the changes after a dot product. stride=1 means the filter moves 1 block at a time(1-d so a row of blocks, each block containing the embeddings)
imagine a CNN for image. lets say we have a tensor of 4x3x64x64 for the data. meaning, we have 4 images, with 3 sets for each(red, green, blue) and each set being 64x64. to pass this to a convolution, we need the number of feature maps for each image(3) before the dimensions of each. likewise for 1-D convolutions, we need the number of features before the elements in the sentence.
now, lets say we have a sentence with 4 words and the embedding matrix has 10 coefficients for each word. first we pass in the indices of each word, using the vocab to the embedding layer, where we get the coefficients of each word. since the input for the convolutional layer should be 3-d, we use `unsqueeze` to add an extra dimension, before passing to the embedding layer. do we pass in 1x4 tensor into the layer and get 1x4x10 tensor as output(10 coefficients for each word). since the features should be before elements(here number of elements is 4), we permute and change the dimensions to 1x10x4. 
so we have 10 sets of "words", like we had 3 sets of images for each color. now we can pass it to the convolutional layer and get whatever number of outputs we set. lets say we set 10. now we get 10 outputs, each being words, so the dimensions are 1x10x4.
now, pass the 10 "words" to the RELU and then we take the mean across each word to get 10 numbers. these numbers are then passed into the Linear layer to get the classification result.
```python
class SentimentAnalysisCNN(nn.Module):
	def __init__(self, vocab_size, embed_dim):
			super().__init__()
			self.embedding = nn.Embedding(vocab_size, embed_dim)
			self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, 
			stride=1, padding=1)
			self.fc = nn.Linear(embed_dim, 2)
	def forward(self, text):
		embedded = self.embedding(text).permute(0, 2, 1)
		conved = F.relu(self.conv(embedded))
		conved = conved.mean(dim=2)
		return self.fc(conved)

model = SentimentAnalysisCNN(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters, lr=0.1)

#training
for epoch in range(10):
	for sentence, label in data:
		model.zero_grad()
		sentence = torch.LongTensor([word_to_idx.get(w, 0) for w in sentence]).unsqueeze(0)
		outputs = model(sentence)
		label = torch.LongTensor([int(label)])
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

#predicting
for sample in book_samples:
	input_tensor = torch.tensor([word_to_idx[w] for w in sample], 
	dtype=torch.long).unsqueeze(0)
	outputs = model(input_tensor)
	_, predicted = torch.max(outputs.data)
```

## RNNs for Text Classification
RNNs are better for text classification, even better than CNNs as they have memory. they can detect patterns in data on the long run.
LSTMs - useful for sentiment analysis
GRUs - useful for spam detection without reading the whole thing

# Text Generation

