#datacamp #deeplearning 
# Training Robust Neural Networks
## OOP and `PyTorch`
usually it is convenient to define the datasets and even the models with python's object oriented programing approach. it is much more simpler. to define a dataset with a class in python,
```python
from torch.utils.data import Dataset

class Data(Dataset): #inheriting the dataset module of pytorch
	def __init__(self, csv_path):  #constructor
		super().__init__()   #calling constructor of Dataset module
		df = pd.read_csv(csv_path)
		self.data = df.to_numpy()
	def __len__(self):              #required to overload
		return self.data.shape[0]
	def __getitems__(self, idx):     #required to overload
		features = self.data[idx, :-1]
		label = self.data[idx, -1]
		return features, label
```

`len` returns the length of the dataset and `getitems` returns the features and labels of a particular index.
to create a model from OOP instead of sequential
![[Pasted image 20240118001158.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240118001158.png/?raw=true)

```python
dataset = Data("df_path.csv")
from torch.utils.data import DataLoader
dataloader_train = DataLoader(dataset, batch_size=2, shuffle=True)
class Net(nn.module):
	def __init__(self):
		self.fc1 = nn.Linear(9, 16)
		self.fc2 = nn.Linear(16, 8)
		self.fc3 = nn.Linear(8, 1)
	def forward(self, x):  #defines what happens when input x passed to model
		x = nn.functional.relu(self.fc1(x))
		x = nn.functional.relu(self.fc2(x))
		x = nn.functional.sigmoid(self.fc3(x))
		return x
```
## Optimizers
the forward pass begins by computing all the values of the neurons and finally the predictions. then with the predictions the loss is calculated and that loss is sent to the optimizer, with the parameters. the optimizer calculates the gradients and updates the parameters.
![[Pasted image 20240118223204.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240118223204.png/?raw=true)
the update of parameters is done in different ways. the learning rate is the main hyperparameter that decides how the parameters change. it gets multiplied by the gradients and are subtracted.
although SGD is simple, it is rarely used. for example, using the same learning rate for each epoch might not be optimal. so `Adagrad` is used(Adaptive Gradients)
```python
optimizer = optim.Adagrad(net.parameters(), lr=0.01)
```
here the learning rate decreases as the epochs increase.
but sometimes it decreases too fast, so `RMSprop` is used(Root Mean Square Propagation), where the LR is based on the size of the gradients of the last epoch.
```python
optimizer = optim.RMSprop(net.parameters(), lr=0.01)
```
another widely used optimizer is the Adaptive Moment Estimation(Adam)
```python
optimizer = optim.Adam(net.parameters(), lr=0.01)
```
the difference to this and RMSprop is that the momentum also gets adaptive.

## Vanishing and Exploding Gradients
vanishing gradients is when the gradients get smaller and smaller as the epochs go on and finally becomes negligible. then the model stops learning.
![[Pasted image 20240118225909.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240118225909.png/?raw=true)

exploding gradients occur when the gradients get bigger and bigger and "explode". this diverges training to just those parameters.
![[Pasted image 20240118230042.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240118230042.png/?raw=true)

to avoid unstable gradients, we can use 3 steps
1. Proper weights initialization
2. Good Activations
3. Batch Normalization
good weights initialization can be achieved by using a distribution to sample the weights from. one good one is the He/Kaiming distribution
```python
import torch.nn as nn
import torch.nn.init as init
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(9, 16)
		self.fc2 = nn.Linear(16, 8)
		self.fc3 = nn.Linear(8, 1)

		init.kaiming_uniform_(self.fc1.weight)
		init.kaiming_uniform_(self.fc2.weight)
		init.kaiming_uniform_(self.fc3.weight,
							nonlinearity="sigmoid")
	def __forward__(self, x):
		x = nn.functional.relu(self.fc1(x))
		x = nn.functional.relu(self.fc2(x))
		x = nn.functional.sigmoid(self.fc3(x))
		return x
```

next activations. the RELU is a good activation function but it has the problem of some neurons becoming dead due to the null property of RELU. An ELU gives out the 
max(-1, x) value, so there are no null weights. but the overall weights value decreases with this.
![[Pasted image 20240118231118.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240118231118.png/?raw=true)

Batch normalization is used to spread out values between layers and making sure all the weights across layers are normalized. to do this first we subtract everything from the mean and divide by the std to normalize the values.
then the layer's outputs are scaled and shifted using parameters learnt from training. just like how the weights are learnt. for implementing,
```python
import torch.nn as nn
import torch.nn.init as init
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(9, 16)
		self.bn1 = nn.BatchNorm1d(16)
		self.fc2 = nn.Linear(16, 8)
		self.bn2 = nn.BatchNorm1d(8)
		self.fc3 = nn.Linear(8, 1)

		init.kaiming_uniform_(self.fc1.weight)
		init.kaiming_uniform_(self.fc2.weight)
		init.kaiming_uniform_(self.fc3.weight,
							nonlinearity="sigmoid")
	def forward(self, x):
		x = nn.functional.elu(self.bn1(self.fc1(x)))
		x = nn.functional.elu(self.bn2(self.fc2(x)))
		x = nn.functional.sigmoid(self.fc3(x))
		return x
```

# Images and CNNs
## Images in `PyTorch`
images are made up of pixels. pixels are numbers containing the information about the color of the image. in grayscale images, each pixel is a number from 0 to 255, 0 being black and 1 being white.
for colored images, each pixel is a sequence of 3 numbers denoting the intensity of Red, Green and Blue respectively.
![[Pasted image 20240118232809.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240118232809.png/?raw=true)

to load images in `PyTorch` we first define the transforms we are going to perform on the images. like turning them to tensors and resizing them.
```python
from torchvision.datasets import ImageFolder
from torchvision import transforms
train_transforms = transforms.Compose([
									   transforms.ToTensor(),
									   transforms.Resize([128, 128])
])
dataset_train = ImageFolder("images path", transform=train_transforms)

dataloader = DataLoader(dataset_train, shuffle=True, batch_size=1)
image, label = next(iter(dataloader_train))
print(image.shape) #returns [1, 3, 128, 128]
```

the size is the batch size, the number of pixel colors for each image, and the width and length. we should reshape to just 3-d shapes to display the image, so,
```python
image = image.squeeze().permute(1, 2, 0)
print(image.shape) #returns [128, 128, 3]
```
squeeze removes the 1 dimension and permute changes the order from 0, 1, 2 to 1, 2, 0
![[Pasted image 20240118234308.jpg]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240118234308.png/?raw=true)
to display the image,
```python
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
```
another transformation we can apply to the images are random flips and rotations. this is known as data augmentation and is used so that models learn to ignore these random things and avoid overfitting
```python
train_transforms = transforms.Compose([
									   transforms.RandomHorizontalFlip(),
									   transforms.RandomRotation(45),
									   transforms.ToTensor(),
									   transforms.Resize([128, 128])
])
```
Data augmentation allows the model to learn from more examples of larger diversity, making it robust to real-world distortions. It tends to improve the model's performance, but it does not create more information than is already contained in the original images.

## CNNs
we can't really use the neural networks we have been using until now for images. that is because we would have a huge amount of parameters after 1 layer itself. this means training will be very slow 
![[Pasted image 20240119000028.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119000028.png/?raw=true)

also linear models can't recognize spatial patterns. if for example there is a pattern in one corner, the linear layer can't learn to recognize the same pattern in other corners.
![[Pasted image 20240119000044.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119000044.png/?raw=true)

A convolutional layer outputs feature maps which are used to compute predictions. feature maps are done by doing convolution operations on the input with a filter. the filter should be of a size with lesser dimensions than the input. then dot product is performed on the filter and the grid with matching dimensions as the filter from the input. we get this grid by sliding the filter on the input and thus all possible grids of the same dimensions are operated and the resulting grid is summed up and the result is the value of 1 space in the feature map.
![[Pasted image 20240119001334.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119001334.png/?raw=true)

here we have a 3x3 filter sliding through the input which is 5x5. each grid of 3x3 in the input is operated on and results in a singular space in the feature map. many filters can be slid through the input and each filter gives a feature map. then activations can be applied to feature maps and they can be further processed again
![[Pasted image 20240119001558.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119001558.png/?raw=true)

to make sure the feature map is of the same dimensions as the input we use padding to add a black pixels around the corners. this also makes sure corners are operated on multiple times instead of just once.
![[Pasted image 20240119001715.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119001715.png/?raw=true)

another operation on convolutions is Max Pooling. here we reduce the height and width of the output of the feature maps by picking the max value from a grid of given dimensions of the filter used for pooling. if a filter of 2x2 is used, the grid is divided into 2x2 grids and each grid outputs 1 value which is the max in that 2x2 grid. this reduces the length and width by 2.
![[Pasted image 20240119001934.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119001934.png/?raw=true)

```python
class Net(nn.Module):
	def __init__(self, num_classes):
			super().__init__()
			self.feature_extractor = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.ELU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ELU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Flatten()
			)
			self.classifier = nn.Linear(64*16*16, num_classes)
	def forward(x):
		x = self.feature_extractor(x)
		x = self.classifier(x)
		return x
```
so we first give 3 feature maps indicating the red, green and blue pixels of shape(3x64x64). then after the first layer, the number is (32x64x64). this is because each of the layer(red, blue and green) get operated on by a filter and then all the grids' values add up to give 1 single grid of values. like this, we use 32 sets of (3x3x3) filters to get 33 grids.
and after max pooling, it is (32x32x32). then after the second layer (64x32x32) and after max pooling,
(64x16x16) and this is passed to the linear layer.

## Training Image Classifiers
before applying data augmentation, we have to keep the data in mind. both the editing image and the original image are now a part of the dataset. now lets say we are classifying flowers. if we apply a color change augmentation, this might confuse the model as an important factor to classify flowers could be color. instead we may use `RandomHorizontalFlip` as an augmentation
so choose the data augmentation with the data and task in mind.
for a clouds dataset, we can use flips, rotates and color contrasts for adapting to different sunlight colors.
the training is similar to last time except now we use a new loss function called cross entropy.
```python
net = Net(num_classes=7)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):
	for image, labels in dataloader:
		optimizer.zero_grad()
		outputs = net(image)
		loss = criterion(outputs, label)
		loss.backward()
		optimizer.step()
```

## Evaluating Model Performance
to predict for test data, we create a similar test data loaders and then pass it to the model
```python
test_transforms = transforms.Compose([
									  transforms.ToTensor(),
									  transforms.Resize((64, 64))
])
dataset = ImageFolder(
					  "path",
					  transform=test_transforms
)
```
we can also use metrics other than accuracy like precision and recall. both of them can be calculated separately for all classes or we can aggregate them to give one single result. we can aggregate in 3 ways:
1. Micro average: where the true positives, true negatives and others are sum of them across all classes. then the formula is applied and precision/recall is calculated
2. Macro average: here the mean of precision across all classes is taken
3. Weighted average: here the precision is calculated across all classes and the weighted mean is taken. the weights are proportional to the size of data in each class.
![[Pasted image 20240119005347.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119005347.png/?raw=true)

to evaluate,
![[Pasted image 20240119005425.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119005425.png/?raw=true)
the predictions are calculated a little differently for images in `PyTorch`
```python
with torch.no_grad():
	for images, labels in dataloader_test:
		outputs = net(images)
		_, preds = torch.max(outputs, 1)
		metric(preds, labels)
metric.compute() 
```
and to calculate across each class average should be "none"

to see the vocab in `PyTorch`:
```python
dataset.class_to_idx.items()
```

Using larger images, more convolutional layers, and a classifier with more than one linear layer should improve the performance.

# Sequences and RNNs
## Sequences in `PyTorch`
sequential data is data ordered within time or space. some examples are
1. Time series data
2. Text
3. Audio waves
all of the above examples are ordered and this order is very important for making predictions.
unlike for other data, the train test split should not be random. we should make sure that the test data should contain future data so that we can see if our model can predict the future. this avoids look ahead bias.
to create a sequence we first define the sequence length. that is the number of data points in one training example. then we create the training data by creating all possible sequences of that sequence length while maintaining the order. for example, lets say we have a data frame of 960 rows and we decide that the sequence length is 96. so we feed in the results of rows 0-95 and the prediction that should be done is row 96 then we feed 1-96 and predict 97 and so on until we feed row 862-958 and predict row with index 959.
so we need to loop for 864 times meaning the length of the data frame - the sequence length
for a time series data, to create a sequence:
```python
import numpy as np
def create_sequence(df, seq_length):
	xs, ys = [], []
	for i in range(len(df) - seq_length):
		x = df.iloc[i:i+seq_length, 0] #96 rows. the upper limit is excluded
		y = df.iloc[i+seq_length, 1] #the row after 96 rows
		xs.append(x)
		ys.append(y)
	return np.array(xs), np.array(ys)
```
then we create a dataset from the sequence
```python
X_train, y_train = create_sequences(train_df, seq_length)

from torch.utils.data import TensorDataset
dataset = TensorDataset(torch.tensor(X_train).float(),
						torch.tensor(y_train).float())
```
this same technique also applies for speech recognition and LLMs.

## RNNs
usual neural networks only work in one direction. however, RNNs or Recurrent Neural Networks works both in the forward and backward direction. this is to remember the previous input while deciding the new output.
for example lets take a single neuron receiving the sequence one by one.
first 1 element of the sequence goes in and the neuron gives two outputs, one is the output that is further passed down to other layers like usual and the other is passed back to itself, that is the same neuron. so when the next element of the sequence come in as input, this element and the hidden output of last time is taken into consideration and the 2 outputs are decided. this goes on until the end of the sequence.
![[Pasted image 20240119160756.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119160756.png/?raw=true)

since the first input won't have any hidden input, that is considered 0 for all first inputs.
the above diagram denotes how a single neuron works over time.
depending on the lengths of the input and output sequence we can have 4 types of RNNs
1. Sequence to Sequence architecture, where if a sequence is given as input a sequence is given as output. that is for each part of a sequence there is an output     ![[Pasted image 20240119160942.png]]
2. Sequence to Vector architectures are used when we need one single output for a whole sequence of inputs. this is used in time series data and also to predict the next word, when given a sentence.                                                                                 ![[Pasted image 20240119161105.png]]
3. Vector to sequence architecture is used when we pass a single input and require a sequence of outputs. this can be used for text generation                                            ![[Pasted image 20240119161211.png]]
4. Encoder Decoder architecture is when we require a sequence as output when we give a sequence as input. however unlike sequence to sequence, the outputs are done only after the input sequence is over. this is used for decoding codes and stuff                                                                                                                                ![[Pasted image 20240119161343.png]]
all of the X marks in the above images mean that either they are discarded or they are fed in as zeroes to the model
```python
class Net(nn.Module):
	def __init__(self):
		self.rnn = nn.RNN(
		input_size=1,
		hidden_size=32,
		num_layers=2,
		batch_first=True)
		self.fc = nn.Linear(32, 1)
	def forward(x):
		h0 = torch.zeros(2, x.size(0), 32)
		out, _ = self.rnn(x, h0)
		out = self.fc(out[:, -1, :])
		return out
```
so first we pass in zeros as the first hidden layer which has the dimensions                    (2x`seq_length`x32) each sequence should have 32 0s as 32 is the hidden size.
so we have 96x32 and since there are 2 layers, (2x96x32).
now, the sequences are passed to the RNN and we get outputs for each layer in out. that is the dimensions would be 2x96x32 for out. from this we only need the last output, that is the output after the last sequence input. so the 96th output. to get that we use `out[:, -1, :]` and get the 32 outputs of the final sequence. this is passed to the linear layer which returns a single output.

## LSTM and GRU cells
one problem with plain RNNs is that they have a short term memory. all they remember is the previous part's output at a time. to introduce long term memory, two new types were used
1. LSTM(Long Short Term Memory)
2. GRU(Gated Recurrent Unit)
to visualize both of them, lets first check how RNN cells work
![[Pasted image 20240119221254.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119221254.png/?raw=true)
RNN cells take in 2 inputs and 2 outputs
LSTM has 3 inputs and 3 outputs. two of them being memory based. two of them is as usual, the input and the hidden layer of the previous part of the sequence. the third is something called the long term memory.
![[Pasted image 20240119221525.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119221525.png/?raw=true)

the long term memory goes through a gate called forget to forget unnecessary parts. then, the input gate takes in both the short term memory and the actual input and determine which are important enough to use in long term memory. and finally the output gate is used to determine y and short term memory. **here both y and h are the same.**
```python
class Net(nn.Module):
	def __init__(self):
		self.lstm = nn.LSTM(
		input_size=1,
		hidden_size=32,
		num_layers=2,
		batch_first=True)
		self.fc = nn.Linear(32, 1)
	def forward(x):
		h0 = torch.zeros(2, x.size(0), 32)
		c0 = torch.zeros(2, x.size(0), 32)
		out, _ = self.lstm(x, (h0, c0))
		out = self.fc(out[:, -1, :])
		return out
```
next is the GRU cell. it simplifies the LSTM cell by merging long term and short term memories by just having one memory.
![[Pasted image 20240119221958.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119221958.png/?raw=true)
here there is no output gate and the whole memory state is just given as output.
```python
class Net(nn.Module):
	def __init__(self):
		self.gru = nn.GRU(
		input_size=1,
		hidden_size=32,
		num_layers=2,
		batch_first=True)
		self.fc = nn.Linear(32, 1)
	def forward(x):
		h0 = torch.zeros(2, x.size(0), 32)
		out, _ = self.gru(x, h0)
		out = self.fc(out[:, -1, :])
		return out
```
to see which to use, it is always good to try both and see which is better on your data.
![[Pasted image 20240119222221.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119222221.png/?raw=true)

## Evaluating RNNs
we use Mean squared error as our loss function, since this is a regression task.
now RNNs expect data to be of 3-dimensions like 
(batch size, sequence length, number of features)
if we have 32 batches of sequence length 96 and each part of the sequence will have a number of features, here it is 1.
but the shape of the data in the data loaders is 32x96 and doesn't take the 1 feature as a shape. to change this we use `.view(32, 96, 1)`
now when we predict the outputs and pass to the loss function, we need the same shapes for both target and prediction. the predictions are of shape (batch size x 1). but the targets are of shape (batch size). so we can squeeze the data to be of lower dimensions.
```python
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(
					   net.parameters(), lr=0.01
)
#training
for epoch in range(num_epochs):
	for seqs, labels in dataloader_train:
		seqs = seqs.view(32, 96, 1)
		outputs = net(seqs).squeeze()
		loss = criterion(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
#evaluating
mse = torchmetrics.MeanSquaredError()
net.eval()
with torch.no_grad():
	for seqs, labels in test_dataloader:
		seqs = seqs.view(32, 96, 1)
		outputs = net(seqs).squeeze()
		mse(outputs, labels)
mse.compute()
```

# Multi Input and Multi Output Models
## Multi input models
multi input can be used for many cases, like when we have 2 different types of data or when we have more info on the same data, etc.
![[Pasted image 20240119225326.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119225326.png/?raw=true)

lets take a dataset containing 964 images of alphabets from 30 languages. we have two inputs - the image of the alphabet and the language it belongs to through a one hot encoded list.
in a multi input model, we first send in the inputs in their own architectures and then after a few layers, we concatenate them to give the output we need.
![[Pasted image 20240119225544.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119225544.png/?raw=true)

lets define the dataset 
```python
from PIL import image

class OmniGlotDataset(Dataset):
	def __init__(self, transform=None, samples):
		self.transform = transform
		self.samples = samples
	def __len__(self):
		return len(self.samples)
	def __getitems__(self, idx):
		img_path, alphabet, label = self.samples[idx]
		img = Image.open(img_path).convert("L")
		img = self.transform(img)
		return img, alphabet, label

dataset_train = OmniglotDataset(
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
    ]),
    samples=samples,
)

dataloader_train = DataLoader(
    dataset_train, shuffle=True, batch_size=3,
)
```
and now the neural networks
```python
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.image_layer = nn.Sequential(
		nn.Conv2d(1, 16, kernel_size=3, padding=1),
		nn.MaxPool2d(kernel_size=2),
		nn.ELU(),
		nn.Flatten(),
		nn.Linear(16*32*32, 128)
		)
		self.alphabet_layer = nn.Sequential(
		nn.Linear(30, 8),
		nn.ELU()
		)
		self.classifier = nn.Linear(128+8, 964)
	def forward(self, x_image, x_alphabet):
		image_x = self.image_layer(x_image)
		alphabet_x = self.alphabet_layer(x_alphabet)
		x = torch.cat((image_x, alphabet_x), dim=1) #column wise concat
		return self.classifier(x)
```
finally the training
```python
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

for epoch in range(10):
	for img, alpha, labels in dataloader_train:
		optimizer.zero_grad()
		outputs = net(img, alpha)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
```

## Multi Output models
just like multi input, multi output is also used in many places.
![[Pasted image 20240119231815.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119231815.png/?raw=true)

lets take the same data as before and predict both the character and the language it belongs to. the architecture is first an image layer giving some embeddings and using those embeddings for both ,predicting the character and the language.
![[Pasted image 20240119231942.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119231942.png/?raw=true)

the dataset can be defined the same as before. the neural network as a small change
```python
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.image = nn.Sequential(
		nn.Conv2d(1, 16, kernel_size=3, padding=1),
		nn.MaxPool2d(kernel_size=2),
		nn.ELU(),
		nn.Flatten(),
		nn.Linear(16*32*32, 128)
		)
		self.classifier_alpha = nn.Linear(128, 30)
		self.classifier_char = nn.Linear(128, 964)
	def forward(self, x):
		image_x = self.image(x)
		alpha_labels = self.classifier_alpha(image_x)
		char_labels = self.classifier_char(image_x)
		return alpha_labels, char_labels
```
and finally the training
```python
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

for epoch in range(10):
	for img, alpha_l, char_l in dataloader_train:
		optimizer.zero_grad()
		output_alpha, output_char = net(img)
		loss_alpha = criterion(output_alpha, alpha_l)
		loss_char = criterion(output_char, char_l)
		loss = loss_alpha + loss_char
		loss.backward()
		optimizer.step()
```

## Loss Weighting
the evaluating part of multi output models is the same as usual, except now we calculate the metrics for both outputs.
recalling the training loop from before, we added both losses so that we weigh both losses equally. if we want the model to focus on one loss more than the other, we can use loss weighting, where we multiply the loss of interest with a weight.
usually we multiply with weights that add up to 1
![[Pasted image 20240119233545.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119233545.png/?raw=true)

however, if loss weighting is used, we must remember to equally scale both losses before weighting. this is to make sure that one loss isn't negligible compared to the other.
![[Pasted image 20240119233727.png]](https://github.com/Golden-Exp/FastAi/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240119233727.png/?raw=true)

