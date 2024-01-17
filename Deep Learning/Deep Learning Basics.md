#datacamp #deeplearning
# `PyTorch`
## Deep learning
deep learning is a subset of machine learning, where the model finds patterns in data by itself and not by feature engineering like in traditional ml algorithms.
deep learning can be applied for big data like images, audios and even tabular.
the fundamental model structure in deep learning is the neural network.
![[Pasted image 20240116111937.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116111937.png/?raw=true)

`PyTorch` is a deep learning framework used more than any other framework. 
```python
import torch
```
just like `numpy`'s arrays we have tensors in `pytorch` it has most of the attributes arrays have.

## Neural networks
```python
import torch.nn as nn
input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])
linear_layer = nn.Linear(in_features=3, out_features=2)
output = linear_layer(input_tensor)
print(output) #prints a tensor with 2 elements
linear_layer.bias, linear_layer.weight
```
the Linear layer takes an input and applies a linear function to it and returns a number of outputs as mentioned while creating the layer.
each layer has a weight and bias attribute.
the linear layer matrix multiplies the inputs to the weights and adds the bias and returns them as outputs.
![[Pasted image 20240116112813.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116112813.png/?raw=true)

so if we have inputs as a 1x3 tensor it gets matrix multiplied by a 3x2 matrix with random weights and the remaining 2 elements in the 1x2 matrix gets added to the bias and returned as a tensor.
we can add multiple layers using Sequential.
```python
model = nn.Sequential(
					  nn.Linear(10, 18),
					  nn.Linear(18, 20),
					  nn.Linear(20, 5)
)
```

## Activation Functions
to add non linearity to our functions we use activation functions. this is because even with multiple linear layers, its still linear in the end. we need to be flexible enough to adapt to any function.
![[Pasted image 20240116113725.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116113725.png/?raw=true)

one of these is the sigmoid function. sigmoid transforms any input into a number between 0 and 1. this is useful for classifications and the number can be used as a probability for that particular category. a threshold of 0.5 can be used, so if the number is above 0.5, it can be classified as 1.
```python
import torch
import torch.nn as nn
model = nn.Sequential(
					  nn.Linear(10, 4),
					  nn.Linear(4, 2),
					  nn.Sigmoid()	
)
```
sigmoid is used for binary classification. so it can use sigmoid for one category and 1 - the answer for the other category.
for many categories we use Softmax.
![[Pasted image 20240116114302.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116114302.png/?raw=true)

this is basically using sigmoid for finding probabilities and to make them add up to 1, we pass it to the exponent function and divide by the sum of all exponents of all categories' probabilities. this makes sure that higher probabilities give higher results and all of them add up to 1.
```python
import torch
import torch.nn as nn
model = nn.Sequential(
					  nn.Linear(10, 4),
					  nn.Linear(4, 2),
					  nn.Softmax(dim=-1)	
)
```

# Training
## Forward pass
When we pass input data through a neural network in the forward direction to generate outputs, or predictions, the input data flows through the model layers. At each layer, computations performed on the data generate intermediate representations, which are passed to each subsequent layer until the final output is generated. The purpose of the forward pass is to propagate input data through the network and produce predictions or outputs based on the model's learned parameters (weights and biases).
there are two passes in a training loop, the forward and the backward pass.
backward passes are used to modify the network's weights after a forward pass, by comparing the predictions from the forward pass to the actual values.

for **binary classification**, a forward pass consists of multiple layers and then a sigmoid function to get the outputs, for each row.
![[Pasted image 20240116115951.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116115951.png/?raw=true)

for **multi class classification**, a forward pass consists of multiple layers and then a softmax function to restrict the probabilities to sum up to 1. note that the final layer before softmax should have to give out the same number of outputs as the number of classes.
![[Pasted image 20240116120009.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116120009.png/?raw=true)

for **regression**, we won't have any activations and just the layers, remain to give a single number for each row.
![[Pasted image 20240116120126.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116120126.png/?raw=true)

## Loss Functions
loss functions are used to assess how the model is performing with its current coefficients. the loss is calculated by comparing the actual values to the predictions. if the predictions are way off from the actuals, the loss gets higher. our goal is to reduce the loss.
lets say we have a probability for each class we want to classify. we also know which category the row corresponds to. so we need to compare the probability of the actual class the row corresponds to. but the actual class would just be an integer, that is the index of its position in the list of categories(vocab). to convert the integer into an array to compare, we use `one_hot` encoding.
![[Pasted image 20240116121331.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116121331.png/?raw=true)

```python
import torch.nn.Functional as F
F.one_hot(torch.tensor(2), num_classes=3) #returns tensor([0, 0, 1])
```
so we pass the index of the category to get an array of 0s with a 1 being in the index given.
this is used to then calculate the loss function which is usually the cross entropy function for classification.
```python
from torch.nn import CrossEntropyLoss
criterion = CrossEntropyLoss()
criterion(predictions.double(), one_hot_targets.double())
```
`.double()` is used to convert into a double datatype that is used in this function.

## Derivatives for upgrading
derivative of a particular point is used to see how much a function changes(increases or decreases) when we change the point there.
derivative of a point with respect to the function gives the rate of change of the function at that point.
so derivative of a weight with respect to the loss calculated gives the rate at which the loss changes when the weight is changed.
so we calculate the derivatives or slopes of all weights in all layers with respect to the loss. this is done in the back propagation step.
![[Pasted image 20240116221642.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116221642.png/?raw=true)

![[Pasted image 20240116125824.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116125824.png/?raw=true)

this gradient calculation can be done by calling `loss.backward()` in `pytorch`.
```python
loss = criterion(preds, target)
loss.backward()
model[0].weight.grad #each layer can be accessed by their index
```
then we update the weights according to these using learning rates
![[Pasted image 20240116130303.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116130303.png/?raw=true)

most of the loss functions are non convex. meaning there might be multiple local minimums.
![[Pasted image 20240116222443.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116222443.png/?raw=true)

so sometimes we may land in a local minimum and think that is the lowest point of the function but actually, it isn't. To find the global minima of functions like these, we use gradient descent. to update weights we use optimizers. most common optimizer is the stochastic Gradient.
```python
import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.step()
```
this optimizer calculates gradients and updates the coefficients for us, using SGD, when we use step

## Training Loop
![[Pasted image 20240116224717.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240116224717.png/?raw=true)
we'll try this on a regression model. for regression we use a loss called mean squared error, which is the difference between the predictions and the targets squared and averaged.  this is also known as L2 loss. mean absolute error is known as L1 loss.
```python
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(torch.tensor(features).float(),
						torch.tensor(target).float())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = nn.Sequential(
					  nn.Linear(4, 2),
					  nn.Linear(2, 1))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
	for data in dataloader:
		optimizer.zero_grad()
		feature, target = data
		pred = model(feature)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
show_results(model, dataloader) -> shows actual and predicted values.
```
the data is loaded as a Dataset and then into a data loader which splits the data into batches. since batch size here is 4, each batch would have 4 rows of data(if its tabular)
then we create the model and set the loss function and the optimizer. then for the training loop, we train for a certain number of epochs. for each epoch we take 1 batch from the data loader and calculate the preds for the whole batch and calculate loss with it and the targets. then we calculate the gradients and update the parameters using the optimizer. 

# Neural Network Architectures
## Activation functions between layers
remember that activation functions like sigmoid and softmax have values between 0 and 1. so their derivatives will also be lower, and thus the change in weights will be too small to count. This behavior is called saturation. This property of the sigmoid function creates a challenge during backpropagation because each local gradient is a function of the previous gradient. For high and low values of x, the gradient will be so small that it can prevent the weight from changing or updating. This phenomenon is called vanishing gradients and makes training the network challenging.
![[Pasted image 20240117190449.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117190449.png/?raw=true)

so for an alternative between layers we use the `Relu` function. `ReLU` is the max(x, 0), that is if x is negative 0 is given as output, else the number itself.
![[Pasted image 20240117192427.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117192427.png/?raw=true)

there is also something known as a leaky `ReLU`, where the negative numbers are multiplied by a small coefficient like 0.01. this makes sure that all the gradients are non null.
![[Pasted image 20240117193812.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117193812.png/?raw=true)
These types of activation layers add a non linearity to the model and can adapt to more complex functions. 

## NN Architecture 
when the neurons in the NN are fully connected to each other, it is known as a fully connected neural network. that is, the value of a neuron is calculated by taking into account all of the neurons in the previous layers. we always use fully connected layers.
![[Pasted image 20240117195046.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117195046.png/?raw=true)

take each neuron to be the output of its layer. for example, in the below diagram, the input is `nn.Linear(4, 4)` where the first 4 is the number of features. then the second and third is also 4,4 and finally the final layer is `nn.Linear(4, 2)`.
![[Pasted image 20240117195131.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117195131.png/?raw=true)

now if we take the values of each neurons, it can be calculated by the number of neurons in the previous layer and bias. so the number of learnable parameters for a neuron is N(number of neurons previously) + 1(bias).
to count the parameters in `PyTorch` we use the `numel` function
```python
total = 0
for parameter in model.parameters():
	total += parameter.numel()
```
![[Pasted image 20240117200128.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117200128.png/?raw=true)

## Learning Rate and Momentum
the optimization technique SGD has two hyper parameters:
1. The Learning Rate 
2. The Momentum
the learning rate determines the size of steps we take towards the minimum point of the loss function. if too small, it might take many epochs to get to the minimum, but if too large we might pass over the minimum.
![[Pasted image 20240117201212.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117201212.png/?raw=true)

![[Pasted image 20240117201304.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117201304.png/?raw=true)

notice that in the below diagram, as we get closer to the minimum, the step size also decreases, this is because the gradients also get lesser and lesser and since the step size is gradient * LR, we get small step sizes.

momentum is most useful with non convex functions, where as we approach the local minima, the step size decreases and we need a momentum to overcome the local minima.
![[Pasted image 20240117201523.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117201523.png/?raw=true)

![[Pasted image 20240117201537.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117201537.png/?raw=true)
the momentum keeps the step size large, even when the gradient is small, if the previous step size was large.
![[Pasted image 20240117201655.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117201655.png/?raw=true)
**"In summary, two parameters of an optimizer can be adjusted when training a model: the learning rate and the momentum. The learning rate controls the step size taken by the optimizer. Typical learning rate values range from ten raised to minus two, to ten raised to minus four. If the learning rate is too high, the optimizer may never be able to minimize the loss function. If it is too low, training may take longer. The other parameter, momentum, controls the inertia of the optimizer. Without momentum, the optimizer may get stuck in a local optimum. Momentum usually ranges from zero-point-eighty-five to zero-point-ninety-nine."**

***A good rule of thumb is to start with a learning rate of 0.001 and a momentum of 0.95.***
## Transfer Learning and Finetuning
the initial layer weights are decided at random and are small so that the neurons don't explode with larger inputs. if we want a particular distribution to sample from for the weights we can use the `init` function
```python
layer = nn.Linear(64, 128)
nn.init.uniform_(layer.weight)
```
these type of initializations are only rarely useful. most of the time we use transfer learning, that is we use a previously trained model and train it on a new dataset. since we use pretrained weights, it will be more efficient by cutting time. to save and load,
```python
import torch
layer = nn.Linear(64, 128)
torch.save(layer, 'layer.pth')

new_layer = torch.load('layer.pth')
```

a version of transfer learning is finetuning, where we use a smaller learning rate. we can also **freeze** a part of the NN and only train newly added parts which are usually closer to the output layer. to freeze in `PyTorch`,
```python
import torch.nn as nn
model = nn.Sequential(nn.Linear(64, 128),
					  nn.Linear(128, 256))
for name, param in model.parameters():
	if name == '0.weight':
		param.requires_grad = False
```
now if we train, only the un-frozen layers will be trained.

# Evaluating and Improving models
## Loading Data
to create a dataset first, we extract the necessary columns from the data frame and turn them into arrays. then we pass them to`TensorDataset` to turn them into datasets, which contains pairs of dependent and independent variables.
```python
from torch.utils.data import TensorDataset
dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
dataset[0] #returns a tuple of 2 elements(dep and indep)
```
then we pass the dataset to the data loaders, where the data is split into batches. we can specify the batch size and the data will be split accordingly. 
```python
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for inputs, outputs in dataloader:
	print("inputs: ", inputs)
	print("outputs: ", outputs)
```

## Evaluating model performance
two important metrics that can be used are the loss and the accuracy.
the training and valid loss are two separate loss and need to be calculated separately.
for training loss, it is calculated for each epoch. so each loss calculated for a batch is averaged and returned at the end of the epoch.
```python
for epoch in range(num_epochs):
	training_loss = 0.0
	for data in dataloader:
		#forward pass
		optimizer.zero_grad()
		feature, target = data
		pred = model(feature)
		loss = criterion(pred, target)
		#loss
		training_loss += loss
		#backward
		loss.backward()
		optimizer.step()
	epoch_loss = training_loss/len(dataloader)
```
for validation loss its a little different because some layers behave differently for valid sets.
```python
validation_loss = 0.0
model.eval()  #switches the model to evaluation mode
with torch.no_grad():
	for i, data in enumerate(validationdataloader, 0):
		feature, target = data
		pred = model(feature)
		loss = criterion(pred, target)
		validation_loss += loss
epoch_loss = validation_loss/len(validationdataloader)
model.train() #switching back to train mode
```

valid loss helps to detect overfitting. this can be seen by the rise of validation loss, but the training loss getting better.
![[Pasted image 20240117224207.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117224207.png/?raw=true)

for calculating accuracy,
```python
import torchmetrics
metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
for data in dataloader:
	features, target = data
	outputs = model(features)
	acc = metric(outputs, target.argmax(dim=-1)) #argmax given if one hot encoded labels
	acc = metric.compute()
	metric.reset()
```

## Fighting overfitting
overfitting can have 3 causes:
1. the data is too small
2. the weights are too high
3. model has too much capacity
all three of them can be countered by:
1. data augmentation
2. weight decay
3. dropouts
data augmentation is the process of rotating/cropping the same images multiple times and using that as an input to the model. this is used when the data is too small and getting extra data is costly
![[Pasted image 20240117225743.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240117225743.png/?raw=true)

dropout is a regularization technique where some random neurons are shut down to become 0 so that the model doesn't heavily depend on one neuron. this should be done carefully, coz its behavior is different in different modes
```python
model = nn.Sequential(
					  nn.Linear(6, 4),
					  nn.ReLU(),
					  nn.Dropout(p=0.5))
```
p value is the probability that a neuron might be shut down. dropouts are mostly given after activations.
weight decay is another form of regularization, where, a penalty is added to the loss such that the penalty is proportional to weights. this keeps the weights in check. `weight_decay` is the constant that is multiplied to the weights before adding to the loss.
```python
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

## improving model
we can choose LR and momentum using hyperparameter tuning by randomly sampling from the uniform distribution.
```python
values = []
for idx in range(10):
    factor = np.random.uniform(2, 4)
    lr = 10 ** -factor
    momentum = np.random.uniform(0.85, 0.99)
    values.append((lr, momentum))
```
