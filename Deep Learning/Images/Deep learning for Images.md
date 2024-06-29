#deeplearning #datacamp 

```python
from torchvision import datasets
import torchvision.transforms as transforms

train_dataset = ImageFolder(root=train_dir, transform=transforms.ToTensor())
train_dataset.classes #gives number of categories
train_dataset.class_to_idx #vocab

class BinaryCNN(nn.Module):
	def __init__(self):
		super(BinaryCNN, self).__init__()
		self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(16*height*width, 1)
		self.sig = nn.Sigmoid()
	def forward(self, x):
		x = self.pool(self.relu(self.conv(x)))
		x = self.fc1(self.flatten(x))
		x = self.sig(x)
		return x
```

multiple layers of CNN
```python
conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
model = BinaryCNN()
model.add_module('conv2', conv2)
```
`nn.Sequential` can also be used to pass all layers into a single block.

```python
from torchvision.transforms import functional as F
image = PIL.Image.open("dog.png")
num_channels = F.get_image_num_channels(image)
print(num_channels)
```

saving models
```python
torch.save(model.state_dict(), "BinaryCNN.pth")
new_model = model()
new_model.load_state_dict(torch.load('BinaryCNN.pth'))
```
pre trained models
```python
from torchvision.models import (
								resnet18, ResNet18_Weights
) 
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
transform = weights.transforms()

im_tensor = transform(image)
image_reshaped = im_tensor.unsqueeze(0)

model.eval()
with torch.no_grad():
	pred = model(image_reshaped).squeeze(0)
pred_cls = pred.softmax(0)
cls_id = pred_cls.argmax().item()
cls_name = weights.meta["categories"][cls_id] #predicted category
```

# Object Detection
```python
from torchvision.utils import draw_bounding_boxes

bbox = [x_min, y_min, x_max, y_max]
bbox = bbox.unsqueeze(0)
bbox_image = draw_bounding_boxes(
								 image_tensor, bbox, width=3, colors="red"
)
transform = transforms.Compose([
								transforms.ToPILImage()
])
img = transform(bbox_image)
plt.imshow(img)
```
ROI - Region Of Interest
IOU - Intersection over Union - measure of how accurate the bounding box predicted by the model is
0 - very bad
1 - perfect
```python
from torchvision.ops import box_iou
iou = box_iou(bbox1, bbox2)
print(iou)
```

```python
model.eval()
with torch.no_grad():
	output = model(image) #dictionary with different box predictions and their confidence and labels
	boxes = output[0]["boxes"]
	scores = output[0]["scores"]
```
NMS - Non Max Suppression - technique used to select the most relevant bounding box.
it discards boxes with low confidence scores
```python
from torchvision.ops import nms
box_indices = nms(
				  boxes=boxes,
				  scores=scores,
				  iou_threshold=0.5
)
filtered_boxes = boxes[box_indices]
```
the output is the indices of the filtered boxes

R-CNN for object detection
3 modules in Regional CNN
1. generation of region proposals
2. extracts features from the proposed region using CNN
3. features used to predict category
![[Pasted image 20240216081128.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240216081128.png/?raw=true)

using a pretrained model as backbone(core CNN architecture) is very useful here.
the model has a features block for generating features, pooling layer to reduce size and linear layer for classification. we only need the features block for feature extraction.
```python
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
vgg = vgg16(weights=VGG16_Weights.DEFAULT)
backbone = nn.Sequential(
						 *list(vgg.features.children())
)
```
`.features` is the features layer and `.children()` extracts the layers. we convert the layers into a list and unpack them using the * symbol.
the reason why we are extracting each layer instead of just passing features to a new sequential, is because features is already sequential so it will just be a sequential in a sequential.
now from the features we need to have two layers, one for classification and the other for regression(bounding boxes)
```python
input_dimension = nn.Sequential(*list(vgg.classifier.children()))[0].in_features #extracting input dimension
classifier = nn.Sequential(
						   nn.Linear(input_dimension, 512),
						   nn.ReLU(),
						   nn.Linear(512, num_classes)
)
regressor = nn.Sequential(
						  nn.Linear(input_dimension, 32),
						  nn.ReLU(),
						  nn.Linear(32, 4)
)
```
all of them together
```python
class ObjectDetectorCNN(nn.Module):
	def __init__(self):
		super(ObjectDetectorCNN, self).__init__()
		vgg = vgg16(weights=VGG16_Weights.DEFAULT)
		self.backbone = nn.Sequential(
						 *list(vgg.features.children()))
		input_dimension = nn.Sequential(*list(vgg.classifier.children()))
		[0].in_features 
		self. classifier = nn.Sequential(
								   nn.Linear(input_dimension, 512),
								   nn.ReLU(),
								   nn.Linear(512, num_classes))
		self.regressor = nn.Sequential(
								  nn.Linear(input_dimension, 32),
								  nn.ReLU(),
								  nn.Linear(32, 4))
	def forward(self, x):
		features = self.backbone(x)
		cls = self.classifier(features)
		bbox = self.regressor(features)
		return bbox, cls
```

Anchor box - predefined bounding boxes with different shapes and aspect ratios.
![[Pasted image 20240217220436.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240217220436.png/?raw=true)
Faster R-CNN
follows a different architecture
it consists of
1. Backbone(CNNs)
2. RPN(Regional Proposal Network) - generates bounding boxes
3. Classifier and Regressor

![[Pasted image 20240217220456.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240217220456.png/?raw=true)

first anchor boxes are generated over potential regions and then the classifier and regressor predict if an object is present and the coordinates of the bounding box. then finally ROI pooling is used to finetune the boxes, then it is connected to fully connected layers where we predict the object
![[Pasted image 20240217220711.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240217220711.png/?raw=true)

```python
#creating an R-CNN Model
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
anchor_generator = AnchorGenerator(
					sizes=((32, 64, 128), ),
					aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = MultiScaleRoIAlign(
					featmap_names=["0"],
					output_size=7,
					sampling_ratio=2)
backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
backbone.out_channels = 1280
model = FasterRCNN(
				   backbone=backbone,
				   num_classes=num_classes,
				   rpn_anchor_generator=anchor_generator,
				   box_roi_pool=roi_pooler)

#using a pretrained RCNN model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model=torchvision.models.detection.fasterrcnn_resenet50_fpn(weights="DEFAULT")

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
#replace the box predictor with one with desired number of classes
model.roi_heads.box_predictor = FasterRCNNPredictor(in_features, num_classes)
```

![[Pasted image 20240217224235.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240217224235.png/?raw=true)

# Image Segmentation
image segmentation is the process of classifying each pixel into multiple segments like object, foreground, background, etc.
there are 3 types
1. semantic: each object is assigned a particular class.![[Pasted image 20240217224825.png]]
2. instance: here, objects of one particular class is separated. ![[Pasted image 20240217224948.png]]
3. Panoptic: here, instance and semantic is combined to give different types of segmentation among the same and different classes![[Pasted image 20240217225059.png]]
to create a "mask", we assign just the object's pixels 1 and others 0. this is our mask. to get the object alone from the original, multiply with the mask
![[Pasted image 20240217225247.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240217225247.png/?raw=true)


for Instance Segmentation we use R-CNN to predict a third thing - mask
![[Pasted image 20240217230142.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240217230142.png/?raw=true)

```python
from torchvision.models.detection import  maskrcnn_resnet50_fpn

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = Image.open("image.png")
transform = transforms.Compose([
			transforms.ToTensor()
])
image_t = transform(image).unsqueeze(0)
with torch.no_grad():
	prediction = model(image_t)
```
prediction is a dictionary with keys like "labels", "scores" and "masks". they also have bounding boxes
the masks produced by mask RCNN is a soft mask. each pixel is a probability that the pixel belongs to an object. we can create binary masks from this by assigning thresholds

```python
masks = prediction[0]["masks"]
labels = prediction[0]["labels"]

for i in range(2):
	plt.imshow(image)
	plt.imshow(
	masks[i, 0],
	cmap="jet",
	alpha=0.5
	)
```
![[Pasted image 20240217233125.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240217233125.png/?raw=true)

the model predicts the labels available in the image and each label has its own mask.
the shape of the masks tensor is (number of labels x 1 x height x width)

U - Net is used for semantic segmentation.
![[Pasted image 20240218225750.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240218225750.png/?raw=true)

![[Pasted image 20240218233625.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240218233625.png/?raw=true)

first, the original image is passed into a CNN to get some feature maps and is down sampled(max pooling) to get lower resolution images with very much info extracted. repeat this process a few times for encoder. then we transition to the decoder with the bottleneck which also down samples and then up samples(transpose convolutions).
transpose convolutions increase the dimensions of the feature maps by adding a few layers of 0s around the image.
now after the bottleneck, we have really detailed feature maps with very less spatial clarity. so we concatenate the corresponding feature maps of the encoder to the feature maps generated before and pass them to the next layer together. this increases the accuracy of segmentation.
```python
class UNet(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(UNet, self)
		self.enc1 = self.conv_block(in_channels, 64)
		self.enc2 = self.conv_block(64, 128)
		self.enc3 = self.conv_block(128, 256)
		self.enc4 = self.conv_block(256, 512)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
		self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
		self.dec1 = self.conv_block(512, 256)
		self.dec2 = self.conv_block(256, 128)
		self.dec3 = self.conv_block(128, 64)
		self.out = nn.Conv2d(64, out_channels, kernel_size=1)
	def forward(x):
		x1 = self.enc1(x)
		x2 = self.enc2(self.pool(x1))
		x3 = self.enc3(self.pool(x2))
		x4 = self.enc4(self.pool(x3))
		x = self.upconv3(x4)
		x = torch.cat([x, x3], dim=1)
		x = self.dec1(x)
		x = self.upconv2(x)
		x = torch.cat([x, x2], dim=1)
		x = self.dec2(x)
		x = self.upconv1(x)
		x = torch.cat([x, x1], dim=1)
		x = self.dec1(x)
		return self.out(x)
	def conv_block(self, in_channels, out_channels):
		return nn.Sequential(
		nn.Conv2d(in_channels, out_channels),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_channels, out_channels),
		nn.ReLU(inplace=True))
```

panoptic segmentations involve using both instance and semantic
![[Pasted image 20240219001425.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240219001425.png/?raw=true)

![[Pasted image 20240219001443.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240219001443.png/?raw=true)

```python
model = UNet()

with torch.no_grad():
	semantic_masks = model(image_tensor)
semantic_mask = torch.argmax(semantic_masks, dim=1)

model = MaskRCNN()

with torch.no_grad():
	instance_masks = model(image_tensor)[0]["masks"]

panoptic_mask = torch.clone(semantic_mask)
instance_id = 3
for mask in instance_masks:
	panoptic_mask[mask > 0.5] = instance_id
	instance_id +=1
```
`argmax` is done for semantic masks to convert it into a single image with multiple objects masked instead of separate masks. argmax goes through each pixel in every mask and returns the index of whichever mask has the highest confidence. that index acts as the id and as the differentiating color for that object
for each mask a unique id is assigned to the pixels with more confidence. so that we can differentiate between objects. the id starts from 3 because semantic has 3 things segmented already. the objects, floor and sky(for this image)
![[Pasted image 20240219002052.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240219002052.png/?raw=true)

# Image Generation
GANs are used to generate images. Generative Adversarial Networks are used to create images from noice - picked from a standard normal distribution
![[Pasted image 20240219193311.png]](https://github.com/Golden-Exp/DataCamp/blob/main/Deep%20Learning/Attachments/Pasted%20image%2020240219193311.png/?raw=true)

The generator tries to create a realistic image that the discriminator can't differentiate and the discriminator tries its best to classify if the image is fake or not. this conflicting approach helps train both the generator and the discriminator better.
below is a simple linear generator
```python
class Generator(nn.Module):
	def __init__(self, in_dim, out_dim):
			super(Generator, self).__init__()
			self.generator = nn.Sequential(
				gen_block(in_dim, 256),
				gen_block(256, 512),
				gen_block(512, 1024),
				nn.Linear(1024, out_dim),
				nn.Sigmoid())
	def forward(x):
		return self.generator(x)
	def gen_block(self, in_dim, out_dim):
		return nn.Sequential(
			nn.Linear(in_dim, out_dim),
			nn.BatchNorm1d(out_dim),
			nn.ReLU(inplace=True))

class Discriminator(nn.Module):
	def __init__(self, in_dim):
		super(Discriminator, self).__init__()
		self.disc = nn.Sequential(
			disc_block(in_dim, 1024),
			disc_block(1024, 512),
			disc_block(512, 256),
			nn.Linear(256, 1))
	def forward(x):
		return self.disc(x)
	def disc_block(self, in_dim, out_dim):
		return nn.Sequential(
			nn.Linear(in_dim, out_dim),
			nn.leakyReLU(0.2))
```

to create GANs with CNNs
DCGANs(Deep Convolutional GANs)
- use only strided convolutions(stride > 1)
- don't using pooling or linear layers
- use batch normalization
- use RELU and tanh for last layer
- leaky RELU in discriminator

for generator we need to up sample from noise to an image with features. so we use transpose convolutions.
```python
class DCGenerator(nn.Module):
	def __init__(self, in_dim, kernel_size=4, stride=2):
		super(Generator, self).__init__()
		self.in_dim = in_dim
		self.gen = nn.Sequential(
			dc_gen_block(in_dim, 1024, kernel_size, stride),
			dc_gen_block(1024, 512, kernel_size, stride),
			dc_gen_block(512, 256, kernel_size, stride),
			nn.Conv2dTranspose2d(256, 3, kernel_size, stride=stride),
			nn.Tanh())
	def forward(x):
		x = x.view(len(x), self.in_dim, 1, 1)
		return self.gen(x)
	def dc_gen_block(in_dim, out_dim, kernel_size, stride):
		return nn.Sequential(
			nn.ConvTranspose2d(
				in_dim, 
				out_dim,
				kernel_size,
				stride=stride),
			nn.BatchNorm2d(out_dim),
			nn.ReLU())

class Discriminator(nn.Module):
	def __init__(self, kernel_size=4, stride=2):
		super(Discriminator, self).__init__()
		self.disc = nn.Sequential(
			dc_disc_block(3, 512, kernel_size, stride),
			dc_disc_block(512, 1024, kernel_size, stride),
			nn.Conv2d(1024, 1, kernel_size, stride=stride))
	def forward(x):
		x = self.disc(x)
		return x.view(len(x), -1)
	def dc_disc_block(in_dim, out_dim, kernel_size=4, stride=2):
		return nn.Sequential(
			nn.Conv2d(
				in_dim, 
				out_dim,
				kernel_size,
				stride=stride),
				nn.BatchNorm2d(out_dim),
				nn.LeakyReLU(0.2))
```
each subsequent transpose convolution is used to create detailed features, while also increasing the spatial dimensions and finally we have one that gives 3 feature maps, red, green and blue.

training GANs by defining loss functions
```python
def gen_loss(gen, disc, num_images, z_dim):
	noise = torch.randn(num_images, z_dim)
	fake = gen(noise)
	disc_pred = disc(fake)
	criterion = nn.BCEWithLogitsLoss()
	loss = criterion(disc_pred, torch.ones_like(disc_pred)) 
	#calculates loss between prediction and a tensor of ones. if prediction is real(label 1) then loss is low. else loss is high
	return loss

def disc_loss(gen, disc, real, num_images, z_dim):
	criterion = nn.BCEWithLogitsLoss()
	noise = torch.randn(num_images, z_dim)
	fake = gen(noise)
	disc_pred_fake = disc(fake)
	fake_loss = criterion(disc_pred_fake, torch.zeros_like(disc_pred_fake))
	disc_pred_real = disc(real)
	real_loss = criterion(disc_pred_real, torch.ones_like(disc_pred_real))
	return (real_loss + fake_loss)/2

for epoch in range(num_epochs):
	for real in dataloader:
		cur_batch_size = len(real)
		disc_opt.zero_grad()
		disc_loss = disc_loss(gen, disc, real, cur_batch_size, z_dim=16)
		disc_loss.backward()
		disc_opt.step()
		gen_opt.zero_grad()
		gen_loss = gen_loss(gen, disc, cur_batch_size, z_dim=16)
		gen_loss.backward()
		gen_opt.step()
```

generating images
```python
noise = torch.randn(num_images, 16)
with torch.no_grad():
	fake = gen(noise)
for i in range(num_images):
	image_tensor = fake[i, :, :, :]
	image_tensor = image_tensor.permute(1, 2, 0)
	plt.imshow(image_tensor)
	plt.show()
```

Fréchet Inception distance is used as a metric to see how good the fake image is.
we extract features from both real and fake images using a pretrained model called inception. then we calculate the means and co-variances for both and see the Fréchet distance between them. if its low, its good. usually its lower than 10 to be decent
```python
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64)#extracting 64th layer from model
fid.update((fake * 255).to(torch.uint8), real=False)
fid.update((real * 255).to(torch.uint8), real=True)
fid.compute()
```