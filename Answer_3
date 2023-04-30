from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
sample_submission = pd.read_csv('/prediction.csv')
train = pd.read_csv('/train_val.csv')
test = pd.read_csv('/test.csv')
train.head()
train_img = []
for img_name in tqdm(train['image_index']):
    image_path = '/train_val/' + str(img_name) + '.png'
    img = imread(image_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    train_img.append(img)
train_x = np.array(train_img)
train_y = train['type'].values
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
encoder = preprocessing.LabelEncoder()
train_x = train_x.reshape(800, 1, 128, 128)
train_x  = torch.from_numpy(train_x)
train_y = encoder.fit_transform(train_y)
train_y = train_y.astype(int);
train_y = torch.from_numpy(train_y)
train_x.shape, train_y.shape
val_x = val_x.reshape(200, 1, 128, 128)
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)
val_x  = torch.from_numpy(val_x)
val_y = encoder.fit_transform(val_y)
val_x.shape, val_y.shape
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),        )
        self.linear_layers = Sequential(
            Linear(4 * 32 * 32, 10)        )   
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
model = Net()
optimizer = Adam(model.parameters(), lr=0.07)
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
def train(epoch):
    model.train()
    tr_loss = 0
    x_train, y_train = Variable(train_x), Variable(train_y)
    y_train = y_train.long()
    x_val, y_val = Variable(val_x), Variable(val_y)
    y_val = y_val.long()
    if torch.cuda.is_available():
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()
        x_train = x_train.cuda()
    optimizer.zero_grad()
    output_train = model(x_train)
    output_val = model(x_val)
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train)
    val_losses.append(loss_val)
    loss_train.backward()
    optimizer.step()
n_epochs = 90
train_losses = []
val_losses = []
for epoch in range(n_epochs):
    train(epoch)
with torch.no_grad():    
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()
with torch.no_grad():
    output = model(train_x)
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
accuracy_score(train_y, predictions)
with torch.no_grad():
    output = model(val_x)
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
accuracy_score(val_y, predictions)
test_img = []
for img_name in tqdm(test['image_index']):
    image_path = 'Chart/test/' + str(img_name) + '.png'
    img = imread(image_path, as_gray=True)
    img /= 255.0
    img = img.astype('float32')
    test_img.append(img)
test_x = np.array(test_img)
test_x  = torch.from_numpy(test_x)
test_x.shape
test_x = test_x.reshape(50, 1, 128, 128)
with torch.no_grad():
    output = model(test_x)
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)
predictions = encoder.inverse_transform(predictions)
sample_submission['type'] = predictions
sample_submission.to_csv('/pred.csv', index=False)
