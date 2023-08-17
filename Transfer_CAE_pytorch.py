import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(2)
np.random.seed(1)
random.seed(1254)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class EncoderX(nn.Module):
    def __init__(self, depth, num_filter):
        super(EncoderX, self).__init__()
        self.depth = depth
        self.num_filter = num_filter
        self.layers = nn.ModuleList()

        for i in range(depth):
            if i != 0:
                input_channels = num_filter * 2
            else:
                input_channels = 3 

            conv1 = nn.Conv2d(input_channels, num_filter, kernel_size=3, stride=1, padding=1)
            conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=9, stride=1, padding=4)
            self.layers.extend([conv1, nn.ReLU(), conv2, nn.ReLU()])

    def forward(self, x):
        intermediates = []
        for i in range(self.depth):
            conv1 = self.layers[i * 4]
            conv2 = self.layers[i * 4 + 2]
            
            x = conv1(x)
            x = torch.relu(x)
            intermediate = x.clone()  # Save intermediate output for concatenation
            intermediates.append(intermediate)
            x = conv2(x)
            x = torch.relu(x)
            x = torch.cat([intermediate, x], dim=1)
            x = nn.functional.max_pool2d(x, kernel_size=2)

        return intermediates, x

class DecoderX(nn.Module):
    def __init__(self, depth, num_filter):
        super(DecoderX, self).__init__()
        self.depth = depth
        self.num_filter = num_filter
        self.layers = nn.ModuleList()
        input_channels = self.num_filter * 2

        for i in range(depth):
            conv3 = nn.Conv2d(input_channels, num_filter, kernel_size=3, stride=1, padding=1)
            conv4 = nn.Conv2d(num_filter, num_filter, kernel_size=9, stride=1, padding=4)

            self.layers.extend([conv3, nn.ReLU(), conv4, nn.ReLU()])

        self.layers.append(nn.Conv2d(num_filter*2, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, intermediates, x):
        intermediate2 = []
        for i in range(self.depth):
            conv3 = self.layers[i * 4]
            conv4 = self.layers[i * 4 + 2]

            x = conv3(x)
            x = torch.relu(x)
            intermediate = x.clone()  # Save intermediate output for concatenation
            intermediate2.append(intermediate)
            x = conv4(x)
            x = torch.relu(x)
            # intermediate = intermediates[self.depth - i - 1]
            x = torch.cat([intermediate, x], dim=1)
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.layers[-1](x)

        return x

class MyModel(nn.Module):
    def __init__(self, depth, num_filter):
        super(MyModel, self).__init__()
        self.encoder = EncoderX(depth, num_filter)
        self.decoder = DecoderX(depth, num_filter)

    def forward(self, x):
        intermediates, x = self.encoder(x)
        x = self.decoder(intermediates, x)

        return x

class FineTuneModel(nn.Module):
    def __init__(self, base_model, num_classes, dense_1, dense_2, drop):
        super(FineTuneModel, self).__init__()
        self.model = base_model
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Sequential(nn.Linear(3*128*128, dense_1), nn.ReLU(), nn.Dropout(drop))
        self.dense_2 = nn.Sequential(nn.Linear(dense_1, dense_2), nn.ReLU(), nn.Dropout(drop))
        self.output_layer = nn.Linear(dense_2, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.output_layer(x)
        # x = F.softmax(x, dim=1)
        return x

def select_data(radar, all_data):

    if radar == 10:
        data = h5py.File(all_data[0], "r")
    elif radar == 24:
        data = h5py.File(all_data[1], "r")
    elif radar == 77:
        data = h5py.File(all_data[2], "r")

    X_train = np.array(data["train_img"])
    Y_train = np.array(data["train_labels"])
    X_test = np.array(data["test_img"])
    Y_test = np.array(data["test_labels"])
    print(radar, 'ghz Dataset''s Number of training samples: ', len(Y_train))
    print(radar, 'ghz Dataset''s Number of test samples: ', len(X_test))    
    data.close()

    X_train = X_train/255.
    X_test = X_test/255.
    Y_train = convert_to_one_hot(Y_train, 11).T
    Y_test = convert_to_one_hot(Y_test, 11).T

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).permute(1, 0)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).permute(1, 0)

    return X_train.cuda(), Y_train.cuda(), X_test.cuda(), Y_test.cuda()

def convert_to_one_hot(labels, num_classes):
    labels = np.eye(num_classes)[labels.reshape(-1)]

    return labels

def main():
    radars = [10]
    epochs = 100
    batch_size = 16
    dense_1 = [64, 64, 128, 128, 256, 256]
    dense_2 = [32, 64, 64, 128, 128, 256]
    learn_rate = [0.0002, 0.0001]
    acc_hist = []
    hist_hist = []
    drop = 0.5
    num_class = 11

    # Dataset loading and preprocessing
    all_data = []
    all_data.append('../mydataset/hdf5/10GHz_dataset.hdf5')
    all_data.append('../mydataset/hdf5/24GHz_dataset.hdf5')
    all_data.append('../mydataset/hdf5/77GHz_dataset.hdf5')

    model = torch.load('./weights/10GHz_5_64.pth')
    model.to(device)
    # print(model)

    for i in range(len(radars)):
        radar = radars[i]
        (X_train, Y_train, X_test, Y_test) = select_data(radar, all_data)
        dataloader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)
        Y_test2 = torch.max(Y_test, 1)
        Y_test2 = Y_test2[1]

        for k in range(len(dense_1)):
            for m in range(len(learn_rate)):
                train_loss, val_loss, train_cnt, val_cnt = 0, 0, 0, 0
                model2 = FineTuneModel(model, num_class, dense_1[k], dense_2[k], drop)
                model2.to(device)
                optimizer = optim.Adam(model2.parameters(), lr=learn_rate[m], weight_decay=1e-06)
                criterion = nn.CrossEntropyLoss()
                for epoch in range(epochs):
                    correct, total = 0, 0
                    for inputs, labels in dataloader:
                        model2.zero_grad()
                        model2.train()
                        outputs = model2(inputs)
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_cnt += 1

                    with torch.no_grad():
                        outputs = model2(X_test)
                        # print("output shape: ", outputs.shape)
                        # print("Y_test shape: ", Y_test.shape)
                        loss = criterion(outputs, Y_test)
                        val_loss += loss.item()
                        val_cnt += 1
                        _, predicted = torch.max(outputs, 1)
                        correct = (predicted == Y_test2).sum().item()
                        total = len(predicted)
                
                    print(f'Epoch {epoch+1}|{radar} GHz, batch_size={batch_size}, '
                        f'dense_1={dense_1[k]}, dense_2={dense_2[k]}, learn_rate={learn_rate[m]}, '
                        f'train_loss={train_loss/train_cnt:.4f}, val_loss={val_loss/val_cnt:.4f}, Accuracy={correct/total:.4f}')

                    # acc_hist.append(val_loss)
                    # hist_hist.append(train_loss) 

if __name__ == "__main__":
    main()
