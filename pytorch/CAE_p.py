import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
import wandb
from time import time

# wandb.init(project="baseline_pytorch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def set_seed():
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(1)
    random.seed(1254)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    print("Shape of X_train: ", X_train.size())
    print("Shape of Y_train: ", Y_train.size())
    print("Shape of X_test: ", X_test.size())
    print("Shape of Y_test: ", Y_test.size())

    return X_train.cuda(), Y_train.cuda(), X_test.cuda(), Y_test.cuda()

def convert_to_one_hot(labels, num_classes):
    labels = np.eye(num_classes)[labels.reshape(-1)]

    return labels

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
        # print("Final shape: ", x.shape)
        
        return x

def main():
    # Parameter setting
    set_seed()
    depth = [5, 6]
    num_filter = [16, 32, 64]
    radars = [24]
    epochs = 100
    batch_size = 16
    lr = 0.001
    num_class = 11
    im_width = 128
    im_height = 128
    inChannel = 3
    acc_hist = []

    # Dataset loading and preprocessing
    all_data = []
    all_data.append('../../dataset/hdf5/10GHz_dataset.hdf5')
    all_data.append('../../dataset/hdf5/24GHz_dataset.hdf5')
    all_data.append('../../dataset/hdf5/77GHz_dataset.hdf5')

    loss_fn = nn.MSELoss()
    
    for i in range(len(radars)):
        radar = radars[i]
        (X_train, Y_train, X_test, Y_test) = select_data(radar, all_data)
        dataloader = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)
        input_img = X_train.size(1)

        for d in range(len(depth)):
            for f in range(len(num_filter)):
                train_loss, val_loss, train_cnt, val_cnt = 0, 0, 0, 0
                model = MyModel(depth[d], num_filter[f])
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                time1 = time()

                for epoch in range(epochs):
                    for batch_X, _ in dataloader:
                        model.zero_grad()
                        model.train()
                        outputs = model(batch_X)
                        loss = loss_fn(outputs, batch_X)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_cnt += 1
                    
                    if((epoch + 1) % 20 == 0):
                        model.eval()
                        for batch_X, _ in val_dataloader:
                            with torch.no_grad():
                                outputs = model(batch_X)
                                loss = nn.MSELoss()(outputs, batch_X)
                                val_loss += loss.item()
                                val_cnt += 1
                        print(f'Epoch {epoch+1} | {radars[i]} GHz, depth={depth[d]}, num_filter={num_filter[f]}, '
                            f'train_loss={train_loss/train_cnt:.4f}, val_loss={val_loss/val_cnt:.4f}')
                        # wandb.log({"val_loss": val_loss/val_cnt})
                    
                    # wandb.log({"train_loss": train_loss/train_cnt})
                
                torch.save(model.state_dict(), '../../weights/' + str(radars[i]) + 'GHz_' + str(depth[d]) + '_' + str(num_filter[f]) + '.pth')
                time2 = time()                 
                print("Training Time: ", (time2 - time1) / 60)
                
if __name__ == "__main__":
    main()