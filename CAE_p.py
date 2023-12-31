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
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import wandb
from time import time
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from model.Attention import build_upsample_layer

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# wandb.init(project="baseline_pytorch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False

set_seed()

def load_data(radar):
    if(radar == 10):
        root = "../../dataset/spectrogram/spectrogram_10/"
    elif (radar == 24):
        root = "../../dataset/spectrogram/spectrogram_24/"
    elif(radar == 77):
        root = "../../dataset/spectrogram/spectrogram_77/"
    
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
    ])
    dataset = ImageFolder(root=root, transform=transform)
    train_ratio = 0.8
    num_data = len(dataset)
    # print(num_data)
    num_train = int(train_ratio * num_data)
    indices = list(range(num_data))
    random.shuffle(indices)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return dataset, train_sampler, test_sampler


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
    # Y_train = convert_to_one_hot(Y_train, 11).T
    # Y_test = convert_to_one_hot(Y_test, 11).T

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).permute(1, 0)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).permute(1, 0)

    print("Shape of X_train: ", X_train.size())
    print("Shape of Y_train: ", Y_train.size())
    print("Shape of X_test: ", X_test.size())
    print("Shape of Y_test: ", Y_test.size())
    print(X_test)
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
        # self.upsample = build_upsample_layer(
        #     cfg=dict(type='nearest-exact'),
        #     scale_factor=2)


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
            # x = self.upsample(x)
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest-exact')
        
        x = self.layers[-1](x)

        return x


class MyModel(nn.Module):
    def __init__(self, depth, num_filter):
        super(MyModel, self).__init__()
        self.encoder = EncoderX(depth, num_filter)
        self.decoder = DecoderX(depth, num_filter)

    def forward(self, x):
        set_seed()
        intermediates, x = self.encoder(x)
        x = self.decoder(intermediates, x)
        
        return x

def main():
    # Parameter setting
    depth = [5, 6]
    num_filter = [64]
    radars = [10, 77]
    epochs = 100
    batch_size = 16
    lr = 0.001
    num_class = 11
    im_width = 128
    im_height = 128
    inChannel = 3

    loss_fn = nn.MSELoss()

    for i in range(len(radars)):
        radar = radars[i]
        # (X_train, Y_train, X_test, Y_test) = select_data(radar, all_data)
        dataset, train_sampler, test_sampler = load_data(radar)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, pin_memory = True)
        val_dataloader = DataLoader(dataset, batch_size=batch_size,sampler=test_sampler, shuffle=False, pin_memory = True)

        for d in range(len(depth)):
            for f in range(len(num_filter)):
                train_loss, val_loss, train_cnt, val_cnt = 0, 0, 0, 0
                model = MyModel(depth[d], num_filter[f])
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model, device_ids=[0, 1])
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                time1 = time()

                for epoch in range(epochs):
                    model.train()
                    for batch_X, label in dataloader:
                        model.train()
                        batch_X = batch_X.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = loss_fn(outputs, batch_X)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_cnt += 1
                    
                    if((epoch + 1) % 20 == 0):
                        model.eval()
                        for batch_X, _ in val_dataloader:
                            batch_X = batch_X.to(device)
                            with torch.no_grad():
                                outputs = model(batch_X)
                                loss = nn.MSELoss()(outputs, batch_X)
                                val_loss += loss.item()
                                val_cnt += 1
                        print(f'Epoch {epoch+1} | {radars[i]} GHz, depth={depth[d]}, num_filter={num_filter[f]}, '
                            f'train_loss={train_loss/train_cnt}, val_loss={val_loss/val_cnt}')
                        # wandb.log({"val_loss": val_loss/val_cnt, "train_loss": train_loss/train_cnt})
                
                torch.save(model.state_dict(), '../../weights/nearest_exact/' + str(radars[i]) + 'GHz_' + str(depth[d]) + '_' + str(num_filter[f]) + '.pth')
                time2 = time()                 
                print("Training Time: ", (time2 - time1) / 60)
                
if __name__ == "__main__":
    main()
