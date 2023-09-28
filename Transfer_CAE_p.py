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
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import wandb
from time import time
from model.Attention import ResidualAttention, SimpleAttention
from torchvision.datasets import ImageFolder

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

wandb.init(project="baseline_pytorch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
time1 = time()

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

def load_data(model, radar):
    root = []
    if(radar == 10):
        root = "../../dataset/spectrogram/spectrogram_10/"
        model.load_state_dict(torch.load('../../weights/nearest_exact/10GHz_5_64.pth'))
    elif (radar == 24):
        root = "../../dataset/spectrogram/spectrogram_24/"
        model.load_state_dict(torch.load('../../weights/nearest_exact/24GHz_5_64.pth'))
    elif(radar == 77):
        root = "../../dataset/spectrogram/spectrogram_77/"
        model.load_state_dict(torch.load('../../weights/nearest_exact/77GHz_5_64.pth'))
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
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

    return model, dataset, train_sampler, test_sampler

def stable(dl):
    set_seed()

    return dl

def select_data(radar, model):

    if radar == 10:
        data = h5py.File('../../dataset/hdf5/10GHz_dataset.hdf5', "r")
        model.load_state_dict(torch.load('../../weights/10GHz_5_64.pth'))

    elif radar == 24:
        data = h5py.File('../../dataset/hdf5/24GHz_dataset.hdf5', "r")
        model.load_state_dict(torch.load('../../weights/24GHz_5_64.pth'))

    elif radar == 77:
        data = h5py.File('../../dataset/hdf5/77GHz_dataset.hdf5', "r")
        model.load_state_dict(torch.load('../../weights/77GHz_5_64.pt'))

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

    return X_train.cuda(), Y_train.cuda(), X_test.cuda(), Y_test.cuda(), model

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

class FineTuneModel(nn.Module):
    def __init__(self, base_model, num_classes, dense_1, dense_2, drop):
        super(FineTuneModel, self).__init__()
        # pre-train Model
        self.model = base_model
        # Residual Attention
        # self.resatt = ResidualAttention()
        # Simple Attention
        # self.simam = SimpleAttention()
        self.flatten = nn.Flatten()
        # self.dense_1 = nn.Sequential(nn.Linear(1000, dense_1), nn.ReLU(), nn.Dropout(drop))    # For adding ResidualAttention
        self.dense_1 = nn.Sequential(nn.Linear(3*128*128, dense_1), nn.ReLU(), nn.Dropout(drop))    # For Original Model
        self.dense_2 = nn.Sequential(nn.Linear(dense_1, dense_2), nn.ReLU(), nn.Dropout(drop))
        self.output_layer = nn.Linear(dense_2, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        # x = self.resatt(x)
        # x = self.simam(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.output_layer(x)
        
        return x

def main():

    radar = 77
    epochs = 100
    batch_size = [8, 16]    
    dense_1 = [64, 64, 128, 128, 256, 256]
    dense_2 = [32, 64, 64, 128, 128, 256]
    learn_rate = [0.0002, 0.0001]
    drop = 0.5
    num_class = 11
    acc_hist = []
    hist_hist = []
    model = []
    model = MyModel(5, 64)
    # (X_train, Y_train, X_test, Y_test, model) = select_data(radar, model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    model, dataset, train_sampler, test_sampler = load_data(model, radar)
        
    for i in range(len(batch_size)):
        dataloader = DataLoader(dataset, batch_size=batch_size[i], sampler=train_sampler, shuffle=False, pin_memory = True)
        val_dataloader = DataLoader(dataset, batch_size=batch_size[i],sampler=test_sampler, shuffle=False, pin_memory = True)

        for k in range(len(dense_1)):
            for m in range(len(learn_rate)):
                train_loss, train_cnt, val_loss, val_cnt = 0, 0, 0, 0
                model2 = FineTuneModel(model, num_class, dense_1[k], dense_2[k], drop).to(device)
                optimizer = optim.Adam(model2.parameters(), lr=learn_rate[m], weight_decay=1e-06)
                
                for epoch in range(epochs):
                    for inputs, labels in dataloader:
                        model2.train()
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model2(inputs)
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_cnt += 1
                    
                    # wandb.log({"train_loss": train_loss/train_cnt})
                    if((epoch + 1) % 20 == 0):
                        correct, total = 0, 0
                        model2.eval()
                        for input, labels in val_dataloader:
                            with torch.no_grad():
                                input = input.to(device)
                                labels = labels.to(device)
                                outputs = model2(input)
                                loss = nn.CrossEntropyLoss()(outputs, labels)
                                val_loss += loss.item()
                                val_cnt += 1
                                _, predicted = torch.max(outputs, 1)
                                correct += (predicted == labels).sum().item()
                                total += len(predicted)
                
                        print(f'Epoch {epoch + 1}, {radar} GHz, batch_size={batch_size[i]}, '
                                f'dense_1={dense_1[k]}, dense_2={dense_2[k]}, learn_rate={learn_rate[m]}, '
                                f'train_loss={train_loss/train_cnt}, val_loss={val_loss/val_cnt}, Accuracy={correct/total}')
                    if((epoch + 1) == 100):
                        wandb.log({"accuracy": correct/total})

    time2 = time()
    print("Total Time: ", int((time2 - time1) / 60))

if __name__ == "__main__":
    seed = 42
    set_seed()
    main()
