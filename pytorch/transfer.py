import os
import random
import h5py
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F
from time import time
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import wandb
from time import time
# from lion_pytorch import Lion
from torchvision.datasets import ImageFolder
from model.attention import *
from model.function import *
from model.convolution import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
wandb.init(project="baseline_pytorch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
time1 = time()

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_c),
            nn.ReLU()
        )
  
    def forward(self, x):
        x = self.conv1(x)
        concate = x.clone()
        x = self.conv2(x)
        
        return concate, x

class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        real_c = int(out_c/2)
        self.conv = DoubleConv(in_c, real_c)
        self.max_pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        concate, x = self.conv(x)
        x = torch.cat([concate, x], dim=1)
        x = self.max_pool(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        real_c = int(out_c/2)
        self.conv = DoubleConv(in_c, real_c)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    
    def forward(self, x):
        concate, x = self.conv(x)
        x = torch.cat([concate, x], dim=1)
        x = self.upsample(x)

        return x

class UNet(nn.Module):
    def __init__(self, channel, num_filter):
        super().__init__()
        self.encoder0 = DoubleConv(channel, num_filter)
        self.encoder1 = Encoder(num_filter, num_filter*2)
        self.encoder2 = Encoder(num_filter*2, num_filter*4)
        self.encoder3 = Encoder(num_filter*4, num_filter*4)
        self.ecaatt = ECAAttention()
        self.bot1 = DoubleConv(num_filter*4, num_filter*8)
        self.bot2 = DoubleConv(num_filter*8, num_filter*8)
        self.bot3 = DoubleConv(num_filter*8, num_filter*4)
        self.decoder1 = Decoder(num_filter*4, num_filter*2)
        self.decoder2 = Decoder(num_filter*2, num_filter)
        self.decoder3 = Decoder(num_filter, num_filter)
        self.decoder0 = nn.Conv2d(num_filter, channel, kernel_size=1)

    def forward(self, x):
        _, x = self.encoder0(x)
        x = self.encoder1(x)
        x = self.ecaatt(x)
        x = self.encoder2(x)
        x = self.ecaatt(x)
        x = self.encoder3(x)
        x = self.ecaatt(x)
        _, x = self.bot1(x)
        _, x = self.bot2(x)
        _, x = self.bot3(x)
        x = self.decoder1(x)
        x = self.ecaatt(x)
        x = self.decoder2(x)
        x = self.ecaatt(x)
        x = self.decoder3(x)
        x = self.ecaatt(x)
        x = self.decoder0(x)
        
        return x

class ClassModel(nn.Module):
    def __init__(self, base_model, num_classes, dense_1, dense_2, drop):
        super().__init__()
        # Load Pre-train Model
        self.model = base_model
        # New Method 
        # self.resatt = ResidualAttention()
        # self.simatt = SimpleAttention()
        # self.nonsquare = NonSquareKernel()
        # self.ecaatt = ECAAttention()
        # Classificate Layer
        self.flatten = nn.Flatten()
        # Input layer 1000 for ResidualAttention
        # Input layer 45*64*78 for NonSquareKernel
        # Input layer 3*128*128 for Original Model
        self.dense_1 = nn.Sequential(nn.Linear(3*128*128, dense_1), nn.ReLU(), nn.Dropout(drop))    # For Original Model
        self.dense_2 = nn.Sequential(nn.Linear(dense_1, dense_2), nn.ReLU(), nn.Dropout(drop))
        self.output_layer = nn.Linear(dense_2, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)
        # x, _, _ = self.dat(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.output_layer(x)
        
        return x

def score(all_labels, all_predictions):
    cm = confusion_matrix(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)

    return cm, precision, recall, f1, accuracy

def main():

    radar = 10
    epochs = 200
    batch_size = [16, 32]
    dense_1 = [128, 128, 256, 256]
    dense_2 = [64, 128, 128, 256]
    learn_rate = [0.0002, 0.0001]
    drop = 0.5
    num_class = 11
    acc_hist = []
    cnt = 0
    patience = 10
    # accuracy_fn = Accuracy(task="MULTICLASS",num_classes=5).to(device)
    # (X_train, Y_train, X_test, Y_test, model) = select_data(radar, model)
    for i in range (len(batch_size)):
        for k in range(len(dense_1)):
            for m in range(len(learn_rate)):
                cnt += 1
                model = []
                model2 = []
                valloss_hist = []
                all_predictions = []
                all_labels = []
                model = UNet(3, 64)
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model, device_ids=[0, 1])
                model.to(device)
                model, dataset, train_sampler, test_sampler = load_data(model, radar)
                dataloader = DataLoader(dataset, batch_size=batch_size[i], sampler=train_sampler, shuffle=False, pin_memory = True)
                val_dataloader = DataLoader(dataset, batch_size=batch_size[i],sampler=test_sampler, shuffle=False, pin_memory = True)

                train_loss, train_cnt, val_loss, val_cnt, kernel_hist, loss_cnt, best_loss = 0, 0, 0, 0, 0, 0, None
                model2 = ClassModel(model, num_class, dense_1[k], dense_2[k], drop).to(device)
                optimizer = optim.RAdam(model2.parameters(), lr=learn_rate[m])
                # optimizer = Lion(model2.parameters(), lr=learn_rate[m], weight_decay=1e-2, use_triton=True)

                
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
                    # if((epoch + 1) % 20 == 0):
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
                            # predictions = (torch.sigmoid(outputs) > 0.5).float()
                            # all_predictions.extend(predictions.cpu().numpy())
                            # all_labels.extend(labels.cpu().numpy())
                    
                    # cm, precision, recall, f1, accuracy = score(all_labels, all_predictions)
                    
                    valloss_hist.append(val_loss/val_cnt)
                    acc = correct/total
                    # print("precision: ", precision)
                    # print("recall: ", recall)
                    # print("f1: ", f1)
                    # print("accuracy: ", accuracy)
                    print(f'Epoch {epoch + 1}, {radar} GHz, batch_size={batch_size[i]}, '
                            f'dense_1={dense_1[k]}, dense_2={dense_2[k]}, learn_rate={learn_rate[m]}, '
                            f'train_loss={train_loss/train_cnt}, val_loss={val_loss/val_cnt}, Accuracy={round(acc, 5)}')
                    
                    if((epoch + 1) == 200):
                        # Record Accuracy
                        wandb.log({"accuracy": acc})
                        acc_hist.append(round(acc, 5))
                        # Draw Validation Plot
                        x_axis = range(len(valloss_hist))
                        plt.plot(x_axis, valloss_hist, marker='o')
                        min_val = min(valloss_hist)
                        min_idx = valloss_hist.index(min_val)
                        plt.plot(min_idx, min_val, marker='o', color='red')
                        plt.xlabel('Epoch')
                        plt.ylabel('validation loss')
                        plt.title('valloss_hist')
                        plt.savefig('./valloss/valloss_hist_plot' + str(cnt) + '.png')
                        plt.clf()
                    # Early Stopping
                    # result, best_loss, loss_cnt = earlystopping(val_loss/val_cnt, best_loss, loss_cnt, patience)
                    # if result:    
                    #     print(f"Early stopping on Epoch {epoch + 1}")
                    #     break
                    

    time2 = time()
    print("Total Time: ", int((time2 - time1) / 60))
    print("Best accuracy: ", max(acc_hist))
    wandb.log({"Best accuracy": max(acc_hist)})

if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    main()
