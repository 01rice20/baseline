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
from model.attention import *
from model.function import *
from model.convolution import *
from model.module import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def main():
    # Parameter setting
    depth = [5]
    num_filter = [64]
    radars = [10, 24, 77]
    epochs = 100
    batch_size = 64
    lr = 0.001
    num_class = 11
    im_width = 128
    im_height = 128
    inChannel = 3

    loss_fn = nn.MSELoss()

    for i in range(len(radars)):
        radar = radars[i]
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
                
                torch.save(model.state_dict(), '../../weights/baseline_bilinear/' + str(radars[i]) + 'GHz_b' + str(batch_size[x]) + '_d' + str(depth) + '_l' + str(num_filter) + '.pth')
                time2 = time()                 
                print("Training Time: ", (time2 - time1) / 60)
                
if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    main()
