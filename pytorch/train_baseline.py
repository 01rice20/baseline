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
from torchvision.datasets import ImageFolder
from model.attention import *
from model.function import *
from model.convolution import *
from model.module import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
wandb.init(project="baseline_pytorch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
time1 = time()

def load_data_baseline(model, radar):
    root = []
    if(radar == 10):
        root = "../../dataset/spectrogram/spectrogram_10/"
        model.load_state_dict(torch.load('../../weights/baseline_bilinear_64/10GHz_5_64.pth'))
    elif (radar == 24):
        root = "../../dataset/spectrogram/spectrogram_24/"
        model.load_state_dict(torch.load('../../weights/baseline_bilinear_64/24GHz_5_64.pth'))
    elif(radar == 77):
        root = "../../dataset/spectrogram/spectrogram_77/"
        model.load_state_dict(torch.load('../../weights/baseline_bilinear_64/77GHz_5_64.pth'))
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


def main():

    radar = 10
    epochs = 100
    batch_size = [32, 64]
    dense_1 = [64, 64, 128, 128, 256, 256]
    dense_2 = [32, 64, 64, 128, 128, 256]
    learn_rate = [0.0002, 0.0001]
    drop = 0.5
    num_class = 11
    depth = 5
    num_filter = 64
    acc_hist = []
    model = []
    model = MyModel(depth, num_filter)
    bestscore = best_score()
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    model, dataset, train_sampler, test_sampler = load_data_baseline(model, radar)
    
    for i in range(len(batch_size)):
        dataloader = DataLoader(dataset, batch_size=batch_size[i], sampler=train_sampler, shuffle=False, pin_memory = True)
        val_dataloader = DataLoader(dataset, batch_size=batch_size[i], sampler=test_sampler, shuffle=False, pin_memory = True)

        for k in range(len(dense_1)):
            for m in range(len(learn_rate)):
                model2 = []
                train_loss, train_cnt, val_loss, val_cnt = 0, 0, 0, 0
                model2 = Classificate(model, num_filter, num_class, dense_1[k], dense_2[k], drop).to(device)
                optimizer = optim.Adam(model2.parameters(), lr=learn_rate[m], weight_decay=1e-06)
                
                for epoch in range(epochs):
                    model2.train()
                    for inputs, labels in dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model2(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_cnt += 1
                    print(f'Epoch {epoch + 1}, {radar} GHz, batch_size={batch_size[i]}, '
                        f'dense_1={dense_1[k]}, dense_2={dense_2[k]}, learn_rate={learn_rate[m]}, '
                        f'train_loss={train_loss/train_cnt}')
                    
                    wandb.log({"train_loss": train_loss/train_cnt})
                    if((epoch + 1) % 5 == 0):
                        model2.eval()
                        v_pred, v_true = [], []
                        istest = False
                        isfinal = False
                        for inputs, labels in val_dataloader:
                            with torch.no_grad():
                                inputs = inputs.to(device)
                                labels = labels.to(device)
                                outputs = model2(inputs)
                                loss = criterion(outputs, labels)
                                val_loss += loss.item()
                                val_cnt += 1
                                _, predicted = torch.max(outputs, 1)
                                v_pred.append(predicted)
                                v_true.append(labels)

                        v_pred = torch.cat(v_pred, dim=0).cpu().numpy()
                        v_true = torch.cat(v_true, dim=0).cpu().numpy()
                        
                        if((epoch + 1) == 100):
                            istest = True
                            isfinal = False
                            cm, precision, recall, f1, accuracy = score(v_pred, v_true, istest)
                            bestscore(cm, precision, recall, f1, accuracy, isfinal)

                            print(f'test_precision={round(precision, 5)}, test_recall={round(recall, 5)}, '
                                    f'test_f1={round(f1, 5)}, test_accuracy={round(accuracy, 5)}')
                            wandb.log({"accuracy": accuracy}) 
                        else:
                            istest = False
                            isfinal = False
                            accuracy = score(v_pred, v_true, istest)
                            vloss_hist = val_loss/val_cnt
                            print(f'val_loss={vloss_hist}, val_acc={round(accuracy, 5)}')
                            # wandb.log({"val_loss": vloss_hist, "val_accuracy": accuracy})
                        
    time2 = time()
    print("Total Time: ", int((time2 - time1) / 60))
    isfinal = True
    result = bestscore(cm, precision, recall, f1, accuracy, isfinal)
    wandb.log({"Best Precision": result['precision'], "Best Recall": result['recall'], 
                "Best F1": result['f1'], "Best Recall": result['recall'], "Best accuracy": result['accuracy']})

if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    main()

