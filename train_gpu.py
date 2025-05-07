import numpy as np 
import pandas as pd 
import os, sys, random
import cv2
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _utils

from random import choice

from skimage import io
from PIL import Image, ImageOps

import glob
import logging
import matplotlib.pyplot as plt

import torchvision.models as models
from tqdm import tqdm
from sklearn.utils import shuffle

import time
import warnings
from torch.optim.lr_scheduler import StepLR
from torch.nn.parameter import Parameter

from albumentations import (
    ShiftScaleRotate,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    RandomResizedCrop,
    Lambda,
    Compose
)
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop

warnings.filterwarnings("ignore")

# 🧠 No torch_xla, xm, xmp, pl — cleaned for GPU
# ✅ This script will now run on CPU/GPU

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

norm_mean = [0.143]
norm_std = [0.144]

RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)

def randomErase(image, **kwargs):
    return RandomErasing(image)

def sample_normalize(image, **kwargs):
    image = image / 255
    channel = image.shape[2]
    mean, std = image.reshape((-1, channel)).mean(axis=0), image.reshape((-1, channel)).std(axis=0)
    return (image - mean) / (std + 1e-3)

transform_train = Compose([
    RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), ratio=(0.75, 1.333), p=0.5),
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0, p=0.8),
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    Lambda(image=sample_normalize),
    ToTensorV2(),
    Lambda(image=randomErase)
])

transform_val = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
])

transform_test = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
])

def read_image(path, image_size=512):
    img = Image.open(path)
    w, h = img.size
    long = max(w, h)
    w, h = int(w / long * image_size), int(h / long * image_size)
    img = img.resize((w, h), Image.ANTIALIAS)
    delta_w, delta_h = image_size - w, image_size - h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return np.array(ImageOps.expand(img, padding).convert("RGB"))

class BAATrainDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        image = transform_train(image=read_image(f"{self.file_path}/{num // 1000}/{num}.png"))['image']
        return (image, Tensor([row['male']])), row['zscore']

    def __len__(self):
        return len(self.df)

class BAAValDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            df['id'] = df['Image ID'].astype('int32')  # convert and standardize to 'id'
            df['male'] = df['male'].astype('float32')
            df['boneage'] = df['boneage'].astype('float32')  # fix typo: bonage -> boneage
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = transform_val(image=read_image(f"{self.file_path}/{int(row['id'])}.png"))['image']
        return (image, Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)

class BAATestDataset(Dataset):
    def __init__(self, df, file_path):
        def preprocess_df(df):
            df['boneage'] = df['boneage'].astype('float32')
            df['id'] = df['Image ID'].astype('int32')
            return df

        self.df = preprocess_df(df)
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = transform_test(image=read_image(f"{self.file_path}/{int(row['id'])}.png"))['image']
        return (image, Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)

def create_data_loader(train_df, val_df, test_df, train_root, val_root, test_root):
    return (
        BAATrainDataset(train_df, train_root),
        BAAValDataset(val_df, val_root),
        BAATestDataset(test_df, test_root)
    )
    
def L1_penalty(net, alpha):
    loss = 0
    for param in net.fc.parameters():
        loss += torch.sum(torch.abs(param))
    return alpha * loss

def train_fn(net, train_loader, loss_fn, epoch, optimizer, device):
    global total_size
    global training_loss

    net.fine_tune()
    train_pbar = tqdm(train_loader)
    train_pbar.set_description(f"Epoch {epoch + 1}")

    for batch_idx, data in enumerate(train_pbar):
        size = len(data[1])
        image, gender = data[0]
        image, gender = image.to(device), gender.to(device)
        label = data[1].to(device)
        batch_size = len(data[1])

        optimizer.zero_grad()
        y_pred = net(image, gender).squeeze()

        loss = loss_fn(y_pred, label)
        total_loss = loss + L1_penalty(net, 1e-5)
        total_loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        training_loss += batch_loss
        total_size += batch_size
        train_pbar.set_postfix({'loss': batch_loss / batch_size})

    return training_loss / total_size

def evaluate_fn(net, val_loader, device):
    net.fine_tune(False)
    global mae_loss
    global val_total_size

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])
            image, gender = data[0]
            image, gender = image.to(device), gender.to(device)
            label = data[1].to(device)

            y_pred = net(image, gender) * boneage_div + boneage_mean
            y_pred = y_pred.squeeze()

            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()
            mae_loss += batch_loss

    return mae_loss

def test_fn(net, test_loader, device):
    net.train(False)
    global test_mae_loss
    global test_total_size

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            test_total_size += len(data[1])
            image, gender = data[0]
            image, gender = image.to(device), gender.to(device)
            label = data[1].to(device)

            y_pred = net(image, gender) * boneage_div + boneage_mean
            y_pred = y_pred.squeeze()

            batch_loss = F.l1_loss(y_pred, label, reduction='sum').item()
            test_mae_loss += batch_loss

    return test_mae_loss

def reduce_fn(vals):
    return sum(vals)

def map_fn(index, flags):
    # Setup
    root = '/content/drive/My Drive/BAA'
    model_name = 'rsa50_4.48'
    path = f'{root}/{model_name}'

    os.makedirs(path, exist_ok=True)
    seed_everything(seed=flags['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mymodel = BAA_New(32, *get_My_resnet50()).to(device)

    # Dataloaders (assumes datasets are defined globally or passed in)
    train_loader = DataLoader(train_set, batch_size=flags['batch_size'],
                              shuffle=True, num_workers=flags['num_workers'], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=flags['batch_size'],
                            shuffle=False, num_workers=flags['num_workers'])
    test_loader = DataLoader(test_set, batch_size=flags['batch_size'],
                             shuffle=False, num_workers=flags['num_workers'])

    net = mymodel.train()
    best_loss = float('inf')

    loss_fn = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=flags['lr'], weight_decay=0)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Training Loop
    for epoch in range(flags['num_epochs']):
        training_loss = 0.0
        total_size = 0.0
        mae_loss = 0.0
        val_total_size = 0.0
        test_mae_loss = 0.0
        test_total_size = 0.0

        start_time = time.time()
        train_fn(net, train_loader, loss_fn, epoch, optimizer, device)

        evaluate_fn(net, val_loader, device)
        test_fn(net, test_loader, device)

        scheduler.step()
        torch.save(net.state_dict(), os.path.join(path, f'{model_name}.bin'))

        train_loss = training_loss / total_size
        val_mae = mae_loss / val_total_size
        test_mae = test_mae_loss / test_total_size

        print(f'Test size: {test_total_size}')
        print(f'training loss: {train_loss}, val loss: {val_mae}, test loss: {test_mae}, '
              f'time: {time.time() - start_time}, lr: {optimizer.param_groups[0]["lr"]}')

        if best_loss >= test_mae:
            best_loss = test_mae
            shutil.copy(f'{path}/{model_name}.bin', f'{path}/best_{model_name}.bin')
            
def map_ensemble_fn(index, flags):
    root = '/content/drive/My Drive/BAA'
    model_name = 'final_ensemble_3.88'
    path = f'{root}/{model_name}'
    os.makedirs(path, exist_ok=True)

    seed_everything(seed=flags['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure new_model is defined properly
    base_model = BAA_New(32, *get_My_resnet50()).to(device)
    new_model = Graph_BAA(base_model).to(device)
    net = Ensemble(new_model).to(device)

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=flags['batch_size'], shuffle=True,
                              num_workers=flags['num_workers'], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=flags['batch_size'], shuffle=False,
                            num_workers=flags['num_workers'])
    test_loader = DataLoader(test_set, batch_size=flags['batch_size'], shuffle=False,
                             num_workers=flags['num_workers'])

    net.fine_tune()
    best_loss = float('inf')

    loss_fn = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=flags['lr'], weight_decay=0)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(flags['num_epochs']):
        training_loss = 0.0
        total_size = 0.0
        mae_loss = 0.0
        val_total_size = 0.0
        test_mae_loss = 0.0
        test_total_size = 0.0

        start_time = time.time()

        train_fn(net, train_loader, loss_fn, epoch, optimizer, device)
        evaluate_fn(net, val_loader, device)
        test_fn(net, test_loader, device)

        scheduler.step()

        torch.save(net.state_dict(), os.path.join(path, f'{model_name}.bin'))

        train_loss = training_loss / total_size
        val_mae = mae_loss / val_total_size
        test_mae = test_mae_loss / test_total_size

        print(f'Test size: {test_total_size}')
        print(f'training loss: {train_loss}, val loss: {val_mae}, test loss: {test_mae}, '
              f'time: {time.time() - start_time}, lr: {optimizer.param_groups[0]["lr"]}')

        if best_loss >= test_mae:
            best_loss = test_mae
            shutil.copy(f'{path}/{model_name}.bin', f'{path}/best_{model_name}.bin')
            
if __name__ == "__main__":
    from model import Ensemble, Graph_BAA, BAA_New, get_My_resnet50, BAA_Base
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_type')
    parser.add_argument('lr', type=float)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('num_epochs', type=int)
    parser.add_argument('seed', type=int)
    args = parser.parse_args()

    if args.model_type == 'ensemble':
        model = BAA_New(32, *get_My_resnet50())
        model.load_state_dict(torch.load('/content/drive/MyDrive/BAA/MRSA_50++_4.03/best_MRSA_50++_4.03.bin'))
        new_model = Graph_BAA(model)
        ensemble = Ensemble(new_model)
    else:
        model = BAA_New(32, *get_My_resnet50())

    flags = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'num_workers': 2,
        'num_epochs': args.num_epochs,
        'seed': args.seed
    }

    train_df = pd.read_csv('/content/drive/My Drive/Dataset/mini_dataset/train.csv')
    val_df = pd.read_csv('/content/drive/My Drive/Dataset/mini_dataset/validation.csv')
    test_df = val_df.copy()  # Reusing val set as test set (no separate .xlsx in mini_dataset)

    boneage_mean = train_df['boneage'].mean()
    boneage_div = train_df['boneage'].std()

    train_set, val_set, test_set = create_data_loader(
        train_df,
        val_df,
        test_df,
        '/content/drive/My Drive/Dataset/mini_dataset/boneage-training-dataset',
        '/content/drive/My Drive/Dataset/mini_dataset/boneage-validation-dataset',
        '/content/drive/My Drive/Dataset/mini_dataset/boneage-validation-dataset'  # reuse for test
    )

    torch.set_default_tensor_type('torch.FloatTensor')
    if args.model_type == 'ensemble':
        map_ensemble_fn(0, flags)
    else:
        map_fn(0, flags)