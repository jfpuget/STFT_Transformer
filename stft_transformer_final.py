#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

fname = 'stft_tranformer'

FOLD = 0

cont_epoch = -1

fname = fname + '_' + str(FOLD)

checkpoint_path = Path('../checkpoints') / fname

if cont_epoch < 0:
    if checkpoint_path.exists():
        sys.exit()
    else:
        checkpoint_path.mkdir()
elif not checkpoint_path.exists():
    sys.exit()

input_path = Path('../input/')
data_path = Path('../data')

if not data_path.exists():
    sys.exit()

PERIOD = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 576

POSWEIGHT=10
SR=32000

import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
pd.options.display.max_columns = 100

from skimage.transform import rescale, resize, downscale_local_mean
from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, PitchShift, AddBackgroundNoise, AddShortNoises, Gain
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import timm

from scipy.special import logit, expit

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def seed_torch(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

train = pd.read_csv(input_path / 'train_001.csv')
train.head()

train_ff1010 = pd.read_csv('../input/train_ff1010.csv')
train_ff1010['primary_label'] = 'nocall'
train_ff1010

columns = ['length', 'primary_label', 'secondary_labels', 'filename']

train = pd.concat((train[columns], train_ff1010[columns])).reset_index(drop=True)

primary_labels = set(train.primary_label.unique())

primary_labels

secondary_labels = set([s for labels in train.secondary_labels for s in eval(labels)])
secondary_labels

len(primary_labels), len(secondary_labels), len(secondary_labels - primary_labels)

res = [[label for label in eval(secondary_label) if label != 'rocpig1'] 
                             for secondary_label in train['secondary_labels']]

train['secondary_labels'] = res

BIRD_CODE = {}
INV_BIRD_CODE = {}
for i,label in enumerate(sorted(primary_labels)):
    BIRD_CODE[label] = i
    INV_BIRD_CODE[i] = label

NOCALL_CODE = BIRD_CODE['nocall']
NOCALL_CODE

train['class'] = [BIRD_CODE[label] for label in train.primary_label]
train['weight'] = train.groupby('class')['class'].transform('count')
train['weight'] = 1 / np.sqrt(train['weight'])
train['weight'] /= train['weight'].mean()
train.loc[train.primary_label == 'nocall', 'weight'] = 1


def get_sample_clip(data_path, sample, period, train_aug):
    filename = sample['filename']
    length = sample['length']
    base_period = PERIOD * SR
    if train_aug:
        start = np.random.choice([0, max(0, length - period)])   
    else:
        start = 0
        
    if not filename.startswith('ff1010'):
        file_idx = int(np.floor(start / base_period))
        start = start - base_period * file_idx
        filename = '.'.join(filename.split('.')[:-1])
        filename = '%s_%d.npy' % (filename, file_idx)
    path = data_path / filename
    clip = np.load(path)
        
    clip = clip[start : start + period]

    if period > length:
        start = np.random.randint(period - length)
        tmp = np.zeros(period, dtype=clip.dtype)
        tmp[start : start + length] = clip
        clip = tmp
    return clip
    

def get_melspec(data_path, sample, train_aug, no_calls, other_samples, display=None):
    sr = SR
    
    if train_aug is not None:
        sr_scale_max = 1.1
        sr_scale_min = 1 / sr_scale_max
        sr_scale = sr_scale_min + (sr_scale_max - sr_scale_min)*np.random.random_sample()
        sr = int(sr*sr_scale)
    sr = max(32000, sr)
    
    period = PERIOD * sr
    if train_aug is not None:
        freq_scale_max = 1.1
        freq_scale_min = 1 / freq_scale_max
        freq_scale = freq_scale_min + (freq_scale_max - freq_scale_min)*np.random.random_sample()
        period = int(np.round(period * freq_scale))
        
    clip = get_sample_clip(data_path, sample, period, train_aug)
    if other_samples is not None:
        for another_sample in other_samples:
            another_clip = get_sample_clip(data_path, another_sample, period, train_aug)
            weight = np.random.random_sample() * 0.8 + 0.2
            clip = clip + weight*another_clip
        
    if no_calls is not None:
        no_calls = no_calls[SR]
        no_calls_clip = np.random.choice(no_calls)
        no_calls_length = no_calls_clip.shape[0]
        no_calls_period = period
        no_calls_start = np.random.randint(no_calls_length - no_calls_period)
        no_calls_clip = no_calls_clip[no_calls_start : no_calls_start + no_calls_period]
        clip = clip + np.random.random_sample() * no_calls_clip

    if train_aug is not None:
        clip = train_aug(clip, sample_rate=sr)

    n_fft = 1024
    win_length = n_fft#//2
    hop_length = int((len(clip) - win_length + n_fft) / IMAGE_WIDTH) + 1 
    spect = np.abs(librosa.stft(y=clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    if spect.shape[1] < IMAGE_WIDTH:
        #print('too large hop length, len(clip)=', len(clip))
        hop_length = hop_length - 1
        spect = np.abs(librosa.stft(y=clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    if spect.shape[1] > IMAGE_WIDTH:
        spect = spect[:, :IMAGE_WIDTH]
    n_mels = IMAGE_HEIGHT // 2
    if train_aug is not None:
        power = 1.5 + np.random.rand()
        spect = np.power(spect, power)
    else:
        spect = np.square(spect)
    spect = librosa.feature.melspectrogram(S=spect, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=300, fmax=16000)
    spect = librosa.power_to_db(spect)
    #print(spect.shape)
    spect = resize(spect, (IMAGE_HEIGHT, IMAGE_WIDTH), preserve_range=True, anti_aliasing=True)
    spect = spect - spect.min()
    smax = spect.max()
    if smax >= 0.001:
        spect = spect / smax
    else:
        spect[...] = 0
    if display:
        plt.imshow(spect)
        plt.show()
    # clip, sr = librosa.load(path, sr=None, mono=False)
    return spect

class BirdDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 data_path: Path,
                 target=True,
                 train_aug=None,
                ):
        super(BirdDataset, self).__init__()
        self.data = data
        self.data_path = data_path
        self.target = target
        self.train_aug = train_aug
        self.no_calls = None

    def __len__(self):
        return len(self.data)

    def inv_stem(self, x):
        x1 = x.transpose(0, 1).view(24, 24, 16, 16)
        y = torch.zeros(384, 384, dtype=x.dtype)
        for i in range(24):
            for j in range(24):
                y[i*16:(i+1)*16, j*16:(j+1)*16] = x1[i, j]
        return y
    
    def __getitem__(self, idx: int):
        sample = self.data.loc[idx, :]
        if self.train_aug:
            no_calls = self.no_calls
        else:
            no_calls = None
        primary_label = sample['primary_label']
        if primary_label == 'nocall' or not self.train_aug:
            other_samples = None
        else:
            num_samples = np.random.choice([0, 1, 2])
            other_samples = [self.data.loc[np.random.randint(len(self.data)), :] for i in range(num_samples)]
        melspec = get_melspec(self.data_path, sample, self.train_aug, no_calls, other_samples)
        melspec = torch.from_numpy(melspec)
        melspec = self.inv_stem(melspec)
            
        input_dict = {
            "spect": melspec,
        }
        if self.target:
            labels = np.zeros(len(BIRD_CODE), dtype=np.float32)
            primary_label = sample['primary_label']
            labels[BIRD_CODE[primary_label]] = 1
            if other_samples is not None:
                for another_sample in other_samples:
                    ebird_code = another_sample['primary_label']
                    labels[BIRD_CODE[ebird_code]] = 1
            if np.sum(labels) >= 2:
                labels[NOCALL_CODE] = 0 # not a nocall
            secondary_mask = np.ones(len(BIRD_CODE), dtype=np.float32)
            extra_labels = sample['secondary_labels']
            for extra_label in extra_labels:
                secondary_mask[BIRD_CODE[extra_label]] = 0
            if other_samples is not None:
                for another_sample in other_samples:
                    extra_labels = another_sample['secondary_labels']
                    for extra_label in extra_labels:
                        secondary_mask[BIRD_CODE[extra_label]] = 0
            secondary_mask = np.maximum(secondary_mask, labels)

            input_dict['secondary_mask'] = torch.from_numpy(secondary_mask)
            input_dict['target'] = torch.from_numpy(labels)
        return input_dict 

train_aug = Compose([
        AddGaussianNoise(p=0.2),
        AddGaussianSNR(p=0.2),
        Gain(min_gain_in_db=-15,max_gain_in_db=15,p=0.3)
    ])

device = torch.device('cuda')


class BirdLoss(nn.Module):
    def __init__(self, pos_weight):
        super(BirdLoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, target, secondary_mask):
        loss = F.binary_cross_entropy_with_logits(logits, target, weight=None, pos_weight=self.pos_weight, reduction='none')
        loss = (loss * secondary_mask).mean()
        return loss
    
criterion = BirdLoss(pos_weight=torch.tensor(POSWEIGHT).to(device))

class Backbone(nn.Module):

    
    def __init__(self, name='resnet18', pretrained=True):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained)
        
        if 'regnet' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'vit' in name:
            self.out_features = self.net.head.in_features
        elif backbone == 'vit_deit_base_distilled_patch16_384':
            self.out_features = 768
        elif 'csp' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'res' in name: #works also for resnest
            self.out_features = self.net.fc.in_features
        elif 'efficientnet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'densenet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'senet' in name:
            self.out_features = self.net.fc.in_features
        elif 'inception' in name:
            self.out_features = self.net.last_linear.in_features

        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x
    
class BirdModel(nn.Module):
    def __init__(self, backbone, out_dim, embedding_size=512, 
                 loss=False, pretrained=True):
        super(BirdModel, self).__init__()
        self.backbone_name = backbone
        self.loss = loss
        self.embedding_size = embedding_size
        self.out_dim = out_dim
        
        self.backbone = Backbone(backbone, pretrained=pretrained)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
            
        self.head = nn.Linear(self.embedding_size, out_dim)
        
    def forward(self, input_dict, get_embeddings=False, get_attentions=False):

        x = input_dict['spect']
        x = x.unsqueeze(1)
        x = x.expand(-1, 3, -1, -1)

        x = self.backbone(x)
        
        if 'vit' not in backbone:
            x = self.global_pool(x)
            x = x[:,:,0,0]
        if 'vit_deit_base_distilled_patch16_384' == backbone:
            x = x[0] + x[1]
        
        x = self.neck(x)

        logits = self.head(x)
        
        output_dict = {'logits':logits,
                      }
        if self.loss:
            target = input_dict['target']
            secondary_mask = input_dict['secondary_mask']
            loss = criterion(logits, target, secondary_mask)
            
            output_dict['loss'] = loss
            
        return output_dict

def train_epoch(loader, model, optimizer, scheduler, scaler, device):
 
    model.train()
    model.zero_grad()
    train_loss = []
    bar = tqdm(range(len(loader)))
    load_iter = iter(loader)
    batch = load_iter.next()
    batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }
    
    for i in bar:
        
        input_dict = batch.copy()
        if i + 1 < len(loader):
            batch = load_iter.next()
            batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }
            
        with autocast():
            out_dict = model(input_dict)
        loss = out_dict['loss']
        loss_np = loss.detach().cpu().numpy()
        #loss.backward()
        scaler.scale(loss).backward()
        
        if (i+1) % GRADIENT_ACCUMULATION == 0 or i == len(loader) - 1:
            #optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            scheduler.step()            

        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.4f, smth: %.4f' % (loss_np, smooth_loss))
    return train_loss


def val_epoch(loader, model, device):

    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []
    
    with torch.no_grad():
        if 1:
            bar = tqdm(range(len(loader)))
            load_iter = iter(loader)
            batch = load_iter.next()
            batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }

            for i in bar:
                input_dict = batch.copy()
                if i + 1 < len(loader):
                    batch = load_iter.next()
                    batch = {k:batch[k].to(device, non_blocking=True) for k in batch.keys() }
                    
                out_dict = model(input_dict)
                logits = out_dict['logits']
                loss = out_dict['loss']
                target = input_dict['target']
                loss_np = loss.detach().cpu().numpy()
                LOGITS.append(logits.detach())
                TARGETS.append(target.detach())
                val_loss.append(loss_np) 
                
                smooth_loss = sum(val_loss[-100:]) / min(len(val_loss), 100)
                bar.set_description('loss: %.4f, smth: %.4f' % (loss_np, smooth_loss))

            val_loss = np.mean(val_loss)
    
    LOGITS = (torch.cat(LOGITS).cpu().numpy())
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    y_pred = 1 * (LOGITS >= -1)
    score_5 = f1_score(TARGETS, y_pred, average="samples")
    y_pred = 1 * (LOGITS >= -0.5)
    score_6 = f1_score(TARGETS, y_pred, average="samples")
    y_pred = 1 * (LOGITS >= 0.0)
    score_7 = f1_score(TARGETS, y_pred, average="samples")
    y_pred = 1 * (LOGITS >= 0.5)
    score_8 = f1_score(TARGETS, y_pred, average="samples")
    y_pred = 1 * (LOGITS >= 1.0)
    score_9 = f1_score(TARGETS, y_pred, average="samples")
    
    return val_loss, score_5, score_6,  score_7,score_8, score_9, LOGITS


TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
EPOCHS=60
WORKERS=4
SEED=0
FP16=False
NFOLDS = 5
backbone = 'vit_deit_base_distilled_patch16_384'
VALID_BATCH_SIZE = 4 * TRAIN_BATCH_SIZE

kfolds = StratifiedKFold(5, shuffle=True, random_state=0)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, fold, seed, fname=fname):
    checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
        }
    torch.save(checkpoint, '../checkpoints/%s/%s_%d_%d_%d.pt' % (fname, fname, fold, seed, epoch))

def load_checkpoint(backbone, epoch, fold, seed, fname):
    model = BirdModel(backbone, 
                      out_dim=len(BIRD_CODE), 
                      loss=True, 
                      pretrained=False,
                     ).to(device)
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                              pct_start=0.1, 
                                              div_factor=1e3, 
                                              max_lr=1e-4, 
                                              epochs=EPOCHS, 
                                              steps_per_epoch=int(np.ceil(len(train_data_loader)/GRADIENT_ACCUMULATION)))
    scaler = GradScaler()
    checkpoint = torch.load('../checkpoints/%s/%s_%d_%d_%d.pt' % (fname, fname, fold, seed, epoch))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    return model, optimizer, scheduler, scaler


device = torch.device('cuda')
for seed in [0]:
    for fold, (train_idx, valid_idx) in enumerate(kfolds.split(train, train['primary_label'])):
        if fold != FOLD:
            continue
        seed_torch(seed)
        
        train_fold = train.iloc[train_idx].reset_index(drop=True)
        
        train_dataset = BirdDataset(train_fold, 
                                    data_path, target=True, train_aug=train_aug)
        
        train_sampler = WeightedRandomSampler(train_fold['weight'].values, len(train_fold))

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=WORKERS,
            shuffle=False,
            pin_memory=True,
            sampler=train_sampler,
        )
                
        valid_dataset_orig = BirdDataset(train.iloc[valid_idx].reset_index(drop=True), 
                                    data_path, target=True, train_aug=None)

        valid_data_loader_orig = DataLoader(
            valid_dataset_orig,
            batch_size=VALID_BATCH_SIZE,
            num_workers=WORKERS,
            shuffle=False,
            pin_memory=True,
        )
        if cont_epoch >= 0:
            model, optimizer, scheduler, scaler = load_checkpoint(backbone, cont_epoch, fold, seed, fname)
        else:
            model = BirdModel(backbone, out_dim=len(BIRD_CODE), 
                          neck="option-F",
                          loss=True, 
                          gem_pooling=False).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                                      pct_start=0.1, 
                                                      div_factor=1e3, 
                                                      max_lr=1e-4, 
                                                      epochs=EPOCHS, 
                                                      steps_per_epoch=int(np.ceil(len(train_data_loader)/GRADIENT_ACCUMULATION)))

            scaler = GradScaler()

        if cont_epoch == -1:
            start_epoch = 0
        else:
            start_epoch = cont_epoch + 1
        for epoch in range(start_epoch, EPOCHS):
            print(time.ctime(), 'Epoch:', epoch, flush=True)
            train_loss = train_epoch(train_data_loader, model, optimizer, scheduler, scaler, device, 
                                     )
            
            (val_loss, score_5, score_6,  score_7,  score_8, score_9, _
            ) = val_epoch(valid_data_loader_orig, model, device)                           
            content = 'Orig %d Ep %d, lr: %.7f, train loss: %.5f, val loss: %.5f, f1: %.4f %.4f %.4f %.4f %.4f'
            values = (fold, 
                     epoch, 
                     optimizer.param_groups[0]["lr"], 
                     np.mean(train_loss),
                     np.mean(val_loss),
                     score_5, score_6,  score_7,  score_8, score_9,
                    )
            print(content % values, flush=True)
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, fold, seed)

print('*' * 40)
print()


# In[ ]:




