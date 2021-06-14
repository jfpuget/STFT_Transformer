#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import librosa
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from pqdm.processes import pqdm

train = pd.read_csv('../input/train_metadata.csv')
train

def ogg2np(ebird, filename):
    input_path = Path('../input/')
    output_path = Path('../data/')
    input_path = input_path / 'train_short_audio' / ebird / filename
    filename = '.'.join(filename.split('.')[:-1])
    output_path = output_path 
    record, sr = sf.read(input_path)
    record = librosa.to_mono(record)
    record = record.astype('float32')
    length = record.shape[0]
    period = int(np.ceil(5 * sr))
    if period == 0:
        filename = '%s_%d.npy' % (filename, 0)
        np.save(output_path / filename, record)
    else:
        for i in range(int(np.ceil(length/period))):  
            filename_i = '%s_%d.npy' % (filename, i)
            record_i = record[i*period : (i+3)*period]
            np.save(output_path / filename_i, record_i)
    return length, sr

res = pqdm(zip(train.primary_label, train.filename), ogg2np, n_jobs=8, argument_type='args')

train['sr'] = [r[1] for r in res]
train['length'] = [r[0] for r in res]
train['duration'] = train['length'] / train['sr']

train['filename'] = [fname[:-4]+'.npy' for fname in train.filename]

train.to_csv('../input/train_001.csv', index=False)

train_ff1010 = pd.read_csv('../input/freefield1010/ff1010bird_metadata.csv')
train_ff1010

train_ff1010 = train_ff1010.sort_values(by='itemid').reset_index(drop=True)
train_ff1010

def get_clip(itemid):
    data_path = Path('../input/freefield1010/wav')
    path = data_path / ('%d.wav' % itemid)
    clip, sr_native = librosa.load(path, sr=None, mono=True, dtype=np.float32)
    sr = 32000
    if sr_native != 0:
        clip = librosa.resample(clip, sr_native, sr, res_type='kaiser_best')
    else:
        print('null sr_native')
    return clip, sr, sr_native

train_ff1010 = train_ff1010[train_ff1010.hasbird == 0].reset_index(drop=True)
train_ff1010

def work_sub(itemid):
    output_path = Path('../data/')
    clip, sr, sr_native = get_clip(itemid)
    clip = clip.astype('float32')
    length = clip.shape[0] 
    filename = 'ff1010_%d_0.npy' % (itemid)
    np.save(output_path / filename, clip)
    
    return sr, sr_native, length


res = pqdm(train_ff1010.itemid, work_sub, n_jobs=8)

train_ff1010['primary_label'] = ''
train_ff1010['secondary_labels'] = None
train_ff1010['sr'] = [r[0] for r in res]
train_ff1010['sr_native'] = [r[1] for r in res]
train_ff1010['length'] = [r[2] for r in res]
train_ff1010['duration'] = train_ff1010['length'] / 32000
train_ff1010['filename'] = ['ff1010_%d_0.npy' % (itemid) for itemid in train_ff1010['itemid']]
train_ff1010

train_ff1010['secondary_labels'] = [[]] * len(train_ff1010)

columns = ['duration', 'length', 'primary_label', 'secondary_labels', 'filename']
train_ff1010[columns].to_csv('../input/train_ff1010.csv', index=False)

