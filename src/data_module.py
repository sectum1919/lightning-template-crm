import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random

class AudioDataset(Dataset):
    def __init__(
        self,
        data_path,
        filelist_path,
        max_frames=500,
        val_max_frames=500,
        phase='train',
        predict=False,
        audio_sr: int=16000,
        n_fft: int=400,
        win_length: int=400,
        hop_length: int=160,
        f_min: int=0,
        f_max: int=8000,
        n_mels: int=80,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.filelist_path = filelist_path
        self.max_frames = max_frames
        self.phase = phase
        self.predict = predict
        self.audio_sr = audio_sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.val_max_frames = val_max_frames
        self.load_data()
        
    def load_data(self):
        self.data = []
        # load file list
        with open(os.path.join(self.filelist_path, f'{self.phase}.csv')) as fp:
            lines = fp.readlines()
            data_root = lines[0].strip('\n')
            for line in lines[1:]:
                uid, audio, sample_count, label = line.strip('\n').split('\t')
                if self.phase == 'valid':
                    if int(sample_count) > self.val_max_frames:
                        print('discard too long')
                        continue
                afn = os.path.join(data_root, audio)
                if os.path.exists(afn):
                    aud = self._load_aud(afn) # (T, )
                    self.data.append((aud, uid, label))
        print(len(self.data))
        
    def __getitem__(self, idx):
        if self.predict:
            return self._getitem_predict(idx)
        return self._getitem_train(idx)
            
    def _getitem_train(self, idx):
        (aud, uid, label) = self.data[idx]
        aud_len = aud.shape[0]
        if aud_len > self.max_frames:
            start = random.randint(0, aud_len-self.max_frames)
            torch_aud = torch.Tensor(aud[start:start+self.max_frames]).float()
            aud_len = self.max_frames
        else:
            torch_aud = torch.Tensor(aud).float()
        return {
            'id': idx,
            'uid': uid,
            'label': label,
            'aud': torch_aud,# (T, )
            'aud_len': aud_len,
        }
           
    def _getitem_predict(self, idx):
        (aud, uid, label) = self.data[idx]
        aud_len = aud.shape[0]
        torch_aud = torch.Tensor(aud).float()
        return {
            'id': idx,
            'uid': uid,
            'label': label,
            'aud': torch_aud,# (T, )
            'aud_len': aud_len,
        }
    
    def _load_aud(self, aud_path):
        waveform, samplerate = torchaudio.load(aud_path, normalize=True, channels_first=True)
        # the "normalize" here means convert to flost32, not volume normalization
        if samplerate != self.audio_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=samplerate, new_freq=self.audio_sr)
        return waveform[0] # only first channel

    def collater(self, samples):
        # uids and labels
        uids = [s['uid'] for s in samples]
        label = [s['label'] for s in samples]
        # collate aud
        aud_source = [s['aud'] for s in samples]
        auds_lengths = torch.LongTensor(np.array([s['aud_len'] for s in samples]))
        max_aud_length = max([s['aud_len'] for s in samples])
        collated_auds = torch.FloatTensor(len(uids), max_aud_length).fill_(0)
        for i, c in enumerate(aud_source):
            collated_auds[i][:c.shape[0]] = c

        batch = {
            'id': torch.LongTensor([s['id'] for s in samples]),
            'uid': uids,
            'label': label,
            'input': collated_auds,
            'aud_len': auds_lengths,
        }
        return batch
    
    def __len__(self):
        return len(self.data)
    
    
class DataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.num_workers = cfg.num_workers
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.val_batch_size = cfg.val_batch_size
        self.persistent_workers = True
        
        self.trainset = AudioDataset(
            phase='train',
            predict = False,
            data_path = self.cfg.data_path,
            filelist_path = self.cfg.filelist_path,
            max_frames = self.cfg.max_frames,
        )
        self.valset = AudioDataset(
            phase='valid', 
            predict = False,
            data_path = self.cfg.data_path,
            filelist_path = self.cfg.filelist_path,
            max_frames = self.cfg.max_frames,
            val_max_frames=self.cfg.val_max_frames,
        )
        self.testset = AudioDataset(
            phase='test', 
            predict = True,
            data_path = self.cfg.data_path,
            filelist_path = self.cfg.filelist_path,
            max_frames = self.cfg.max_frames,
            val_max_frames=self.cfg.val_max_frames,
        )

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          collate_fn=self.trainset.collater,
                          persistent_workers=self.persistent_workers,
                          )

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=self.valset.collater,
                          )

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=self.testset.collater,
                          )