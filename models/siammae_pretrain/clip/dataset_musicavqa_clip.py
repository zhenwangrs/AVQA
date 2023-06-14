# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# AST: https://github.com/YuanGongND/ast
# --------------------------------------------------------
import csv, os, sys
import json

import munch
import torchaudio
import numpy as np
import torch
import torch.nn.functional
import yaml
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
import torch.distributed as dist
import random
import math

from tqdm import tqdm
from transformers import AutoImageProcessor


class MusicavqaDataset(Dataset):
    def __init__(self, config, image_processor=None, clip_image_processor=None):
        self.config = config
        self.data = json.load(open(self.config.dataset.data_json_path))
        self.image_processor = image_processor
        self.clip_image_processor = clip_image_processor
        self.use_fbank = self.config.dataset.use_fbank
        self.fbank_dir = self.config.dataset.fbank_dir
        self.melbins = self.config.dataset.num_mel_bins
        self.freqm = self.config.dataset.freqm
        self.timem = self.config.dataset.timem
        print('using following mask: {:d} freq, {:d} time'.format(self.config.dataset.freqm, self.config.dataset.timem))

        self.norm_mean = self.config.dataset.norm_mean
        self.norm_std = self.config.dataset.norm_std
        print('mean {:.3f} and std {:.3f}'.format(self.norm_mean, self.norm_std))

        self.roll_mag_aug = self.config.dataset.roll_mag_aug
        print(f'size of dataset {self.__len__()}')

    def _load_data(self, dir_path):
        dirs = os.listdir(dir_path)
        data = []
        for dir in tqdm(dirs):
            dir = os.path.join(dir_path, dir)
            files = os.listdir(dir)
            wav_files = [file for file in files if file.endswith('.wav')]
            total_wav = len(wav_files)
            for i in range(1, total_wav + 1):
                wav_file = os.path.join(dir, f'audio_{i}.wav')
                waveform, sr = torchaudio.load(wav_file)
                if waveform.shape[1] < 400:
                    continue
                frame1_file = os.path.join(dir, f'frame_{i-1}.jpg')
                frame2_file = os.path.join(dir, f'frame_{i}.jpg')
                if not os.path.exists(frame1_file) or not os.path.exists(frame2_file):
                    continue
                data.append({
                    'wav': wav_file,
                    'frame1': frame1_file,
                    'frame2': frame2_file
                })
        return data

    def _roll_mag_aug(self, waveform):
        waveform = waveform.numpy()
        idx = np.random.randint(len(waveform))
        rolled_waveform = np.roll(waveform, idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform * mag)

    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        # 498 128, 998, 128
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
            frame_shift=10
        )
        target_length = self.config.dataset.target_length
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def _fbank(self, filename):
        fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav', '.npy'))
        fbank = np.load(fn1)
        return torch.from_numpy(fbank)

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]

        if not self.use_fbank:
            fbank = self._wav2fbank(os.path.join(self.config.dataset.data_dir, datum['wav']))
        else:
            fbank = self._fbank(os.path.join(self.config.dataset.data_dir, datum['wav']))
        # SpecAug for training (not for eval)
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0, 1).unsqueeze(0)  # 1, 128, 1024 (...,freq,time)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)  # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1)  # time, freq
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]

        return fbank.unsqueeze(0).unsqueeze(0), datum['frame1'], datum['frame2']

    def collate_fn(self, batch):
        audio, frame1_paths, frame2_paths = zip(*batch)
        audio_feat = torch.cat(audio, dim=0)
        frame1s = [Image.open(os.path.join(self.config.dataset.data_dir, image_path)) for image_path in frame1_paths]
        frame2s = [Image.open(os.path.join(self.config.dataset.data_dir, image_path)) for image_path in frame2_paths]
        frame1_feats = self.image_processor(images=frame1s, return_tensors="pt").pixel_values
        frame2_feats = self.image_processor(images=frame2s, return_tensors="pt").pixel_values
        # frame2_clip_feats = []
        # for fram2 in frame2s:
        #     frame2_clip_feats.append(self.clip_image_processor(fram2).unsqueeze(0))
        # frame2_clip_feats = torch.cat(frame2_clip_feats, dim=0)
        frame1_clip_feats = self.clip_image_processor(images=frame1s, return_tensors="pt").pixel_values
        return audio_feat, frame1_feats, frame2_feats, frame1_clip_feats

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    config = yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    config = munch.munchify(config)

    image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    dataset = MusicavqaDataset(config, image_processor)
    json.dump(dataset._load_data(config.dataset.data_dir), open(config.dataset.data_json_path, 'w', encoding='utf-8'), indent=4)

    # pretrain_dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True,
    #     collate_fn=dataset.collate_fn
    # )
    # for audio, pixel_values in tqdm(pretrain_dataloader):
    #     break
