import sys
sys.path.append('../siammae_pretrain')

import torch.nn as nn
from torch.utils.data import DataLoader

from models.siammae_pretrain.models_siammae import SiamMAE

import os
import json

import munch
import torchaudio
import numpy as np
import torch
import torch.nn.functional
import yaml
from PIL import Image
from torch.utils.data import Dataset

from tqdm import tqdm


class MusicavqaDataset(Dataset):
    def __init__(self, config, image_processor=None):
        self.config = config
        self.data = json.load(open(self.config.dataset.finetune_json_path, 'r', encoding='utf8')) \
            if os.path.exists(self.config.dataset.finetune_json_path) \
            else self._load_data(self.config.dataset.data_dir)
        self.image_processor = image_processor
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
            dir_audio_data = []
            dir_frame_data = []
            folder_path = os.path.join(dir_path, dir)
            for i in range(1, 61):
                audio_file = os.path.join(folder_path, f'audio_{i}.wav')
                frame_file = os.path.join(folder_path, f'frame_{i}.jpg')
                if not os.path.exists(audio_file) or not os.path.exists(frame_file):
                    continue
                dir_audio_data.append(audio_file)
                dir_frame_data.append(frame_file)
            # 如果长度不够，就用存在的最后一秒的补齐
            if len(dir_audio_data) < 60:
                for i in range(60 - len(dir_audio_data)):
                    dir_audio_data.append(dir_audio_data[-1])
                    dir_frame_data.append(dir_frame_data[-1])
            # 如果长度超过60，就截断
            if len(dir_audio_data) > 60:
                dir_audio_data = dir_audio_data[:60]
                dir_frame_data = dir_frame_data[:60]
            data.append({
                'video_id': dir,
                'audio': dir_audio_data,
                'frame': dir_frame_data
            })
        json.dump(data, open(self.config.dataset.finetune_json_path, 'w', encoding='utf-8'), indent=4)
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

    def wav_to_fbank(self, wav_filename):
        if not self.use_fbank:
            fbank = self._wav2fbank(wav_filename)
        else:
            fbank = self._fbank(wav_filename)
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
        return fbank.unsqueeze(0).unsqueeze(0)

    def __getitem__(self, index):
        data = self.data[index]
        audio_feats, frame_feat, video_id = self.av_to_pt(data)
        return {
            'video_id': video_id,
            'audio_feats': audio_feats,
            'frame_feats': frame_feat,
        }

    def av_to_pt(self, data):
        video_id = data['video_id']
        audio_files = data['audio']
        frame_files = data['frame']
        audio_feats = []
        for audio_file in audio_files:
            audio_feat = self.wav_to_fbank(audio_file)
            if self.roll_mag_aug:
                audio_feat = self._roll_mag_aug(audio_feat)
            audio_feats.append(audio_feat)
        audio_feats = torch.cat(audio_feats, dim=0)
        frame_feats = [Image.open(frame_file) for frame_file in frame_files]
        frame_feats = self.image_processor(images=frame_feats, return_tensors="pt").pixel_values
        return audio_feats, frame_feats, video_id

    def collate_fn(self, batch):
        return batch[0]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    config = yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    config = munch.munchify(config)

    model = SiamMAE(config)
    model.load_state_dict(torch.load('E:/Research/AVQA/models/siammae_pretrain/ckp/model_10.pth'))
    model = model.cuda()
    vit = model.image_mae_encoder
    audio_mae = model.audio_mae

    musicavqa_dataset = MusicavqaDataset(config, model.image_processor)
    pretrain_dataloader = DataLoader(
            musicavqa_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=musicavqa_dataset.collate_fn
    )

    avg_pool = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3))
    for data in tqdm(pretrain_dataloader):
        video_id = data['video_id']
        audio_feats = data['audio_feats'].cuda()
        frame_feats = data['frame_feats'].cuda()
        with torch.no_grad():
            audio_feats = audio_mae.forward_encoder_no_mask(audio_feats).detach().cpu()
            audio_feats_mean = torch.mean(audio_feats, dim=1).squeeze(1)
            # 对audio feats最后一维进行平均池化，[B, N, 768] -> [B, N, 256]
            B, N, _ = audio_feats.shape
            audio_feats = audio_feats.view(B, N, 256, 3)
            audio_feats = avg_pool(audio_feats)
            audio_feats = audio_feats.view(B, N, 256)

            frame_feats = vit(frame_feats, apply_mask=False).last_hidden_state.detach().cpu()
            frame_feats_mean = torch.mean(frame_feats, dim=1).squeeze(1)
            # 对frame feats最后一维进行平均池化，[B, N, 768] -> [B, N, 256]
            B, N, _ = frame_feats.shape
            frame_feats = frame_feats.view(B, N, 256, 3)
            frame_feats = avg_pool(frame_feats)
            frame_feats = frame_feats.view(B, N, 256)

            torch.save({
                'audio_feats': audio_feats,
                'frame_feats': frame_feats,
                'audio_feats_mean': audio_feats_mean,
                'frame_feats_mean': frame_feats_mean,
            }, os.path.join(config.dataset.finetune_pt_dir, video_id + '.pt'))
            # break
