import os

import torchaudio
import numpy as np
import torch
import torch.nn.functional
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class AV2PT(Dataset):
    def __init__(self, config, image_processor):
        self.config = config
        self.image_processor = image_processor
        self.use_fbank = self.config.dataset.use_fbank
        self.fbank_dir = self.config.dataset.fbank_dir
        self.melbins = self.config.dataset.num_mel_bins
        self.freqm = self.config.dataset.freqm
        self.timem = self.config.dataset.timem
        self.norm_mean = self.config.dataset.norm_mean
        self.norm_std = self.config.dataset.norm_std
        self.roll_mag_aug = self.config.dataset.roll_mag_aug

    def _roll_mag_aug(self, waveform):
        waveform = waveform.numpy()
        idx = np.random.randint(len(waveform))
        rolled_waveform = np.roll(waveform, idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform * mag)

    def _wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        if waveform.shape[1] < 400:
            # print(f'padding {filename}')
            waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[1]))
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

    def wav_to_fbank(self, wav_filename):
        fbank = self._wav2fbank(wav_filename)
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

    def av_to_pt(self, audio_files, frame_files, random_augmentation=False):
        audio_feats = []
        for audio_file in audio_files:
            audio_feat = self.wav_to_fbank(audio_file)
            if self.roll_mag_aug:
                audio_feat = self._roll_mag_aug(audio_feat)
            audio_feats.append(audio_feat)
        audio_feats = torch.cat(audio_feats, dim=0)
        frame_feats = [Image.open(frame_file) for frame_file in frame_files]
        if random_augmentation:
            frame_feats = [self.random_augmentation(frame_feat) for frame_feat in frame_feats]
        frame_feats = self.image_processor(images=frame_feats, return_tensors="pt").pixel_values
        return audio_feats, frame_feats

    def random_augmentation(self, image):
        # 保存原图
        # image.save('origin.jpg')
        # 转换为Tensor
        tensor_image = transforms.ToTensor()(image)

        # 随机水平翻转
        # if torch.rand(1) < 0.5:
        #     tensor_image = transforms.functional.hflip(tensor_image)

        # 随机垂直翻转
        if torch.rand(1) < 0.5:
            tensor_image = transforms.functional.vflip(tensor_image)
        else:
            # 随机旋转
            angle = transforms.RandomRotation.get_params([-45, 45])
            tensor_image = transforms.functional.rotate(tensor_image, angle)

        if torch.rand(1) < 0.5:
            # 随机调整亮度、对比度、饱和度和色调
            brightness_factor = torch.rand(1).item() * 0.5 + 0.5  # (0.5, 1.0)
            contrast_factor = torch.rand(1).item() * 0.5 + 0.5  # (0.5, 1.0)
            saturation_factor = torch.rand(1).item() * 0.5 + 0.5  # (0.5, 1.0)
            hue_factor = torch.rand(1).item() * 0.5 - 0.25  # (-0.25, 0.25)
            tensor_image = transforms.functional.adjust_brightness(tensor_image, brightness_factor)
            tensor_image = transforms.functional.adjust_contrast(tensor_image, contrast_factor)
            tensor_image = transforms.functional.adjust_saturation(tensor_image, saturation_factor)
            tensor_image = transforms.functional.adjust_hue(tensor_image, hue_factor)

        if torch.rand(1) < 0.5:
            # 随机裁剪
            i, j, h, w = transforms.RandomCrop.get_params(tensor_image, output_size=(image.height, image.width))
            tensor_image = transforms.functional.crop(tensor_image, i, j, h, w)

        # 转换回PIL图像
        augmented_image = transforms.ToPILImage()(tensor_image)
        # 保存图像
        # augmented_image.save('augmented_image.jpg')

        return augmented_image
