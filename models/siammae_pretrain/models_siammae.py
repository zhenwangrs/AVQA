import torch
import torch.nn as nn
import pathlib

from transformers import ViTMAEModel, AutoImageProcessor

from model_vit_crossmae import ViTMAEForPreTraining
import models_audio_crossmae
from utils import load_encoder_from_pretained_audiomae, load_decoder_from_pretrained_vitmae

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


class SiamMAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_mae = models_audio_crossmae.__dict__[config.model.model_name](
            norm_pix_loss=config.model.norm_pix_loss,
            in_chans=config.model.in_channels,
            audio_exp=config.model.audio_exp,
            img_size=(config.dataset.target_length, 128),
            alpha=config.model.alpha,
            mode=config.model.mode,
            use_custom_patch=config.model.use_custom_patch,
            split_pos=config.model.split_pos,
            pos_trainable=config.model.pos_trainable,
            use_nce=config.model.use_nce,
            decoder_mode=config.model.decoder_mode,
            mask_2d=config.model.mask_2d,
            mask_t_prob=config.model.mask_t_prob,
            mask_f_prob=config.model.mask_f_prob,
            no_shift=config.model.no_shift,
        )
        self.audio_mae = load_encoder_from_pretained_audiomae(self.audio_mae, config.model.audiomae_pretrained_pth_path)
        self.audio_mae = load_decoder_from_pretrained_vitmae(self.audio_mae)

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        self.image_mae_for_pretraining = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        self.image_mae_encoder = self.image_mae_for_pretraining.vit
        self.image_mae_decoder_embed = self.image_mae_for_pretraining.decoder.decoder_embed

    def forward(self, audio_feat, frame1_feats, frame2_feats):
        frame1_encoder_feat = self.image_mae_encoder(frame1_feats, apply_mask=False).last_hidden_state
        frame1_encoder_feat = self.image_mae_decoder_embed(frame1_encoder_feat)
        frame2_decoder_output = self.image_mae_for_pretraining(frame2_feats, frame1_encoder_feat, frame1_encoder_feat)
        frame2_decoder_feat = frame2_decoder_output.decoder_last_hidden_state
        loss_frame2_recon = frame2_decoder_output.loss
        loss_audio_recon, pred, mask, _ = self.audio_mae(audio_feat, frame2_decoder_feat, frame2_decoder_feat)
        total_loss = loss_frame2_recon + loss_audio_recon
        return total_loss, loss_frame2_recon, loss_audio_recon

    def extract_feature(self, audio_feat, frame1_feats, frame2_feats):
        frame1_encoder_feat = self.image_mae_encoder(frame1_feats, apply_mask=False).last_hidden_state
        frame1_encoder_feat = self.image_mae_decoder_embed(frame1_encoder_feat)
        frame2_decoder_output = self.image_mae_for_pretraining(frame2_feats, frame1_encoder_feat, frame1_encoder_feat)
        frame2_decoder_feat = frame2_decoder_output.decoder_last_hidden_state
        loss_frame2_recon = frame2_decoder_output.loss
        _, _, _, audio_decoder_last_hidden_state = self.audio_mae(audio_feat, frame2_decoder_feat, frame2_decoder_feat)
        return frame2_decoder_feat, audio_decoder_last_hidden_state


if __name__ == '__main__':
    model = torch.load('E:/Research/AVQA/lib/finetuned.pth')
    pass
