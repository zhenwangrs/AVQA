import open_clip
from open_clip import ClipLoss

import torch
import torch.nn as nn
import platform

plat = platform.system().lower()

if plat == 'windows':
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

from transformers import ViTMAEModel, AutoImageProcessor, CLIPVisionModel, AutoProcessor, CLIPModel

from models.siammae_pretrain.model_vit_crossmae import ViTMAEForPreTraining
from models.siammae_pretrain import models_audio_crossmae
from models.siammae_pretrain.utils import load_encoder_from_pretained_audiomae, load_decoder_from_pretrained_vitmae


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
        hidden_size = 768
        self.audio_mae = load_encoder_from_pretained_audiomae(self.audio_mae, config.model.audiomae_pretrained_pth_path)
        self.audio_mae = load_decoder_from_pretrained_vitmae(self.audio_mae)
        self.audio_mae_proj = nn.Linear(768, hidden_size)

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        self.image_mae_for_pretraining = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        self.image_mae_encoder = self.image_mae_for_pretraining.vit
        self.image_mae_decoder_embed = self.image_mae_for_pretraining.decoder.decoder_embed
        self.image_mae_proj = nn.Linear(768, hidden_size)

        used_clip = "openai/clip-vit-large-patch14"
        self.clip = CLIPModel.from_pretrained(used_clip)
        self.clip_visual = self.clip.vision_model
        self.clip_visual_proj = self.clip.visual_projection
        self.clip_text_proj = self.clip.text_projection
        # 冻结clip的参数
        for param in self.clip.parameters():
            param.requires_grad = False

        self.clip_loss_fn = ClipLoss()
        # self.clip_image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_image_processor = AutoProcessor.from_pretrained(used_clip)
        self.logit_scale = torch.nn.Parameter(torch.tensor(2.6592), requires_grad=True)

    def forward(self, audio_feats, frame1_feats, frame2_feats, frame1_clip_feats):
        frame1_encoder_feat = self.image_mae_encoder(frame1_feats, apply_mask=False).last_hidden_state
        frame1_decoder_feat = self.image_mae_decoder_embed(frame1_encoder_feat)

        frame2_decoder_output = self.image_mae_for_pretraining(
            frame2_feats, frame1_decoder_feat, frame1_decoder_feat, return_dict=True)
        frame2_decoder_feat = frame2_decoder_output.decoder_last_hidden_state
        loss_frame2_recon = frame2_decoder_output.loss

        loss_audio_recon, pred, audio_encoder_feat, audio_decoder_last_hidden_state = \
            self.audio_mae(audio_feats, frame2_decoder_feat, frame2_decoder_feat)

        frame1_clip_feats = self.clip_visual(frame1_clip_feats).last_hidden_state
        frame1_clip_feats = self.clip_visual_proj(frame1_clip_feats)
        frame1_clip_feats_mean = torch.mean(frame1_clip_feats, dim=1)
        frame1_clip_feats_mean = frame1_clip_feats_mean / frame1_clip_feats_mean.norm(dim=-1, keepdim=True)

        frame1_encoder_feat = self.image_mae_proj(frame1_encoder_feat)
        frame1_encoder_feat_mean = torch.mean(frame1_encoder_feat, dim=1)
        frame1_encoder_feat_mean = frame1_encoder_feat_mean / frame1_encoder_feat_mean.norm(dim=-1, keepdim=True)
        clip_loss_frame = self.clip_loss_fn(frame1_encoder_feat_mean, frame1_clip_feats_mean, self.logit_scale.exp())

        audio_no_mask_feats = self.audio_mae.forward_encoder_no_mask(audio_feats)
        audio_no_mask_feats = self.audio_mae_proj(audio_no_mask_feats)
        audio_encoder_feat_mean = torch.mean(audio_no_mask_feats, dim=1)
        audio_encoder_feat_mean = audio_encoder_feat_mean / audio_encoder_feat_mean.norm(dim=-1, keepdim=True)
        clip_loss_audio = self.clip_loss_fn(audio_encoder_feat_mean, frame1_clip_feats_mean, self.logit_scale.exp())

        # loss_frame2_recon = torch.tensor(0.0).cuda()
        # loss_audio_recon = torch.tensor(0.0).cuda()
        # clip_loss_frame = torch.tensor(0.0).cuda()
        # clip_loss_audio = torch.tensor(0.0).cuda()

        total_loss = loss_frame2_recon + loss_audio_recon + clip_loss_frame + clip_loss_audio
        return total_loss, loss_frame2_recon, loss_audio_recon, clip_loss_frame, clip_loss_audio

    def extract_feature(self, audio_feat, frame1_feats, frame2_feats):
        frame1_encoder_feat = self.image_mae_encoder(frame1_feats, apply_mask=False).last_hidden_state
        frame1_encoder_feat = self.image_mae_decoder_embed(frame1_encoder_feat)
        frame2_decoder_output = self.image_mae_for_pretraining(frame2_feats, frame1_encoder_feat, frame1_encoder_feat)
        frame2_decoder_feat = frame2_decoder_output.decoder_last_hidden_state
        _, _, _, audio_decoder_last_hidden_state = self.audio_mae(audio_feat, frame2_decoder_feat, frame2_decoder_feat)
        return frame2_decoder_feat, audio_decoder_last_hidden_state


if __name__ == '__main__':
    model = torch.load('/lib/finetuned.pth')
    pass
